"""Step 3: Train LightGBM (Optuna-tuned params, 5-fold).

Loads features from features/, trains 41 per-target binary classifiers.
Targets are trained in parallel batches for speed.

Output: checkpoints_lgbm/lgbm_predictions.npz (oof_preds, test_preds)

Runtime: ~5-10 minutes.
"""

import gc
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, log_per_target_auc

FEATURES_DIR = Path("features")
SELECTED_DIR = FEATURES_DIR / "selected_features"
CHECKPOINT_DIR = Path("checkpoints_lgbm")
MODELS_DIR = CHECKPOINT_DIR / "models"

# ThreadPoolExecutor: LightGBM releases GIL during C++ training,
# so threads are truly parallel. No data copying, no pickle overhead.
N_CPUS = os.cpu_count() or 8
PARALLEL_TARGETS = min(4, max(2, N_CPUS // 16))  # 64 cores → 4 parallel, 224 → 4 (capped)
THREADS_PER_MODEL = max(1, N_CPUS // PARALLEL_TARGETS)  # 64 cores / 4 = 16 threads each

# Optuna-tuned params (L7, 30 trials)
LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.050216,
    num_leaves=34,
    max_depth=10,
    min_child_samples=102,
    n_estimators=1500,
    subsample=0.521715,
    colsample_bytree=0.205092,
    reg_alpha=8.146025,
    reg_lambda=7.761833,
    min_split_gain=0.424512,
    subsample_freq=2,
    random_state=SEED,
    verbose=-1,
    # ── Speed optimizations ──
    force_col_wise=True,       # faster for #features >> #rows/1000
    max_bin=127,               # was 63 on GPU; 127 is better quality + still fast on CPU
    min_data_in_bin=20,        # default, keep quality
    feature_pre_filter=True,   # skip features that can't improve split
)
EARLY_STOPPING_ROUNDS = 100


def _load_per_target_features(target_cols, all_feature_cols):
    """Load per-target feature selections if available.

    Returns dict {target_name: list_of_col_indices} or None if not available.
    """
    if not SELECTED_DIR.exists():
        return None
    per_target = {}
    col_to_idx = {c: i for i, c in enumerate(all_feature_cols)}
    for target in target_cols:
        fpath = SELECTED_DIR / f"{target}.json"
        if not fpath.exists():
            return None  # incomplete — fall back to all features
        with open(fpath) as f:
            selected = json.load(f)
        per_target[target] = [col_to_idx[c] for c in selected if c in col_to_idx]
    return per_target


def _train_one_target(target_idx, target_name, fold_idx, threads,
                      cat_indices, supports_cat, device_params,
                      X_tr, y_tr, X_val, y_val, X_test,
                      feat_indices=None):
    """Train a single target — runs in a thread (GIL released by LightGBM C++)."""
    # Per-target feature selection
    if feat_indices is not None:
        X_tr = X_tr[:, feat_indices]
        X_val = X_val[:, feat_indices]
        X_test = X_test[:, feat_indices]
        cat_idx_set = set(cat_indices) if cat_indices else set()
        cat_indices = [i for i, fi in enumerate(feat_indices) if fi in cat_idx_set]

    y_t = y_tr[:, target_idx]
    n_neg = (y_t == 0).sum()
    n_pos = (y_t == 1).sum()
    spw = n_neg / max(n_pos, 1)
    params = {**LGBM_PARAMS, "n_jobs": threads, "scale_pos_weight": spw, **device_params}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr[:, target_idx],
            eval_set=[(X_val, y_val[:, target_idx])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
            categorical_feature=cat_indices if supports_cat else "auto",
        )

    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    model.booster_.save_model(str(MODELS_DIR / f"{target_name}_fold{fold_idx}.lgb"))
    importance = model.booster_.feature_importance(importance_type="gain")

    return val_preds, test_preds, importance


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 3: Train LightGBM (5-fold × 41 targets)")
    print("=" * 60)

    # Force CPU: 224 cores with parallel targets is faster than sequential GPU
    device_params, supports_cat = {}, True
    print(f"  LightGBM: CPU mode, {PARALLEL_TARGETS} parallel targets × {THREADS_PER_MODEL} threads")

    # 1. Load features
    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    all_feature_cols = meta["feature_names"]  # full list — never changes
    feature_cols = list(all_feature_cols)
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]
    cat_indices = [feature_cols.index(c) for c in cat_feature_names]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(cat_feature_names)} cat, {len(feature_cols) - len(cat_feature_names)} num")

    # Load per-target feature selections (if available from 01b_select_features.py)
    per_target_feats = _load_per_target_features(target_cols, feature_cols)
    if per_target_feats:
        n_feats = [len(v) for v in per_target_feats.values()]
        print(f"  Per-target selection: {min(n_feats)}-{max(n_feats)} features/target")
    else:
        print(f"  Per-target selection: not available, using all {len(feature_cols)} features")

    # 2. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "lgbm_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # 3. Train
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_targets = len(target_cols)
    oof_preds = np.zeros((n_train, n_targets))
    test_preds_sum = np.zeros((n_test, n_targets))
    fold_aucs = []
    importance_sum = np.zeros((len(feature_cols), n_targets), dtype=np.float64)

    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n[2/4] Training {N_FOLDS}-Fold × {n_targets} targets...", flush=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        t_fold = time.time()
        print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS} "
              f"(train={len(tr_idx):,}, val={len(val_idx):,}) ──", flush=True)

        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]
        fold_test_preds = np.zeros((n_test, n_targets))

        # Parallel training across targets using threads
        # LightGBM releases GIL → threads are truly parallel, zero copy overhead
        with ThreadPoolExecutor(max_workers=PARALLEL_TARGETS) as executor:
            futures = {}
            for i, col in enumerate(target_cols):
                fi = per_target_feats.get(col) if per_target_feats else None
                futures[executor.submit(
                    _train_one_target, i, col, fold_idx, THREADS_PER_MODEL,
                    cat_indices, supports_cat, device_params,
                    X_tr, y_tr, X_val, y_val, X_test,
                    feat_indices=fi,
                )] = (i, fi)

            done_count = 0
            for future in as_completed(futures):
                i, fi = futures[future]
                val_p, test_p, imp = future.result()
                oof_preds[val_idx, i] = val_p
                fold_test_preds[:, i] = test_p
                # Scatter per-target importance back to full feature array
                if fi is not None:
                    for j, feat_idx in enumerate(fi):
                        if j < len(imp):
                            importance_sum[feat_idx, i] += imp[j]
                else:
                    importance_sum[:, i] += imp
                done_count += 1
                if done_count % 10 == 0 or done_count == n_targets:
                    print(f"    {done_count}/{n_targets} targets done", flush=True)

        fold_auc, _ = compute_macro_auc(y_val, oof_preds[val_idx], target_cols)
        test_preds_sum += fold_test_preds
        fold_aucs.append(fold_auc)
        del X_tr, X_val, y_tr, y_val; gc.collect()
        print(f"  Fold {fold_idx+1} AUC={fold_auc:.4f} ({(time.time()-t_fold)/60:.1f} min)", flush=True)

    test_preds_avg = test_preds_sum / N_FOLDS

    # 4. Results
    oof_auc, per_target_aucs = compute_macro_auc(y_train, oof_preds, target_cols)
    print(f"\n[3/4] Results:")
    print(f"  Per-fold AUC: {['%.4f' % a for a in fold_aucs]}")
    print(f"  OOF Macro ROC-AUC: {oof_auc:.4f}")
    log_per_target_auc(per_target_aucs, y_train, target_cols)

    # Save predictions
    np.savez(cache_file, oof_preds=oof_preds, test_preds=test_preds_avg,
             fold_aucs=np.array(fold_aucs))
    print(f"  Saved: {cache_file}")

    # Save averaged feature importances — always keyed by full feature list
    importance_avg = importance_sum / N_FOLDS
    full_imp = importance_avg

    imp_path = CHECKPOINT_DIR / "feature_importances.json"
    imp_data = {
        "feature_names": all_feature_cols,
        "target_cols": target_cols,
        "importances_per_target": {
            col: dict(sorted(
                zip(all_feature_cols, full_imp[:, i].tolist()),
                key=lambda x: x[1], reverse=True
            ))
            for i, col in enumerate(target_cols)
        },
        "mean_importance": dict(sorted(
            zip(all_feature_cols, full_imp.mean(axis=1).tolist()),
            key=lambda x: x[1], reverse=True
        )),
    }
    with open(imp_path, "w") as f:
        json.dump(imp_data, f, indent=2)
    print(f"  Saved: {imp_path}")
    print(f"  Models: {MODELS_DIR}/ ({N_FOLDS * n_targets} .lgb files)")

    # Save submission
    print("\n[4/4] Saving submission...")
    from utils import verify_submission
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_feat["customer_id"]}).hstack(
        pl.DataFrame(test_preds_avg.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/lgbm.parquet")
    print(f"  Saved: submissions/lgbm.parquet")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={oof_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
