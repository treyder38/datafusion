"""Step 6: Train CatBoost (5-fold x 41 targets).

Loads features from features/, trains 41 per-target binary classifiers.
CatBoost handles categoricals natively — no label encoding needed.

Output: checkpoints_catboost/cb_predictions.npz (oof_preds, test_preds)

Runtime: ~1-3 hours (GPU), ~4-8 hours (CPU).
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, log_per_target_auc

FEATURES_DIR = Path("features")
SELECTED_DIR = FEATURES_DIR / "selected_features"
CHECKPOINT_DIR = Path("checkpoints_catboost")
MODELS_DIR = CHECKPOINT_DIR / "models"

# CatBoost hyperparameters — load Optuna-tuned params if available
CB_PARAMS_DEFAULT = dict(
    iterations=5000,
    early_stopping_rounds=200,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
)

def load_cb_params():
    """Load best params from Optuna tuning, fall back to defaults."""
    params_path = CHECKPOINT_DIR / "best_params.json"
    base = dict(CB_PARAMS_DEFAULT)
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"  Loaded tuned params from {params_path}")
        # Map Optuna params into CatBoost params
        base.update({
            k: v for k, v in tuned.items()
            if k not in ("bootstrap_type",)
        })
        # Bootstrap params
        bt = tuned.get("bootstrap_type", "Bayesian")
        base["bootstrap_type"] = bt
        if bt == "Bayesian" and "bagging_temperature" in tuned:
            base["bagging_temperature"] = tuned["bagging_temperature"]
        elif bt == "MVS" and "subsample" in tuned:
            base["subsample"] = tuned["subsample"]
        for k, v in tuned.items():
            print(f"    {k}: {v}")
    else:
        print(f"  No tuned params found, using defaults")
    return base


def detect_task_type():
    """Detect GPU availability for CatBoost."""
    try:
        cb = CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=0)
        cb.fit([[0, 0], [1, 1]], [0, 1])
        return "GPU", "0"
    except Exception:
        return "CPU", None


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 6: Train CatBoost (5-fold x 41 targets)")
    print("=" * 60)

    task_type, devices = detect_task_type()
    print(f"  CatBoost task_type: {task_type}")

    cb_params = load_cb_params()

    # 1. Load features
    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    # CatBoost handles NaN and categoricals natively — use pandas
    X_train = train_feat.select(feature_cols).to_pandas()
    X_test = test_feat.select(feature_cols).to_pandas()
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    # Cast cat columns to str for CatBoost
    for col in cat_feature_names:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(cat_feature_names)} cat, {len(feature_cols) - len(cat_feature_names)} num")

    # Load per-target feature selections (if available from 01b_select_features.py)
    per_target_feats = {}
    if SELECTED_DIR.exists():
        for target in target_cols:
            fpath = SELECTED_DIR / f"{target}.json"
            if fpath.exists():
                with open(fpath) as f:
                    per_target_feats[target] = [c for c in json.load(f) if c in feature_cols]
        if len(per_target_feats) == len(target_cols):
            n_feats = [len(v) for v in per_target_feats.values()]
            print(f"  Per-target selection: {min(n_feats)}-{max(n_feats)} features/target")
        else:
            per_target_feats = {}
            print(f"  Per-target selection: not available, using all {len(feature_cols)} features")
    else:
        print(f"  Per-target selection: not available, using all {len(feature_cols)} features")

    # 2. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "cb_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # 3. Train
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_targets = len(target_cols)
    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds_sum = np.zeros((n_test, n_targets), dtype=np.float32)
    fold_aucs = []
    # Accumulate feature importances across folds
    importance_sum = np.zeros((len(feature_cols), n_targets), dtype=np.float64)

    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n[2/4] Training {N_FOLDS}-Fold x {n_targets} targets...", flush=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        t_fold = time.time()
        print(f"\n  -- Fold {fold_idx+1}/{N_FOLDS} "
              f"(train={len(tr_idx):,}, val={len(val_idx):,}) --", flush=True)

        fold_test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

        for i, col in enumerate(target_cols):
            y = y_train[:, i]

            params = dict(cb_params)
            params["random_seed"] = SEED + fold_idx
            params["auto_class_weights"] = "Balanced"
            if task_type == "GPU":
                params["task_type"] = "GPU"
                params["devices"] = devices

            # Per-target feature selection
            if col in per_target_feats:
                sel_cols = per_target_feats[col]
                sel_cats = [c for c in cat_feature_names if c in sel_cols]
                X_tr_t = X_train[sel_cols].iloc[tr_idx]
                X_va_t = X_train[sel_cols].iloc[val_idx]
                X_te_t = X_test[sel_cols]
            else:
                sel_cols = feature_cols
                sel_cats = cat_feature_names
                X_tr_t = X_train.iloc[tr_idx]
                X_va_t = X_train.iloc[val_idx]
                X_te_t = X_test

            # Detect pseudo-categorical numeric features (nunique < 50)
            # CatBoost uses Ordered Target Statistics for categoricals — superior to threshold splits
            for col_name in sel_cols:
                if col_name not in sel_cats:  # Only check numeric features
                    n_unique = X_tr_t[col_name].nunique()
                    if n_unique < 50:
                        sel_cats.append(col_name)

            tr_pool = Pool(X_tr_t, y[tr_idx], cat_features=sel_cats)
            va_pool = Pool(X_va_t, y[val_idx], cat_features=sel_cats)
            te_pool = Pool(X_te_t, cat_features=sel_cats)

            cb = CatBoostClassifier(**params)
            cb.fit(tr_pool, eval_set=va_pool, verbose=0)

            oof_preds[val_idx, i] = cb.predict_proba(va_pool)[:, 1].astype(np.float32)
            fold_test_preds[:, i] = cb.predict_proba(te_pool)[:, 1].astype(np.float32)

            # Save model weights
            model_path = MODELS_DIR / f"{col}_fold{fold_idx}.cbm"
            cb.save_model(str(model_path))

            # Accumulate feature importances (scatter back to full array)
            imp = cb.get_feature_importance()
            if col in per_target_feats:
                for j, sc in enumerate(sel_cols):
                    if sc in feature_cols:
                        importance_sum[feature_cols.index(sc), i] += imp[j]
            else:
                importance_sum[:, i] += imp

            del cb, tr_pool, va_pool, te_pool; gc.collect()
            if (i + 1) % 10 == 0 or i == n_targets - 1:
                print(f"    {i+1}/{n_targets} targets done", flush=True)

        fold_auc, _ = compute_macro_auc(y_train[val_idx], oof_preds[val_idx], target_cols)
        test_preds_sum += fold_test_preds
        fold_aucs.append(fold_auc)
        del fold_test_preds; gc.collect()
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

    # Save averaged feature importances
    importance_avg = importance_sum / N_FOLDS
    imp_path = CHECKPOINT_DIR / "feature_importances.json"
    imp_data = {
        "feature_names": feature_cols,
        "target_cols": target_cols,
        "importances_per_target": {
            col: dict(sorted(
                zip(feature_cols, importance_avg[:, i].tolist()),
                key=lambda x: x[1], reverse=True
            ))
            for i, col in enumerate(target_cols)
        },
        "mean_importance": dict(sorted(
            zip(feature_cols, importance_avg.mean(axis=1).tolist()),
            key=lambda x: x[1], reverse=True
        )),
    }
    with open(imp_path, "w") as f:
        json.dump(imp_data, f, indent=2)
    print(f"  Saved: {imp_path}")
    print(f"  Models: {MODELS_DIR}/ ({N_FOLDS * n_targets} .cbm files)")

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
    submit.write_parquet("submissions/catboost.parquet")
    print(f"  Saved: submissions/catboost.parquet")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={oof_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
