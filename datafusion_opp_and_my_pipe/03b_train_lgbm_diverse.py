"""Step 3b: Train diverse LightGBM variants for loss-diversity ensemble.

Trains 3 LGBM variants with different objectives/regularization to capture
different regions of the ROC curve. Per-target hill climbing in blend
assigns weight=0 to any variant that doesn't help.

Variants:
  1. lgbm_focal  — focal loss (gamma=2.0), focuses on hard boundary samples
  2. lgbm_high_spw — 2× scale_pos_weight, aggressively upweights rare positives
  3. lgbm_low_reg — low regularization + higher colsample, captures more signal

Output: checkpoints_lgbm_diverse/lgbm_diverse_predictions.npz
        (oof_focal, test_focal, oof_high_spw, test_high_spw, oof_low_reg, test_low_reg)

Runtime: ~15-30 minutes (3× base LGBM).
"""

import gc
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import lightgbm as lgb

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, log_per_target_auc, effective_number_weight

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_lgbm_diverse")

N_CPUS = os.cpu_count() or 8
PARALLEL_TARGETS = min(4, max(2, N_CPUS // 16))
THREADS_PER_MODEL = max(1, N_CPUS // PARALLEL_TARGETS)

# Base params shared across all variants (from Optuna tuning)
BASE_PARAMS = dict(
    metric="auc",
    learning_rate=0.050216,
    num_leaves=34,
    max_depth=10,
    min_child_samples=102,
    n_estimators=1500,
    subsample=0.521715,
    subsample_freq=2,
    random_state=SEED,
    verbose=-1,
)

EARLY_STOPPING_ROUNDS = 100

# ── Variant definitions ──────────────────────────────────────────

VARIANTS = {
    "focal": {
        "label": "Focal Loss (gamma=2.0)",
        "params": {
            **BASE_PARAMS,
            "objective": None,  # custom objective
            "colsample_bytree": 0.205092,
            "reg_alpha": 8.146025,
            "reg_lambda": 7.761833,
            "min_split_gain": 0.424512,
        },
        "use_focal": True,
    },
    "high_spw": {
        "label": "High scale_pos_weight (2×)",
        "params": {
            **BASE_PARAMS,
            "objective": "binary",
            "colsample_bytree": 0.205092,
            "reg_alpha": 8.146025,
            "reg_lambda": 7.761833,
            "min_split_gain": 0.424512,
        },
        "spw_multiplier": 2.0,
    },
    "low_reg": {
        "label": "Low regularization + wider colsample",
        "params": {
            **BASE_PARAMS,
            "objective": "binary",
            "colsample_bytree": 0.5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_split_gain": 0.01,
        },
    },
}


# ── Focal loss for LightGBM ─────────────────────────────────────

def focal_loss_objective(y_true, y_pred):
    """Focal loss custom objective for LightGBM.

    LightGBM signature: (y_true, y_pred) → (grad, hess).
    Down-weights easy samples, focuses on hard boundary cases.
    """
    p = np.clip(expit(y_pred), 1e-7, 1 - 1e-7)
    gamma = 2.0
    focal_weight = np.where(y_true == 1, (1 - p) ** gamma, p ** gamma)
    grad = focal_weight * (p - y_true)
    hess = focal_weight * p * (1 - p)
    hess = np.maximum(hess, 1e-7)
    return grad, hess


def focal_loss_eval(y_true, y_pred):
    """AUC eval metric for focal loss (LightGBM uses raw predictions)."""
    p = expit(y_pred)
    auc = roc_auc_score(y_true, p)
    return "auc", auc, True  # name, value, is_higher_better


# ── Training ─────────────────────────────────────────────────────

def _train_one_target(target_idx, target_name, fold_idx, threads,
                      cat_indices, variant_key, variant_config,
                      X_tr, y_tr, X_val, y_val, X_test):
    """Train a single target for one variant."""
    y_t = y_tr[:, target_idx]
    n_neg = int((y_t == 0).sum())
    n_pos = int((y_t == 1).sum())

    spw = effective_number_weight(n_pos, n_neg)
    if "spw_multiplier" in variant_config:
        spw *= variant_config["spw_multiplier"]

    params = {**variant_config["params"], "n_jobs": threads}

    use_focal = variant_config.get("use_focal", False)

    if use_focal:
        params["objective"] = focal_loss_objective
        params.pop("metric", None)
        eval_fn = focal_loss_eval
    else:
        params["scale_pos_weight"] = spw
        eval_fn = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = lgb.LGBMClassifier(**params)

        fit_kwargs = dict(
            eval_set=[(X_val, y_val[:, target_idx])],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
            categorical_feature=cat_indices,
        )
        if use_focal:
            fit_kwargs["eval_metric"] = eval_fn

        model.fit(X_tr, y_t, **fit_kwargs)

    if use_focal:
        raw_val = model.predict(X_val, raw_score=True)
        val_preds = expit(raw_val)
        raw_test = model.predict(X_test, raw_score=True)
        test_preds = expit(raw_test)
    else:
        val_preds = model.predict_proba(X_val)[:, 1]
        test_preds = model.predict_proba(X_test)[:, 1]

    return val_preds, test_preds


def train_variant(variant_key, variant_config, X_train, X_test, y_train,
                  target_cols, cat_indices, kf_splits):
    """Train one LGBM variant across all folds and targets."""
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_targets = len(target_cols)
    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds_sum = np.zeros((n_test, n_targets), dtype=np.float32)
    fold_aucs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf_splits):
        t_fold = time.time()
        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]
        fold_test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=PARALLEL_TARGETS) as executor:
            futures = {}
            for i, col in enumerate(target_cols):
                futures[executor.submit(
                    _train_one_target, i, col, fold_idx, THREADS_PER_MODEL,
                    cat_indices, variant_key, variant_config,
                    X_tr, y_tr, X_val, y_val, X_test,
                )] = i

            done_count = 0
            for future in as_completed(futures):
                i = futures[future]
                val_p, test_p = future.result()
                oof_preds[val_idx, i] = val_p
                fold_test_preds[:, i] = test_p
                done_count += 1
                if done_count % 10 == 0 or done_count == n_targets:
                    print(f"      {done_count}/{n_targets} targets done", flush=True)

        fold_auc, _ = compute_macro_auc(y_val, oof_preds[val_idx], target_cols)
        test_preds_sum += fold_test_preds
        fold_aucs.append(fold_auc)
        del X_tr, X_val, y_tr, y_val
        gc.collect()
        print(f"    Fold {fold_idx+1} AUC={fold_auc:.4f} ({(time.time()-t_fold)/60:.1f} min)", flush=True)

    test_preds_avg = test_preds_sum / len(kf_splits)
    oof_auc, _ = compute_macro_auc(y_train, oof_preds, target_cols)
    return oof_preds, test_preds_avg, oof_auc, fold_aucs


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 3b: Train Diverse LightGBM Variants")
    print("=" * 60)
    print(f"  Variants: {', '.join(VARIANTS.keys())}")
    print(f"  LightGBM: CPU mode, {PARALLEL_TARGETS} parallel targets × {THREADS_PER_MODEL} threads")

    # 1. Load features (same as base LGBM)
    print("\n[1/3] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]
    cat_indices = [feature_cols.index(c) for c in cat_feature_names]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    # Load kNN features (from 01d_knn_features.py) — no masking needed (OOF-safe)
    knn_feat_path = FEATURES_DIR / "knn_features_train.parquet"
    if knn_feat_path.exists():
        knn_train = pl.read_parquet(knn_feat_path).to_numpy().astype(np.float32)
        knn_test = pl.read_parquet(FEATURES_DIR / "knn_features_test.parquet").to_numpy().astype(np.float32)
        X_train = np.hstack([X_train, knn_train])
        X_test = np.hstack([X_test, knn_test])
        print(f"  Added {knn_train.shape[1]} kNN features")
    else:
        print(f"  kNN features not found (optional)")

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(cat_feature_names)} cat, {len(feature_cols) - len(cat_feature_names)} num")

    # 2. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "lgbm_diverse_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # 3. Pre-compute CV splits (same splits for all variants)
    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    kf_splits = list(kf.split(np.arange(X_train.shape[0]), y_train))

    # 4. Train each variant
    print(f"\n[2/3] Training {len(VARIANTS)} variants × {N_FOLDS} folds × {len(target_cols)} targets...")
    results = {}
    for vkey, vconfig in VARIANTS.items():
        print(f"\n  ── Variant: {vconfig['label']} [{vkey}] ──")
        oof, test, oof_auc, fold_aucs = train_variant(
            vkey, vconfig, X_train, X_test, y_train, target_cols, cat_indices, kf_splits
        )
        results[vkey] = (oof, test, oof_auc, fold_aucs)
        print(f"  {vkey} OOF AUC: {oof_auc:.4f}  folds: {['%.4f' % a for a in fold_aucs]}")

    # 5. Save
    print(f"\n[3/3] Saving...")
    save_dict = {}
    for vkey, (oof, test, oof_auc, fold_aucs) in results.items():
        save_dict[f"oof_{vkey}"] = oof
        save_dict[f"test_{vkey}"] = test
        save_dict[f"auc_{vkey}"] = np.array(oof_auc)
    np.savez_compressed(cache_file, **save_dict)
    print(f"  Saved: {cache_file}")

    # Summary
    print(f"\n{'='*60}")
    print("Variant Summary:")
    for vkey, (_, _, oof_auc, _) in sorted(results.items(), key=lambda x: -x[1][2]):
        print(f"  {vkey:<12s} OOF={oof_auc:.4f}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
