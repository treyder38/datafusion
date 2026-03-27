"""Step 4: Train XGBoost (4-fold x 41 targets).

Loads features from features/, trains 41 per-target binary classifiers.
Targets are trained in parallel batches for speed (XGBoost releases GIL).

Output: checkpoints_xgboost/xgb_predictions.npz (oof_preds, test_preds)

Runtime: ~10-20 minutes.
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
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, log_per_target_auc, effective_number_weight

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_xgboost")
MODELS_DIR = CHECKPOINT_DIR / "models"

N_CPUS = os.cpu_count() or 8
PARALLEL_TARGETS = min(4, max(2, N_CPUS // 16))
THREADS_PER_MODEL = max(1, N_CPUS // PARALLEL_TARGETS)

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=50,
    n_estimators=2000,
    subsample=0.7,
    colsample_bytree=0.3,
    reg_alpha=5.0,
    reg_lambda=5.0,
    gamma=0.5,
    random_state=SEED,
    verbosity=0,
)
EARLY_STOPPING_ROUNDS = 100

# Rarity-tier hyperparameters based on positive rate
def get_rarity_tier_params(n_pos):
    """Get HP tier based on number of positive samples."""
    if n_pos < 500:  # Ultra-rare
        return {"learning_rate": 0.02, "max_depth": 6, "n_estimators": 2000, "early_stopping": 300}
    elif n_pos < 2000:  # Rare
        return {"learning_rate": 0.03, "max_depth": 7, "n_estimators": 1800, "early_stopping": 200}
    elif n_pos < 10000:  # Moderate
        return {"learning_rate": 0.04, "max_depth": 7, "n_estimators": 1600, "early_stopping": 150}
    else:  # Common/Abundant
        return None  # Use default Optuna-tuned params


def focal_loss_objective(y_pred, dtrain):
    """Focal loss for XGBoost: down-weights easy-to-classify samples.

    Uses the correct XGBoost custom objective signature: (preds, DMatrix).
    Proper focal cross-entropy: L = -(1-p)^gamma * log(p) for y=1,
                                  -p^gamma * log(1-p) for y=0.
    """
    from scipy.special import expit
    y_true = dtrain.get_label()
    p = np.clip(expit(y_pred), 1e-7, 1 - 1e-7)
    gamma = 2.0

    # Focal CE gradient w.r.t. raw logit f (where p = sigmoid(f)):
    # For y=1: dL/df = (1-p)^gamma * (gamma*p*log(p) + p - 1)
    # For y=0: dL/df = p^gamma * (gamma*(1-p)*log(1-p) - p + 1) ... but simplified:
    # Standard approach: use the normal BCE gradient scaled by focal weight
    # grad_bce = p - y, focal_weight = ((1-p)^gamma if y=1 else p^gamma)
    focal_weight = np.where(y_true == 1, (1 - p) ** gamma, p ** gamma)
    grad = focal_weight * (p - y_true)

    # Hessian: use |grad| * (1 - |grad|) approximation, clamped positive
    # This is the standard numerically-stable focal loss hessian
    hess = focal_weight * p * (1 - p)
    hess = np.maximum(hess, 1e-7)  # GBDTs require non-negative hessians

    return grad, hess


def load_xgb_params():
    """Load best params from Optuna tuning, fall back to defaults."""
    params = dict(XGB_PARAMS)
    params_path = CHECKPOINT_DIR / "best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"  Loaded tuned params from {params_path}")
        params.update(tuned)
        for k, v in tuned.items():
            print(f"    {k}: {v}")
    else:
        print(f"  No tuned params found, using defaults")
    return params


def _train_one_target(target_idx, target_name, fold_idx, threads,
                      X_tr, y_tr, X_val, y_val, X_test,
                      xgb_params, fold_aucs_per_target=None, has_oof_features=False, n_base_features=0):
    """Train a single target — runs in a thread (XGBoost releases GIL)."""
    y_t = y_tr[:, target_idx]

    # Mask own-target OOF feature to prevent leakage
    X_tr_masked = X_tr.copy() if has_oof_features else X_tr
    X_val_masked = X_val.copy() if has_oof_features else X_val
    if has_oof_features and n_base_features > 0:
        # OOF features start at index n_base_features
        oof_col_idx = n_base_features + target_idx
        X_tr_masked[:, oof_col_idx] = np.nan
        X_val_masked[:, oof_col_idx] = np.nan
    n_neg = int((y_t == 0).sum())
    n_pos = int((y_t == 1).sum())
    pos_rate = n_pos / (n_pos + n_neg)

    spw = effective_number_weight(n_pos, n_neg)
    params = {**xgb_params, "nthread": threads, "scale_pos_weight": spw}

    # Apply rarity-tier HP
    tier_params = get_rarity_tier_params(n_pos)
    if tier_params is not None:
        params["learning_rate"] = tier_params["learning_rate"]
        params["max_depth"] = tier_params["max_depth"]
        params["n_estimators"] = tier_params["n_estimators"]
        early_stop = tier_params["early_stopping"]
    else:
        early_stop = EARLY_STOPPING_ROUNDS

    # Difficulty override: after fold 1, if AUC < 0.73, use aggressive HP
    if fold_idx > 0 and fold_aucs_per_target is not None and target_idx in fold_aucs_per_target:
        target_auc = fold_aucs_per_target[target_idx]
        if target_auc < 0.73:
            params["learning_rate"] = 0.02
            params["n_estimators"] = 3500
            early_stop = 300

    # Use focal loss for rare targets (pos_rate < 5%)
    if pos_rate < 0.05:
        params["objective"] = focal_loss_objective

    model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stop)
    model.fit(
        X_tr_masked, y_tr[:, target_idx],
        eval_set=[(X_val_masked, y_val[:, target_idx])],
        verbose=False,
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    model.save_model(str(MODELS_DIR / f"{target_name}_fold{fold_idx}.json"))
    importance = model.feature_importances_[:n_base_features]  # exclude OOF cols

    return val_preds, test_preds, importance


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Step 4: Train XGBoost ({N_FOLDS}-fold × 41 targets)")
    print("=" * 60)
    print(f"  XGBoost: CPU mode, {PARALLEL_TARGETS} parallel targets × {THREADS_PER_MODEL} threads")

    xgb_params = load_xgb_params()
    xgb_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    # 1. Load features
    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    all_feature_cols = meta["feature_names"]
    feature_cols = list(all_feature_cols)
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    # Load cross-target OOF features (from 01c_add_oof_features.py)
    oof_feat_path = FEATURES_DIR / "oof_features_train.parquet"
    if oof_feat_path.exists():
        oof_feats_train = pl.read_parquet(oof_feat_path).to_numpy().astype(np.float32)
        oof_feats_test = pl.read_parquet(FEATURES_DIR / "oof_features_test.parquet").to_numpy().astype(np.float32)
        X_train = np.hstack([X_train, oof_feats_train])
        X_test = np.hstack([X_test, oof_feats_test])
        has_oof_features = True
        print(f"  Added {oof_feats_train.shape[1]} cross-target OOF features")
    else:
        has_oof_features = False
        print(f"  OOF features not found (optional)")

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
    print(f"  Features: {len(feature_cols)}")

    # 2. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "xgb_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # 3. Train
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_targets = len(target_cols)
    oof_preds = np.zeros((n_train, n_targets))
    test_preds_sum = np.zeros((n_test, n_targets))
    fold_aucs = []
    # Only track importance for base features (not OOF columns)
    n_base_features_for_imp = len(feature_cols)
    importance_sum = np.zeros((n_base_features_for_imp, n_targets), dtype=np.float64)

    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n[2/4] Training {N_FOLDS}-Fold × {n_targets} targets...", flush=True)

    # Number of base features (without OOF columns) — used for masking and importance
    n_base_features = len(feature_cols)
    fold_aucs_per_target = {}  # Track per-target AUC from fold 1 for difficulty override

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        t_fold = time.time()
        print(f"\n  ── Fold {fold_idx+1}/{N_FOLDS} "
              f"(train={len(tr_idx):,}, val={len(val_idx):,}) ──", flush=True)

        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]
        fold_test_preds = np.zeros((n_test, n_targets))

        with ThreadPoolExecutor(max_workers=PARALLEL_TARGETS) as executor:
            futures = {}
            for i, col in enumerate(target_cols):
                futures[executor.submit(
                    _train_one_target, i, col, fold_idx, THREADS_PER_MODEL,
                    X_tr, y_tr, X_val, y_val, X_test,
                    xgb_params, fold_aucs_per_target, has_oof_features, n_base_features,
                )] = i

            done_count = 0
            for future in as_completed(futures):
                i = futures[future]
                val_p, test_p, imp = future.result()
                oof_preds[val_idx, i] = val_p
                fold_test_preds[:, i] = test_p
                importance_sum[:, i] += imp
                done_count += 1
                if done_count % 10 == 0 or done_count == n_targets:
                    print(f"    {done_count}/{n_targets} targets done", flush=True)

        # Compute per-target AUC for difficulty override in next fold
        _, per_target_aucs = compute_macro_auc(y_val, oof_preds[val_idx], target_cols)
        fold_aucs_per_target = {i: per_target_aucs[target_cols[i]] for i in range(n_targets)}

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

    # Save averaged feature importances
    importance_avg = importance_sum / N_FOLDS
    imp_path = CHECKPOINT_DIR / "feature_importances.json"
    imp_data = {
        "feature_names": all_feature_cols,
        "target_cols": target_cols,
        "importances_per_target": {
            col: dict(sorted(
                zip(all_feature_cols, importance_avg[:, i].tolist()),
                key=lambda x: x[1], reverse=True
            ))
            for i, col in enumerate(target_cols)
        },
        "mean_importance": dict(sorted(
            zip(all_feature_cols, importance_avg.mean(axis=1).tolist()),
            key=lambda x: x[1], reverse=True
        )),
    }
    with open(imp_path, "w") as f:
        json.dump(imp_data, f, indent=2)
    print(f"  Saved: {imp_path}")
    print(f"  Models: {MODELS_DIR}/ ({N_FOLDS * n_targets} .json files)")

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
    submit.write_parquet("submissions/xgboost.parquet")
    print(f"  Saved: submissions/xgboost.parquet")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={oof_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
