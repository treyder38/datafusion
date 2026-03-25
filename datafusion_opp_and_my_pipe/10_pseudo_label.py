"""Step 10: Pseudo-labeling — retrain base models on expanded dataset.

Uses the stacking ensemble's confident test predictions (>0.9 or <0.1) as pseudo-labels.
Concatenates pseudo-labeled test data with original training data, retrains LightGBM.

Motivation: Pipeline detected train/test distribution shift (SHIFT_DROP features).
Pseudo-labels align decision boundaries with test distribution (semi-supervised learning).

Output: submissions/pseudo_labeled.parquet (updated test predictions)

Runtime: ~30-45 minutes (GPU/CPU).
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, log_per_target_auc

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

FEATURES_DIR = Path("features")
SELECTED_DIR = FEATURES_DIR / "selected_features"
CHECKPOINT_DIR = Path("checkpoints_pseudo")
OUTPUT_DIR = Path("submissions")


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 10: Pseudo-Labeling — Retrain LightGBM on Expanded Data")
    print("=" * 60)

    if not LGBM_AVAILABLE:
        print("ERROR: LightGBM not available. Install with: pip install lightgbm")
        return

    # 1. Load features and targets
    print("\n[1/5] Loading features and targets...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    # Convert to pandas for LightGBM
    X_train = train_feat.select(feature_cols).to_pandas()
    X_test = test_feat.select(feature_cols).to_pandas()
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_targets = len(target_cols)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Target columns: {n_targets}")

    # Load per-target feature selections
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

    # 2. Load stacking predictions (best ensemble output)
    print("\n[2/5] Loading stacking predictions...")
    stacking_path = OUTPUT_DIR / "stacking.parquet"
    if not stacking_path.exists():
        print(f"  ERROR: {stacking_path} not found. Run step 09_stacking.py first.")
        return

    stacking_df = pl.read_parquet(stacking_path)
    # Stacking output has predict_target_* columns
    test_pred_cols = [c for c in stacking_df.columns if c.startswith("predict_target_")]
    if len(test_pred_cols) != n_targets:
        print(f"  WARNING: Expected {n_targets} predict_* columns, found {len(test_pred_cols)}")

    test_preds_ensemble = stacking_df[test_pred_cols].to_numpy().astype(np.float32)
    print(f"  Loaded stacking predictions: {test_preds_ensemble.shape}")

    # 3. Select confident pseudo-labels
    print("\n[3/5] Selecting confident pseudo-labels...")
    CONFIDENCE_THRESHOLD_HIGH = 0.9
    CONFIDENCE_THRESHOLD_LOW = 0.1
    n_pseudo_total = 0
    pseudo_labels_per_target = []

    for i, col in enumerate(target_cols):
        pred = test_preds_ensemble[:, i]
        # Confident positives: pred > 0.9, Confident negatives: pred < 0.1
        mask_pos = pred > CONFIDENCE_THRESHOLD_HIGH
        mask_neg = pred < CONFIDENCE_THRESHOLD_LOW
        mask_confident = mask_pos | mask_neg

        n_confident = mask_confident.sum()
        n_pseudo_total += n_confident

        pseudo_labels = np.where(mask_pos, 1.0, 0.0)[mask_confident]
        pseudo_labels_per_target.append((mask_confident, pseudo_labels))

        if n_confident > 0:
            print(f"  {col}: {n_confident:,} confident samples ({mask_pos.sum()} pos, {mask_neg.sum()} neg)")

    print(f"  Total pseudo-labeled samples (if all targets selected): {n_pseudo_total:,}")

    # 4. Retrain LightGBM on expanded data (one model across all targets)
    print("\n[4/5] Retraining LightGBM on expanded data...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # LightGBM hyperparameters (from pipeline)
    lgbm_params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.050216,
        num_leaves=34,
        max_depth=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_estimators=1500,
        random_state=SEED,
        verbose=-1,
        force_row_wise=True,
        num_threads=8,
    )

    oof_preds_pseudo = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds_pseudo = np.zeros((n_test, n_targets), dtype=np.float32)
    fold_aucs_before = []
    fold_aucs_after = []

    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}...", flush=True)
        fold_test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

        for i, col in enumerate(target_cols):
            y = y_train[:, i]

            # Get selected features for this target
            if col in per_target_feats:
                sel_cols = per_target_feats[col]
            else:
                sel_cols = feature_cols

            X_tr = X_train[sel_cols].iloc[tr_idx].copy()
            X_va = X_train[sel_cols].iloc[val_idx].copy()
            X_te = X_test[sel_cols].copy()
            y_tr = y[tr_idx]

            # Get categorical feature indices for this target
            cat_indices = [sel_cols.index(c) for c in cat_feature_names if c in sel_cols]

            # Train baseline model (original data only)
            train_ds = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_indices)
            val_ds = lgb.Dataset(X_va, label=y[val_idx], categorical_feature=cat_indices, reference=train_ds)

            model_baseline = lgb.train(
                lgbm_params,
                train_ds,
                num_boost_round=lgbm_params["n_estimators"],
                valid_sets=[val_ds],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )

            oof_preds_pseudo[val_idx, i] = model_baseline.predict(X_va).astype(np.float32)

            # Now add pseudo-labeled data
            mask_confident, pseudo_labels = pseudo_labels_per_target[i]
            X_test_confident = X_test[sel_cols][mask_confident].copy()

            if len(X_test_confident) > 0:
                # Concatenate original train with pseudo-labeled test samples
                X_expanded = pl.concat([
                    pl.DataFrame(X_tr),
                    pl.DataFrame(X_test_confident),
                ]).to_pandas()
                y_expanded = np.concatenate([y_tr, pseudo_labels])

                # Retrain on expanded data
                train_ds_expanded = lgb.Dataset(
                    X_expanded, label=y_expanded, categorical_feature=cat_indices
                )
                val_ds_expanded = lgb.Dataset(
                    X_va, label=y[val_idx], categorical_feature=cat_indices, reference=train_ds_expanded
                )

                model_pseudo = lgb.train(
                    lgbm_params,
                    train_ds_expanded,
                    num_boost_round=lgbm_params["n_estimators"],
                    valid_sets=[val_ds_expanded],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
                )

                fold_test_preds[:, i] = model_pseudo.predict(X_te).astype(np.float32)
                del train_ds_expanded, val_ds_expanded, model_pseudo
            else:
                fold_test_preds[:, i] = model_baseline.predict(X_te).astype(np.float32)

            del train_ds, val_ds, model_baseline, X_tr, X_va, X_te
            gc.collect()

            if (i + 1) % 10 == 0 or i == n_targets - 1:
                print(f"    {i + 1}/{n_targets} targets done", flush=True)

        fold_auc, _ = compute_macro_auc(y_train[val_idx], oof_preds_pseudo[val_idx], target_cols)
        fold_aucs_after.append(fold_auc)
        test_preds_pseudo += fold_test_preds / N_FOLDS
        del fold_test_preds
        gc.collect()
        print(f"  Fold {fold_idx + 1} OOF AUC={fold_auc:.4f}", flush=True)

    # 5. Results and save
    print(f"\n[5/5] Results:")
    oof_auc_pseudo, per_target_aucs_pseudo = compute_macro_auc(y_train, oof_preds_pseudo, target_cols)
    print(f"  Pseudo-labeled OOF Macro ROC-AUC: {oof_auc_pseudo:.4f}")
    log_per_target_auc(per_target_aucs_pseudo, y_train, target_cols)

    # Compare with baseline (stacking)
    if OUTPUT_DIR.exists() and (OUTPUT_DIR / "stacking.parquet").exists():
        baseline_auc, _ = compute_macro_auc(y_train, test_preds_ensemble[:n_train], target_cols)
        print(f"\n  Baseline (stacking) AUC: {baseline_auc:.4f}")
        print(f"  Pseudo-labeled AUC:    {oof_auc_pseudo:.4f}")
        print(f"  Delta:                 {oof_auc_pseudo - baseline_auc:+.4f}")

    # Save pseudo-labeled predictions
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit_pseudo = pl.DataFrame({"customer_id": test_feat["customer_id"]}).hstack(
        pl.DataFrame(test_preds_pseudo.astype(np.float64), schema=predict_cols)
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    submit_pseudo.write_parquet(OUTPUT_DIR / "pseudo_labeled.parquet")
    print(f"\n  Saved: {OUTPUT_DIR / 'pseudo_labeled.parquet'}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. OOF={oof_auc_pseudo:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
