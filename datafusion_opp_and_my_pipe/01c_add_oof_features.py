"""Step 1c: Extract and save cross-target OOF features.

Loads OOF predictions from all base models (NN, LGBM, XGBoost, PyBoost, CatBoost).
Computes consensus OOF (mean across available models) to reduce noise.
Saves as parquet for reuse in base model retraining with cross-talk signal.

Output:
- features/oof_features_train.parquet (750K × 41 consensus OOF columns)
- features/oof_features_test.parquet (test samples × 41 consensus OOF columns)

Runtime: ~1-2 minutes.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from utils import SEED, DATA_DIR, N_FOLDS

FEATURES_DIR = Path("features")


def load_nn():
    """Load NN OOF predictions from fold checkpoints."""
    nn_dir = Path("checkpoints_nn")
    oof_parts, test_parts = {}, []
    n_targets = np.load(nn_dir / "fold_0.npz")["val_preds"].shape[1]
    for fi in range(N_FOLDS):
        d = np.load(nn_dir / f"fold_{fi}.npz")
        for idx, pred in zip(d["val_idx"], d["val_preds"]):
            oof_parts[int(idx)] = pred
        test_parts.append(d["test_preds"])
    n_train = max(oof_parts.keys()) + 1
    oof = np.zeros((n_train, n_targets), dtype=np.float32)
    for idx, pred in oof_parts.items():
        oof[idx] = pred
    return oof, np.mean(test_parts, axis=0).astype(np.float32)


def main():
    print("=" * 60)
    print("Step 1c: Extract Cross-Target OOF Features")
    print("=" * 60)

    # Load target column names
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    target_cols = meta["target_cols"]
    n_targets = len(target_cols)

    # Load OOF from all available models
    print("\n[1/3] Loading OOF from base models...")
    oof_list, test_list = [], []
    model_names = []

    # NN (always available)
    try:
        oof_nn, test_nn = load_nn()
        oof_list.append(oof_nn)
        test_list.append(test_nn)
        model_names.append("NN")
        print(f"  NN: {oof_nn.shape}")
    except Exception as e:
        print(f"  NN: ERROR - {e}")

    # LGBM
    try:
        d = np.load("checkpoints_lgbm/lgbm_predictions.npz")
        oof_list.append(d["oof_preds"].astype(np.float32))
        test_list.append(d["test_preds"].astype(np.float32))
        model_names.append("LGBM")
        print(f"  LGBM: {d['oof_preds'].shape}")
    except Exception as e:
        print(f"  LGBM: not found")

    # XGBoost
    try:
        d = np.load("checkpoints_xgboost/xgb_predictions.npz")
        oof_list.append(d["oof_preds"].astype(np.float32))
        test_list.append(d["test_preds"].astype(np.float32))
        model_names.append("XGBoost")
        print(f"  XGBoost: {d['oof_preds'].shape}")
    except Exception as e:
        print(f"  XGBoost: not found")

    # PyBoost
    try:
        d = np.load("checkpoints_pyboost/pyboost_predictions.npz")
        oof_list.append(d["oof_preds"].astype(np.float32))
        test_list.append(d["test_preds"].astype(np.float32))
        model_names.append("PyBoost")
        print(f"  PyBoost: {d['oof_preds'].shape}")
    except Exception as e:
        print(f"  PyBoost: not found")

    # CatBoost
    try:
        d = np.load("checkpoints_catboost/cb_predictions.npz")
        oof_list.append(d["oof_preds"].astype(np.float32))
        test_list.append(d["test_preds"].astype(np.float32))
        model_names.append("CatBoost")
        print(f"  CatBoost: {d['oof_preds'].shape}")
    except Exception as e:
        print(f"  CatBoost: not found")


    if not oof_list:
        print("\nERROR: No OOF predictions found. Run base models first.")
        return

    # Compute consensus OOF (mean across available models)
    print(f"\n[2/3] Computing consensus OOF ({len(model_names)} models)...")
    print(f"  Models: {', '.join(model_names)}")
    oof_consensus = np.mean(oof_list, axis=0).astype(np.float32)
    test_consensus = np.mean(test_list, axis=0).astype(np.float32)
    print(f"  Consensus OOF shape: {oof_consensus.shape}")
    print(f"  Consensus test shape: {test_consensus.shape}")

    # Create DataFrames with proper column naming
    print(f"\n[3/3] Saving OOF features...")
    oof_col_names = [f"oof_{target_cols[i]}" for i in range(n_targets)]

    # Train OOF features
    oof_df = pl.DataFrame(oof_consensus, schema=oof_col_names)
    oof_path = FEATURES_DIR / "oof_features_train.parquet"
    oof_df.write_parquet(oof_path)
    print(f"  Saved: {oof_path}")

    # Test OOF features
    test_df = pl.DataFrame(test_consensus, schema=oof_col_names)
    test_path = FEATURES_DIR / "oof_features_test.parquet"
    test_df.write_parquet(test_path)
    print(f"  Saved: {test_path}")

    print(f"\nDone. OOF features ready for base model retraining.")
    print(f"Usage: modify base scripts to load and use these features with per-target masking.")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
