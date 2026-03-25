"""Step 5: Train PyBoost (single-fit, no OOF).

Loads features from features/, trains one SketchBoost multi-output model.

Output: checkpoints_pyboost/pyboost_predictions.npz (train_preds, test_preds)

Runtime: ~20-40 minutes on a CUDA GPU.
"""

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import DATA_DIR, SEED, compute_macro_auc, log_per_target_auc

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_pyboost")
MODELS_DIR = CHECKPOINT_DIR / "models"

PARAMS = dict(
    ntrees=5000,
    lr=0.0566,
    max_depth=7,
    min_data_in_leaf=88,
    lambda_l2=2.076,
    subsample=0.88,
    colsample=0.76,
    max_bin=64,
    es=200,
    verbose=100,
    gd_steps=1,
    use_hess=False,
)


def make_validation_split(y_train):
    """Single holdout split for early stopping only."""
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(splitter.split(np.arange(len(y_train)), y_train))
    return tr_idx, val_idx


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 5: Train PyBoost (single-fit SketchBoost)")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("\nERROR: PyBoost requires NVIDIA GPU with CUDA.")
            print("  This script uses cupy + py-boost which only work on CUDA GPUs.")
            print("  Skipping PyBoost training.")
            return
    except ImportError:
        print("\nERROR: torch not found. Install PyTorch with CUDA support.")
        return

    try:
        import cupy as cp
        from py_boost import SketchBoost
    except ImportError:
        print("\nERROR: cupy and/or py-boost not installed.")
        print("  pip install py-boost cupy-cuda12x")
        return

    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(cat_feature_names)} cat, {len(feature_cols) - len(cat_feature_names)} num")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "pyboost_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    tr_idx, val_idx = make_validation_split(y_train)
    X_fit, X_val = X_train[tr_idx], X_train[val_idx]
    y_fit, y_val = y_train[tr_idx], y_train[val_idx]

    print("\n[2/4] Training SketchBoost with a single validation split...")
    print(f"  Holdout: train={len(tr_idx):,}, val={len(val_idx):,}", flush=True)

    model = SketchBoost("bce", **PARAMS)
    model.fit(X_fit, y_fit, eval_sets=[{"X": X_val, "y": y_val}])

    train_preds = sigmoid(model.predict(X_train)).astype(np.float32)
    test_preds = sigmoid(model.predict(X_test)).astype(np.float32)

    with open(MODELS_DIR / "pyboost.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_auc, per_target_aucs = compute_macro_auc(y_train, train_preds, target_cols)
    print("\n[3/4] Results:")
    print(f"  Train Macro ROC-AUC: {train_auc:.4f}")
    log_per_target_auc(per_target_aucs, y_train, target_cols)

    np.savez_compressed(cache_file, train_preds=train_preds, test_preds=test_preds)
    print(f"  Saved: {cache_file}")
    print("  Saved: checkpoints_pyboost/models/pyboost.pkl")

    print("\n[4/4] Saving submission...")
    from utils import verify_submission

    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_feat["customer_id"]}).hstack(
        pl.DataFrame(test_preds.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/pyboost.parquet")
    print("  Saved: submissions/pyboost.parquet")

    del X_fit, X_val, y_fit, y_val, model
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Train AUC={train_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
