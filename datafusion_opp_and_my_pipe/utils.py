"""Shared constants and helper functions."""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

SEED = 1234
DATA_DIR = str(Path(__file__).resolve().parent.parent) + "/"
N_FOLDS = 5


def get_device():
    """Auto-detect best available torch device: cuda > mps > cpu."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def detect_lgbm_device():
    """Detect the best available LightGBM device with a safe CPU fallback.

    Returns (device_params_dict, device_name_str, supports_cat_features_bool).
    CUDA backend does not support categorical_feature — the third return value
    tells callers whether to pass cat indices to .fit().
    """
    import lightgbm as lgb

    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)

    # Prefer "gpu" (OpenCL) over "cuda" — OpenCL supports categorical_feature,
    # CUDA does not and crashes with illegal memory access on cat features.
    candidates = ["gpu"] if sys.platform.startswith("linux") else ["gpu"]
    for device_type in candidates:
        try:
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=1,
                num_leaves=2,
                min_child_samples=1,
                max_bin=63,
                device_type=device_type,
                verbose=-1,
            )
            model.fit(X, y)
            return {"device_type": device_type, "max_bin": 63}, device_type.upper(), True
        except Exception:
            continue
    return {}, "CPU", True


def compute_macro_auc(y_true, y_pred, target_cols):
    """Compute macro ROC-AUC with per-target breakdown."""
    aucs = {}
    for i, col in enumerate(target_cols):
        y_t = y_true[:, i]
        if y_t.sum() >= 2 and (len(y_t) - y_t.sum()) >= 2:
            aucs[col] = roc_auc_score(y_t, y_pred[:, i])
    return float(np.mean(list(aucs.values()))), aucs


def to_ranks(arr):
    """Convert predictions to per-column ranks."""
    return np.column_stack([rankdata(arr[:, i]) for i in range(arr.shape[1])])


def verify_submission(submit, sample):
    """Assert submission matches sample format."""
    assert submit.shape == sample.shape, f"Shape mismatch: {submit.shape} vs {sample.shape}"
    assert submit.columns == sample.columns, "Column mismatch"
    for col in submit.columns:
        assert submit[col].dtype == sample[col].dtype, f"Dtype mismatch for {col}"
    print(f"  Format verified: {submit.shape[0]:,} rows, {submit.shape[1]} cols")
