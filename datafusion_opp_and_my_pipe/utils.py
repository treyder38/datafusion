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


def log_per_target_auc(aucs, y_true, target_cols, top_worst=5):
    """Log per-target AUC breakdown with weak target warnings.

    Args:
        aucs: dict {target_name: auc} from compute_macro_auc.
        y_true: (n_samples, n_targets) array for pos_rate calculation.
        target_cols: list of target names.
        top_worst: number of worst targets to highlight.
    """
    if not aucs:
        return

    sorted_targets = sorted(aucs.items(), key=lambda x: x[1])
    n_targets = len(target_cols)

    # Pos rates
    pos_rates = {}
    for i, col in enumerate(target_cols):
        pos_rates[col] = float(y_true[:, i].mean())

    # Worst targets
    worst = sorted_targets[:top_worst]
    print(f"\n  Per-target AUC ({n_targets} targets):")
    print(f"    Best:  {sorted_targets[-1][0]} = {sorted_targets[-1][1]:.4f}")
    print(f"    Worst: {sorted_targets[0][0]} = {sorted_targets[0][1]:.4f}")
    print(f"    Std:   {np.std(list(aucs.values())):.4f}")

    print(f"\n  Weak targets (bottom {top_worst}):")
    for col, auc in worst:
        pr = pos_rates.get(col, 0)
        flag = " *** CRITICAL" if auc < 0.70 else " ** LOW" if auc < 0.75 else ""
        print(f"    {col:<12s}  AUC={auc:.4f}  pos_rate={pr:.4f} ({pr*100:.1f}%){flag}")

    # Quantile summary
    auc_values = np.array(list(aucs.values()))
    print(f"\n  Distribution: min={auc_values.min():.4f}  Q25={np.percentile(auc_values, 25):.4f}  "
          f"median={np.median(auc_values):.4f}  Q75={np.percentile(auc_values, 75):.4f}  "
          f"max={auc_values.max():.4f}")


def to_ranks(arr):
    """Convert predictions to per-column ranks."""
    return np.column_stack([rankdata(arr[:, i]) for i in range(arr.shape[1])])


def load_zero_importance_mask(importances_json, feature_names):
    """Return boolean mask and list of indices for features with non-zero mean importance.

    Args:
        importances_json: Path to feature_importances.json from a previous run.
        feature_names: list of feature names in the current feature matrix.

    Returns:
        (keep_indices, dropped_count) or (None, 0) if file doesn't exist.
    """
    import json
    from pathlib import Path

    path = Path(importances_json)
    if not path.exists():
        return None, 0

    with open(path) as f:
        data = json.load(f)

    mean_imp = data.get("mean_importance", {})
    if not mean_imp:
        return None, 0

    zero_features = {name for name, val in mean_imp.items() if val == 0.0}
    keep = [i for i, name in enumerate(feature_names) if name not in zero_features]
    dropped = len(feature_names) - len(keep)
    return np.array(keep, dtype=np.int64) if dropped > 0 else None, dropped


def verify_submission(submit, sample):
    """Assert submission matches sample format."""
    assert submit.shape == sample.shape, f"Shape mismatch: {submit.shape} vs {sample.shape}"
    assert submit.columns == sample.columns, "Column mismatch"
    for col in submit.columns:
        assert submit[col].dtype == sample[col].dtype, f"Dtype mismatch for {col}"
    print(f"  Format verified: {submit.shape[0]:,} rows, {submit.shape[1]} cols")
