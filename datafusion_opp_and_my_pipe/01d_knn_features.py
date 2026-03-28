"""Step 1d: Generate kNN-based features for each target.

For each target, computes neighbor-based features using OOF methodology:
  - knn_mean:          mean target value of k nearest neighbors
  - knn_weighted_mean: distance-weighted mean (closer neighbors count more)
  - knn_topk_mean:     mean of top-k/2 closest neighbors only

Uses DAE embeddings (256-dim learned representation from NN step) as the
feature space with cosine distance. DAE captures non-linear patterns that
raw Euclidean distance in high-dimensional feature space cannot.

Fallback: if DAE embeddings are not available, uses raw numeric features
with cosine distance and top-100 variance selection.

Anti-leakage: kNN is fit on train fold only, queried on val fold (OOF style).

Output:
- features/knn_features_train.parquet (750K × 123 columns: 41 targets × 3 features)
- features/knn_features_test.parquet  (test × 123 columns)

Runtime: ~15-30 min on CPU.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS

FEATURES_DIR = Path("features")
NN_CHECKPOINT_DIR = Path("checkpoints_nn")
K = 8  # number of neighbors (not 16 — too many smooths out signal)
K_TOP = 4  # for knn_topk_mean: use only closest K_TOP neighbors


def compute_knn_features(distances, indices, y_train_fold, k_top):
    """Compute 3 kNN features from precomputed neighbors.

    Args:
        distances: (n_query, k) distances to neighbors
        indices: (n_query, k) indices of neighbors in train fold
        y_train_fold: (n_train_fold,) target values for train fold
        k_top: number of closest neighbors for topk feature

    Returns:
        (n_query, 3) array: [knn_mean, knn_weighted_mean, knn_topk_mean]
    """
    # Gather neighbor targets
    neighbor_targets = y_train_fold[indices]  # (n_query, k)

    # 1. knn_mean: simple average of all k neighbors' targets
    knn_mean = neighbor_targets.mean(axis=1)

    # 2. knn_weighted_mean: distance-weighted (closer = higher weight)
    weights = 1.0 / (distances + 1e-6)  # inverse distance
    weights_norm = weights / weights.sum(axis=1, keepdims=True)
    knn_weighted_mean = (neighbor_targets * weights_norm).sum(axis=1)

    # 3. knn_topk_mean: only closest k_top neighbors
    knn_topk_mean = neighbor_targets[:, :k_top].mean(axis=1)

    return np.column_stack([knn_mean, knn_weighted_mean, knn_topk_mean]).astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Step 1d: kNN Features (k={K}, top-k={K_TOP})")
    print("=" * 60)

    # Load metadata
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    target_cols = meta["target_cols"]
    feature_cols = meta["feature_names"]
    n_targets = len(target_cols)

    # Load data
    print("\n[1/4] Loading data...")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    # Try DAE embeddings first (256-dim learned representation from NN step)
    dae_train_path = NN_CHECKPOINT_DIR / "dae_train_emb.npy"
    dae_test_path = NN_CHECKPOINT_DIR / "dae_test_emb.npy"
    use_dae = dae_train_path.exists() and dae_test_path.exists()

    if use_dae:
        print("  Using DAE embeddings (256-dim learned space) + cosine distance")
        X_train = np.load(str(dae_train_path)).astype(np.float32)
        X_test = np.load(str(dae_test_path)).astype(np.float32)
    else:
        print("  DAE embeddings not found, falling back to raw numeric features")
        train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
        test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
        num_cols = [c for c in feature_cols if not c.startswith("cat_feature_")]
        print(f"  Using {len(num_cols)} numeric features for kNN distance")
        X_train = np.nan_to_num(train_feat.select(num_cols).to_numpy().astype(np.float32), nan=0.0)
        X_test = np.nan_to_num(test_feat.select(num_cols).to_numpy().astype(np.float32), nan=0.0)
        # Reduce dimensionality for raw features
        MAX_FEATURES = 100
        if len(num_cols) > MAX_FEATURES:
            variances = np.var(X_train, axis=0)
            top_var_idx = np.argsort(variances)[::-1][:MAX_FEATURES]
            X_train = X_train[:, top_var_idx]
            X_test = X_test[:, top_var_idx]
            print(f"  Reduced to top {MAX_FEATURES} features by variance")

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # OOF kNN feature generation
    print(f"\n[2/4] Computing OOF kNN features ({N_FOLDS}-fold)...")
    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Output arrays: 3 features per target
    knn_train = np.zeros((n_train, n_targets * 3), dtype=np.float32)
    # For test: average across folds (like OOF for test)
    knn_test_sum = np.zeros((n_test, n_targets * 3), dtype=np.float32)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        t_fold = time.time()

        # L2-normalize for cosine distance (cosine = 1 - dot(a/|a|, b/|b|))
        # sklearn NearestNeighbors with metric='cosine' handles this, but
        # pre-normalizing + brute force is faster for dense data
        X_tr_norm = normalize(X_train[tr_idx], norm='l2')
        X_val_norm = normalize(X_train[val_idx], norm='l2')
        X_test_norm = normalize(X_test, norm='l2')

        # Fit kNN on train fold with cosine distance
        nn = NearestNeighbors(n_neighbors=K, metric='cosine', n_jobs=-1, algorithm='brute')
        nn.fit(X_tr_norm)

        # Query val fold
        val_distances, val_indices = nn.kneighbors(X_val_norm)
        # Query test
        test_distances, test_indices = nn.kneighbors(X_test_norm)

        # Compute features for each target
        for t_idx in range(n_targets):
            y_tr_t = y_train[tr_idx, t_idx]
            col_start = t_idx * 3

            # Val fold (OOF)
            val_feats = compute_knn_features(val_distances, val_indices, y_tr_t, K_TOP)
            knn_train[val_idx, col_start:col_start+3] = val_feats

            # Test (accumulate across folds)
            test_feats = compute_knn_features(test_distances, test_indices, y_tr_t, K_TOP)
            knn_test_sum[:, col_start:col_start+3] += test_feats

        elapsed = time.time() - t_fold
        print(f"  Fold {fold_idx+1}/{N_FOLDS}: {elapsed:.0f}s")

    # Average test predictions across folds
    knn_test = knn_test_sum / N_FOLDS

    # Column names
    print(f"\n[3/4] Evaluating kNN features...")
    col_names = []
    for t_col in target_cols:
        col_names.extend([f"knn_mean_{t_col}", f"knn_wmean_{t_col}", f"knn_topk_{t_col}"])

    # Quick evaluation: how good are kNN features alone per target?
    for t_idx, t_col in enumerate(target_cols):
        y_t = y_train[:, t_idx]
        if y_t.sum() < 2:
            continue
        col_start = t_idx * 3
        # knn_mean is the simplest — check its AUC
        knn_mean_auc = roc_auc_score(y_t, knn_train[:, col_start])
        if (t_idx + 1) % 10 == 0 or t_idx < 3:
            print(f"  {t_col}: knn_mean AUC={knn_mean_auc:.4f}")

    # Save
    print(f"\n[4/4] Saving kNN features...")
    knn_train_df = pl.DataFrame(knn_train, schema=col_names)
    knn_test_df = pl.DataFrame(knn_test, schema=col_names)

    train_path = FEATURES_DIR / "knn_features_train.parquet"
    test_path = FEATURES_DIR / "knn_features_test.parquet"
    knn_train_df.write_parquet(train_path)
    knn_test_df.write_parquet(test_path)

    print(f"  Saved: {train_path} ({knn_train_df.shape})")
    print(f"  Saved: {test_path} ({knn_test_df.shape})")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
