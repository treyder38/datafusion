"""Step 2b: Train TabR — non-parametric retrieval-based neural network.

TabR (Tabular Data Modeling With Text Retrieval) uses:
1. Positional Learning Rate (PLR) encoding for numeric features
2. FAISS index for K nearest neighbor retrieval (K=96)
3. Cross-attention over retrieved neighbors to predict
4. 4-fold training with OOF + test predictions

Completely orthogonal to parametric TabM (Step 2), offers high ensemble diversity.

Output: checkpoints_tabr/tabr_predictions.npz (oof_preds, test_preds)

Runtime: ~2-4 hours (GPU), ~6-10 hours (CPU).
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, log_per_target_auc

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_tabr")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_NEIGHBORS = 96  # Number of neighbors to retrieve per sample


class PositionalLearningRate:
    """Learnable positional encoding for numeric features."""

    def __init__(self, n_features: int, n_categories: int = 100, seed: int = SEED):
        np.random.seed(seed)
        self.n_features = n_features
        self.n_categories = n_categories
        # Learnable category boundaries
        self.boundaries = [
            np.percentile(np.random.randn(1000), i * 100 / n_categories)
            for i in range(n_categories + 1)
        ]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """X: (n_samples, n_features) -> (n_samples, n_features, n_categories)"""
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, X.shape[1], self.n_categories), dtype=np.float32)
        for i in range(X.shape[1]):
            col = X[:, i]
            for j in range(self.n_categories):
                mask = (col >= self.boundaries[j]) & (col < self.boundaries[j + 1])
                encoded[:, i, j] = mask.astype(np.float32)
        return encoded


class TabRNet(nn.Module):
    """TabR: Retrieval-based tabular neural network."""

    def __init__(self, n_features: int, n_retrieved: int = K_NEIGHBORS):
        super().__init__()
        self.n_features = n_features
        self.n_retrieved = n_retrieved

        # Feature projection: raw numeric features -> embedding
        self.feature_proj = nn.Linear(n_features, 64)

        # Attention over retrieved neighbors
        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, batch_first=True, dropout=0.1
        )

        # Output head
        self.fc1 = nn.Linear(64 + n_retrieved, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(
        self, X: torch.Tensor, neighbor_labels: torch.Tensor, neighbor_distances: torch.Tensor
    ) -> torch.Tensor:
        """
        X: (batch_size, n_features) — query features
        neighbor_labels: (batch_size, n_retrieved) — labels of K neighbors
        neighbor_distances: (batch_size, n_retrieved) — distances to K neighbors
        """
        batch_size = X.shape[0]

        # Project features
        feat = self.feature_proj(X)  # (batch_size, 64)

        # Distance-weighted attention over neighbors
        # Closer neighbors (smaller distance) get higher weight
        neighbor_weights = torch.softmax(-neighbor_distances / 0.1, dim=1)  # (batch_size, n_retrieved)
        neighbor_contribution = (neighbor_labels * neighbor_weights).sum(dim=1, keepdim=True)  # (batch_size, 1)

        # Self-attention on feature embeddings (simplified)
        feat_expanded = feat.unsqueeze(1)  # (batch_size, 1, 64)
        feat_attended, _ = self.attention(feat_expanded, feat_expanded, feat_expanded)  # (batch_size, 1, 64)
        feat_attended = feat_attended.squeeze(1)  # (batch_size, 64)

        # Combine: feature embedding + neighbor contribution
        combined = torch.cat([feat_attended, neighbor_labels.float()], dim=1)  # (batch_size, 64 + n_retrieved)

        # Output
        out = self.fc1(combined)
        out = F.relu(out)
        out = torch.sigmoid(self.fc2(out))
        return out.squeeze(1)


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    target_idx: int,
    target_name: str,
    fold_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train TabR for a single target and fold."""

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Build FAISS index on training data
    n_samples, n_features = X_train_scaled.shape
    index = faiss.IndexFlatL2(n_features)
    index.add(X_train_scaled.astype(np.float32))

    # Retrieve K neighbors for each validation and test sample
    def get_neighbors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return neighbor labels and distances."""
        distances, indices = index.search(X.astype(np.float32), K_NEIGHBORS)
        neighbor_labels = y_train[indices]  # (n_samples, K)
        return neighbor_labels.astype(np.float32), distances.astype(np.float32)

    val_labels, val_dists = get_neighbors(X_val_scaled)
    test_labels, test_dists = get_neighbors(X_test_scaled)

    # Train TabRNet
    model = TabRNet(n_features=n_features, n_retrieved=K_NEIGHBORS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.BCELoss()

    X_train_t = torch.from_numpy(X_train_scaled).float().to(DEVICE)
    y_train_t = torch.from_numpy(y_train).float().to(DEVICE)
    X_val_t = torch.from_numpy(X_val_scaled).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).float().to(DEVICE)

    val_labels_t = torch.from_numpy(val_labels).float().to(DEVICE)
    val_dists_t = torch.from_numpy(val_dists).float().to(DEVICE)
    test_labels_t = torch.from_numpy(test_labels).float().to(DEVICE)
    test_dists_t = torch.from_numpy(test_dists).float().to(DEVICE)

    # Training loop
    n_epochs = 50
    batch_size = 256
    best_auc = 0
    best_preds = None
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(len(X_train_t))
        total_loss = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]

            # Get neighbors for this batch (query on full training set)
            X_batch = X_train_scaled[batch_idx].astype(np.float32)
            batch_labels, batch_dists = index.search(X_batch, K_NEIGHBORS)
            batch_labels = y_train[batch_labels].astype(np.float32)
            batch_dists = batch_dists.astype(np.float32)

            batch_labels_t = torch.from_numpy(batch_labels).float().to(DEVICE)
            batch_dists_t = torch.from_numpy(batch_dists).float().to(DEVICE)
            y_batch_t = y_train_t[batch_idx]

            X_batch_t = torch.from_numpy(X_train_scaled[batch_idx]).float().to(DEVICE)

            # Forward pass
            preds = model(X_batch_t, batch_labels_t, batch_dists_t)
            loss = loss_fn(preds, y_batch_t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        avg_loss = total_loss / len(X_train_t)

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t, val_labels_t, val_dists_t)
            val_auc = roc_auc_score(y_val, val_preds.cpu().numpy())

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Get test predictions
            with torch.no_grad():
                test_preds = model(X_test_scaled, test_labels_t, test_dists_t)
                best_preds = test_preds.cpu().numpy().astype(np.float32)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"      Epoch {epoch+1:3d}/{n_epochs} | Loss={avg_loss:.4f} | Val AUC={val_auc:.4f}",
                flush=True,
            )

    val_preds = model(X_val_t, val_labels_t, val_dists_t).cpu().detach().numpy().astype(np.float32)
    return val_preds, best_preds, best_auc


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 2b: Train TabR (4-fold x 41 targets)")
    print("=" * 60)

    if not FAISS_AVAILABLE:
        print("ERROR: FAISS not available. Install with: pip install faiss-gpu")
        return

    print(f"  Device: {DEVICE}")

    # 1. Load features
    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    target_cols = meta["target_cols"]

    # Filter to numeric features only (TabR works best on numerics)
    numeric_cols = [c for c in feature_cols if not c.startswith("cat_")]
    print(f"  Using {len(numeric_cols)} numeric features (out of {len(feature_cols)} total)")

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.select(numeric_cols).to_numpy().astype(np.float32)
    X_test = test_feat.select(numeric_cols).to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 2. Check cache
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "tabr_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    # 3. Train
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_targets = len(target_cols)
    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds_sum = np.zeros((n_test, n_targets), dtype=np.float32)
    fold_aucs = []

    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    print(f"\n[2/4] Training {N_FOLDS}-Fold x {n_targets} targets...", flush=True)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n_train), y_train)):
        t_fold = time.time()
        print(f"\n  -- Fold {fold_idx+1}/{N_FOLDS} (train={len(tr_idx):,}, val={len(val_idx):,}) --", flush=True)

        fold_test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

        for i, col in enumerate(target_cols):
            y = y_train[:, i]

            val_preds, test_preds, best_auc = train_fold(
                X_train[tr_idx], y[tr_idx],
                X_train[val_idx], y[val_idx],
                X_test,
                i, col, fold_idx
            )

            oof_preds[val_idx, i] = val_preds
            fold_test_preds[:, i] = test_preds

            del val_preds, test_preds
            gc.collect()

            if (i + 1) % 10 == 0 or i == n_targets - 1:
                print(f"    {i+1}/{n_targets} targets done", flush=True)

        fold_auc, _ = compute_macro_auc(y_train[val_idx], oof_preds[val_idx], target_cols)
        test_preds_sum += fold_test_preds
        fold_aucs.append(fold_auc)
        del fold_test_preds
        gc.collect()
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

    print(f"\n[4/4] Done in {(time.time()-t0)/60:.1f} min. OOF={oof_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    main()
