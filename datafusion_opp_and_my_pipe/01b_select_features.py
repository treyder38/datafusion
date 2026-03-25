"""Step 1b: Per-target feature selection using quick LightGBM.

For each of the 41 targets, trains a lightweight LGBM on a balanced subsample
and selects top-K features by gain importance. Saves per-target feature lists
to features/selected_features/.

Inspired by DL pipeline (select_features.py) which achieves 0.8471 with
per-target top-300 selection + XGBoost.

Runtime: ~15-20 minutes.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from utils import SEED, effective_number_weight

FEATURES_DIR = Path("features")
SELECTED_DIR = FEATURES_DIR / "selected_features"

TOP_K = 350
SUBSAMPLE_SIZE = 300_000

LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1,
    force_col_wise=True,
)


def subsample_balanced(y, size=SUBSAMPLE_SIZE):
    """Balanced subsampling: if pos < size/2, keep all pos + sample neg."""
    rng = np.random.default_rng(SEED)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if pos_idx.size < size // 2:
        n_neg = min(size - pos_idx.size, neg_idx.size)
        neg_take = rng.choice(neg_idx, size=n_neg, replace=False)
        return np.sort(np.concatenate([pos_idx, neg_take]))
    else:
        return np.sort(rng.choice(np.arange(len(y)), size=min(size, len(y)), replace=False))


def select_features_for_target(X, y, feature_names):
    """Train one lightweight LGBM and return top-K features by gain importance."""
    from sklearn.model_selection import StratifiedShuffleSplit

    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    spw = effective_number_weight(n_pos, n_neg)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(splitter.split(X, y))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw)
        model.fit(
            X[tr_idx],
            y[tr_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    importance = model.booster_.feature_importance(importance_type="gain")
    top_indices = np.argsort(importance)[::-1][:TOP_K]
    return [feature_names[i] for i in top_indices]


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Step 1b: Per-target feature selection (top {TOP_K})")
    print("=" * 60)

    # Load meta
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    target_cols = meta["target_cols"]

    print(f"\n[1/2] Loading data...")
    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")
    X_all = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_all = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    print(f"  X: {X_all.shape}, targets: {len(target_cols)}")

    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/2] Selecting top-{TOP_K} features per target...")
    results = {}
    skipped = 0

    for i, target in enumerate(target_cols):
        out_path = SELECTED_DIR / f"{target}.json"
        if out_path.exists():
            with open(out_path) as f:
                results[target] = json.load(f)
            skipped += 1
            continue

        y = y_all[:, i]
        n_pos = int(y.sum())

        # Subsample for speed
        idx = subsample_balanced(y)
        X_sub = X_all[idx]
        y_sub = y[idx]

        selected = select_features_for_target(X_sub, y_sub, feature_cols)
        results[target] = selected

        with open(out_path, "w") as f:
            json.dump(selected, f)

        print(f"  {target:<14s}  pos={n_pos:>6,}  selected={len(selected)}  "
              f"({i+1}/{len(target_cols)})", flush=True)

    if skipped:
        print(f"  Skipped {skipped} targets (already computed)")

    # Save summary
    summary = {target: len(feats) for target, feats in results.items()}
    all_selected = set()
    for feats in results.values():
        all_selected.update(feats)

    with open(SELECTED_DIR / "summary.json", "w") as f:
        json.dump({
            "top_k": TOP_K,
            "per_target_counts": summary,
            "total_unique_features": len(all_selected),
            "total_features": len(feature_cols),
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")
    print(f"  Unique features used: {len(all_selected)}/{len(feature_cols)}")
    print(f"  Saved to: {SELECTED_DIR}/")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
