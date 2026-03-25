"""Step 8: Rank per-target blend (NN + LGBM + XGBoost + PyBoost + CatBoost + LGBM meta).

Per-target weight optimization on train predictions via grid search (6 models).

Output: submissions/blend.parquet

Runtime: ~5-10 minutes.
"""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from utils import DATA_DIR, SEED, compute_macro_auc, to_ranks, verify_submission


def load_nn():
    d = np.load("checkpoints_nn/nn_predictions.npz")
    return d["train_preds"], d["test_preds"]


def _gen_weight_combos(n_models, step):
    """Generate all weight combinations that sum to ~1.0."""
    grid = np.arange(0, 1.01, step)

    def _recurse(depth, remaining):
        if depth == n_models - 1:
            if remaining >= -0.001:
                yield (remaining,)
            return
        for w in grid:
            if w > remaining + 0.001:
                break
            for rest in _recurse(depth + 1, remaining - w):
                yield (w,) + rest

    return _recurse(0, 1.0)


def optimize_per_target(train_ranks, y, target_cols, n_models, step=0.10):
    """Per-target weight optimization for N-model rank blend."""
    n_targets = len(target_cols)
    default_w = [1.0 / n_models] * n_models
    weights = np.zeros((n_targets, n_models))
    combos = list(_gen_weight_combos(n_models, step))
    for i in range(n_targets):
        y_t = y[:, i]
        if y_t.sum() < 2 or (len(y_t) - y_t.sum()) < 2:
            weights[i] = default_w
            continue
        ranks_i = np.column_stack([train_ranks[m][:, i] for m in range(n_models)])
        best_auc, best_w = 0.0, default_w[:]
        for combo in combos:
            blended = ranks_i @ np.array(combo)
            auc = roc_auc_score(y_t, blended)
            if auc > best_auc:
                best_auc = auc
                best_w = list(combo)
        weights[i] = best_w
    return weights


def main():
    t0 = time.time()
    model_names = ["NN", "LGBM", "XGBoost", "PyBoost", "CatBoost", "LGBM_meta"]
    n_models = len(model_names)
    print("=" * 60)
    print(f"Step 8: Rank per-target blend ({' + '.join(model_names)})")
    print("=" * 60)

    train_tgt = pl.read_parquet(f"{DATA_DIR}train_target.parquet")
    target_cols = [c for c in train_tgt.columns if c.startswith("target_")]
    y = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    print("\n[1/3] Loading predictions...")
    train_nn, test_nn = load_nn()
    nn_auc, _ = compute_macro_auc(y, train_nn, target_cols)
    print(f"  NN: train {nn_auc:.5f}")

    d = np.load("checkpoints_lgbm/lgbm_predictions.npz")
    train_lgbm, test_lgbm = d["train_preds"], d["test_preds"]
    lgbm_auc, _ = compute_macro_auc(y, train_lgbm, target_cols)
    print(f"  LGBM: train {lgbm_auc:.5f}")

    d = np.load("checkpoints_xgboost/xgb_predictions.npz")
    train_xgb, test_xgb = d["train_preds"], d["test_preds"]
    xgb_auc, _ = compute_macro_auc(y, train_xgb, target_cols)
    print(f"  XGBoost: train {xgb_auc:.5f}")

    d = np.load("checkpoints_pyboost/pyboost_predictions.npz")
    train_pb, test_pb = d["train_preds"], d["test_preds"]
    pb_auc, _ = compute_macro_auc(y, train_pb, target_cols)
    print(f"  PyBoost: train {pb_auc:.5f}")

    d = np.load("checkpoints_catboost/cb_predictions.npz")
    train_cb, test_cb = d["train_preds"], d["test_preds"]
    cb_auc, _ = compute_macro_auc(y, train_cb, target_cols)
    print(f"  CatBoost: train {cb_auc:.5f}")

    d = np.load("checkpoints_lgbm_meta/lgbm_predictions.npz")
    train_lgbm_meta, test_lgbm_meta = d["train_preds"], d["test_preds"]
    lgbm_meta_auc, _ = compute_macro_auc(y, train_lgbm_meta, target_cols)
    print(f"  LGBM_meta: train {lgbm_meta_auc:.5f}")

    print(f"\n[2/3] Optimizing per-target weights ({n_models} models, step=0.10)...")
    train_ranks = [
        to_ranks(train_nn),
        to_ranks(train_lgbm),
        to_ranks(train_xgb),
        to_ranks(train_pb),
        to_ranks(train_cb),
        to_ranks(train_lgbm_meta),
    ]
    test_ranks = [
        to_ranks(test_nn),
        to_ranks(test_lgbm),
        to_ranks(test_xgb),
        to_ranks(test_pb),
        to_ranks(test_cb),
        to_ranks(test_lgbm_meta),
    ]

    weights = optimize_per_target(train_ranks, y, target_cols, n_models, step=0.10)

    n_targets = len(target_cols)
    blend_train = np.zeros_like(train_nn)
    blend_test = np.zeros_like(test_nn)
    for i in range(n_targets):
        w = weights[i]
        blend_train[:, i] = sum(w[j] * train_ranks[j][:, i] for j in range(n_models))
        blend_test[:, i] = sum(w[j] * test_ranks[j][:, i] for j in range(n_models))

    blend_auc, _ = compute_macro_auc(y, blend_train, target_cols)
    print(f"  Blend train AUC: {blend_auc:.5f}")

    avg_w = weights.mean(axis=0)
    for name, w in zip(model_names, avg_w):
        print(f"    {name}: {w:.2f}")

    print("\n[3/3] Saving...")
    test_ids = pl.read_parquet(f"{DATA_DIR}test_main_features.parquet", columns=["customer_id"])
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_ids["customer_id"]}).hstack(
        pl.DataFrame(blend_test.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/blend.parquet")
    print("  Saved: submissions/blend.parquet")

    Path("blend_artifacts").mkdir(exist_ok=True)
    np.savez_compressed(
        "blend_artifacts/blend_data.npz",
        train_nn=train_nn,
        test_nn=test_nn,
        train_lgbm=train_lgbm,
        test_lgbm=test_lgbm,
        train_xgb=train_xgb,
        test_xgb=test_xgb,
        train_pb=train_pb,
        test_pb=test_pb,
        train_cb=train_cb,
        test_cb=test_cb,
        train_lgbm_meta=train_lgbm_meta,
        test_lgbm_meta=test_lgbm_meta,
        blend_train=blend_train,
        blend_test=blend_test,
        weights=weights,
    )
    with open("blend_artifacts/blend_weights.pkl", "wb") as f:
        pickle.dump(
            {
                "model_names": model_names,
                "target_cols": target_cols,
                "weights": weights,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("  Saved: blend_artifacts/blend_data.npz")
    print("  Saved: blend_artifacts/blend_weights.pkl")

    print(f"\nDone in {time.time() - t0:.1f}s. Blend train AUC={blend_auc:.5f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
