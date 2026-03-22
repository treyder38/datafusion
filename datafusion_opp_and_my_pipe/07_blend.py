"""Step 7: Rank per-target blend (NN + LGBM + PyBoost + CatBoost + LGBM meta).

Per-target weight optimization on OOF via grid search (5 models).

Output: submissions/blend.parquet

Runtime: ~5-10 minutes.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc, to_ranks, verify_submission


def load_nn():
    nn_dir = Path("checkpoints_nn")
    oof_parts, test_parts = {}, []
    n_targets = np.load(nn_dir / "fold_0.npz")["val_preds"].shape[1]
    for fi in range(N_FOLDS):
        d = np.load(nn_dir / f"fold_{fi}.npz")
        for idx, pred in zip(d["val_idx"], d["val_preds"]):
            oof_parts[int(idx)] = pred
        test_parts.append(d["test_preds"])
    n_train = max(oof_parts.keys()) + 1
    oof = np.zeros((n_train, n_targets))
    for idx, pred in oof_parts.items():
        oof[idx] = pred
    return oof, np.mean(test_parts, axis=0)


def optimize_per_target(oof_ranks, y, target_cols, n_models, step=0.10):
    """Per-target weight optimization for N-model rank blend."""
    n_targets = len(target_cols)
    default_w = [1.0 / n_models] * n_models
    weights = np.zeros((n_targets, n_models))
    for i in range(n_targets):
        y_t = y[:, i]
        if y_t.sum() < 2 or (len(y_t) - y_t.sum()) < 2:
            weights[i] = default_w
            continue
        best_auc, best_w = 0.0, default_w[:]
        # Grid search over n_models-1 free weights
        grid = np.arange(0, 1.01, step)
        for w0 in grid:
            for w1 in np.arange(0, 1.01 - w0, step):
                for w2 in np.arange(0, 1.01 - w0 - w1, step):
                    for w3 in np.arange(0, 1.01 - w0 - w1 - w2, step):
                        w4 = 1.0 - w0 - w1 - w2 - w3
                        if w4 < -0.001:
                            continue
                        blended = (w0 * oof_ranks[0][:, i] + w1 * oof_ranks[1][:, i]
                                   + w2 * oof_ranks[2][:, i] + w3 * oof_ranks[3][:, i]
                                   + w4 * oof_ranks[4][:, i])
                        auc = roc_auc_score(y_t, blended)
                        if auc > best_auc:
                            best_auc = auc
                            best_w = [w0, w1, w2, w3, w4]
        weights[i] = best_w
    return weights


def main():
    t0 = time.time()
    model_names = ["NN", "LGBM", "PyBoost", "CatBoost", "LGBM_meta"]
    n_models = len(model_names)
    print("=" * 60)
    print(f"Step 7: Rank per-target blend ({' + '.join(model_names)})")
    print("=" * 60)

    # Load targets
    train_tgt = pl.read_parquet(f"{DATA_DIR}train_target.parquet")
    target_cols = [c for c in train_tgt.columns if c.startswith("target_")]
    y = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    # Load predictions
    print("\n[1/3] Loading predictions...")
    oof_nn, test_nn = load_nn()
    nn_auc, _ = compute_macro_auc(y, oof_nn, target_cols)
    print(f"  NN: OOF {nn_auc:.5f}")

    d = np.load("checkpoints_lgbm/lgbm_predictions.npz")
    oof_lgbm, test_lgbm = d["oof_preds"], d["test_preds"]
    lgbm_auc, _ = compute_macro_auc(y, oof_lgbm, target_cols)
    print(f"  LGBM: OOF {lgbm_auc:.5f}")

    d = np.load("checkpoints_pyboost/pyboost_predictions.npz")
    oof_pb, test_pb = d["oof_preds"], d["test_preds"]
    pb_auc, _ = compute_macro_auc(y, oof_pb, target_cols)
    print(f"  PyBoost: OOF {pb_auc:.5f}")

    d = np.load("checkpoints_catboost/cb_predictions.npz")
    oof_cb, test_cb = d["oof_preds"], d["test_preds"]
    cb_auc, _ = compute_macro_auc(y, oof_cb, target_cols)
    print(f"  CatBoost: OOF {cb_auc:.5f}")

    d = np.load("checkpoints_lgbm_meta/lgbm_predictions.npz")
    oof_lgbm_meta, test_lgbm_meta = d["oof_preds"], d["test_preds"]
    lgbm_meta_auc, _ = compute_macro_auc(y, oof_lgbm_meta, target_cols)
    print(f"  LGBM_meta: OOF {lgbm_meta_auc:.5f}")

    # Rank per-target optimization
    print(f"\n[2/3] Optimizing per-target weights ({n_models} models, step=0.10)...")
    oof_ranks = [to_ranks(oof_nn), to_ranks(oof_lgbm), to_ranks(oof_pb),
                 to_ranks(oof_cb), to_ranks(oof_lgbm_meta)]
    test_ranks = [to_ranks(test_nn), to_ranks(test_lgbm), to_ranks(test_pb),
                  to_ranks(test_cb), to_ranks(test_lgbm_meta)]

    weights = optimize_per_target(oof_ranks, y, target_cols, n_models, step=0.10)

    # Build blended predictions
    n_targets = len(target_cols)
    blend_oof = np.zeros_like(oof_nn)
    blend_test = np.zeros_like(test_nn)
    for i in range(n_targets):
        w = weights[i]
        blend_oof[:, i] = sum(w[j] * oof_ranks[j][:, i] for j in range(n_models))
        blend_test[:, i] = sum(w[j] * test_ranks[j][:, i] for j in range(n_models))

    blend_auc, _ = compute_macro_auc(y, blend_oof, target_cols)
    print(f"  Blend OOF: {blend_auc:.5f}")

    # Weight summary
    avg_w = weights.mean(axis=0)
    for name, w in zip(model_names, avg_w):
        print(f"    {name}: {w:.2f}")

    # Save submission
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
    print(f"  Saved: submissions/blend.parquet")

    # Save artifacts for stacking
    Path("blend_artifacts").mkdir(exist_ok=True)
    np.savez_compressed("blend_artifacts/blend_data.npz",
                        oof_nn=oof_nn, test_nn=test_nn,
                        oof_lgbm=oof_lgbm, test_lgbm=test_lgbm,
                        oof_pb=oof_pb, test_pb=test_pb,
                        oof_cb=oof_cb, test_cb=test_cb,
                        oof_lgbm_meta=oof_lgbm_meta, test_lgbm_meta=test_lgbm_meta,
                        weights=weights)

    print(f"\nDone in {time.time()-t0:.1f}s. Blend OOF={blend_auc:.5f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
