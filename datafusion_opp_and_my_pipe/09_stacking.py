"""Step 9: Stacking without OOF.

Meta-features: 6x41 base + pairwise interactions + aggregate rank features.
LGBM stacker is trained once on full train with a single holdout split for early stopping.
Final combo: alpha * rank(meta) + (1 - alpha) * rank(blend).

Output: submissions/stacking.parquet

Runtime: ~3-5 minutes.
"""

import os
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import DATA_DIR, SEED, compute_macro_auc, to_ranks, verify_submission


def make_validation_split(y_train):
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(splitter.split(np.arange(len(y_train)), y_train))
    return tr_idx, val_idx


def build_meta_features(*train_arrays):
    """Build meta-feature matrix from N models."""
    parts = list(train_arrays)
    for a, b in combinations(train_arrays, 2):
        parts.append(np.abs(a - b))
    for a, b in combinations(train_arrays, 2):
        parts.append(a * b)

    stacked = np.stack(train_arrays, axis=0)
    parts.append(stacked.std(axis=0).astype(np.float32))
    parts.append(stacked.max(axis=0).astype(np.float32))
    parts.append(stacked.min(axis=0).astype(np.float32))

    for arr in train_arrays:
        parts.append(to_ranks(arr).astype(np.float32))

    return np.hstack(parts).astype(np.float32)


def stack_lgbm_meta(X_train, y_train, X_test, target_cols, device_params):
    """Train one LightGBM stacker per target without cross-validation."""
    import lightgbm as lgb

    n_targets = y_train.shape[1]
    train_preds = np.zeros((X_train.shape[0], n_targets), dtype=np.float32)
    test_preds = np.zeros((X_test.shape[0], n_targets), dtype=np.float32)
    best_iterations = np.zeros(n_targets, dtype=np.int32)
    model_dir = Path("checkpoints_stacking/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    tr_idx, val_idx = make_validation_split(y_train)
    params = dict(
        objective="binary",
        metric="auc",
        num_leaves=8,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_samples=1000,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )

    t0 = time.time()
    for t_idx, target_name in enumerate(target_cols):
        y_t = y_train[:, t_idx]

        search_model = lgb.LGBMClassifier(**params, **device_params)
        search_model.fit(
            X_train[tr_idx],
            y_t[tr_idx],
            eval_set=[(X_train[val_idx], y_t[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)],
        )
        best_iteration = getattr(search_model, "best_iteration_", None)
        if best_iteration is None or best_iteration <= 0:
            best_iteration = params["n_estimators"]
        best_iterations[t_idx] = int(best_iteration)

        final_model = lgb.LGBMClassifier(
            **{**params, "n_estimators": int(best_iteration)},
            **device_params,
        )
        final_model.fit(X_train, y_t)
        train_preds[:, t_idx] = final_model.predict_proba(X_train)[:, 1]
        test_preds[:, t_idx] = final_model.predict_proba(X_test)[:, 1]

        with open(model_dir / f"{target_name}.pkl", "wb") as f:
            pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        if (t_idx + 1) % 10 == 0 or t_idx == n_targets - 1:
            elapsed = time.time() - t0
            print(f"    LGBM meta: {t_idx + 1}/{n_targets} targets ({elapsed:.1f}s)", flush=True)

    return train_preds, test_preds, best_iterations


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


def optimize_rank_blend(train_list, y, target_cols, n_models, step=0.10):
    """Per-target weight optimization for N-model rank blend."""
    train_ranks = [to_ranks(arr) for arr in train_list]
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
    return weights, train_ranks


def main():
    t0 = time.time()
    model_names = ["NN", "LGBM", "XGBoost", "PyBoost", "CatBoost", "LGBM_meta"]
    n_models = len(model_names)
    print("=" * 60)
    print(f"Step 9: Stacking (single-fit, {n_models} models)")
    print("=" * 60)

    device_params = {}
    print("  LightGBM meta: CPU mode")

    train_tgt = pl.read_parquet(f"{DATA_DIR}train_target.parquet")
    target_cols = [c for c in train_tgt.columns if c.startswith("target_")]
    y = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    print("\n[1/4] Loading predictions...")
    d = np.load("blend_artifacts/blend_data.npz")
    train_nn = d["train_nn"].astype(np.float32)
    test_nn = d["test_nn"].astype(np.float32)
    train_lgbm = d["train_lgbm"].astype(np.float32)
    test_lgbm = d["test_lgbm"].astype(np.float32)
    train_xgb = d["train_xgb"].astype(np.float32)
    test_xgb = d["test_xgb"].astype(np.float32)
    train_pb = d["train_pb"].astype(np.float32)
    test_pb = d["test_pb"].astype(np.float32)
    train_cb = d["train_cb"].astype(np.float32)
    test_cb = d["test_cb"].astype(np.float32)
    train_lgbm_meta = d["train_lgbm_meta"].astype(np.float32)
    test_lgbm_meta = d["test_lgbm_meta"].astype(np.float32)
    blend_train = d["blend_train"].astype(np.float32)
    blend_test = d["blend_test"].astype(np.float32)

    for name, preds in [
        ("NN", train_nn),
        ("LGBM", train_lgbm),
        ("XGBoost", train_xgb),
        ("PyBoost", train_pb),
        ("CatBoost", train_cb),
        ("LGBM_meta", train_lgbm_meta),
    ]:
        auc, _ = compute_macro_auc(y, preds, target_cols)
        print(f"  {name}: {auc:.4f}")

    print(f"\n  Computing rank blend baseline ({n_models} models)...")
    blend_weights, train_ranks = optimize_rank_blend(
        [train_nn, train_lgbm, train_xgb, train_pb, train_cb, train_lgbm_meta],
        y,
        target_cols,
        n_models,
    )
    test_ranks = [
        to_ranks(test_nn),
        to_ranks(test_lgbm),
        to_ranks(test_xgb),
        to_ranks(test_pb),
        to_ranks(test_cb),
        to_ranks(test_lgbm_meta),
    ]

    baseline_train = np.zeros_like(train_nn)
    baseline_test = np.zeros_like(test_nn)
    for i in range(len(target_cols)):
        w = blend_weights[i]
        baseline_train[:, i] = sum(w[j] * train_ranks[j][:, i] for j in range(n_models))
        baseline_test[:, i] = sum(w[j] * test_ranks[j][:, i] for j in range(n_models))

    baseline_auc, _ = compute_macro_auc(y, baseline_train, target_cols)
    print(f"  Rank blend baseline: {baseline_auc:.4f}")

    print("\n[2/4] Building meta-features...")
    X_meta_train = build_meta_features(
        train_nn,
        train_lgbm,
        train_xgb,
        train_pb,
        train_cb,
        train_lgbm_meta,
    )
    X_meta_test = build_meta_features(
        test_nn,
        test_lgbm,
        test_xgb,
        test_pb,
        test_cb,
        test_lgbm_meta,
    )
    print(f"  Meta-features: {X_meta_train.shape[1]}")

    print("\n[3/4] LGBM meta stacking...")
    lgbm_meta_train, lgbm_meta_test, best_iterations = stack_lgbm_meta(
        X_meta_train,
        y,
        X_meta_test,
        target_cols,
        device_params,
    )
    lgbm_meta_auc, _ = compute_macro_auc(y, lgbm_meta_train, target_cols)
    print(f"  LGBM meta train AUC: {lgbm_meta_auc:.4f} (vs baseline: {lgbm_meta_auc - baseline_auc:+.4f})")

    print("\n[4/4] Optimizing combo...")
    meta_train_rank = to_ranks(lgbm_meta_train)
    best_combo_auc, best_alpha = 0.0, 0.5
    for alpha in np.arange(0, 1.05, 0.05):
        combo = alpha * meta_train_rank + (1 - alpha) * blend_train
        auc, _ = compute_macro_auc(y, combo, target_cols)
        if auc > best_combo_auc:
            best_combo_auc = auc
            best_alpha = float(alpha)

    meta_test_rank = to_ranks(lgbm_meta_test)
    combo_test = best_alpha * meta_test_rank + (1 - best_alpha) * blend_test
    print(f"  Combo: LGBM meta {best_alpha:.0%} + blend {1 - best_alpha:.0%}, train AUC {best_combo_auc:.4f}")

    print(f"\n{'=' * 60}")
    results = [
        ("Rank blend", baseline_auc, baseline_test),
        ("LGBM meta", lgbm_meta_auc, lgbm_meta_test),
        ("Combo", best_combo_auc, combo_test),
    ]
    best_name, best_auc, best_preds = max(results, key=lambda x: x[1])
    for name, auc, _ in sorted(results, key=lambda x: -x[1]):
        marker = " <<<" if name == best_name else ""
        print(f"  {name:<20s} {auc:.4f}{marker}")

    test_ids = pl.read_parquet(f"{DATA_DIR}test_main_features.parquet", columns=["customer_id"])
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_ids["customer_id"]}).hstack(
        pl.DataFrame(best_preds.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/stacking.parquet")

    ckpt_dir = Path("checkpoints_stacking")
    ckpt_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        ckpt_dir / "stacking_predictions.npz",
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        meta_train=lgbm_meta_train,
        meta_test=lgbm_meta_test,
        combo_test=combo_test,
        best_iterations=best_iterations,
    )
    with open(ckpt_dir / "stacking_artifacts.pkl", "wb") as f:
        pickle.dump(
            {
                "blend_weights": blend_weights,
                "best_alpha": best_alpha,
                "model_names": model_names,
                "target_cols": target_cols,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\n  Saved: submissions/stacking.parquet")
    print("  Saved: checkpoints_stacking/stacking_predictions.npz")
    print("  Saved: checkpoints_stacking/stacking_artifacts.pkl")
    print(f"  Best: {best_name} (train AUC {best_auc:.4f})")
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
