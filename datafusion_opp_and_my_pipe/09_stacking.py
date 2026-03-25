"""Step 9: Stacking — LGBM meta-learner + combo (7 models including TabR).

Meta-features: 7x41 base + C(7,2)x41 pairwise |diffs| + C(7,2)x41 pairwise prods + aggregates.
LGBM meta-learner trained with 4-fold OOF.
Final combo: alpha * rank(meta) + (1-alpha) * rank(blend).

Output: submissions/stacking.parquet

Runtime: ~3-5 minutes.
"""

import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import (
    SEED,
    DATA_DIR,
    compute_macro_auc,
    to_ranks,
    verify_submission,
)

N_META_FOLDS = 4


def apply_logit_transform(*arrays):
    """Apply logit transformation to probability arrays.

    Logit maps [0, 1] → (-∞, +∞), making probability distributions more linear.
    This helps tree-based meta-learners find better weight combinations.
    Probabilities are clipped to [1e-6, 1-1e-6] to avoid infinities.
    """
    transformed = []
    for arr in arrays:
        # Clip to avoid infinities at boundaries
        clipped = np.clip(arr, 1e-6, 1 - 1e-6)
        # Apply logit: log(p / (1-p))
        logit_arr = logit(clipped).astype(np.float32)
        transformed.append(logit_arr)
    return transformed


def build_meta_features(*oof_arrays):
    """Build meta-feature matrix from N models.

    For 5 models: 5*41 base + C(5,2)*41 diffs + C(5,2)*41 prods
                  + 41 std + 41 max + 41 min + 5*41 ranks = extended features.
    """
    parts = list(oof_arrays)  # base predictions
    for a, b in combinations(oof_arrays, 2):
        parts.append(np.abs(a - b))
    for a, b in combinations(oof_arrays, 2):
        parts.append(a * b)

    # Disagreement / aggregate features
    stacked = np.stack(oof_arrays, axis=0)  # (n_models, n_samples, n_targets)
    parts.append(stacked.std(axis=0).astype(np.float32))   # std across models
    parts.append(stacked.max(axis=0).astype(np.float32))   # max across models
    parts.append(stacked.min(axis=0).astype(np.float32))   # min across models

    # Rank-based features per model
    for arr in oof_arrays:
        parts.append(to_ranks(arr).astype(np.float32))

    return np.hstack(parts).astype(np.float32)


def stack_lgbm_meta(X_train, y_train, X_test, target_cols, device_params):
    """LGBM per-target stacking."""
    import lightgbm as lgb

    n_train, n_targets = y_train.shape
    n_test = X_test.shape[0]
    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds = np.zeros((n_test, n_targets), dtype=np.float32)

    kf = MultilabelStratifiedKFold(n_splits=N_META_FOLDS, shuffle=True, random_state=SEED)
    params = dict(objective="binary", metric="auc", num_leaves=8, max_depth=3,
                  learning_rate=0.05, n_estimators=200, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                  min_child_samples=1000, random_state=SEED, verbose=-1, n_jobs=-1)

    t0 = time.time()
    for t_idx in range(n_targets):
        y_t = y_train[:, t_idx]
        for tr_idx, val_idx in kf.split(np.arange(n_train), y_train):
            model = lgb.LGBMClassifier(**params, **device_params)
            model.fit(X_train[tr_idx], y_t[tr_idx],
                      eval_set=[(X_train[val_idx], y_t[val_idx])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            oof_preds[val_idx, t_idx] = model.predict_proba(X_train[val_idx])[:, 1]
            test_preds[:, t_idx] += model.predict_proba(X_test)[:, 1] / N_META_FOLDS

        if (t_idx + 1) % 10 == 0 or t_idx == n_targets - 1:
            elapsed = time.time() - t0
            print(f"    LGBM meta: {t_idx+1}/{n_targets} targets ({elapsed:.1f}s)", flush=True)

    return oof_preds, test_preds


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


def optimize_rank_blend(oof_list, y, target_cols, n_models, step=0.10):
    """Per-target weight optimization for N-model rank blend."""
    oof_ranks = [to_ranks(arr) for arr in oof_list]
    n_targets = len(target_cols)
    default_w = [1.0 / n_models] * n_models
    weights = np.zeros((n_targets, n_models))
    combos = list(_gen_weight_combos(n_models, step))
    for i in range(n_targets):
        y_t = y[:, i]
        if y_t.sum() < 2 or (len(y_t) - y_t.sum()) < 2:
            weights[i] = default_w
            continue
        ranks_i = np.column_stack([oof_ranks[m][:, i] for m in range(n_models)])
        best_auc, best_w = 0.0, default_w[:]
        for combo in combos:
            blended = ranks_i @ np.array(combo)
            auc = roc_auc_score(y_t, blended)
            if auc > best_auc:
                best_auc = auc
                best_w = list(combo)
        weights[i] = best_w
    return weights, oof_ranks


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 9: Stacking (LGBM meta + combo)")
    print("=" * 60)

    # Keep the meta-learner on CPU. This stage is small enough that CPU is fast,
    # and GPU/OpenCL builds can fail on some drivers during program compilation.
    device_params = {}
    print("  LightGBM meta: CPU mode")

    # Load targets
    train_tgt = pl.read_parquet(f"{DATA_DIR}train_target.parquet")
    target_cols = [c for c in train_tgt.columns if c.startswith("target_")]
    y = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    n_train, n_targets = y.shape

    # Load predictions — dynamic model list (TabR optional)
    print("\n[1/4] Loading predictions...")
    d = np.load("blend_artifacts/blend_data.npz")
    model_names = []
    oof_arrays, test_arrays = [], []

    oof_nn = d["oof_nn"].astype(np.float32)
    test_nn = d["test_nn"].astype(np.float32)
    model_names.append("NN"); oof_arrays.append(oof_nn); test_arrays.append(test_nn)

    if "oof_tabr" in d:
        oof_tabr = d["oof_tabr"].astype(np.float32)
        test_tabr = d["test_tabr"].astype(np.float32)
        model_names.append("TabR"); oof_arrays.append(oof_tabr); test_arrays.append(test_tabr)

    oof_lgbm = d["oof_lgbm"].astype(np.float32); test_lgbm = d["test_lgbm"].astype(np.float32)
    model_names.append("LGBM"); oof_arrays.append(oof_lgbm); test_arrays.append(test_lgbm)
    oof_xgb = d["oof_xgb"].astype(np.float32); test_xgb = d["test_xgb"].astype(np.float32)
    model_names.append("XGBoost"); oof_arrays.append(oof_xgb); test_arrays.append(test_xgb)
    oof_pb = d["oof_pb"].astype(np.float32); test_pb = d["test_pb"].astype(np.float32)
    model_names.append("PyBoost"); oof_arrays.append(oof_pb); test_arrays.append(test_pb)
    oof_cb = d["oof_cb"].astype(np.float32); test_cb = d["test_cb"].astype(np.float32)
    model_names.append("CatBoost"); oof_arrays.append(oof_cb); test_arrays.append(test_cb)
    oof_lgbm_meta = d["oof_lgbm_meta"].astype(np.float32); test_lgbm_meta = d["test_lgbm_meta"].astype(np.float32)
    model_names.append("LGBM_meta"); oof_arrays.append(oof_lgbm_meta); test_arrays.append(test_lgbm_meta)

    n_models = len(model_names)
    n_test = test_nn.shape[0]
    for name, oof in zip(model_names, oof_arrays):
        auc, _ = compute_macro_auc(y, oof, target_cols)
        print(f"  {name}: {auc:.4f}")
    print(f"  Total models: {n_models}")

    # Rank blend baseline
    print(f"\n  Computing rank blend baseline ({n_models} models)...")
    blend_weights, oof_ranks = optimize_rank_blend(oof_arrays, y, target_cols, n_models)
    test_ranks = [to_ranks(t) for t in test_arrays]

    baseline_oof = np.zeros_like(oof_nn)
    baseline_test = np.zeros_like(test_nn)
    for i in range(n_targets):
        w = blend_weights[i]
        baseline_oof[:, i] = sum(w[j] * oof_ranks[j][:, i] for j in range(n_models))
        baseline_test[:, i] = sum(w[j] * test_ranks[j][:, i] for j in range(n_models))

    baseline_auc, _ = compute_macro_auc(y, baseline_oof, target_cols)
    print(f"  Rank blend baseline: {baseline_auc:.4f}")

    # Build meta-features with logit calibration
    print(f"\n[2/4] Building meta-features with logit calibration ({n_models} models)...")
    oof_logit = apply_logit_transform(*oof_arrays)
    test_logit = apply_logit_transform(*test_arrays)

    X_meta_train = build_meta_features(*oof_logit)
    X_meta_test = build_meta_features(*test_logit)
    print(f"  Meta-features: {X_meta_train.shape[1]} (from logit-transformed probabilities)")

    # LGBM stacking
    print("\n[3/4] LGBM meta stacking...")
    lgbm_meta_oof, lgbm_meta_test = stack_lgbm_meta(
        X_meta_train, y, X_meta_test, target_cols, device_params
    )
    lgbm_meta_auc, _ = compute_macro_auc(y, lgbm_meta_oof, target_cols)
    print(f"  LGBM meta OOF: {lgbm_meta_auc:.4f} (vs baseline: {lgbm_meta_auc - baseline_auc:+.4f})")

    # Combo
    print("\n[4/4] Optimizing combo...")
    meta_oof_rank = to_ranks(lgbm_meta_oof)
    best_combo_auc, best_alpha = 0, 0.5
    for alpha in np.arange(0, 1.05, 0.05):
        combo = alpha * meta_oof_rank + (1 - alpha) * baseline_oof
        auc, _ = compute_macro_auc(y, combo, target_cols)
        if auc > best_combo_auc:
            best_combo_auc = auc
            best_alpha = alpha

    meta_test_rank = to_ranks(lgbm_meta_test)
    combo_test = best_alpha * meta_test_rank + (1 - best_alpha) * baseline_test
    print(f"  Combo: LGBM meta {best_alpha:.0%} + blend {1-best_alpha:.0%}, "
          f"OOF {best_combo_auc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    results = [
        ("Rank blend", baseline_auc, baseline_test),
        ("LGBM meta", lgbm_meta_auc, lgbm_meta_test),
        ("Combo", best_combo_auc, combo_test),
    ]
    best_name, best_auc, best_preds = max(results, key=lambda x: x[1])
    for name, auc, _ in sorted(results, key=lambda x: -x[1]):
        marker = " <<<" if name == best_name else ""
        print(f"  {name:<20s} {auc:.4f}{marker}")

    # Save
    test_ids = pl.read_parquet(f"{DATA_DIR}test_main_features.parquet", columns=["customer_id"])
    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_ids["customer_id"]}).hstack(
        pl.DataFrame(best_preds.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/stacking.parquet")
    print(f"\n  Saved: submissions/stacking.parquet")
    print(f"  Best: {best_name} (OOF {best_auc:.4f})")
    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
