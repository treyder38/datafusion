"""Step 4: Train XGBoost (single-fit, no OOF).

Loads features from features/, trains 41 per-target binary classifiers.
Targets are trained in parallel batches for speed (XGBoost releases GIL).

Output: checkpoints_xgboost/xgb_predictions.npz (train_preds, test_preds)

Runtime: ~10-20 minutes.
"""

import gc
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import DATA_DIR, SEED, compute_macro_auc, effective_number_weight, log_per_target_auc

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_xgboost")
MODELS_DIR = CHECKPOINT_DIR / "models"

N_CPUS = os.cpu_count() or 8
PARALLEL_TARGETS = min(4, max(2, N_CPUS // 16))
THREADS_PER_MODEL = max(1, N_CPUS // PARALLEL_TARGETS)

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=50,
    n_estimators=2000,
    subsample=0.7,
    colsample_bytree=0.3,
    reg_alpha=5.0,
    reg_lambda=5.0,
    gamma=0.5,
    random_state=SEED,
    verbosity=0,
)
EARLY_STOPPING_ROUNDS = 100


def load_xgb_params():
    """Load best params from Optuna tuning, fall back to defaults."""
    params = dict(XGB_PARAMS)
    params_path = CHECKPOINT_DIR / "best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"  Loaded tuned params from {params_path}")
        params.update(tuned)
        for k, v in tuned.items():
            print(f"    {k}: {v}")
    else:
        print("  No tuned params found, using defaults")
    return params


def make_validation_split(y_train):
    """Single holdout split for early stopping only."""
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(splitter.split(np.arange(len(y_train)), y_train))
    return tr_idx, val_idx


def _best_iteration_or_default(model, default_n_estimators):
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None or best_iteration < 0:
        return default_n_estimators
    return int(best_iteration) + 1


def _train_one_target(
    target_idx,
    target_name,
    threads,
    X_fit,
    y_fit,
    X_val,
    y_val,
    X_train,
    y_train,
    X_test,
    xgb_params,
):
    """Train a single target with a holdout for early stopping, then refit on full train."""
    y_fit_t = y_fit[:, target_idx]
    fit_spw = effective_number_weight(int((y_fit_t == 1).sum()), int((y_fit_t == 0).sum()))
    fit_params = {**xgb_params, "nthread": threads, "scale_pos_weight": fit_spw}

    model = xgb.XGBClassifier(**fit_params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_fit,
        y_fit_t,
        eval_set=[(X_val, y_val[:, target_idx])],
        verbose=False,
    )

    best_n_estimators = _best_iteration_or_default(model, xgb_params["n_estimators"])

    y_full_t = y_train[:, target_idx]
    full_spw = effective_number_weight(int((y_full_t == 1).sum()), int((y_full_t == 0).sum()))
    final_params = {
        **xgb_params,
        "nthread": threads,
        "n_estimators": best_n_estimators,
        "scale_pos_weight": full_spw,
    }
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_full_t, verbose=False)

    train_preds = final_model.predict_proba(X_train)[:, 1]
    test_preds = final_model.predict_proba(X_test)[:, 1]
    importance = final_model.feature_importances_

    with open(MODELS_DIR / f"{target_name}.pkl", "wb") as f:
        pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_preds, test_preds, importance, best_n_estimators


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 4: Train XGBoost (single-fit x 41 targets)")
    print("=" * 60)
    print(f"  XGBoost: CPU mode, {PARALLEL_TARGETS} parallel targets x {THREADS_PER_MODEL} threads")

    xgb_params = load_xgb_params()

    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    all_feature_cols = meta["feature_names"]
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(all_feature_cols)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "xgb_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    tr_idx, val_idx = make_validation_split(y_train)
    X_fit = X_train[tr_idx]
    X_val = X_train[val_idx]
    y_fit = y_train[tr_idx]
    y_val = y_train[val_idx]
    print(f"\n[2/4] Training {len(target_cols)} targets with a single validation split...")
    print(f"  Holdout: train={len(tr_idx):,}, val={len(val_idx):,}", flush=True)

    n_targets = len(target_cols)
    train_preds = np.zeros((X_train.shape[0], n_targets), dtype=np.float32)
    test_preds = np.zeros((X_test.shape[0], n_targets), dtype=np.float32)
    importance_sum = np.zeros((len(all_feature_cols), n_targets), dtype=np.float64)
    best_iterations = np.zeros(n_targets, dtype=np.int32)

    with ThreadPoolExecutor(max_workers=PARALLEL_TARGETS) as executor:
        futures = {}
        for i, col in enumerate(target_cols):
            futures[
                executor.submit(
                    _train_one_target,
                    i,
                    col,
                    THREADS_PER_MODEL,
                    X_fit,
                    y_fit,
                    X_val,
                    y_val,
                    X_train,
                    y_train,
                    X_test,
                    xgb_params,
                )
            ] = i

        done_count = 0
        for future in as_completed(futures):
            i = futures[future]
            train_p, test_p, imp, best_iter = future.result()
            train_preds[:, i] = train_p
            test_preds[:, i] = test_p
            importance_sum[:, i] += imp
            best_iterations[i] = best_iter
            done_count += 1
            if done_count % 10 == 0 or done_count == n_targets:
                print(f"    {done_count}/{n_targets} targets done", flush=True)

    train_auc, per_target_aucs = compute_macro_auc(y_train, train_preds, target_cols)
    print("\n[3/4] Results:")
    print(f"  Train Macro ROC-AUC: {train_auc:.4f}")
    log_per_target_auc(per_target_aucs, y_train, target_cols)

    np.savez(
        cache_file,
        train_preds=train_preds,
        test_preds=test_preds,
        best_iterations=best_iterations,
    )
    print(f"  Saved: {cache_file}")

    imp_path = CHECKPOINT_DIR / "feature_importances.json"
    imp_data = {
        "feature_names": all_feature_cols,
        "target_cols": target_cols,
        "importances_per_target": {
            col: dict(
                sorted(
                    zip(all_feature_cols, importance_sum[:, i].tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            for i, col in enumerate(target_cols)
        },
        "mean_importance": dict(
            sorted(
                zip(all_feature_cols, importance_sum.mean(axis=1).tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        ),
    }
    with open(imp_path, "w") as f:
        json.dump(imp_data, f, indent=2)
    print(f"  Saved: {imp_path}")
    print(f"  Models: {MODELS_DIR}/ ({n_targets} .pkl files)")

    print("\n[4/4] Saving submission...")
    from utils import verify_submission

    sample = pl.read_parquet(f"{DATA_DIR}sample_submit.parquet")
    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = pl.DataFrame({"customer_id": test_feat["customer_id"]}).hstack(
        pl.DataFrame(test_preds.astype(np.float64), schema=predict_cols)
    )
    verify_submission(submit, sample)
    Path("submissions").mkdir(exist_ok=True)
    submit.write_parquet("submissions/xgboost.parquet")
    print("  Saved: submissions/xgboost.parquet")

    del X_fit, X_val, y_fit, y_val
    gc.collect()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Train AUC={train_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
