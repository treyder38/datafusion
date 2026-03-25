"""Step 3: Train LightGBM (single-fit, no OOF).

Loads features from features/, trains 41 per-target binary classifiers.
Targets are trained in parallel batches for speed.

Output: checkpoints_lgbm/lgbm_predictions.npz (train_preds, test_preds)

Runtime: ~5-10 minutes.
"""

import gc
import json
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import DATA_DIR, SEED, compute_macro_auc, effective_number_weight, log_per_target_auc

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_lgbm")
MODELS_DIR = CHECKPOINT_DIR / "models"

N_CPUS = os.cpu_count() or 8
PARALLEL_TARGETS = min(4, max(2, N_CPUS // 16))
THREADS_PER_MODEL = max(1, N_CPUS // PARALLEL_TARGETS)

LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.050216,
    num_leaves=34,
    max_depth=10,
    min_child_samples=102,
    n_estimators=1500,
    subsample=0.521715,
    colsample_bytree=0.205092,
    reg_alpha=8.146025,
    reg_lambda=7.761833,
    min_split_gain=0.424512,
    subsample_freq=2,
    random_state=SEED,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 100


def load_lgbm_params():
    """Load best params from Optuna tuning, fall back to defaults."""
    params = dict(LGBM_PARAMS)
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
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        return default_n_estimators
    return int(best_iteration)


def _train_one_target(
    target_idx,
    target_name,
    threads,
    cat_indices,
    supports_cat,
    device_params,
    X_fit,
    y_fit,
    X_val,
    y_val,
    X_train,
    y_train,
    X_test,
    lgbm_params,
):
    """Train one target with a single validation split, then refit on full train."""
    y_fit_t = y_fit[:, target_idx]
    fit_spw = effective_number_weight(int((y_fit_t == 1).sum()), int((y_fit_t == 0).sum()))
    fit_params = {**lgbm_params, "n_jobs": threads, "scale_pos_weight": fit_spw, **device_params}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = lgb.LGBMClassifier(**fit_params)
        model.fit(
            X_fit,
            y_fit_t,
            eval_set=[(X_val, y_val[:, target_idx])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
            categorical_feature=cat_indices if supports_cat else "auto",
        )

    best_n_estimators = _best_iteration_or_default(model, lgbm_params["n_estimators"])

    y_full_t = y_train[:, target_idx]
    full_spw = effective_number_weight(int((y_full_t == 1).sum()), int((y_full_t == 0).sum()))
    final_params = {
        **lgbm_params,
        "n_jobs": threads,
        "n_estimators": best_n_estimators,
        "scale_pos_weight": full_spw,
        **device_params,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        final_model = lgb.LGBMClassifier(**final_params)
        final_model.fit(
            X_train,
            y_full_t,
            categorical_feature=cat_indices if supports_cat else "auto",
        )

    train_preds = final_model.predict_proba(X_train)[:, 1]
    test_preds = final_model.predict_proba(X_test)[:, 1]
    importance = final_model.booster_.feature_importance(importance_type="gain")

    with open(MODELS_DIR / f"{target_name}.pkl", "wb") as f:
        pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_preds, test_preds, importance, best_n_estimators


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 3: Train LightGBM (single-fit x 41 targets)")
    print("=" * 60)

    device_params, supports_cat = {}, True
    print(f"  LightGBM: CPU mode, {PARALLEL_TARGETS} parallel targets x {THREADS_PER_MODEL} threads")

    lgbm_params = load_lgbm_params()

    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    all_feature_cols = meta["feature_names"]
    feature_cols = list(all_feature_cols)
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]
    cat_indices = [feature_cols.index(c) for c in cat_feature_names]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(cat_feature_names)} cat, {len(feature_cols) - len(cat_feature_names)} num")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "lgbm_predictions.npz"
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
    importance_sum = np.zeros((len(feature_cols), n_targets), dtype=np.float64)
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
                    cat_indices,
                    supports_cat,
                    device_params,
                    X_fit,
                    y_fit,
                    X_val,
                    y_val,
                    X_train,
                    y_train,
                    X_test,
                    lgbm_params,
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
    submit.write_parquet("submissions/lgbm.parquet")
    print("  Saved: submissions/lgbm.parquet")

    del X_fit, X_val, y_fit, y_val
    gc.collect()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Train AUC={train_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
