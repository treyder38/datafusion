"""Step 7: Train LGBM with cross-target meta-features (single-fit, no OOF).

Uses train predictions from NN, LGBM, XGBoost, PyBoost, CatBoost as additional features.
For each target_i: base features + meta-features (excluding the own target columns
from every base model to reduce direct leakage).

Output: checkpoints_lgbm_meta/lgbm_predictions.npz (train_preds, test_preds)

Runtime: ~10-15 minutes.
"""

import gc
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import DATA_DIR, SEED, compute_macro_auc, effective_number_weight, log_per_target_auc

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_lgbm_meta")
MODELS_DIR = CHECKPOINT_DIR / "models"

N_CPUS = os.cpu_count() or 8

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
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 100


def load_lgbm_meta_params():
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


def load_nn_predictions():
    d = np.load("checkpoints_nn/nn_predictions.npz")
    return d["train_preds"].astype(np.float32), d["test_preds"].astype(np.float32)


def load_model_predictions():
    """Load train and test predictions from all 5 base models."""
    print("\n  Loading train predictions from 5 models...")

    nn_train, nn_test = load_nn_predictions()
    print(f"    NN: train {nn_train.shape}, test {nn_test.shape}")

    d = np.load("checkpoints_lgbm/lgbm_predictions.npz")
    lgbm_train = d["train_preds"].astype(np.float32)
    lgbm_test = d["test_preds"].astype(np.float32)
    print(f"    LGBM: train {lgbm_train.shape}, test {lgbm_test.shape}")

    d = np.load("checkpoints_xgboost/xgb_predictions.npz")
    xgb_train = d["train_preds"].astype(np.float32)
    xgb_test = d["test_preds"].astype(np.float32)
    print(f"    XGBoost: train {xgb_train.shape}, test {xgb_test.shape}")

    d = np.load("checkpoints_pyboost/pyboost_predictions.npz")
    pb_train = d["train_preds"].astype(np.float32)
    pb_test = d["test_preds"].astype(np.float32)
    print(f"    PyBoost: train {pb_train.shape}, test {pb_test.shape}")

    d = np.load("checkpoints_catboost/cb_predictions.npz")
    cb_train = d["train_preds"].astype(np.float32)
    cb_test = d["test_preds"].astype(np.float32)
    print(f"    CatBoost: train {cb_train.shape}, test {cb_test.shape}")

    meta_train = np.hstack([lgbm_train, nn_train, xgb_train, pb_train, cb_train])
    meta_test = np.hstack([lgbm_test, nn_test, xgb_test, pb_test, cb_test])
    print(f"    Combined meta: {meta_train.shape}")
    return meta_train, meta_test


def best_iteration_or_default(model, default_n_estimators):
    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is None or best_iteration <= 0:
        return default_n_estimators
    return int(best_iteration)


def main():
    t0 = time.time()
    n_models = 5
    print("=" * 60)
    print(f"Step 7: LGBM with cross-target meta-features ({n_models} models, single-fit)")
    print("=" * 60)

    device_params, supports_cat = {}, True
    print(f"  LightGBM: CPU mode, {N_CPUS} threads")

    lgbm_params = load_lgbm_meta_params()

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

    X_train_base = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    X_test_base = test_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    n_targets = len(target_cols)
    n_base = len(feature_cols)

    meta_train, meta_test = load_model_predictions()
    X_train_full = np.hstack([X_train_base, meta_train])
    X_test_full = np.hstack([X_test_base, meta_test])
    del X_train_base, X_test_base, meta_train, meta_test
    gc.collect()

    n_total = X_train_full.shape[1]
    print(f"\n  Features: {n_base} base + {n_models}x{n_targets} meta = {n_total} total")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "lgbm_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    tr_idx, val_idx = make_validation_split(y_train)
    print("\n[2/4] Training single-fit meta models...")
    print(f"  Holdout: train={len(tr_idx):,}, val={len(val_idx):,}", flush=True)

    exclude_cols_per_target = []
    for i in range(n_targets):
        exclude_cols_per_target.append([n_base + m * n_targets + i for m in range(n_models)])

    X_fit_full = X_train_full[tr_idx].copy()
    X_val_full = X_train_full[val_idx].copy()
    X_train_maskable = X_train_full.copy()
    X_test_maskable = X_test_full.copy()
    y_fit = y_train[tr_idx]
    y_val = y_train[val_idx]

    train_preds = np.zeros((X_train_full.shape[0], n_targets), dtype=np.float32)
    test_preds = np.zeros((X_test_full.shape[0], n_targets), dtype=np.float32)
    importance_sum = np.zeros((n_total, n_targets), dtype=np.float64)
    best_iterations = np.zeros(n_targets, dtype=np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for i, col in enumerate(target_cols):
            exclude_cols = exclude_cols_per_target[i]
            saved = {}
            for c in exclude_cols:
                saved[c] = (
                    X_fit_full[:, c].copy(),
                    X_val_full[:, c].copy(),
                    X_train_maskable[:, c].copy(),
                    X_test_maskable[:, c].copy(),
                )
                X_fit_full[:, c] = np.nan
                X_val_full[:, c] = np.nan
                X_train_maskable[:, c] = np.nan
                X_test_maskable[:, c] = np.nan

            y_fit_t = y_fit[:, i]
            fit_spw = effective_number_weight(int((y_fit_t == 1).sum()), int((y_fit_t == 0).sum()))
            search_params = {**lgbm_params, "scale_pos_weight": fit_spw, **device_params}
            search_model = lgb.LGBMClassifier(**search_params)
            search_model.fit(
                X_fit_full,
                y_fit_t,
                eval_set=[(X_val_full, y_val[:, i])],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
                categorical_feature=cat_indices if supports_cat else "auto",
            )
            best_n_estimators = best_iteration_or_default(search_model, lgbm_params["n_estimators"])
            best_iterations[i] = best_n_estimators

            y_full_t = y_train[:, i]
            full_spw = effective_number_weight(int((y_full_t == 1).sum()), int((y_full_t == 0).sum()))
            final_params = {
                **lgbm_params,
                "n_estimators": best_n_estimators,
                "scale_pos_weight": full_spw,
                **device_params,
            }
            final_model = lgb.LGBMClassifier(**final_params)
            final_model.fit(
                X_train_maskable,
                y_full_t,
                categorical_feature=cat_indices if supports_cat else "auto",
            )

            train_preds[:, i] = final_model.predict_proba(X_train_maskable)[:, 1]
            test_preds[:, i] = final_model.predict_proba(X_test_maskable)[:, 1]
            importance_sum[:, i] += final_model.booster_.feature_importance(importance_type="gain")

            with open(MODELS_DIR / f"{col}.pkl", "wb") as f:
                pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)

            for c in exclude_cols:
                X_fit_full[:, c] = saved[c][0]
                X_val_full[:, c] = saved[c][1]
                X_train_maskable[:, c] = saved[c][2]
                X_test_maskable[:, c] = saved[c][3]

            del saved, search_model, final_model
            if (i + 1) % 10 == 0 or i == n_targets - 1:
                print(f"    {i + 1}/{n_targets} targets done", flush=True)

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

    model_names = ["lgbm", "nn", "xgboost", "pyboost", "catboost"]
    meta_col_names = [f"meta_{m_name}_{tcol}" for m_name in model_names for tcol in target_cols]
    all_feature_names = list(all_feature_cols) + meta_col_names

    imp_path = CHECKPOINT_DIR / "feature_importances.json"
    imp_data = {
        "feature_names": all_feature_names,
        "target_cols": target_cols,
        "importances_per_target": {
            col: dict(
                sorted(
                    zip(all_feature_names, importance_sum[:, i].tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            for i, col in enumerate(target_cols)
        },
        "mean_importance": dict(
            sorted(
                zip(all_feature_names, importance_sum.mean(axis=1).tolist()),
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
    submit.write_parquet("submissions/lgbm_meta.parquet")
    print("  Saved: submissions/lgbm_meta.parquet")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Train AUC={train_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
