"""Step 6: Train CatBoost (single-fit, no OOF).

Loads features from features/, trains 41 per-target binary classifiers.
CatBoost handles categoricals natively.

Output: checkpoints_catboost/cb_predictions.npz (train_preds, test_preds)

Runtime: ~1-3 hours (GPU), ~4-8 hours (CPU).
"""

import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostClassifier, Pool
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import DATA_DIR, SEED, compute_macro_auc, log_per_target_auc

FEATURES_DIR = Path("features")
SELECTED_DIR = FEATURES_DIR / "selected_features"
CHECKPOINT_DIR = Path("checkpoints_catboost")
MODELS_DIR = CHECKPOINT_DIR / "models"

CB_PARAMS_DEFAULT = dict(
    iterations=5000,
    early_stopping_rounds=200,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
)


def load_cb_params():
    """Load best params from Optuna tuning, fall back to defaults."""
    params_path = CHECKPOINT_DIR / "best_params.json"
    base = dict(CB_PARAMS_DEFAULT)
    if params_path.exists():
        with open(params_path) as f:
            tuned = json.load(f)
        print(f"  Loaded tuned params from {params_path}")
        base.update({k: v for k, v in tuned.items() if k not in ("bootstrap_type",)})
        bt = tuned.get("bootstrap_type", "Bayesian")
        base["bootstrap_type"] = bt
        if bt == "Bayesian" and "bagging_temperature" in tuned:
            base["bagging_temperature"] = tuned["bagging_temperature"]
        elif bt == "MVS" and "subsample" in tuned:
            base["subsample"] = tuned["subsample"]
        for k, v in tuned.items():
            print(f"    {k}: {v}")
    else:
        print("  No tuned params found, using defaults")
    return base


def detect_task_type():
    """Detect GPU availability for CatBoost."""
    try:
        cb = CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=0)
        cb.fit([[0, 0], [1, 1]], [0, 1])
        return "GPU", "0"
    except Exception:
        return "CPU", None


def make_validation_split(y_train):
    """Single holdout split for early stopping only."""
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(splitter.split(np.arange(len(y_train)), y_train))
    return tr_idx, val_idx


def best_iterations_or_default(model, default_iterations):
    best_iteration = model.get_best_iteration()
    if best_iteration is None or best_iteration < 0:
        return default_iterations
    return int(best_iteration) + 1


def main():
    t0 = time.time()
    print("=" * 60)
    print("Step 6: Train CatBoost (single-fit x 41 targets)")
    print("=" * 60)

    task_type, devices = detect_task_type()
    print(f"  CatBoost task_type: {task_type}")

    cb_params = load_cb_params()

    print("\n[1/4] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    test_feat = pl.read_parquet(FEATURES_DIR / "test_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.select(feature_cols).to_pandas()
    X_test = test_feat.select(feature_cols).to_pandas()
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    for col in cat_feature_names:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Features: {len(cat_feature_names)} cat, {len(feature_cols) - len(cat_feature_names)} num")

    per_target_feats = {}
    if SELECTED_DIR.exists():
        for target in target_cols:
            fpath = SELECTED_DIR / f"{target}.json"
            if fpath.exists():
                with open(fpath) as f:
                    per_target_feats[target] = [c for c in json.load(f) if c in feature_cols]
        if len(per_target_feats) == len(target_cols):
            n_feats = [len(v) for v in per_target_feats.values()]
            print(f"  Per-target selection: {min(n_feats)}-{max(n_feats)} features/target")
        else:
            per_target_feats = {}
            print(f"  Per-target selection: not available, using all {len(feature_cols)} features")
    else:
        print(f"  Per-target selection: not available, using all {len(feature_cols)} features")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CHECKPOINT_DIR / "cb_predictions.npz"
    if cache_file.exists():
        print(f"\n  Predictions exist at {cache_file}! Delete to retrain.")
        return

    tr_idx, val_idx = make_validation_split(y_train)
    print("\n[2/4] Training with a single validation split...")
    print(f"  Holdout: train={len(tr_idx):,}, val={len(val_idx):,}", flush=True)

    n_targets = len(target_cols)
    train_preds = np.zeros((X_train.shape[0], n_targets), dtype=np.float32)
    test_preds = np.zeros((X_test.shape[0], n_targets), dtype=np.float32)
    importance_sum = np.zeros((len(feature_cols), n_targets), dtype=np.float64)
    best_iterations = np.zeros(n_targets, dtype=np.int32)

    for i, col in enumerate(target_cols):
        y_t = y_train[:, i]

        params = dict(cb_params)
        params["random_seed"] = SEED
        params["auto_class_weights"] = "Balanced"
        if task_type == "GPU":
            params["task_type"] = "GPU"
            params["devices"] = devices

        if col in per_target_feats:
            sel_cols = per_target_feats[col]
            sel_cats = [c for c in cat_feature_names if c in sel_cols]
            X_fit_t = X_train[sel_cols].iloc[tr_idx]
            X_val_t = X_train[sel_cols].iloc[val_idx]
            X_train_t = X_train[sel_cols]
            X_test_t = X_test[sel_cols]
        else:
            sel_cols = feature_cols
            sel_cats = cat_feature_names
            X_fit_t = X_train.iloc[tr_idx]
            X_val_t = X_train.iloc[val_idx]
            X_train_t = X_train
            X_test_t = X_test

        fit_pool = Pool(X_fit_t, y_t[tr_idx], cat_features=sel_cats)
        val_pool = Pool(X_val_t, y_t[val_idx], cat_features=sel_cats)

        search_model = CatBoostClassifier(**params)
        search_model.fit(fit_pool, eval_set=val_pool, verbose=0)
        best_iters = best_iterations_or_default(search_model, params["iterations"])
        best_iterations[i] = best_iters

        final_params = dict(params)
        final_params["iterations"] = best_iters
        final_params.pop("early_stopping_rounds", None)
        final_model = CatBoostClassifier(**final_params)

        train_pool = Pool(X_train_t, y_t, cat_features=sel_cats)
        test_pool = Pool(X_test_t, cat_features=sel_cats)
        final_model.fit(train_pool, verbose=0)

        train_preds[:, i] = final_model.predict_proba(train_pool)[:, 1].astype(np.float32)
        test_preds[:, i] = final_model.predict_proba(test_pool)[:, 1].astype(np.float32)

        with open(MODELS_DIR / f"{col}.pkl", "wb") as f:
            pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        imp = final_model.get_feature_importance()
        if col in per_target_feats:
            for j, sc in enumerate(sel_cols):
                importance_sum[feature_cols.index(sc), i] += imp[j]
        else:
            importance_sum[:, i] += imp

        del fit_pool, val_pool, train_pool, test_pool, search_model, final_model
        gc.collect()
        if (i + 1) % 10 == 0 or i == n_targets - 1:
            print(f"    {i + 1}/{n_targets} targets done", flush=True)

    train_auc, per_target_aucs = compute_macro_auc(y_train, train_preds, target_cols)
    print("\n[3/4] Results:")
    print(f"  Train Macro ROC-AUC: {train_auc:.4f}")
    log_per_target_auc(per_target_aucs, y_train, target_cols)

    np.savez(cache_file, train_preds=train_preds, test_preds=test_preds, best_iterations=best_iterations)
    print(f"  Saved: {cache_file}")

    imp_path = CHECKPOINT_DIR / "feature_importances.json"
    imp_data = {
        "feature_names": feature_cols,
        "target_cols": target_cols,
        "importances_per_target": {
            col: dict(
                sorted(
                    zip(feature_cols, importance_sum[:, i].tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            for i, col in enumerate(target_cols)
        },
        "mean_importance": dict(
            sorted(
                zip(feature_cols, importance_sum.mean(axis=1).tolist()),
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
    submit.write_parquet("submissions/catboost.parquet")
    print("  Saved: submissions/catboost.parquet")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Train AUC={train_auc:.4f}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    np.random.seed(SEED)
    main()
