"""Step 5a: Optuna hyperparameter tuning for CatBoost.

Tunes on a single fold with a subset of targets for speed.
Saves best params to checkpoints_catboost/best_params.json.

Usage:
    python 05a_tune_catboost.py [--n-trials 50] [--n-targets 10]

After tuning, delete checkpoints_catboost/cb_predictions.npz
and re-run 05_train_catboost.py (it will load best_params.json automatically).

Runtime: ~2-4 hours (GPU), depends on n_trials.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import polars as pl
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils import SEED, DATA_DIR, N_FOLDS, compute_macro_auc

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_catboost")


def detect_task_type():
    """Detect GPU availability for CatBoost."""
    try:
        cb = CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=0)
        cb.fit([[0, 0], [1, 1]], [0, 1])
        return "GPU", "0"
    except Exception:
        return "CPU", None


def objective(trial, X_train, y_train, tr_idx, val_idx,
              cat_feature_names, target_indices, target_cols,
              task_type, devices):
    """Optuna objective: train on 1 fold, evaluate on subset of targets."""

    depth = trial.suggest_int("depth", 6, 10)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 0.1, 50.0, log=True)
    random_strength = trial.suggest_float("random_strength", 0.1, 10.0, log=True)
    # GPU limits border_count to 128 and only supports SymmetricTree
    if task_type == "GPU":
        border_count = trial.suggest_categorical("border_count", [64, 128])
        grow_policy = "SymmetricTree"
    else:
        border_count = trial.suggest_categorical("border_count", [64, 128, 254])
        grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"])
    max_ctr_complexity = trial.suggest_int("max_ctr_complexity", 1, 4)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 500, log=True)
    boosting_type = trial.suggest_categorical("boosting_type", ["Plain", "Ordered"])

    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "MVS"])
    params = dict(
        iterations=5000,
        early_stopping_rounds=150,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        max_ctr_complexity=max_ctr_complexity,
        boosting_type=boosting_type,
        bootstrap_type=bootstrap_type,
        random_strength=random_strength,
        min_data_in_leaf=min_data_in_leaf,
        grow_policy=grow_policy,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        logging_level="Silent",
    )
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 5.0)
    else:  # MVS
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    if task_type == "GPU":
        params["task_type"] = "GPU"
        params["devices"] = devices

    aucs = []
    for i in target_indices:
        y = y_train[:, i]

        tr_pool = Pool(X_train.iloc[tr_idx], y[tr_idx], cat_features=cat_feature_names)
        va_pool = Pool(X_train.iloc[val_idx], y[val_idx], cat_features=cat_feature_names)

        cb = CatBoostClassifier(**params)
        cb.fit(tr_pool, eval_set=va_pool)

        val_pred = cb.predict_proba(va_pool)[:, 1]
        y_val = y[val_idx]

        if y_val.sum() >= 2 and (len(y_val) - y_val.sum()) >= 2:
            aucs.append(roc_auc_score(y_val, val_pred))

        del cb, tr_pool, va_pool; gc.collect()

    mean_auc = float(np.mean(aucs))
    return mean_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-targets", type=int, default=20,
                        help="Number of targets to evaluate (subset for speed)")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Step 5a: Optuna tuning for CatBoost")
    print("=" * 60)

    task_type, devices = detect_task_type()
    print(f"  CatBoost task_type: {task_type}")
    print(f"  Trials: {args.n_trials}, Targets per trial: {args.n_targets}")

    # Load features
    print("\n[1/3] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.select(feature_cols).to_pandas()
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    for col in cat_feature_names:
        X_train[col] = X_train[col].astype(str)

    print(f"  X_train: {X_train.shape}")

    # Use first fold only for tuning
    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(iter(kf.split(np.arange(len(X_train)), y_train)))
    print(f"  Fold 1: train={len(tr_idx):,}, val={len(val_idx):,}")

    # Select diverse subset of targets (spread across different class ratios)
    n_targets = min(args.n_targets, len(target_cols))
    pos_rates = y_train.mean(axis=0)
    sorted_indices = np.argsort(pos_rates)
    # Evenly spaced targets across the pos_rate spectrum
    target_indices = sorted_indices[np.linspace(0, len(sorted_indices)-1, n_targets, dtype=int)]
    print(f"  Tuning on {n_targets} targets: {[target_cols[i] for i in target_indices[:5]]}...")

    # Run Optuna
    print(f"\n[2/3] Running Optuna ({args.n_trials} trials)...", flush=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="catboost_tuning",
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, tr_idx, val_idx,
            cat_feature_names, target_indices, target_cols,
            task_type, devices,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Results
    print(f"\n[3/3] Results:")
    print(f"  Best AUC: {study.best_value:.5f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Save best params
    params_path = CHECKPOINT_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n  Saved: {params_path}")

    # Save full study results
    trials_path = CHECKPOINT_DIR / "optuna_trials.json"
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2)
    print(f"  Saved: {trials_path}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print(f"\nNext: delete checkpoints_catboost/cb_predictions.npz and re-run 05_train_catboost.py")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    import warnings
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(SEED)
    main()
