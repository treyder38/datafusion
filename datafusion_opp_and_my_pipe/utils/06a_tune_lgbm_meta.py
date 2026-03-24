"""Step 6a: Optuna hyperparameter tuning for LightGBM meta.

Tunes on a single fold with a subset of targets for speed.
Uses base features + OOF meta-features from 4 models (NN, LGBM, PyBoost, CatBoost).
Masks own-target meta columns to prevent leakage.
Saves best params to checkpoints_lgbm_meta/best_params.json.

Usage:
    python 06a_tune_lgbm_meta.py [--n-trials 50] [--n-targets 20]

After tuning, delete checkpoints_lgbm_meta/lgbm_predictions.npz
and re-run 07_train_lgbm_meta.py (it will load best_params.json automatically).
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from utils.utils import SEED, N_FOLDS, effective_number_weight

FEATURES_DIR = Path("features")
CHECKPOINT_DIR = Path("checkpoints_lgbm_meta")

TUNE_TRAIN_SIZE = 150_000


def load_nn_predictions():
    """Load NN OOF predictions from fold checkpoints."""
    nn_dir = Path("checkpoints_nn")
    n_targets = np.load(nn_dir / "fold_0.npz")["val_preds"].shape[1]
    oof_parts = {}
    for fi in range(N_FOLDS):
        d = np.load(nn_dir / f"fold_{fi}.npz")
        for idx, pred in zip(d["val_idx"], d["val_preds"]):
            oof_parts[int(idx)] = pred
    n_train = max(oof_parts.keys()) + 1
    nn_oof = np.zeros((n_train, n_targets), dtype=np.float32)
    for idx, pred in oof_parts.items():
        nn_oof[idx] = pred
    return nn_oof


def load_meta_features():
    """Load OOF meta-features from all 4 models."""
    nn_oof = load_nn_predictions()

    d = np.load("checkpoints_lgbm/lgbm_predictions.npz")
    lgbm_oof = d["oof_preds"].astype(np.float32)

    d = np.load("checkpoints_pyboost/pyboost_predictions.npz")
    pb_oof = d["oof_preds"].astype(np.float32)

    d = np.load("checkpoints_catboost/cb_predictions.npz")
    cb_oof = d["oof_preds"].astype(np.float32)

    return np.hstack([lgbm_oof, nn_oof, pb_oof, cb_oof])


def objective(trial, X_train, y_train, tr_idx, val_idx,
              cat_indices, target_indices, target_cols,
              n_base, n_targets_total, n_models):
    """Optuna objective: train on 1 fold, evaluate on subset of targets."""

    num_leaves = trial.suggest_int("num_leaves", 16, 128)
    min_child_samples = trial.suggest_int("min_child_samples", 5, 200, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    min_split_gain = trial.suggest_float("min_split_gain", 0.0, 1.0)
    max_bin = trial.suggest_categorical("max_bin", [127, 255, 511])

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        min_child_samples=min_child_samples,
        n_estimators=2000,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_split_gain=min_split_gain,
        subsample_freq=1,
        max_bin=max_bin,
        random_state=SEED,
        verbose=-1,
        force_col_wise=True,
        n_jobs=-1,
    )

    # Pre-slice train/val
    X_tr = X_train[tr_idx].copy()
    X_val = X_train[val_idx].copy()

    aucs = []
    for i in target_indices:
        y_t = y_train[:, i]

        # Mask own-target meta columns (same logic as 07_train_lgbm_meta.py)
        exclude_cols = [n_base + m * n_targets_total + i for m in range(n_models)]
        for c in exclude_cols:
            X_tr[:, c] = np.nan
            X_val[:, c] = np.nan

        n_neg = int((y_t[tr_idx] == 0).sum())
        n_pos = int((y_t[tr_idx] == 1).sum())
        spw = effective_number_weight(n_pos, n_neg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = lgb.LGBMClassifier(**params, scale_pos_weight=spw)
            model.fit(
                X_tr, y_t[tr_idx],
                eval_set=[(X_val, y_t[val_idx])],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(0),
                ],
                categorical_feature=cat_indices,
            )

        val_pred = model.predict_proba(X_val)[:, 1]
        y_val = y_t[val_idx]
        if y_val.sum() >= 2 and (len(y_val) - y_val.sum()) >= 2:
            aucs.append(roc_auc_score(y_val, val_pred))

        # Restore masked columns
        for c in exclude_cols:
            X_tr[:, c] = X_train[tr_idx, c]
            X_val[:, c] = X_train[val_idx, c]

        del model; gc.collect()

    return float(np.mean(aucs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-targets", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Step 6a: Optuna tuning for LightGBM meta")
    print("=" * 60)
    print(f"  Trials: {args.n_trials}, Targets per trial: {args.n_targets}")

    # Load base features
    print("\n[1/3] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]
    cat_indices = [feature_cols.index(c) for c in cat_feature_names]
    n_base = len(feature_cols)
    n_targets_total = len(target_cols)
    n_models = 4

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_base = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)

    # Load meta features
    print("  Loading meta-features from 4 models...")
    meta_train = load_meta_features()
    X_train = np.hstack([X_base, meta_train])
    del X_base, meta_train; gc.collect()

    n_total = X_train.shape[1]
    print(f"  X_train: {X_train.shape} ({n_base} base + {n_models}x{n_targets_total} meta = {n_total})")

    # First fold, subsample train
    kf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, val_idx = next(iter(kf.split(np.arange(len(X_train)), y_train)))
    if len(tr_idx) > TUNE_TRAIN_SIZE:
        rng = np.random.RandomState(SEED)
        tr_idx = rng.choice(tr_idx, size=TUNE_TRAIN_SIZE, replace=False)
        tr_idx.sort()
    print(f"  Fold 1: train={len(tr_idx):,}, val={len(val_idx):,}")

    # Diverse target subset
    n_targets = min(args.n_targets, len(target_cols))
    pos_rates = y_train.mean(axis=0)
    sorted_indices = np.argsort(pos_rates)
    target_indices = sorted_indices[np.linspace(0, len(sorted_indices) - 1, n_targets, dtype=int)]
    print(f"  Tuning on {n_targets} targets: {[target_cols[i] for i in target_indices[:5]]}...")

    # Optuna
    print(f"\n[2/3] Running Optuna ({args.n_trials} trials)...", flush=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="lgbm_meta_tuning",
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, tr_idx, val_idx,
            cat_indices, target_indices, target_cols,
            n_base, n_targets_total, n_models,
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

    params_path = CHECKPOINT_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n  Saved: {params_path}")

    trials_path = CHECKPOINT_DIR / "optuna_trials.json"
    trials_data = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials
    ]
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2)
    print(f"  Saved: {trials_path}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")
    print(f"\nNext: delete checkpoints_lgbm_meta/lgbm_predictions.npz and re-run 07_train_lgbm_meta.py")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(SEED)
    main()
