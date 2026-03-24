"""Step 3a: Optuna hyperparameter tuning for LightGBM.

Tunes on a single fold with a subset of targets for speed.
Supports per-target feature selection (from 01b_select_features.py).
Saves best params to checkpoints_lgbm/best_params.json.

Usage:
    python 03a_tune_lgbm.py [--n-trials 50] [--n-targets 20]

After tuning, delete checkpoints_lgbm/lgbm_predictions.npz
and re-run 03_train_lgbm.py (it will load best_params.json automatically).
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
SELECTED_DIR = FEATURES_DIR / "selected_features"
CHECKPOINT_DIR = Path("checkpoints_lgbm")

TUNE_TRAIN_SIZE = 150_000


def load_per_target_features(target_cols, feature_cols):
    """Load per-target feature selections if available."""
    if not SELECTED_DIR.exists():
        return None
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    per_target = {}
    for target in target_cols:
        fpath = SELECTED_DIR / f"{target}.json"
        if not fpath.exists():
            return None
        with open(fpath) as f:
            selected = json.load(f)
        per_target[target] = [col_to_idx[c] for c in selected if c in col_to_idx]
    return per_target


def objective(trial, X_train, y_train, tr_idx, val_idx,
              cat_indices, target_indices, target_cols,
              per_target_feats):
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

    aucs = []
    for i in target_indices:
        col = target_cols[i]
        y_t = y_train[:, i]

        # Per-target feature selection
        fi = per_target_feats.get(col) if per_target_feats else None
        if fi is not None:
            X_tr = X_train[np.ix_(tr_idx, fi)]
            X_val = X_train[np.ix_(val_idx, fi)]
            cat_idx_set = set(cat_indices)
            cat_fi = [j for j, feat_idx in enumerate(fi) if feat_idx in cat_idx_set]
        else:
            X_tr = X_train[tr_idx]
            X_val = X_train[val_idx]
            cat_fi = cat_indices

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
                categorical_feature=cat_fi,
            )

        val_pred = model.predict_proba(X_val)[:, 1]
        y_val = y_t[val_idx]
        if y_val.sum() >= 2 and (len(y_val) - y_val.sum()) >= 2:
            aucs.append(roc_auc_score(y_val, val_pred))

        del model; gc.collect()

    return float(np.mean(aucs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-targets", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Step 3a: Optuna tuning for LightGBM")
    print("=" * 60)
    print(f"  Trials: {args.n_trials}, Targets per trial: {args.n_targets}")

    # Load features
    print("\n[1/3] Loading features...")
    with open(FEATURES_DIR / "meta.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    cat_feature_names = meta["cat_cols"]
    target_cols = meta["target_cols"]
    cat_indices = [feature_cols.index(c) for c in cat_feature_names]

    train_feat = pl.read_parquet(FEATURES_DIR / "train_features.parquet")
    train_tgt = pl.read_parquet(FEATURES_DIR / "targets.parquet")

    X_train = train_feat.drop("customer_id").to_numpy().astype(np.float32)
    y_train = train_tgt.select(target_cols).to_numpy().astype(np.float32)
    print(f"  X_train: {X_train.shape}")

    # Per-target feature selection
    per_target_feats = load_per_target_features(target_cols, feature_cols)
    if per_target_feats:
        n_feats = [len(v) for v in per_target_feats.values()]
        print(f"  Per-target selection: {min(n_feats)}-{max(n_feats)} features/target")
    else:
        print(f"  No per-target selection, using all {len(feature_cols)} features")

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
        study_name="lgbm_tuning",
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, tr_idx, val_idx,
            cat_indices, target_indices, target_cols,
            per_target_feats,
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
    print(f"\nNext: delete checkpoints_lgbm/lgbm_predictions.npz and re-run 03_train_lgbm.py")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(SEED)
    main()
