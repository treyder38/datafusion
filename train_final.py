import polars as pl
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings
from pathlib import Path

SEED = 42
N_FOLDS = 5
FEATURES_FILE = "selected_features_final.txt"


def check_gpu():
    try:
        model = CatBoostClassifier(task_type="GPU", iterations=1, verbose=0)
        model.fit([[0, 1], [1, 0]], [0, 1])
        return True
    except Exception as e:
        print(f"GPU недоступен: {e}")
        return False


def train_and_evaluate(X, Y, X_test, feature_subset, tgt_cols, cat_cols_set,
                       task_type, n_folds=5, params=None, early_stopping=50):
    if params is None:
        params = dict(
            iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
            task_type=task_type, random_seed=SEED, verbose=0,
            loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
        )

    subset_cat_idx = [i for i, c in enumerate(feature_subset) if c in cat_cols_set]
    X_sub = X[feature_subset]
    n_targets = Y.shape[1]

    oof_preds = np.zeros((len(X_sub), n_targets))
    test_preds = np.zeros((len(X_test), n_targets))
    auc_scores = []

    for ti in range(n_targets):
        y = Y[:, ti]
        n_pos = y.sum()
        if n_pos < n_folds:
            oof_preds[:, ti] = y.mean()
            test_preds[:, ti] = y.mean()
            auc_scores.append(0.5)
            print(f"  {tgt_cols[ti]}: SKIP (n_pos={int(n_pos)}), AUC=0.5000")
            continue

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        fold_aucs = []

        for _, (tr_idx, val_idx) in enumerate(skf.split(X_sub, y)):
            X_tr, X_val = X_sub.iloc[tr_idx], X_sub.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            m = CatBoostClassifier(**params, early_stopping_rounds=early_stopping)
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=subset_cat_idx)

            val_pred = m.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, ti] = val_pred
            test_preds[:, ti] += m.predict_proba(X_test[feature_subset])[:, 1] / n_folds
            fold_aucs.append(roc_auc_score(y_val, val_pred))

        mean_auc = np.mean(fold_aucs)
        auc_scores.append(mean_auc)
        print(f"  {tgt_cols[ti]}: AUC={mean_auc:.4f} (+/- {np.std(fold_aucs):.4f})")

    macro_auc = np.mean(auc_scores)
    print(f"\n  >>> Macro AUC: {macro_auc:.4f}")
    return oof_preds, test_preds, auc_scores, macro_auc


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)

    # ── Загрузка списка признаков ─────────────────────────────────────────────

    features_path = Path(FEATURES_FILE)
    if not features_path.exists():
        raise FileNotFoundError(f"Файл с признаками не найден: {FEATURES_FILE}")

    best_features = [l.strip() for l in features_path.read_text().splitlines() if l.strip()]
    print(f"Загружено признаков из {FEATURES_FILE}: {len(best_features)}")

    task_type = "GPU" if check_gpu() else "CPU"
    print(f"CatBoost device: {task_type}")

    # ── Загрузка данных ───────────────────────────────────────────────────────

    train_main = pl.read_parquet("train_main_features.parquet")
    test_main = pl.read_parquet("test_main_features.parquet")
    target = pl.read_parquet("train_target.parquet")
    train_extra = pl.read_parquet("train_extra_features.parquet")
    test_extra = pl.read_parquet("test_extra_features.parquet")

    train_full = train_main.join(train_extra, on="customer_id", how="left")
    test_full = test_main.join(test_extra, on="customer_id", how="left")
    del train_main, test_main, train_extra, test_extra
    gc.collect()

    print(f"Full: train={train_full.shape}, test={test_full.shape}")

    num_cols = [c for c in train_full.columns if c.startswith("num_feature")]
    target_cols = [c for c in target.columns if c.startswith("target")]

    # ── Feature Engineering ───────────────────────────────────────────────────

    train_full = train_full.with_columns(
        pl.sum_horizontal(*[pl.col(c).is_null().cast(pl.Int16) for c in num_cols]).alias("total_null_count")
    )
    test_full = test_full.with_columns(
        pl.sum_horizontal(*[pl.col(c).is_null().cast(pl.Int16) for c in num_cols]).alias("total_null_count")
    )

    fill_values = {}
    for c in num_cols:
        median_val = train_full[c].median()
        fill_values[c] = median_val if median_val is not None else 0.0

    train_full = train_full.with_columns([pl.col(c).fill_null(v) for c, v in fill_values.items()])
    test_full = test_full.with_columns([pl.col(c).fill_null(v) for c, v in fill_values.items()])

    # ── Подготовка данных ─────────────────────────────────────────────────────

    all_cat_cols = [c for c in best_features if c.startswith("cat_feature")]
    num_in_best = [c for c in best_features if not c.startswith("cat_feature")]

    test_ids = test_full.select("customer_id")

    X_train = train_full.select(best_features).with_columns([
        pl.col(c).cast(pl.Float32) for c in num_in_best
    ]).to_pandas()

    X_test = test_full.select(best_features).with_columns([
        pl.col(c).cast(pl.Float32) for c in num_in_best
    ]).to_pandas()

    del train_full, test_full
    gc.collect()

    Y_train = target.select(target_cols).to_pandas().values

    for c in all_cat_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].fillna(-999).astype(int)
            X_test[c] = X_test[c].fillna(-999).astype(int)

    remaining_nan = X_train[num_in_best].isnull().sum().sum()
    if remaining_nan > 0:
        print(f"WARN: осталось {remaining_nan} NaN в числовых, заполняем 0")
        X_train[num_in_best] = X_train[num_in_best].fillna(0)
        X_test[num_in_best] = X_test[num_in_best].fillna(0)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}")

    # ── Финальная модель ──────────────────────────────────────────────────────

    final_params = dict(
        iterations=3000, learning_rate=0.02, depth=8, l2_leaf_reg=1,
        random_strength=0.5, bagging_temperature=0.8,
        task_type=task_type, random_seed=SEED, verbose=0,
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
    )

    print(f"=== Финальная модель: {len(best_features)} признаков, {N_FOLDS} фолдов ===")
    oof_final, test_final, aucs_final, macro_final = train_and_evaluate(
        X_train, Y_train, X_test, best_features, target_cols, set(all_cat_cols),
        task_type, n_folds=N_FOLDS, params=final_params, early_stopping=100,
    )

    # ── Per-target тюнинг для слабых таргетов ────────────────────────────────

    weak_targets_idx = [i for i, a in enumerate(aucs_final) if a < 0.6]
    print(f"Слабых таргетов (AUC < 0.6): {len(weak_targets_idx)}")

    if weak_targets_idx:
        print("Пробуем агрессивные параметры для слабых таргетов...\n")
        aggressive_params = dict(
            iterations=5000, learning_rate=0.01, depth=10, l2_leaf_reg=0.5,
            random_strength=1.0, bagging_temperature=1.0,
            task_type=task_type, random_seed=SEED, verbose=0,
            loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
        )
        subset_cat_idx = [i for i, c in enumerate(best_features) if c in all_cat_cols]

        for ti in weak_targets_idx:
            y = Y_train[:, ti]
            if y.sum() < N_FOLDS:
                continue

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            fold_aucs = []
            ti_test_pred = np.zeros(len(X_test))

            for _, (tr_idx, val_idx) in enumerate(skf.split(X_train[best_features], y)):
                X_tr = X_train[best_features].iloc[tr_idx]
                X_val = X_train[best_features].iloc[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                m = CatBoostClassifier(**aggressive_params, early_stopping_rounds=150)
                m.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=subset_cat_idx)

                val_pred = m.predict_proba(X_val)[:, 1]
                fold_aucs.append(roc_auc_score(y_val, val_pred))
                ti_test_pred += m.predict_proba(X_test[best_features])[:, 1] / N_FOLDS

            new_auc = np.mean(fold_aucs)
            old_auc = aucs_final[ti]
            if new_auc > old_auc:
                print(f"  {target_cols[ti]}: {old_auc:.4f} -> {new_auc:.4f} (IMPROVED)")
                test_final[:, ti] = ti_test_pred
                aucs_final[ti] = new_auc
            else:
                print(f"  {target_cols[ti]}: {old_auc:.4f} -> {new_auc:.4f} (keeping original)")

        print(f"\nОбновленный Macro AUC: {np.mean(aucs_final):.4f}")

    # ── Сохранение submission ─────────────────────────────────────────────────

    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = test_ids.hstack(pl.DataFrame(test_final, schema=predict_cols))
    submit.write_parquet("submission_final.parquet")

    print(f"\nSubmission shape: {submit.shape}")
    print("Saved: submission_final.parquet")


if __name__ == "__main__":
    main()
