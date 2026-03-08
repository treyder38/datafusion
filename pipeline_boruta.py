import polars as pl
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from BorutaShap import BorutaShap
import gc
import warnings
from pathlib import Path

SEED = 42
N_FOLDS = 5


def save_feature_list(feature_names, output_path):
    output_path = Path(output_path)
    output_path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    print(f"Список итоговых признаков сохранен в: {output_path}")


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


def run_boruta(X, y, feature_subset, cat_cols_set, task_type, n_trials=50):
    """
    Классический BorutaShap с verbose=True для отображения прогресса.
    Категориальные признаки не трогает — добавляются в итоговый список.
    """
    num_subset = [c for c in feature_subset if c not in cat_cols_set]
    cat_in_subset = [c for c in feature_subset if c in cat_cols_set]

    boruta_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        task_type=task_type,
        random_seed=SEED,
        verbose=0,
        loss_function="Logloss",
        auto_class_weights="Balanced",
    )

    selector = BorutaShap(
        model=boruta_model,
        importance_measure="shap",
        classification=True,
    )

    print(f"Запуск BorutaShap: {len(num_subset)} числовых признаков, {n_trials} итераций")
    selector.fit(
        X=X[num_subset],
        y=y,
        n_trials=n_trials,
        random_state=SEED,
        verbose=True,   # печатает прогресс каждой итерации
    )

    confirmed = list(selector.accepted)
    tentative = list(selector.tentative)
    rejected = list(selector.rejected)

    print(f"Boruta — confirmed: {len(confirmed)}, tentative: {len(tentative)}, "
          f"rejected: {len(rejected)}, cat (не трогали): {len(cat_in_subset)}")

    return confirmed + tentative + cat_in_subset


def main():
    warnings.filterwarnings("ignore")
    np.random.seed(SEED)

    task_type = "GPU" if check_gpu() else "CPU"
    print(f"CatBoost device: {task_type}")

    # ── 1. Загрузка данных ───────────────────────────────────────────────────

    train_main = pl.read_parquet("train_main_features.parquet")
    test_main = pl.read_parquet("test_main_features.parquet")
    target = pl.read_parquet("train_target.parquet")
    train_extra = pl.read_parquet("train_extra_features.parquet")
    test_extra = pl.read_parquet("test_extra_features.parquet")

    print(f"Main:  train={train_main.shape}, test={test_main.shape}")
    print(f"Extra: train={train_extra.shape}, test={test_extra.shape}")

    train_full = train_main.join(train_extra, on="customer_id", how="left")
    test_full = test_main.join(test_extra, on="customer_id", how="left")
    del train_main, test_main, train_extra, test_extra
    gc.collect()

    print(f"Full: train={train_full.shape}, test={test_full.shape}")

    feature_cols = [c for c in train_full.columns if c != "customer_id"]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c.startswith("num_feature")]
    target_cols = [c for c in target.columns if c.startswith("target")]

    print(f"Категориальных: {len(cat_cols)}, Числовых: {len(num_cols)}, Таргетов: {len(target_cols)}")

    # ── 2. Feature Engineering ───────────────────────────────────────────────

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

    print("Числовые NaN заполнены медианой")

    # ── 3. Удаление бесполезных признаков ────────────────────────────────────

    all_feature_cols = [c for c in train_full.columns if c != "customer_id"]
    n_train = len(train_full)
    cols_to_drop = []

    for c in all_feature_cols:
        null_pct = train_full[c].null_count() / n_train
        if null_pct > 0.99:
            cols_to_drop.append(c)
            continue
        if train_full[c].n_unique() <= 1:
            cols_to_drop.append(c)

    print(f"Удаляем {len(cols_to_drop)} признаков (>99% null или константные)")

    if cols_to_drop:
        train_full = train_full.drop(cols_to_drop)
        test_full = test_full.drop([c for c in cols_to_drop if c in test_full.columns])

    all_feature_cols = [c for c in train_full.columns if c != "customer_id"]
    all_cat_cols = [c for c in all_feature_cols if c.startswith("cat_feature")]
    print(f"Осталось признаков: {len(all_feature_cols)}")

    # ── 4. Подготовка данных для CatBoost ────────────────────────────────────

    num_feature_cols_list = [c for c in all_feature_cols if c not in set(all_cat_cols)]

    train_f32 = train_full.select(all_feature_cols).with_columns([
        pl.col(c).cast(pl.Float32) for c in num_feature_cols_list
    ])
    test_f32 = test_full.select(all_feature_cols).with_columns([
        pl.col(c).cast(pl.Float32) for c in num_feature_cols_list
    ])

    test_ids = test_full.select("customer_id")
    del train_full, test_full
    gc.collect()

    X_train = train_f32.to_pandas()
    del train_f32
    gc.collect()

    X_test = test_f32.to_pandas()
    del test_f32
    gc.collect()

    Y_train = target.select(target_cols).to_pandas().values

    cat_indices = [i for i, c in enumerate(all_feature_cols) if c in all_cat_cols]
    for c in all_cat_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].fillna(-999).astype(int)
            X_test[c] = X_test[c].fillna(-999).astype(int)

    remaining_nan = X_train[num_feature_cols_list].isnull().sum().sum()
    if remaining_nan > 0:
        print(f"WARN: осталось {remaining_nan} NaN в числовых, заполняем 0")
        X_train[num_feature_cols_list] = X_train[num_feature_cols_list].fillna(0)
        X_test[num_feature_cols_list] = X_test[num_feature_cols_list].fillna(0)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}")

    # ── 5. Быстрая оценка важности признаков (все таргеты) ──────────────────

    quick_params = dict(
        iterations=200, learning_rate=0.1, depth=6, l2_leaf_reg=3,
        task_type=task_type, random_seed=SEED, verbose=0,
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
    )

    importances = np.zeros(len(all_feature_cols))
    n_valid = 0
    print(f"Оценка важности на всех {len(target_cols)} таргетах:")
    for ti in range(len(target_cols)):
        y = Y_train[:, ti]
        if y.sum() < N_FOLDS:
            print(f"  {target_cols[ti]}: SKIP (n_pos={int(y.sum())})")
            continue
        model = CatBoostClassifier(**quick_params)
        model.fit(X_train, y, cat_features=cat_indices)
        importances += model.get_feature_importance()
        auc = roc_auc_score(y, model.predict_proba(X_train)[:, 1])
        n_valid += 1
        print(f"  {target_cols[ti]}: train AUC = {auc:.4f}")

    importances /= n_valid

    feat_imp = sorted(zip(all_feature_cols, importances), key=lambda x: -x[1])
    selected_features = [name for name, imp in feat_imp if imp > 0]
    print(f"Признаков с importance > 0: {len(selected_features)}")
    print(f"Убрано с нулевой важностью: {len([1 for _, imp in feat_imp if imp == 0])}")

    # ── 6+7. Per-target: Boruta → индивидуальная модель ─────────────────────
    # Для каждого таргета:
    #   1. Boruta отбирает признаки именно для этого таргета
    #   2. На этих признаках обучается своя модель (5-fold CV)
    #   3. Предсказания собираются в итоговую матрицу

    print(f"\n{'=' * 60}")
    print("Per-target Boruta + обучение")
    print(f"{'=' * 60}")

    BORUTA_SAMPLE = 350_000
    rng = np.random.default_rng(SEED)

    if len(X_train) > BORUTA_SAMPLE:
        sample_idx = rng.choice(len(X_train), size=BORUTA_SAMPLE, replace=False)
        X_boruta_base = X_train[selected_features].iloc[sample_idx].reset_index(drop=True)
        Y_boruta = Y_train[sample_idx]
    else:
        X_boruta_base = X_train[selected_features].reset_index(drop=True)
        Y_boruta = Y_train

    final_params = dict(
        iterations=3000, learning_rate=0.02, depth=8, l2_leaf_reg=1,
        random_strength=0.5, bagging_temperature=0.8,
        task_type=task_type, random_seed=SEED, verbose=0,
        loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced",
    )

    test_final = np.zeros((len(X_test), len(target_cols)))
    aucs_final = []
    target_features_map = {}  # какие признаки у какого таргета

    for ti in range(len(target_cols)):
        y_full = Y_train[:, ti]
        y_boruta = Y_boruta[:, ti]

        print(f"\n{'=' * 60}")
        print(f"[{ti+1}/{len(target_cols)}] {target_cols[ti]}")
        print(f"{'=' * 60}")

        if y_full.sum() < N_FOLDS:
            print(f"SKIP (n_pos={int(y_full.sum())})")
            test_final[:, ti] = y_full.mean()
            aucs_final.append(0.5)
            target_features_map[target_cols[ti]] = []
            continue

        # Boruta для этого таргета
        ti_features = run_boruta(
            X=X_boruta_base,
            y=y_boruta,
            feature_subset=selected_features,
            cat_cols_set=set(all_cat_cols),
            task_type=task_type,
            n_trials=30,
        )

        # fallback если Boruta ничего не нашёл
        if not ti_features:
            ti_features = selected_features[:20]
            print(f"  Boruta не нашёл признаков, fallback к топ-20 по importance")

        target_features_map[target_cols[ti]] = ti_features
        print(f"  Признаков для таргета: {len(ti_features)}")

        # обучение модели на признаках этого таргета
        ti_cat_idx = [i for i, c in enumerate(ti_features) if c in all_cat_cols]
        X_sub = X_train[ti_features]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aucs = []

        for _, (tr_idx, val_idx) in enumerate(skf.split(X_sub, y_full)):
            X_tr, X_val = X_sub.iloc[tr_idx], X_sub.iloc[val_idx]
            y_tr, y_val = y_full[tr_idx], y_full[val_idx]

            m = CatBoostClassifier(**final_params, early_stopping_rounds=100)
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=ti_cat_idx)

            val_pred = m.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, val_pred))
            test_final[:, ti] += m.predict_proba(X_test[ti_features])[:, 1] / N_FOLDS

        mean_auc = np.mean(fold_aucs)
        aucs_final.append(mean_auc)
        print(f"  AUC={mean_auc:.4f} (+/- {np.std(fold_aucs):.4f})")

    macro_final = np.mean(aucs_final)
    print(f"\n>>> Macro AUC: {macro_final:.4f}")

    # сохраняем признаки каждого таргета
    features_out = Path("selected_features_per_target.txt")
    lines = []
    for tname, feats in target_features_map.items():
        lines.append(f"{tname}: {','.join(feats)}")
    features_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Признаки по таргетам сохранены в: {features_out}")

    # ── 9. Сохранение submission ─────────────────────────────────────────────

    predict_cols = [c.replace("target_", "predict_") for c in target_cols]
    submit = test_ids.hstack(pl.DataFrame(test_final, schema=predict_cols))
    submit.write_parquet("submission_automl_boruta.parquet")

    print(f"\nSubmission shape: {submit.shape}")
    print("Saved: submission_automl_boruta.parquet")


if __name__ == "__main__":
    main()
