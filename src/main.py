import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from src.models.model_storage import ModelStorage
from src.utils.drawer import Drawer

PARAMS_MODEL = {
    'n_estimators': [1100],
    'max_features': ['sqrt'],
    'max_depth': [16],
    'min_samples_split': [19],
    'min_samples_leaf': [18]
}


def ensure_logs_directory_exists(path='../../logs'):
    """Ensure the logs directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_and_preprocess_data(filepath):
    """Загрузка и предварительная обработка данных."""
    features = pd.read_parquet(filepath)
    # features = features[features['population'] > 0]

    y_true = features['population'].values

    # Выбор признаков
    start_col = features.columns.get_loc('Shape_Leng')
    features = features.iloc[:, start_col:].drop(columns=['population', 'geometry'])
    X_true = features.values

    # Обработка inf и nan
    X_true = np.where(np.isinf(X_true) | np.isneginf(X_true), np.nan, X_true)
    X_true = np.nan_to_num(X_true, nan=0)

    # Масштабирование данных
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_true)

    return X_sc, y_true, features.columns


def split_data(X, y, test_size=0.01, random_state=0):
    """Разделение данных на обучающую и тестовую выборки."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def print_data_stats(y_true):
    """Вывод статистики по данным."""
    print("Среднее значение population:", np.mean(y_true))
    print("Минимум и максимум:", np.min(y_true), "-", np.max(y_true))
    print("Стандартное отклонение:", np.std(y_true))


def train_random_forest_with_gridsearch(X_train, y_train, model_storage):
    """Обучение RandomForest с GridSearch."""

    gs_model = model_storage.load_latest_model('random_forest_gridsearch')

    if gs_model is None:
        print("Обучение новой модели GridSearch...")
        forest_reg = RandomForestRegressor(bootstrap=True, oob_score=True)

        gs = GridSearchCV(
            estimator=forest_reg,
            param_grid=PARAMS_MODEL,
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        gs.fit(X_train, y_train)
        model_storage.save_model(gs, 'random_forest_gridsearch')
        return gs
    else:
        print("Используем сохраненную модель GridSearch")
        return gs_model


def train_final_random_forest(X_train, y_train, best_params, model_storage):
    """Обучение финальной модели RandomForest."""
    rf_model = model_storage.load_latest_model('random_forest')

    if rf_model is None:
        print("Обучение новой модели RandomForest...")
        rf = RandomForestRegressor(**best_params, oob_score=True, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        model_storage.save_model(rf, 'random_forest')
        return rf
    else:
        print("Используем сохраненную модель RandomForest")
        return rf_model


def evaluate_model(model, X, y_true, model_name=""):
    """Оценка модели и вывод метрик."""
    y_pred = model.predict(X)

    print(f"\nMetrics for {model_name}:")
    print("R2:", r2_score(y_true, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_true, y_pred)))
    print('MAE:', mean_absolute_error(y_true, y_pred))

    return y_pred


def save_results_to_files(y_pred, y_true, feature_importances=None, prefix="", path='../../logs'):
    """Сохранение результатов в файлы."""
    ensure_logs_directory_exists()
    error = y_pred - y_true

    np.savetxt(f'{path}/{prefix}_error.txt', error, newline='\n', delimiter=' ')
    np.savetxt(f'{path}/{prefix}_pred_result.txt', y_pred, newline='\n', delimiter=' ')

    if feature_importances is not None:
        np.savetxt(f'{path}/{prefix}_feature_importance.txt', feature_importances, newline='\n', delimiter=' ')


def train_lasso_model(X_train, y_train, model_storage):
    """Обучение модели Lasso."""
    lasso_model = model_storage.load_latest_model('lasso')

    if lasso_model is None:
        print("Обучение новой модели Lasso...")
        lassocv = LassoCV()
        lassocv.fit(X_train, y_train)
        print("Best alpha:", lassocv.alpha_)

        alg = Lasso(alpha=lassocv.alpha_)
        alg.fit(X_train, y_train)
        model_storage.save_model(alg, 'lasso')
        return alg
    else:
        print("Используем сохраненную модель Lasso")
        return lasso_model


def main():
    try:
        ensure_logs_directory_exists()

        # Загрузка и предобработка данных
        X_sc, y_true, feature_columns = load_and_preprocess_data('../data/processed/processed_buildings.parquet')
        X_train, X_test, y_train, y_test = split_data(X_sc, y_true)
        print_data_stats(y_true)

        # Инициализация менеджера моделей
        model_storage = ModelStorage(PARAMS_MODEL)

        # Обучение и оценка RandomForest
        grid_search = train_random_forest_with_gridsearch(X_train, y_train, model_storage)
        print('\nbest parameters:', grid_search.best_params_)
        print('best estimator:', grid_search.best_estimator_)
        print('best score:', grid_search.best_score_)

        rf_model = train_final_random_forest(X_train, y_train, grid_search.best_params_, model_storage)
        rfy_pred = evaluate_model(rf_model, X_sc, y_true, "RandomForest")

        # Сохранение результатов RandomForest
        save_results_to_files(
            rfy_pred, y_true,
            rf_model.feature_importances_ if hasattr(rf_model, 'feature_importances_') else None,
            prefix='rf'
        )

        print("RF pred min/max:", np.min(rfy_pred), np.max(rfy_pred))
        print("True min/max:", np.min(y_true), np.max(y_true))

        # Обучение и оценка Lasso
        lasso_model = train_lasso_model(X_train, y_train, model_storage)
        ml_y_pred = evaluate_model(lasso_model, X_sc, y_true, "Lasso")

        # Вывод коэффициентов Lasso
        print('\nLasso coefficients:', lasso_model.coef_)
        print('Lasso intercept:', lasso_model.intercept_)

        # Сохранение результатов Lasso
        save_results_to_files(ml_y_pred, y_true, prefix='lasso')

        prediction_plotter = Drawer()

        # Для Random Forest
        prediction_plotter.plot_prediction(y_true, rfy_pred, title="RF Predictions", color="Blues")

        # Для Lasso
        prediction_plotter.plot_prediction(y_true, ml_y_pred, title="Lasso Predictions", color="Reds")

    except Exception as e:
        print("Ошибка в основном потоке выполнения:", e)
        raise


if __name__ == "__main__":
    main()
