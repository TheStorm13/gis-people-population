import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from src.models.model_manager import ModelManager


def load_and_preprocess_data(filepath):
    """Загрузка и предварительная обработка данных."""
    features = pd.read_parquet(filepath)
    features = features[features['population'] > 0]

    y_true = features['population'].values

    # Выбор признаков
    start_col = features.columns.get_loc('Shape_Leng')
    features = features.iloc[:, start_col:].drop(columns=['population', 'geometry'])
    X_true = features.values

    # Обработка inf и nan
    X_true = np.where(np.isinf(X_true) | np.isneginf(X_true), np.nan, X_true)
    X_true = np.nan_to_num(X_true, nan=1e6)

    # Масштабирование данных
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_true)

    return X_sc, y_true, features.columns


def split_data(X, y, test_size=0.15, random_state=0):
    """Разделение данных на обучающую и тестовую выборки."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def print_data_stats(y_true):
    """Вывод статистики по данным."""
    print("Среднее значение population:", np.mean(y_true))
    print("Минимум и максимум:", np.min(y_true), "-", np.max(y_true))
    print("Стандартное отклонение:", np.std(y_true))


def train_random_forest_with_gridsearch(X_train, y_train, model_manager):
    """Обучение RandomForest с GridSearch."""
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt']
    }

    gs_model = model_manager.load_latest_model('random_forest_gridsearch')

    if gs_model is None:
        print("Обучение новой модели GridSearch...")
        forest_reg = RandomForestRegressor(bootstrap=True, oob_score=True)

        gs = GridSearchCV(
            estimator=forest_reg,
            param_grid=params,
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        gs.fit(X_train, y_train)
        model_manager.save_model(gs, 'random_forest_gridsearch')
        return gs
    else:
        print("Используем сохраненную модель GridSearch")
        return gs_model


def train_final_random_forest(X_train, y_train, best_params, model_manager):
    """Обучение финальной модели RandomForest."""
    rf_model = model_manager.load_latest_model('random_forest')

    if rf_model is None:
        print("Обучение новой модели RandomForest...")
        rf = RandomForestRegressor(**best_params, oob_score=True, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)
        model_manager.save_model(rf, 'random_forest')
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


def save_results_to_files(y_pred, y_true, feature_importances=None, prefix=""):
    """Сохранение результатов в файлы."""
    error = y_pred - y_true

    np.savetxt(f'{prefix}_error.txt', error, newline='\n', delimiter=' ')
    np.savetxt(f'{prefix}_pred_result.txt', y_pred, newline='\n', delimiter=' ')

    if feature_importances is not None:
        np.savetxt(f'{prefix}_feature_importance.txt', feature_importances, newline='\n', delimiter=' ')


def train_lasso_model(X_train, y_train, model_manager):
    """Обучение модели Lasso."""
    lasso_model = model_manager.load_latest_model('lasso')

    if lasso_model is None:
        print("Обучение новой модели Lasso...")
        lassocv = LassoCV()
        lassocv.fit(X_train, y_train)
        print("Best alpha:", lassocv.alpha_)

        alg = Lasso(alpha=lassocv.alpha_)
        alg.fit(X_train, y_train)
        model_manager.save_model(alg, 'lasso')
        return alg
    else:
        print("Используем сохраненную модель Lasso")
        return lasso_model


def plot_results(x_true, y_pred_rf, y_pred_ml):
    """Построение графиков результатов."""
    xmin, xmax = np.percentile(x_true, [1, 99])
    ymin_rf, ymax_rf = np.percentile(y_pred_rf, [1, 99])
    ymin_ml, ymax_ml = np.percentile(y_pred_ml, [1, 99])

    common_max = max(x_true.max(), y_pred_rf.max())
    xmax = common_max
    ymax_rf = common_max

    fig, axs = plt.subplots(ncols=1, nrows=2, sharey=True, dpi=600, figsize=(8, 16))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

    # График для Random Forest
    ax = axs[0]
    hb = ax.hexbin(x_true, y_pred_rf, gridsize=50, bins='log', cmap='Blues')
    ax.set(xlim=(xmin, xmax), ylim=(ymin_rf, ymax_rf))
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.set_xlabel('True_value')
    ax.set_ylabel('Prediction_value')
    ax.set_title("RF Result")
    ax.plot([0, xmax], [0, xmax], 'r--')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    # График для Lasso (ML)
    ax = axs[1]
    hb = ax.hexbin(x_true, y_pred_ml, gridsize=50, bins='log', cmap='Blues')
    ax.set(xlim=(xmin, xmax), ylim=(ymin_ml, ymax_ml))
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.set_xlabel('True_value')
    ax.set_ylabel('Prediction_value')
    ax.set_title("ML (Lasso) Result")
    ax.plot([0, xmax], [0, xmax], 'r--')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    plt.show()


def main():
    try:
        # Загрузка и предобработка данных
        X_sc, y_true, feature_columns = load_and_preprocess_data('../../data/processed/processed_buildings.parquet')
        X_train, X_test, y_train, y_test = split_data(X_sc, y_true)
        print_data_stats(y_true)

        # Инициализация менеджера моделей
        model_manager = ModelManager()

        # Обучение и оценка RandomForest
        grid_search = train_random_forest_with_gridsearch(X_train, y_train, model_manager)
        print('\nbest parameters:', grid_search.best_params_)
        print('best estimator:', grid_search.best_estimator_)
        print('best score:', grid_search.best_score_)

        rf_model = train_final_random_forest(X_train, y_train, grid_search.best_params_, model_manager)
        rfy_pred = evaluate_model(rf_model, X_sc, y_true, "RandomForest")

        # Сохранение результатов RandomForest
        save_results_to_files(
            rfy_pred, y_true,
            rf_model.feature_importances_ if hasattr(rf_model, 'feature_importances_') else None,
            prefix='rf'
        )

        # Обучение и оценка Lasso
        lasso_model = train_lasso_model(X_train, y_train, model_manager)
        ml_y_pred = evaluate_model(lasso_model, X_sc, y_true, "Lasso")

        # Вывод коэффициентов Lasso
        print('\nLasso coefficients:', lasso_model.coef_)
        print('Lasso intercept:', lasso_model.intercept_)

        # Сохранение результатов Lasso
        save_results_to_files(ml_y_pred, y_true, prefix='lasso')

        # Построение графиков
        plot_results(y_true, rfy_pred, ml_y_pred)

    except Exception as e:
        print("Ошибка в основном потоке выполнения:", e)
        raise


if __name__ == "__main__":
    main()