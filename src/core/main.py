from config import Config
from src.core.evaluator import ModelEvaluator
from src.core.model_factory import ModelFactory
from src.storage.data_loader import DataLoader
from src.storage.data_preprocessor import DataPreprocessor
from src.storage.model_storage import ModelStorage

from src.utils.drawer import Drawer


def run_standard_models():
    """Запуск стандартного набора моделей"""
    try:
        Config.ensure_logs_directory_exists()

        # Загрузка и предобработка данных
        loader = DataLoader(Config.DATA_PATH)
        data = loader.load_data()

        preprocessor = DataPreprocessor(data)
        X_sc, y_true = preprocessor.preprocess()
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X_sc, y_true,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )

        preprocessor.print_data_stats(y_true)

        # Инициализация хранилища моделей
        model_storage = ModelStorage()

        # Список моделей для запуска
        models_to_run = [
            ('random_forest', Config.get_model_params('random_forest')),
            ('lasso', Config.get_model_params('lasso')),
            ('linear_regression', Config.get_model_params('linear_regression')),
            ('xgboost', Config.get_model_params('xgboost')),
            ('gradient_boosting', Config.get_model_params('gradient_boosting')),
            #('svr', Config.get_model_params('svr'))
        ]

        results = {}

        # Обучение и оценка всех моделей
        for model_name, params in models_to_run:
            print(f"\n=== Training {model_name} model ===")

            model = ModelFactory.create_model(model_name, model_storage, params)
            model.train(X_train, y_train)

            y_pred = ModelEvaluator.evaluate(model, X_sc, y_true, model_name)

            # Сохранение feature importance для моделей, которые это поддерживают
            feature_importances = getattr(model, 'get_feature_importances', lambda: None)()

            ModelEvaluator.save_results(
                y_pred, y_true,
                feature_importances,
                prefix=model_name
            )

            ModelEvaluator.print_prediction_stats(y_pred, y_true, model_name)

            results[model_name] = {
                'model': model,
                'predictions': y_pred
            }

        # Визуализация результатов
        drawer = Drawer()
        for model_name, result in results.items():
            drawer.plot_prediction(
                y_true, result['predictions'],
                title=f"{model_name} Predictions",
                color="viridis"
            )

    except Exception as e:
        print("Ошибка в основном потоке выполнения:", e)
        raise


if __name__ == "__main__":
    run_standard_models()
