from config import Config
from src.core.evaluator import ModelEvaluator
from src.models.lasso_model import LassoModel
from src.models.random_forest_model import RandomForestModel
from src.storage.data_loader import DataLoader
from src.storage.data_preprocessor import DataPreprocessor
from src.storage.model_storage import ModelStorage
from src.utils.drawer import Drawer

def main():
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

        # Инициализация моделей
        model_storage = ModelStorage(Config.RF_PARAMS)

        # Обучение и оценка RandomForest
        rf_model = RandomForestModel(model_storage, Config.RF_PARAMS)
        best_params = rf_model.train_with_gridsearch(X_train, y_train)
        print('\nbest parameters:', best_params)

        rf_model.train(X_train, y_train)
        rfy_pred = ModelEvaluator.evaluate(rf_model, X_sc, y_true, "RandomForest")
        ModelEvaluator.save_results(
            rfy_pred, y_true,
            rf_model.get_feature_importances(),
            prefix='rf'
        )
        ModelEvaluator.print_prediction_stats(rfy_pred, y_true, "RF")

        # Обучение и оценка Lasso
        lasso_model = LassoModel(model_storage)
        lasso_model.train(X_train, y_train)
        ml_y_pred = ModelEvaluator.evaluate(lasso_model, X_sc, y_true, "Lasso")

        print('\nLasso coefficients:', lasso_model.get_coefficients())
        print('Lasso intercept:', lasso_model.get_intercept())

        ModelEvaluator.save_results(ml_y_pred, y_true, prefix='lasso')

        # Визуализация результатов
        drawer = Drawer()
        drawer.plot_prediction(y_true, rfy_pred, title="RF Predictions", color="Blues")
        drawer.plot_prediction(y_true, ml_y_pred, title="Lasso Predictions", color="Reds")

    except Exception as e:
        print("Ошибка в основном потоке выполнения:", e)
        raise

if __name__ == "__main__":
    main()