import os

class Config:
    LOGS_DIR = './logs'
    DATA_PATH = '/home/ilya-groza/Projects/qgis-people-population/data/processed/samara_people_model_processed.parquet'

    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': [150],
            'max_features': ['sqrt'],
            'max_depth': [7],
            'min_samples_split': [8],
            'min_samples_leaf': [8]
        },
        'xgboost': {
            'n_estimators': [100, 150],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'gradient_boosting': {
            'n_estimators': [100, 150],
            'learning_rate': [0.05],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        },
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'selection': ['cyclic', 'random']
        },
        'linear_regression': None,  # Без GridSearch
        'svr': None  # Без GridSearch (можно добавить параметры позже)
    }

    TEST_SIZE = 0.15
    RANDOM_STATE = 0

    @staticmethod
    def ensure_logs_directory_exists():
        if not os.path.exists(Config.LOGS_DIR):
            os.makedirs(Config.LOGS_DIR)

    @staticmethod
    def get_model_params(model_name):
        return Config.MODEL_PARAMS.get(model_name)