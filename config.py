import os

class Config:
    LOGS_DIR = './logs'
    DATA_PATH = '/home/ilya-groza/Projects/qgis-people-population/data/processed/samara_people_model_processed.parquet'

    RF_PARAMS = {
        'n_estimators': [150],
        'max_features': ['sqrt'],
        'max_depth': [7],
        'min_samples_split': [8],
        'min_samples_leaf': [8]
    }

    TEST_SIZE = 0.01
    RANDOM_STATE = 0

    @staticmethod
    def ensure_logs_directory_exists():
        if not os.path.exists(Config.LOGS_DIR):
            os.makedirs(Config.LOGS_DIR)