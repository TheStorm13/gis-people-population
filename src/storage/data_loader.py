import pandas as pd


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.feature_columns = None

    def load_data(self):
        """Загрузка данных из файла"""
        return pd.read_parquet(self.filepath)

    def get_feature_columns(self):
        """Получение названий признаков"""
        return self.feature_columns
