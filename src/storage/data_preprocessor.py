import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()

    def preprocess(self):
        """Основной метод предобработки данных"""
        y_true = self.data['peopleCount'].values
        features = self.data.drop(columns=['peopleCount', 'geometry'])
        self.feature_columns = features.columns

        X_true = features.values
        X_true = self._handle_inf_and_nan(X_true)
        X_sc = self.scaler.fit_transform(X_true)

        return X_sc, y_true

    def _handle_inf_and_nan(self, X):
        """Обработка inf и nan значений"""
        X = np.where(np.isinf(X) | np.isneginf(X), np.nan, X)
        return np.nan_to_num(X, nan=0)

    def split_data(self, X, y, test_size=0.01, random_state=0):
        """Разделение данных на train/test"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def print_data_stats(self, y_true):
        """Вывод статистики по данным"""
        print("Среднее значение population:", np.mean(y_true))
        print("Минимум и максимум:", np.min(y_true), "-", np.max(y_true))
        print("Стандартное отклонение:", np.std(y_true))
