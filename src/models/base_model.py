from abc import ABC, abstractmethod

from sklearn.model_selection import GridSearchCV


class BaseModel(ABC):
    def __init__(self, model_storage, model_name):
        self.model_storage = model_storage
        self.model_name = model_name
        self.params = None
        self.model = None
        self.grid_search = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    def train_with_gridsearch(self, X_train, y_train, estimator):
        print("Запуск GridSearchCV для поиска лучших параметров...")
        self.grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.params,
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        self.grid_search.fit(X_train, y_train)
        return self.grid_search.best_params_

    def predict(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict(X)

    def load_model(self):
        self.model = self.model_storage.load_latest_model(self.model_name)
        return self.model is not None

    def save_model(self):
        if self.model is None:
            raise ValueError("Нет модели для сохранения")
        self.model_storage.save_model(self.model, self.model_name)
