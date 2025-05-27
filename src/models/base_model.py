from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_storage, model_name):
        self.model_storage = model_storage
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

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