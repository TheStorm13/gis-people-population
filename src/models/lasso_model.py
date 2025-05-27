from sklearn.linear_model import LassoCV, Lasso

from src.models.base_model import BaseModel


class LassoModel(BaseModel):
    def __init__(self, model_storage):
        super().__init__(model_storage, 'lasso')

    def train(self, X_train, y_train):
        if not self.load_model():
            print("Обучение новой модели Lasso...")
            lassocv = LassoCV()
            lassocv.fit(X_train, y_train)
            print("Best alpha:", lassocv.alpha_)

            self.model = Lasso(alpha=lassocv.alpha_)
            self.model.fit(X_train, y_train)
            self.save_model()

    def get_coefficients(self):
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.coef_

    def get_intercept(self):
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.intercept_
