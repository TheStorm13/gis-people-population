from sklearn.linear_model import LinearRegression

from src.models.base_model import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self, model_storage):
        super().__init__(model_storage, 'linear_regression')

    def train(self, X_train, y_train):
        if not self.load_model():
            print("Training new Linear Regression model...")
            self.model = LinearRegression()
            self.model.fit(X_train, y_train)
            self.save_model()
