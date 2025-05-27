from sklearn.ensemble import GradientBoostingRegressor
from .base_model import BaseModel

class GradientBoostingModel(BaseModel):
    def __init__(self, model_storage, params):
        super().__init__(model_storage, 'gradient_boosting')
        self.params = params

    def train(self, X_train, y_train):
        if not self.load_model():
            print("Training new Gradient Boosting model...")
            self.model = GradientBoostingRegressor(**self.params)
            self.model.fit(X_train, y_train)
            self.save_model()