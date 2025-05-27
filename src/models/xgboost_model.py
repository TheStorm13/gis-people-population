from xgboost import XGBRegressor
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, model_storage, params):
        super().__init__(model_storage, 'xgboost')
        self.params = params

    def train(self, X_train, y_train):
        if not self.load_model():
            print("Training new XGBoost model...")
            self.model = XGBRegressor(**self.params)
            self.model.fit(X_train, y_train)
            self.save_model()