from xgboost import XGBRegressor

from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, model_storage, params):
        super().__init__(model_storage, 'xgboost')
        self.params = params

    def train(self, X_train, y_train):
        if not self.load_model():
            self.train_with_gridsearch(X_train, y_train, XGBRegressor(random_state=0))


            print("Training new XGBoost model...")
            self.model = XGBRegressor(**self.grid_search.best_params_)
            self.model.fit(X_train, y_train)
            self.save_model()
