from sklearn.ensemble import GradientBoostingRegressor

from src.models.base_model import BaseModel


class GradientBoostingModel(BaseModel):
    def __init__(self, model_storage, params):
        super().__init__(model_storage, 'gradient_boosting')
        self.params = params

    def train(self, X_train, y_train):
        if not self.load_model():
            self.train_with_gridsearch(X_train, y_train, GradientBoostingRegressor(random_state=0))

            print("Training new Gradient Boosting model...")
            self.model = GradientBoostingRegressor(**self.grid_search.best_params_)
            self.model.fit(X_train, y_train)
            self.save_model()
