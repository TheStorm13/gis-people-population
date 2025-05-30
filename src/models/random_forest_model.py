from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, model_storage, params):
        super().__init__(model_storage, 'random_forest')
        self.params = params

    def train(self, X_train, y_train):
        if not self.load_model():
            self.train_with_gridsearch(X_train, y_train, RandomForestRegressor(bootstrap=True, oob_score=True))

            print("Обучение новой модели RandomForest...")
            self.model = RandomForestRegressor(
                **self.grid_search.best_params_,
                oob_score=True,
                random_state=0,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            self.save_model()

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.feature_importances_
