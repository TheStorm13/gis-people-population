from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, model_storage, params):
        super().__init__(model_storage, 'random_forest')
        self.params = params
        self.grid_search = None

    def train(self, X_train, y_train):
        if not self.load_model():
            print("Обучение новой модели RandomForest...")
            self.model = RandomForestRegressor(
                **self.grid_search.best_params_,
                oob_score=True,
                random_state=0,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            self.save_model()

    def train_with_gridsearch(self, X_train, y_train):
        self.grid_search = GridSearchCV(
            estimator=RandomForestRegressor(bootstrap=True, oob_score=True),
            param_grid=self.params,
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1,
            verbose=2
        )
        self.grid_search.fit(X_train, y_train)

        return self.grid_search.best_params_

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.feature_importances_