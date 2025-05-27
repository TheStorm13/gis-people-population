from sklearn.ensemble import GradientBoostingRegressor

from ..models.gradient_boosting_model import GradientBoostingModel
from ..models.lasso_model import LassoModel
from ..models.linear_regression_model import LinearRegressionModel
from ..models.random_forest_model import RandomForestModel
from ..models.svr_model import SVRModel
from ..models.xgboost_model import XGBoostModel


class ModelFactory:
    @staticmethod
    def create_model(model_name, model_storage, params=None):
        if model_name == 'random_forest':
            return RandomForestModel(model_storage, params or {})
        elif model_name == 'lasso':
            return LassoModel(model_storage)
        elif model_name == 'linear_regression':
            return LinearRegressionModel(model_storage)
        elif model_name == 'xgboost':
            return XGBoostModel(model_storage, params or {})
        elif model_name == 'svr':
            return SVRModel(model_storage)
        elif model_name == 'gradient_boosting':
            return GradientBoostingModel(model_storage, params or {})
        else:
            raise ValueError(f"Unknown model type: {model_name}")