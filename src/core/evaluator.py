from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate(model, X, y_true, model_name=""):
        y_pred = model.predict(X)

        print(f"\nMetrics for {model_name}:")
        print("R2:", r2_score(y_true, y_pred))
        print('RMSE:', np.sqrt(mean_squared_error(y_true, y_pred)))
        print('MAE:', mean_absolute_error(y_true, y_pred))

        return y_pred

    @staticmethod
    def save_results(y_pred, y_true, feature_importances=None, prefix="", path='../../logs'):
        error = y_pred - y_true

        np.savetxt(f'{path}/{prefix}_error.txt', error, newline='\n', delimiter=' ')
        np.savetxt(f'{path}/{prefix}_pred_result.txt', y_pred, newline='\n', delimiter=' ')

        if feature_importances is not None:
            np.savetxt(f'{path}/{prefix}_feature_importance.txt', feature_importances, newline='\n', delimiter=' ')

    @staticmethod
    def print_prediction_stats(y_pred, y_true, model_name=""):
        print(f"{model_name} pred min/max:", np.min(y_pred), np.max(y_pred))
        print("True min/max:", np.min(y_true), np.max(y_true))