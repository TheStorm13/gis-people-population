from sklearn.svm import SVR

from src.models.base_model import BaseModel
from src.utils.progress_callback import ProgressCallback


class SVRModel(BaseModel):
    def __init__(self, model_storage):
        super().__init__(model_storage, 'svr')

    def train(self, X_train, y_train):
        if not self.load_model():
            print("Training new SVR model...")
            self.model = SVR()
            self.model.fit(X_train, y_train, callback=ProgressCallback(100))
            self.save_model()
