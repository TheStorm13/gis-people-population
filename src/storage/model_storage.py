import hashlib
from datetime import datetime
from pathlib import Path

import joblib


class ModelStorage:
    def __init__(self, models_root="../../embeddings"):
        self.params_hash = ''
        self.models_root = Path(models_root)
        self.models_root.mkdir(exist_ok=True)

    def save_model(self, model, model_name):
        """Сохраняет модель и метаданные"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_root / f"{model_name}"
        model_dir.mkdir(exist_ok=True)

        # Сохраняем модель
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        # Сохраняем метаданные
        metadata = {
            'timestamp': timestamp,
            'model_name': model_name,
            'model_type': str(type(model)),
            'params_hash': self.params_hash,
        }
        if hasattr(model, 'best_params_'):
            metadata['best_params'] = model.best_params_

        joblib.dump(metadata, model_dir / "metadata.joblib")

        print(f"Model saved to {model_path}")
        return model_dir

    def load_latest_model(self, model_name):
        """Загружает последнюю версию модели по имени"""
        # Ищем все директории, начинающиеся с имени модели
        model_dirs = [d for d in self.models_root.glob(f"{model_name}") if d.is_dir()]
        if not model_dirs:
            return None

        # Сортируем по времени создания (по имени, так как есть timestamp)
        latest_dir = sorted(model_dirs)[-1]
        model_path = latest_dir / f"{model_name}.joblib"

        if not model_path.exists():
            return None

        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def params_to_hash(self, params):
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
