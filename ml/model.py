import os
import joblib
from utils.config_loader import load_config

class EdgeRunnerMLModel:
    def __init__(self):
        config = load_config()
        ml_cfg = config.get('ml', {})
        self.use_ml = ml_cfg.get('use_ml', False)
        self.model_path = ml_cfg.get('model_path', 'models/edgerunner_model.pkl')
        self.model = None
        if self.use_ml:
            self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ML model file not found: {self.model_path}")
        model_obj = joblib.load(self.model_path)
        # Support both old (model only) and new (dict with model and feature_names)
        if isinstance(model_obj, dict) and 'model' in model_obj and 'feature_names' in model_obj:
            self.model = model_obj['model']
            self.feature_names = model_obj['feature_names']
        else:
            self.model = model_obj
            self.feature_names = None

    def predict_signal(self, features_df):
        """
        Given a DataFrame of features (indicators), return model predictions.
        Returns: Array-like of signals (e.g., 1=buy, -1=sell, 0=hold)
        """
        if not self.use_ml or self.model is None:
            raise RuntimeError("ML model is not loaded or ML is disabled in config.")
        # Ensure feature alignment
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            missing = [col for col in self.feature_names if col not in features_df.columns]
            for col in missing:
                features_df[col] = 0
            features_df = features_df[self.feature_names]
        return self.model.predict(features_df)
