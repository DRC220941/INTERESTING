from typing import Dict, List, Tuple
import numpy as np
from src.models.statistical import StatisticalModel
from src.models.bayesian import BayesianModel
from src.models.timeseries import TimeSeriesModel

class ModelFusion:
    """Fusionne les prédictions de plusieurs modèles avec pondération dynamique"""

    def __init__(self):
        self.models = {
            "statistical": StatisticalModel(),
            "bayesian": BayesianModel(),
            "timeseries": TimeSeriesModel()
        }
        self.weights = {
            "statistical": 0.4,
            "bayesian": 0.3,
            "timeseries": 0.3
        }
        self.performance = {name: [] for name in self.models}

    def update_all(self, new_values: List[float]):
        """Met à jour tous les modèles avec de nouvelles valeurs"""
        for model in self.models.values():
            model.update(new_values)

    def predict(self) -> Tuple[List[float], List[float], Dict]:
        """Fusionne les prédictions de tous les modèles"""
        all_predictions = {}
        all_confidences = {}

        for name, model in self.models.items():
            preds, confs = model.predict()
            all_predictions[name] = preds
            all_confidences[name] = confs

        # Fusion pondérée
        fused_predictions = []
        fused_confidences = []
        for i in range(3):
            weighted_pred = sum(
                all_predictions[name][i] * self.weights[name]
                for name in self.models
            )
            fused_predictions.append(round(weighted_pred, 2))

            weighted_conf = sum(
                all_confidences[name][i] * self.weights[name]
                for name in self.models
            )
            fused_confidences.append(round(weighted_conf, 2))

        model_info = {
            name: {
                "predictions": all_predictions[name],
                "confidences": all_confidences[name],
                "weight": self.weights[name]
            }
            for name in self.models
        }

        return fused_predictions, fused_confidences, model_info
