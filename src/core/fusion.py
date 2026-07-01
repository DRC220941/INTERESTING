from typing import Dict, List, Tuple
import numpy as np
from ..models.statistical import StatisticalModel
from ..models.bayesian import BayesianModel
from ..models.timeseries import TimeSeriesModel

class ModelFusion:
    """Fusionne les pr‚dictions de plusieurs modŠles avec pond‚ration dynamique"""

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
        """Met … jour tous les modŠles avec de nouvelles valeurs"""
        for model in self.models.values():
            model.update(new_values)

    def predict(self) -> Tuple[List[float], List[float], Dict]:
        """Fusionne les pr‚dictions de tous les modŠles"""
        all_predictions = {}
        all_confidences = {}

        for name, model in self.models.items():
            preds, confs = model.predict()
            all_predictions[name] = preds
            all_confidences[name] = confs

        # Fusion pond‚r‚e des pr‚dictions (moyenne pond‚r‚e pour chaque position)
        fused_predictions = []
        fused_confidences = []
        for i in range(3):  # Top 3 pr‚dictions
            weighted_pred = sum(
                all_predictions[name][i] * self.weights[name]
                for name in self.models
            )
            fused_predictions.append(round(weighted_pred, 2))

            # Confiance moyenne pond‚r‚e
            weighted_conf = sum(
                all_confidences[name][i] * self.weights[name]
                for name in self.models
            )
            fused_confidences.append(round(weighted_conf, 2))

        # Informations sur les modŠles
        model_info = {
            name: {
                "predictions": all_predictions[name],
                "confidences": all_confidences[name],
                "weight": self.weights[name]
            }
            for name in self.models
        }

        return fused_predictions, fused_confidences, model_info

    def update_weights(self, true_value: float, predictions: Dict[str, List[float]]):
        """Met … jour les poids en fonction de la performance r‚elle"""
        for name in self.models:
            error = abs(true_value - predictions[name][0])  # Erreur sur la 1Šre pr‚diction
            self.performance[name].append(error)

            # R‚duire le poids si l'erreur est ‚lev‚e
            if len(self.performance[name]) > 5:
                avg_error = np.mean(self.performance[name][-5:])
                self.weights[name] = max(0.05, self.weights[name] * (1 - avg_error / 10))

        # Normaliser les poids
        total = sum(self.weights.values())
        for name in self.weights:
            self.weights[name] /= total