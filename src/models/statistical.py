import numpy as np
from typing import List, Tuple

class StatisticalModel:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: List[float] = []

    def update(self, new_values: List[float]):
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        if len(self.history) < 3:
            return [1.0, 1.5, 2.0], [0.5, 0.4, 0.3]

        mean = np.mean(self.history[-self.window_size:])
        std = np.std(self.history[-self.window_size:])

        # Prédictions
        pred_low = max(0.1, mean - std)
        pred_mid = mean
        pred_high = mean + std

        # Confiance de base (entre 0.5 et 0.95)
        base_confidence = min(0.95, max(0.5, 0.7 + (0.25 if std < mean * 0.3 else 0)))

        # Confiances pour chaque prédiction (normalisées entre 0 et 1)
        confidences = [
            min(1.0, base_confidence * 1.1),  # Low prediction
            min(1.0, base_confidence * 1.0),  # Mid prediction
            min(1.0, base_confidence * 0.9)   # High prediction
        ]

        return [
            round(pred_low, 2),
            round(pred_mid, 2),
            round(pred_high, 2)
        ], [round(c, 2) for c in confidences]
