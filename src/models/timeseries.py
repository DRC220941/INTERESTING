import numpy as np
from typing import List, Tuple

class TimeSeriesModel:
    """ModŠle bas‚ sur une r‚gression lin‚aire simple"""

    def __init__(self):
        self.history: List[float] = []
        self.timestamps: List[int] = []

    def update(self, new_values: List[float]):
        """Met … jour l'historique avec de nouvelles valeurs"""
        start_idx = len(self.history)
        for i, v in enumerate(new_values):
            self.history.append(v)
            self.timestamps.append(start_idx + i)

    def predict(self) -> Tuple[List[float], List[float]]:
        """Pr‚dit les 3 prochaines valeurs avec une r‚gression lin‚aire"""
        if len(self.history) < 3:
            return [1.0, 1.5, 2.0], [0.5, 0.4, 0.3]

        # R‚gression lin‚aire : y = a*x + b
        x = np.array(self.timestamps)
        y = np.array(self.history)
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # Pr‚dictions pour les 3 prochains pas de temps
        next_indices = [self.timestamps[-1] + 1, self.timestamps[-1] + 2, self.timestamps[-1] + 3]
        predictions = [a * idx + b for idx in next_indices]

        # Confiance bas‚e sur la qualit‚ de la r‚gression
        y_pred = a * x + b
        mse = np.mean((y - y_pred) ** 2)
        confidence = max(0.5, min(0.95, 1 - np.sqrt(mse) / (np.mean(y) + 1e-6)))

        return [round(p, 2) for p in predictions], [confidence] * 3