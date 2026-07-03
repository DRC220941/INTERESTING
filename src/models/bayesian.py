from typing import List, Tuple
from collections import defaultdict
import numpy as np

class BayesianModel:
    def __init__(self):
        self.value_counts = defaultdict(int)
        self.total = 0
        self.history: List[float] = []

    def update(self, new_values: List[float]):
        for v in new_values:
            self.value_counts[round(v, 1)] += 1
            self.total += 1
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        if self.total < 3:
            return [1.0, 1.5, 2.0], [0.5, 0.4, 0.3]

        # Top 3 valeurs les plus fréquentes
        sorted_values = sorted(
            self.value_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        predictions = [v[0] for v in sorted_values[:3]]
        frequencies = [v[1] / self.total for v in sorted_values[:3]]

        # Calcul des confiances (basé sur la fréquence et la variance)
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies) if len(frequencies) > 1 else 0

        # Confiance = fréquence normalisée (entre 0.5 et 0.95)
        confidences = [
            min(0.95, max(0.5, freq * 2))  # Échelle 0-1 → 0.5-0.95
            for freq in frequencies
        ]

        # Si moins de 3 valeurs uniques, compléter
        while len(predictions) < 3:
            last_pred = predictions[-1]
            last_conf = confidences[-1]
            predictions.append(round(last_pred * 1.2, 2))
            confidences.append(round(last_conf * 0.8, 2))

        return [round(p, 2) for p in predictions], [round(c, 2) for c in confidences]
