from typing import List, Tuple
from collections import defaultdict

class BayesianModel:
    """ModŠle bay‚sien simple bas‚ sur les fr‚quences des valeurs"""

    def __init__(self):
        self.value_counts = defaultdict(int)
        self.total = 0

    def update(self, new_values: List[float]):
        """Met … jour les comptes des valeurs"""
        for v in new_values:
            self.value_counts[round(v, 1)] += 1
            self.total += 1

    def predict(self) -> Tuple[List[float], List[float]]:
        """Retourne 3 pr‚dictions bas‚es sur les fr‚quences"""
        if self.total < 3:
            return [1.0, 1.5, 2.0], [0.5, 0.4, 0.3]

        # Trier les valeurs par fr‚quence d‚croissante
        sorted_values = sorted(
            self.value_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Top 3 valeurs les plus fr‚quentes
        predictions = [v[0] for v in sorted_values[:3]]

        # Confiances bas‚es sur la fr‚quence
        confidences = [min(0.9, v[1] / self.total * 2) for v in sorted_values[:3]]

        # Si moins de 3 valeurs uniques, compl‚ter avec des valeurs par d‚faut
        while len(predictions) < 3:
            predictions.append(predictions[-1] * 1.2)
            confidences.append(confidences[-1] * 0.8)

        return [round(p, 2) for p in predictions], [round(c, 2) for c in confidences]