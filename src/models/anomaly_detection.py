from typing import List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.history = []
        self.is_trained = False

    def update(self, new_values: List[float]):
        """Met à jour l'historique"""
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        """Prédit les 3 prochaines valeurs avec détection d'anomalies"""
        if len(self.history) < 3:
            return [1.0, 1.5, 2.0], [0.85, 0.80, 0.75]

        # Entraîner le modèle si assez de données
        if len(self.history) >= 5 and not self.is_trained:
            self._train()

        # Prédire la moyenne + variations
        mean = np.mean(self.history[-5:])
        std = np.std(self.history[-5:])

        pred1 = max(0.1, mean)
        pred2 = max(0.1, mean + std)
        pred3 = max(0.1, mean - std)

        # Détecter les anomalies dans l'historique
        if self.is_trained:
            X = np.array(self.history[-5:]).reshape(-1, 1)
            anomalies = self.model.predict(X)
            # Si des anomalies sont détectées, réduire la confiance
            confidence_reduction = 0.1 if -1 in anomalies else 0
        else:
            confidence_reduction = 0

        confidences = [
            round(max(0.77, 0.85 - confidence_reduction), 2),
            round(max(0.77, 0.82 - confidence_reduction), 2),
            round(max(0.77, 0.80 - confidence_reduction), 2)
        ]

        return [round(pred1, 2), round(pred2, 2), round(pred3, 2)], confidences

    def _train(self):
        """Entraîne le modèle de détection d'anomalies"""
        if len(self.history) < 5:
            return

        X = np.array(self.history).reshape(-1, 1)
        self.model.fit(X)
        self.is_trained = True

    def is_anomaly(self, value: float) -> bool:
        """Vérifie si une valeur est une anomalie"""
        if not self.is_trained:
            return False
        X = np.array([[value]])
        return self.model.predict(X)[0] == -1
