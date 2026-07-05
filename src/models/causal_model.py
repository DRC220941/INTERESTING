from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict

class CausalModel:
    def __init__(self):
        self.history = []
        self.pattern_history = defaultdict(list)  # {pattern: [next_values]}
        self.conditional_probs = defaultdict(lambda: defaultdict(int))  # {pattern: {next_value: count}}
        self.performance = 0.7  # Performance initiale

    def update(self, new_values: List[float]):
        """Met à jour l'historique et les motifs causaux"""
        self.history.extend(new_values)

        # Si on a assez de données, mettre à jour les probabilités conditionnelles
        if len(self.history) >= 5:
            self._update_conditional_probs()

    def predict(self) -> Tuple[List[float], List[float]]:
        """Prédit les 3 prochaines valeurs en utilisant des règles causales"""
        if len(self.history) < 3:
            return [1.5, 2.0, 2.5], [0.75, 0.75, 0.75]

        # Extraire le dernier motif (3 dernières valeurs)
        last_pattern = tuple(round(v, 2) for v in self.history[-3:])

        # Chercher des motifs similaires dans l'historique
        similar_patterns = self._find_similar_patterns(last_pattern)

        if similar_patterns:
            # Prédire la moyenne des valeurs suivantes pour ces motifs
            next_values = []
            for pattern in similar_patterns:
                # Trouver les valeurs qui suivent ce motif dans l'historique
                for i in range(len(self.history) - 3):
                    if tuple(round(v, 2) for v in self.history[i:i+3]) == pattern:
                        if i + 3 < len(self.history):
                            next_values.append(self.history[i+3])

            if next_values:
                # Calculer les prédictions
                mean_next = np.mean(next_values)
                std_next = np.std(next_values) if len(next_values) > 1 else 0.5

                pred1 = max(0.1, mean_next)
                pred2 = max(0.1, mean_next + std_next)
                pred3 = max(0.1, mean_next - std_next)

                # Calculer les confiances
                confidence = min(0.95, 0.7 + (1 - std_next/mean_next) if mean_next > 0 else 0.7)
                confidences = [confidence, confidence * 0.95, confidence * 0.9]

                # Mettre à jour la performance
                self.performance = min(0.95, self.performance + 0.01)
            else:
                # Pas de motif similaire trouvé, utiliser des valeurs par défaut
                pred1, pred2, pred3 = 1.5, 2.0, 2.5
                confidences = [0.7, 0.7, 0.7]
        else:
            pred1, pred2, pred3 = 1.5, 2.0, 2.5
            confidences = [0.7, 0.7, 0.7]

        return [round(pred1, 2), round(pred2, 2), round(pred3, 2)], [round(c, 2) for c in confidences]

    def _find_similar_patterns(self, pattern: tuple, threshold: float = 0.2) -> List[tuple]:
        """Trouve des motifs similaires dans l'historique"""
        similar = []
        for i in range(len(self.history) - 2):
            current_pattern = tuple(round(v, 2) for v in self.history[i:i+3])
            # Calculer la distance entre les motifs
            distance = sum(abs(p - c) for p, c in zip(pattern, current_pattern)) / len(pattern)
            if distance <= threshold:
                similar.append(current_pattern)
        return similar

    def _update_conditional_probs(self):
        """Met à jour les probabilités conditionnelles"""
        for i in range(len(self.history) - 1):
            current = round(self.history[i], 2)
            next_val = round(self.history[i+1], 2)
            self.conditional_probs[current][next_val] += 1

    def get_performance(self) -> float:
        """Retourne la performance actuelle du modèle"""
        return self.performance
