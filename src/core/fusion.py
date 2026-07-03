from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import random 
from src.models.pattern_detection import PatternDetector
from src.core.hypothesis import HypothesisGenerator
from src.core.memory import LearningMemory

class ModelFusion:
    def __init__(self, learning_mem: LearningMemory):
        self.models = {
            "statistical": None,  # À initialiser dans main.py
            "bayesian": None,
            "timeseries": None,
            "ml": None,
            "anomaly": None
        }
        self.learning_mem = learning_mem
        self.pattern_detector = PatternDetector()
        self.hypothesis_generator = HypothesisGenerator()
        self.default_weights = {
            "statistical": 0.3,
            "bayesian": 0.25,
            "timeseries": 0.25,
            "ml": 0.15,
            "anomaly": 0.05
        }

    def update_all(self, new_values: List[float]):
        """Met à jour tous les modèles avec de nouvelles valeurs"""
        for model in self.models.values():
            if model:
                model.update(new_values)

    def predict(self, history: List[float]) -> Tuple[List[float], List[float], Dict, Dict, Optional[Dict]]:
        """
        Retourne :
        - prédictions fusionnées
        - confiances fusionnées
        - infos des modèles
        - motifs détectés
        - nouvelle hypothèse (si applicable)
        """
        # Détection de motifs
        patterns = self.pattern_detector.detect_all(history)

        # Prédictions et confiances des modèles
        all_predictions = {}
        all_confidences = {}
        model_performance = {}

        for name, model in self.models.items():
            if model:
                try:
                    preds, confs = model.predict()
                    all_predictions[name] = preds
                    all_confidences[name] = confs
                    # Estimer la performance (à améliorer avec des métriques réelles)
                    model_performance[name] = model.get_performance() if hasattr(model, 'get_performance') else 0.8
                except Exception as e:
                    print(f"Erreur avec le modèle {name}: {e}")
                    # Utiliser des valeurs par défaut en cas d'erreur
                    all_predictions[name] = [1.0, 1.5, 2.0]
                    all_confidences[name] = [0.75, 0.75, 0.75]
                    model_performance[name] = 0.5

        # Récupérer les poids actuels (ou utiliser les poids par défaut)
        current_weights = self.learning_mem.get_weights()

        # Fusion pondérée
        fused_predictions = []
        fused_confidences = []
        for i in range(3):
            weighted_pred = sum(
                all_predictions[name][i] * current_weights.get(name, self.default_weights.get(name, 0))
                for name in all_predictions
            )
            weighted_conf = sum(
                all_confidences[name][i] * current_weights.get(name, self.default_weights.get(name, 0))
                for name in all_confidences
            )
            fused_predictions.append(round(weighted_pred, 2))
            fused_confidences.append(round(max(0.77, weighted_conf), 2))

        # Infos des modèles
        model_info = {
            name: {
                "predictions": all_predictions[name],
                "confidences": all_confidences[name],
                "weight": current_weights.get(name, self.default_weights.get(name, 0)),
                "performance": model_performance.get(name, 0.8)
            }
            for name in all_predictions
        }

        # Générer une nouvelle hypothèse (1 fois sur 3)
        new_hypothesis = None
        if random.random() < 0.3:  # 30% de chance de générer une hypothèse
            new_hypothesis = self.hypothesis_generator.generate(patterns, history)

        return fused_predictions, fused_confidences, model_info, patterns, new_hypothesis

    def update_weights(self, actual_value: float, predictions: Dict[str, List[float]]):
        """
        Met à jour les poids des modèles en fonction de la performance
        (appelé après qu'une nouvelle valeur réelle soit connue)
        """
        errors = {}
        for name, preds in predictions.items():
            # Calculer l'erreur (différence absolue moyenne)
            error = abs(preds[0] - actual_value)  # On compare avec la première prédiction
            errors[name] = error
            # Loguer la performance
            self.learning_mem.log_performance(name, error)

        # Mettre à jour les poids en fonction des erreurs (inverse de l'erreur)
        total_error = sum(errors.values())
        if total_error > 0:
            for name, error in errors.items():
                # Nouveau poids = poids actuel * (1 - erreur/total_error)
                new_weight = self.learning_mem.get_weights().get(name, self.default_weights.get(name, 0)) * (1 - error/total_error)
                # Limiter entre 0.05 et 0.5
                new_weight = max(0.05, min(0.5, new_weight))
                self.learning_mem.update_weight(name, new_weight)

        # Normaliser les poids pour qu'ils somment à 1
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalise les poids pour qu'ils somment à 1"""
        weights = self.learning_mem.get_weights()
        total = sum(weights.values())
        if total > 0:
            for name, weight in weights.items():
                self.learning_mem.update_weight(name, weight / total)
