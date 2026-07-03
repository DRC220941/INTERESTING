from typing import List, Dict, Optional
import random
from datetime import datetime

class HypothesisGenerator:
    def __init__(self):
        self.patterns = [
            "Les multiplicateurs supérieurs à 2.0 apparaissent plus fréquemment après une séquence croissante.",
            "Les palindromes (ex: 121, 3443) précèdent souvent des valeurs plus stables.",
            "Les chiffres dominants (ex: 1, 2) influencent la probabilité des prochains multiplicateurs.",
            "Les séquences de longueur 3 sont plus fréquentes après une anomalie.",
            "Les répétitions de chiffres (ex: 11, 222) annoncent une hausse des multiplicateurs.",
            "Les valeurs entre 1.5 et 2.5 sont plus probables après une séquence décroissante.",
            "Les chiffres manquants (ex: 7, 9) réduisent la variabilité des prédictions.",
            "Les motifs symétriques augmentent la confiance des prédictions.",
        ]
        self.used_hypotheses = set()

    def generate(self, patterns: Dict, history: List[float]) -> Optional[Dict]:
        """Génère une nouvelle hypothèse basée sur les motifs détectés"""
        if not patterns:
            return None

        # Sélectionner un motif aléatoire non utilisé
        available_patterns = [p for p in self.patterns if p not in self.used_hypotheses]
        if not available_patterns:
            self.used_hypotheses = set()  # Réinitialiser si toutes les hypothèses ont été utilisées

        hypothesis = random.choice(available_patterns)
        self.used_hypotheses.add(hypothesis)

        # Calculer une confiance initiale basée sur la fréquence des motifs
        confidence = self._calculate_confidence(patterns)

        return {
            "description": hypothesis,
            "confidence": round(confidence, 2),
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

    def _calculate_confidence(self, patterns: Dict) -> float:
        """Calcule une confiance initiale basée sur les motifs détectés"""
        total_patterns = sum(len(v) if isinstance(v, list) else sum(v.values()) if isinstance(v, dict) else 1
                          for v in patterns.values())
        return min(0.95, 0.5 + (total_patterns * 0.05))
