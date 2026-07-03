from typing import List, Tuple, Dict, Optional, Any
import re
from collections import Counter

class PatternDetector:
    def __init__(self):
        self.pattern_types = {
            "repetitions": self._detect_repetitions,
            "symmetries": self._detect_symmetries,
            "palindromes": self._detect_palindromes,
            "dominant_digits": self._detect_dominant_digits,
            "missing_digits": self._detect_missing_digits,
            "length_patterns": self._detect_length_patterns,
            "increasing_sequences": self._detect_increasing_sequences,
            "decreasing_sequences": self._detect_decreasing_sequences,
            "digit_pairs": self._detect_digit_pairs,
        }

    def detect_all(self, values: List[float]) -> Dict[str, Any]:
        """Détecte tous les motifs dans une liste de valeurs"""
        str_values = [self._format_value(v) for v in values]
        results = {}
        for pattern_name, detector in self.pattern_types.items():
            results[pattern_name] = detector(str_values)
        return results

    def _format_value(self, value: float) -> str:
        """Formate une valeur en chaîne sans le point décimal pour l'analyse"""
        return str(value).replace('.', '')

    def _detect_repetitions(self, values: List[str]) -> Dict[str, int]:
        """Détecte les répétitions de chiffres (ex: 11, 222, 3434)"""
        repetitions = {}
        for val in values:
            for digit in set(val):
                count = val.count(digit)
                if count > 1:
                    key = f"repetition_{digit}_{count}"
                    repetitions[key] = repetitions.get(key, 0) + 1
        return repetitions

    def _detect_symmetries(self, values: List[str]) -> List[str]:
        """Détecte les nombres symétriques (ex: 12321, 4554)"""
        symmetrical = []
        for val in values:
            if len(val) > 1 and val == val[::-1]:
                symmetrical.append(val)
        return symmetrical

    def _detect_palindromes(self, values: List[str]) -> List[str]:
        """Détecte les palindromes (identique à symétries pour les nombres)"""
        return self._detect_symmetries(values)

    def _detect_dominant_digits(self, values: List[str]) -> Dict[str, int]:
        """Détecte les chiffres dominants (apparaissant le plus souvent)"""
        digit_counts = Counter()
        for val in values:
            for digit in val:
                digit_counts[digit] += 1
        return dict(digit_counts.most_common())

    def _detect_missing_digits(self, values: List[str]) -> List[str]:
        """Détecte les chiffres manquants (0-9) dans l'historique"""
        all_digits = set('0123456789')
        present_digits = set()
        for val in values:
            present_digits.update(val)
        return sorted(list(all_digits - present_digits))

    def _detect_length_patterns(self, values: List[str]) -> Dict[int, int]:
        """Détecte les motifs de longueur (nombre de chiffres)"""
        length_counts = Counter()
        for val in values:
            length_counts[len(val)] += 1
        return dict(length_counts)

    def _detect_increasing_sequences(self, values: List[str]) -> List[str]:
        """Détecte les séquences strictement croissantes (ex: 123, 13579)"""
        increasing = []
        for val in values:
            if len(val) > 1 and all(val[i] < val[i+1] for i in range(len(val)-1)):
                increasing.append(val)
        return increasing

    def _detect_decreasing_sequences(self, values: List[str]) -> List[str]:
        """Détecte les séquences strictement décroissantes (ex: 321, 97531)"""
        decreasing = []
        for val in values:
            if len(val) > 1 and all(val[i] > val[i+1] for i in range(len(val)-1)):
                decreasing.append(val)
        return decreasing

    def _detect_digit_pairs(self, values: List[str]) -> Dict[str, int]:
        """Détecte les paires de chiffres fréquentes (ex: 12, 23, 45)"""
        pairs = Counter()
        for val in values:
            for i in range(len(val) - 1):
                pair = val[i:i+2]
                pairs[pair] += 1
        return dict(pairs.most_common())
