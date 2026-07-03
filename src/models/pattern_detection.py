from typing import List, Dict
import re

class PatternDetector:
    def __init__(self):
        self.patterns = {
            "repetitions": self._detect_repetitions,
            "symmetries": self._detect_symmetries,
            "palindromes": self._detect_palindromes,
            "dominant_digits": self._detect_dominant_digits,
            "length_patterns": self._detect_length_patterns,
        }

    def detect_all(self, values: List[float]) -> Dict[str, any]:
        results = {}
        str_values = [str(v) for v in values]
        for pattern_name, detector in self.patterns.items():
            results[pattern_name] = detector(str_values)
        return results

    def _detect_repetitions(self, values: List[str]) -> Dict:
        """Détecte les répétitions de chiffres ou de séquences"""
        repetitions = {}
        for val in values:
            # Répétitions de chiffres (ex: 11, 222, 3434)
            for digit in set(val):
                count = val.count(digit)
                if count > 1:
                    repetitions[f"repetition_{digit}"] = repetitions.get(f"repetition_{digit}", 0) + 1
        return repetitions

    def _detect_symmetries(self, values: List[str]) -> List[str]:
        """Détecte les symétries (ex: 12321, 4554)"""
        symmetrical = []
        for val in values:
            if val == val[::-1][:len(val)]:  # Vérifie si symétrique
                symmetrical.append(val)
        return symmetrical

    def _detect_palindromes(self, values: List[str]) -> List[str]:
        """Détecte les palindromes (ex: 12321, 5445)"""
        return [val for val in values if val == val[::-1]]

    def _detect_dominant_digits(self, values: List[str]) -> Dict[str, int]:
        """Détecte les chiffres dominants"""
        digit_counts = {}
        for val in values:
            for digit in val.replace('.', ''):
                digit_counts[digit] = digit_counts.get(digit, 0) + 1
        return digit_counts

    def _detect_length_patterns(self, values: List[str]) -> Dict[int, int]:
        """Détecte les motifs de longueur"""
        length_counts = {}
        for val in values:
            length = len(val)
            length_counts[length] = length_counts.get(length, 0) + 1
        return length_counts
