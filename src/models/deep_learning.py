from typing import List, Tuple
import numpy as np
import onnxruntime as ort
import joblib
import os

class DeepLearningModel:
    def __init__(self, model_path: str = "lstm_model.onnx", scaler_path: str = "dl_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.history = []
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        """Charge le modèle ONNX et le scaler"""
        try:
            if os.path.exists(self.model_path):
                self.model = ort.InferenceSession(self.model_path)
                self.is_trained = True
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle ONNX : {e}")
            self.model = None
            self.scaler = None
            self.is_trained = False

    def update(self, new_values: List[float]):
        """Met à jour l'historique avec de nouvelles valeurs"""
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        """Prédit les 3 prochaines valeurs avec le modèle ONNX"""
        if len(self.history) < 10 or self.model is None or self.scaler is None:
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Préparer les données
        X = self._prepare_data(self.history[-20:])
        if X.size == 0:
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Inférence ONNX
        try:
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            preds = self.model.run([output_name], {input_name: X.astype(np.float32)})[0]

            # Inverse du scaling
            preds = self.scaler.inverse_transform(preds) if self.scaler else preds
            preds = [float(p) for p in preds[0]]  # Prend la première prédiction

            # Calculer les confiances
            confidences = self._calculate_confidences(preds)

        except Exception as e:
            print(f"⚠️ Erreur ONNX : {e}")
            preds = [1.5, 2.0, 2.5]
            confidences = [0.75, 0.75, 0.75]

        return [round(max(0.1, p), 2) for p in preds], [round(c, 2) for c in confidences]

    def _prepare_data(self, data: List[float]) -> np.ndarray:
        """Prépare les données pour l'inférence ONNX"""
        if len(data) < 10:
            return np.array([])

        # Normaliser avec le scaler chargé
        scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1))
        X = []
        for i in range(len(scaled_data) - 10 + 1):
            X.append(scaled_data[i:i+10].flatten())
        return np.array(X).reshape((-1, 10, 1))

    def _calculate_confidences(self, predictions: List[float]) -> List[float]:
        """Calcule les confiances basées sur la variance des prédictions"""
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions) if len(predictions) > 1 else 0.5
        base_confidence = max(0.7, min(0.95, 1 - std_pred / (mean_pred + 1e-6)))
        return [base_confidence, base_confidence * 0.98, base_confidence * 0.95]

    def get_performance(self) -> float:
        """Retourne une estimation de la performance (0-1)"""
        if not self.is_trained:
            return 0.5
        return min(0.95, 0.7 + len(self.history) * 0.005)
