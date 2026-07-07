from typing import List, Tuple
import numpy as np
import onnxruntime as ort
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

class DeepLearningModel:
    def __init__(self, model_path: str = "lstm_model.onnx", scaler_path: str = "dl_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.scaler = None
        self.session = None
        self.history = []
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        """Charge le modèle ONNX et le scaler"""
        if os.path.exists(self.model_path):
            self.session = ort.InferenceSession(self.model_path)
            self.is_trained = True
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

    def update(self, new_values: List[float]):
        """Met à jour l'historique"""
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        """Prédit les 3 prochaines valeurs avec le modèle ONNX"""
        if len(self.history) < 10 or not self.is_trained:
            # Valeurs par défaut si pas assez de données ou modèle non chargé
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Préparer les données
        X = self._prepare_input(self.history[-10:])
        if X is None:
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Exécuter l'inférence ONNX
        try:
            ort_inputs = {self.session.get_inputs()[0].name: X}
            ort_outs = self.session.run(None, ort_inputs)
            predictions = ort_outs[0][0]  # Récupère les 3 prédictions

            # Dénormaliser
            if self.scaler:
                predictions = self.scaler.inverse_transform([predictions])[0]

            preds = [float(p) for p in predictions]
            confidences = self._calculate_confidences(preds)

            return [round(max(0.1, p), 2) for p in preds], [round(c, 2) for c in confidences]

        except Exception as e:
            print(f"Erreur ONNX: {e}")
            return [1.5, 2.0, 2.5], [0.75, 0.75, 0.75]

    def _prepare_input(self, data: List[float]) -> np.ndarray:
        """Prépare les données pour le modèle ONNX"""
        if len(data) < 10:
            return None

        # Normaliser
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(np.array(data).reshape(-1, 1))

        scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1))
        X = scaled_data[-10:].reshape(1, 10, 1).astype(np.float32)  # ONNX attend float32
        return X

    def _calculate_confidences(self, predictions: List[float]) -> List[float]:
        """Calcule les confiances basées sur la variance des prédictions"""
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        base_confidence = max(0.7, min(0.95, 1 - std_pred))
        return [base_confidence, base_confidence * 0.98, base_confidence * 0.95]

    def get_performance(self) -> float:
        """Retourne une estimation de la performance (0-1)"""
        if not self.is_trained:
            return 0.5
        return min(0.95, 0.7 + len(self.history) * 0.005)

    def train_and_save(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîne un modèle LSTM et le sauve en ONNX (à exécuter dans Colab)"""
        # Ce code est pour l'entraînement (à exécuter dans Google Colab)
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import MinMaxScaler

        # Normaliser
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(-1, 10, 1)
        y_train_scaled = scaler.transform(y_train.reshape(-1, 1)).reshape(-1, 3)

        # Construire le modèle
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(10, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3)  # Prédit 3 valeurs
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Entraîner
        model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=8, verbose=1)

        # Sauvegarder le modèle et le scaler
        model.save("lstm_model.h5")
        joblib.dump(scaler, self.scaler_path)

        # Convertir en ONNX
        import tf2onnx
        spec = (tf.TensorSpec((None, 10, 1), tf.float32, name="input"),
                tf.TensorSpec((None, 3), tf.float32, name="output"))
        output_path = self.model_path
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        self.is_trained = True
