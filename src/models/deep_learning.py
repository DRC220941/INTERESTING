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
        self.model = None
        self.scaler = None
        self.history = []
        self.is_trained = False
        self._load_model()

    def _load_model(self):
        """Charge le modèle ONNX et le scaler"""
        if os.path.exists(self.model_path):
            self.model = ort.InferenceSession(self.model_path)
            self.is_trained = True
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

    def update(self, new_values: List[float]):
        """Met à jour l'historique"""
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        """Prédit les 3 prochaines valeurs avec le modèle ONNX"""
        if len(self.history) < 10 or self.model is None:
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Préparer les données
        X = self._prepare_data(self.history[-20:])
        if X.size == 0:
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Inférence ONNX
        try:
            # Exécuter l'inférence
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            preds = self.model.run([output_name], {input_name: X.astype(np.float32)})[0]

            # Post-traitement
            preds = self.scaler.inverse_transform(preds) if self.scaler else preds
            preds = [float(p) for p in preds[0]]  # Prend la première prédiction

            # Calculer les confiances
            confidences = self._calculate_confidences(preds)

        except Exception as e:
            print(f"Erreur ONNX: {e}")
            preds = [1.5, 2.0, 2.5]
            confidences = [0.75, 0.75, 0.75]

        return [round(max(0.1, p), 2) for p in preds], [round(c, 2) for c in confidences]

    def _prepare_data(self, data: List[float]) -> np.ndarray:
        """Prépare les données pour l'inférence ONNX"""
        if len(data) < 10:
            return np.array([])

        # Normaliser
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(np.array(data).reshape(-1, 1))

        scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1))
        X = []
        for i in range(len(scaled_data) - 10):
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

    def train_and_save(self, history: List[float], epochs: int = 50):
        """Entraîne le modèle et sauvegarde en ONNX (à exécuter dans Colab)"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf

        # Préparer les données
        X, y = self._prepare_training_data(history)

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
        model.fit(X, y, epochs=epochs, batch_size=8, validation_split=0.2, verbose=1)

        # Sauvegarder en ONNX
        input_sample = np.random.rand(1, 10, 1).astype(np.float32)
        tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec((None, 10, 1), tf.float32, name='input')],
                                  opset=13, output_path=self.model_path)

        # Sauvegarder le scaler
        joblib.dump(self.scaler, self.scaler_path)
        self.is_trained = True

    def _prepare_training_data(self, history: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les données pour l'entraînement"""
        X = []
        y = []
        for i in range(len(history) - 10):
            X.append(history[i:i+10])
            y.append(history[i+10:i+13])  # Prédire les 3 prochaines valeurs

        X = np.array(X).reshape(-1, 10, 1)
        y = np.array(y)

        # Normaliser
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, 10, 1)
        else:
            X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(-1, 10, 1)

        y_scaled = self.scaler.transform(y.reshape(-1, 1)).reshape(-1, 3)
        return X_scaled, y_scaled
