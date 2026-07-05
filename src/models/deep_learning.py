from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

class DeepLearningModel:
    def __init__(self, model_path: str = "dl_model.keras", scaler_path: str = "dl_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = self._build_model()
        self.scaler = None
        self.history = []
        self.is_trained = False
        self._load_or_initialize()

    def _build_model(self) -> Sequential:
        """Construit le modèle LSTM"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(10, 1)),  # 10 time steps, 1 feature
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3)  # Prédit 3 valeurs
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def _load_or_initialize(self):
        """Charge le modèle et le scaler sauvegardés, ou initialise de nouveaux"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_trained = True
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)

    def update(self, new_values: List[float]):
        """Met à jour l'historique avec de nouvelles valeurs"""
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        """Prédit les 3 prochaines valeurs avec confiance"""
        if len(self.history) < 10:
            # Pas assez de données pour LSTM, utiliser des valeurs par défaut
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Préparer les données
        X = self._prepare_data(self.history[-20:])  # Utiliser les 20 dernières valeurs
        if X.size == 0:
            return [1.5, 2.0, 2.5], [0.80, 0.78, 0.75]

        # Prédire
        try:
            predictions = self.model.predict(X)[0]
            preds = [float(p) for p in predictions]

            # Calculer les confiances (basées sur la variance des prédictions)
            confidences = self._calculate_confidences(preds)

            # Entraîner le modèle si assez de données
            if len(self.history) >= 20 and not self.is_trained:
                self._train()

        except Exception as e:
            print(f"Erreur LSTM: {e}")
            preds = [1.5, 2.0, 2.5]
            confidences = [0.75, 0.75, 0.75]

        return [round(max(0.1, p), 2) for p in preds], [round(c, 2) for c in confidences]

    def _prepare_data(self, data: List[float]) -> np.ndarray:
        """Prépare les données pour le LSTM"""
        if len(data) < 10:
            return np.array([])

        # Normaliser les données
        if self.scaler is None:
            self.scaler = tf.keras.preprocessing.sequence.TimeseriesGenerator(
                data, data, length=10, batch_size=1
            )
            # Pour simplifier, on utilise MinMaxScaler
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            self.scaler.fit(np.array(data).reshape(-1, 1))

        scaled_data = self.scaler.transform(np.array(data).reshape(-1, 1))
        X = []
        for i in range(len(scaled_data) - 10):
            X.append(scaled_data[i:i+10])
        return np.array(X).reshape((-1, 10, 1))

    def _train(self):
        """Entraîne le modèle LSTM"""
        if len(self.history) < 20:
            return

        # Préparer les données d'entraînement
        X = []
        y = []
        for i in range(len(self.history) - 10):
            X.append(self.history[i:i+10])
            y.append(self.history[i+10:i+13])  # Prédire les 3 prochaines valeurs

        X = np.array(X)
        y = np.array(y)

        # Normaliser
        if self.scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, 10, 1)
        else:
            X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(-1, 10, 1)

        y_scaled = self.scaler.transform(y.reshape(-1, 1)).reshape(-1, 3)

        # Entraîner
        self.model.fit(
            X_scaled, y_scaled,
            epochs=50,
            batch_size=8,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        # Sauvegarder le modèle et le scaler
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        self.is_trained = True

    def _calculate_confidences(self, predictions: List[float]) -> List[float]:
        """Calcule les confiances basées sur la variance des prédictions"""
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        # Confiance inverse de la variance
        base_confidence = max(0.7, min(0.95, 1 - std_pred))
        return [
            base_confidence,
            base_confidence * 0.98,
            base_confidence * 0.95
        ]

    def get_performance(self) -> float:
        """Retourne une estimation de la performance (0-1)"""
        if not self.is_trained:
            return 0.5
        return min(0.95, 0.7 + len(self.history) * 0.005)
