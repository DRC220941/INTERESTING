from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Tuple
import sqlite3
import os
from datetime import datetime

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================
class MultiplicateurRequest(BaseModel):
    values: List[float] = []

class PredictionResponse(BaseModel):
    predictions: List[float] = []
    confidences: List[float] = []
    top_features: Dict = {}
    hypotheses: Dict = {"validated": [], "rejected": []}
    warnings: List[str] = []
    timestamp: str = datetime.now().isoformat()

# ============================================================================
# MÉMOIRE DE TRAVAIL (SQLite)
# ============================================================================
class WorkingMemory:
    def __init__(self, db_path: str = "working_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multiplicateurs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_multiple(self, values: List[float]):
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("INSERT INTO multiplicateurs (value) VALUES (?)", [(v,) for v in values])
            conn.commit()

    def get_all(self) -> List[float]:
        with sqlite3.connect(self.db_path) as conn:
            return [row[0] for row in conn.execute("SELECT value FROM multiplicateurs ORDER BY timestamp").fetchall()]

    def reset(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM multiplicateurs")
            conn.commit()

# ============================================================================
# MODÈLES DE PRÉDICTION (Phase 2)
# ============================================================================
import numpy as np
from collections import defaultdict

class StatisticalModel:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: List[float] = []

    def update(self, new_values: List[float]):
        self.history.extend(new_values)

    def predict(self) -> Tuple[List[float], List[float]]:
        if len(self.history) < 3:
            return [1.0, 1.5, 2.0], [0.5, 0.4, 0.3]
        mean = np.mean(self.history[-self.window_size:])
        std = np.std(self.history[-self.window_size:])
        pred_low = max(0.1, mean - std)
        pred_mid = mean
        pred_high = mean + std
        confidence = min(0.95, 0.7 + (0.25 if std < mean * 0.3 else 0))
        return [round(pred_low, 2), round(pred_mid, 2), round(pred_high, 2)], [confidence, confidence * 1.1, confidence * 0.9]

class BayesianModel:
    def __init__(self):
        self.value_counts = defaultdict(int)
        self.total = 0

    def update(self, new_values: List[float]):
        for v in new_values:
            self.value_counts[round(v, 1)] += 1
            self.total += 1

    def predict(self) -> Tuple[List[float], List[float]]:
        if self.total < 3:
            return [1.0, 1.5, 2.0], [0.5, 0.4, 0.3]
        sorted_values = sorted(self.value_counts.items(), key=lambda x: x[1], reverse=True)
        predictions = [v[0] for v in sorted_values[:3]]
        confidences = [min(0.9, v[1] / self.total * 2) for v in sorted_values[:3]]
        while len(predictions) < 3:
            predictions.append(predictions[-1] * 1.2)
            confidences.append(confidences[-1] * 0.8)
        return [round(p, 2) for p in predictions], [round(c, 2) for c in confidences]

class TimeSeriesModel:
    def __init__(self):
        self.history: List[float] = []
        self.timestamps: List[int] = []

    def update(self, new_values: List[float]):
        start_idx = len(self.history)
        for i, v in enumerate(new_values):
            self.history.append(v)
            self.timestamps.append(start_idx + i)

    def predict(self) -> Tuple[List[float], List[float]]:
        if len(self.history) < 3:
            return [1.0, 1.5, 2.0], [0.5, 0.4, 0.3]
        x = np.array(self.timestamps)
        y = np.array(self.history)
        A = np.vstack([x, np.ones(len(x))]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        next_indices = [self.timestamps[-1] + 1, self.timestamps[-1] + 2, self.timestamps[-1] + 3]
        predictions = [a * idx + b for idx in next_indices]
        y_pred = a * x + b
        mse = np.mean((y - y_pred) ** 2)
        confidence = max(0.5, min(0.95, 1 - np.sqrt(mse) / (np.mean(y) + 1e-6)))
        return [round(p, 2) for p in predictions], [confidence] * 3

class ModelFusion:
    def __init__(self):
        self.models = {
            "statistical": StatisticalModel(),
            "bayesian": BayesianModel(),
            "timeseries": TimeSeriesModel()
        }
        self.weights = {"statistical": 0.4, "bayesian": 0.3, "timeseries": 0.3}
        self.performance = {name: [] for name in self.models}

    def update_all(self, new_values: List[float]):
        for model in self.models.values():
            model.update(new_values)

    def predict(self) -> Tuple[List[float], List[float], Dict]:
        all_predictions = {}
        all_confidences = {}
        for name, model in self.models.items():
            preds, confs = model.predict()
            all_predictions[name] = preds
            all_confidences[name] = confs

        fused_predictions = []
        fused_confidences = []
        for i in range(3):
            weighted_pred = sum(all_predictions[name][i] * self.weights[name] for name in self.models)
            fused_predictions.append(round(weighted_pred, 2))
            weighted_conf = sum(all_confidences[name][i] * self.weights[name] for name in self.models)
            fused_confidences.append(round(max(0.77, weighted_conf), 2))

        model_info = {
            name: {"predictions": all_predictions[name], "confidences": all_confidences[name], "weight": self.weights[name]}
            for name in self.models
        }
        return fused_predictions, fused_confidences, model_info

# ============================================================================
# INITIALISATION
# ============================================================================
working_mem = WorkingMemory()
fusion = ModelFusion()

# ============================================================================
# APPLICATION FASTAPI
# ============================================================================
app = FastAPI(
    from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise TOUS les domaines (pour le développement)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)
    title="Moteur de Prédiction de Multiplicateurs",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Moteur de Prédiction de Multiplicateurs",
        "version": "1.0.0",
        "working_memory_count": len(working_mem.get_all())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MultiplicateurRequest):
    if request.values:
        working_mem.add_multiple(request.values)
        fusion.update_all(request.values)

    history = working_mem.get_all()

    if len(history) >= 3:
        predictions, confidences, model_info = fusion.predict()
        warnings = [] if all(c >= 0.77 for c in confidences) else ["⚠️ Confiance < 77%"]
    else:
        predictions = [1.0, 1.5, 2.0]
        confidences = [0.5, 0.4, 0.3]
        warnings = ["Données insuffisantes"]
        model_info = {}

    return PredictionResponse(
        predictions=predictions,
        confidences=confidences,
        top_features={
            "last_value": round(history[-1], 2) if history else None,
            "mean": round(sum(history) / len(history), 2) if history else None,
            "count": len(history),
            "models": model_info
        },
        hypotheses={"validated": [], "rejected": []},
        warnings=warnings
    )

@app.post("/reset")
async def reset_memory():
    working_mem.reset()
    return {"status": "success", "message": "Mémoire de travail réinitialisée"}

@app.get("/history")
async def get_history():
    return {"history": [round(v, 2) for v in working_mem.get_all()], "count": len(working_mem.get_all())}

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
