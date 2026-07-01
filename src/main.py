from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import sqlite3
import os
from datetime import datetime

app = FastAPI(
    title="Moteur de Prédiction de Multiplicateurs",
    version="1.0.0"
)

# ========== MÉMOIRE DE TRAVAIL ==========
class WorkingMemory:
    def __init__(self, db_path="working_memory.db"):
        self.db_path = db_path
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

# ========== INITIALISATION ==========
working_mem = WorkingMemory()

# ========== MODÈLES PYDANTIC ==========
class MultiplicateurRequest(BaseModel):
    values: List[float] = []

class PredictionResponse(BaseModel):
    predictions: List[float] = []
    confidences: List[float] = []
    top_features: dict = {}
    hypotheses: dict = {"validated": [], "rejected": []}
    warnings: List[str] = []
    timestamp: str = datetime.now().isoformat()

# ========== ENDPOINTS ==========
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Moteur de Prédiction",
        "version": "1.0.0",
        "working_memory_count": len(working_mem.get_all())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MultiplicateurRequest):
    if request.values:
        working_mem.add_multiple(request.values)
    history = working_mem.get_all()
    if len(history) < 3:
        return PredictionResponse(
            predictions=[1.0, 1.5, 2.0],
            confidences=[0.5, 0.4, 0.3],
            warnings=["Données insuffisantes"]
        )
    mean_val = sum(history) / len(history)
    return PredictionResponse(
        predictions=[round(mean_val * 0.8, 2), round(mean_val, 2), round(mean_val * 1.5, 2)],
        confidences=[0.75, 0.85, 0.65],
        warnings=[] if all(c >= 0.77 for c in [0.75, 0.85, 0.65]) else ["⚠️ Confiance < 77%"],
        top_features={"last_value": round(history[-1], 2), "mean": round(mean_val, 2), "count": len(history)}
    )

@app.post("/reset")
async def reset_memory():
    working_mem.reset()
    return {"status": "success", "message": "Mémoire réinitialisée"}

@app.get("/history")
async def get_history():
    return {"history": [round(v, 2) for v in working_mem.get_all()], "count": len(working_mem.get_all())}