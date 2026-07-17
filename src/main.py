import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any
from datetime import datetime

# ============================================================================
# IMPORTS DES MODÈLES ET MÉMOIRES
# ============================================================================
from src.core.memory import WorkingMemory, LearningMemory
from src.core.fusion import ModelFusion
from src.models.statistical import StatisticalModel
from src.models.bayesian import BayesianModel
from src.models.timeseries import TimeSeriesModel
from src.models.ml_model import MLModel
from src.models.anomaly_detection import AnomalyDetector
from src.models.causal_model import CausalModel
from src.models.optimization_model import OptimizationModel

# Deep Learning (toujours activé si onnxruntime est installé)
try:
    from src.models.deep_learning import DeepLearningModel
    deep_learning_model = DeepLearningModel()
    print("✅ Deep Learning activé (ONNX Runtime trouvé).")
except ImportError as e:
    print(f"⚠️ ONNX Runtime non disponible : {e}. Deep Learning désactivé.")
    deep_learning_model = None
except Exception as e:
    print(f"⚠️ Erreur lors du chargement du Deep Learning : {e}. Deep Learning désactivé.")
    deep_learning_model = None

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================
class MultiplicateurRequest(BaseModel):
    values: List[float] = []

    @validator('values', pre=True)
    def parse_semicolon_separated(cls, v):
        """Accepte les valeurs séparées par ; ou ,"""
        if isinstance(v, str):
            return [float(x.strip()) for x in v.replace(';', ',').split(',') if x.strip()]
        return v

class PredictionResponse(BaseModel):
    predictions: List[float] = []
    confidences: List[float] = []
    top_features: Dict = {}
    hypotheses: Dict = {"validated": [], "rejected": [], "pending": []}
    warnings: List[str] = []
    patterns: Dict = {}
    new_hypothesis: Optional[Dict] = None
    timestamp: str = datetime.now().isoformat()

# ============================================================================
# INITIALISATION
# ============================================================================
# Mémoires
working_mem = WorkingMemory()
learning_mem = LearningMemory()

# Modèles
statistical_model = StatisticalModel()
bayesian_model = BayesianModel()
timeseries_model = TimeSeriesModel()
ml_model = MLModel()
anomaly_model = AnomalyDetector()
causal_model = CausalModel()
optimization_model = OptimizationModel()

# Fusion
fusion = ModelFusion(learning_mem)
fusion.models = {
    "statistical": statistical_model,
    "bayesian": bayesian_model,
    "timeseries": timeseries_model,
    "ml": ml_model,
    "anomaly": anomaly_model,
    "causal": causal_model,
    "optimization": optimization_model,
    "deep_learning": deep_learning_model,
}

# Initialiser les poids par défaut
for name, weight in fusion.default_weights.items():
    learning_mem.update_weight(name, weight)

# ============================================================================
# APPLICATION FASTAPI + CORS
# ============================================================================
app = FastAPI(
    title="Moteur de Prédiction de Multiplicateurs",
    version="2.0.0",
    description="Moteur IA pour l'analyse et la prédiction de multiplicateurs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Moteur de Prédiction de Multiplicateurs",
        "version": "2.0.0",
        "working_memory_count": working_mem.count(),
        "models": list(fusion.models.keys()),
        "model_weights": learning_mem.get_weights(),
        "deep_learning_available": deep_learning_model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MultiplicateurRequest):
    # Ajouter les nouvelles valeurs à la mémoire de travail
    if request.values:
        working_mem.add_multiple(request.values)
        fusion.update_all(request.values)

    # Récupérer l'historique
    history = working_mem.get_all()

    if len(history) >= 3:
        predictions, confidences, model_info, patterns, new_hypothesis = fusion.predict(history)
        warnings = [] if all(c >= 0.77 for c in confidences) else ["⚠️ Confiance < 77%"]
    else:
        predictions = [1.0, 1.5, 2.0]
        confidences = [0.85, 0.80, 0.75]
        model_info = {}
        patterns = {}
        new_hypothesis = None
        warnings = ["Données insuffisantes"]

    # Récupérer les hypothèses existantes
    all_hypotheses = learning_mem.get_hypotheses()
    validated = [h for h in all_hypotheses if h["status"] == "validated"]
    rejected = [h for h in all_hypotheses if h["status"] == "rejected"]
    pending = [h for h in all_hypotheses if h["status"] == "pending"]

    # Ajouter la nouvelle hypothèse si elle existe
    if new_hypothesis:
        learning_mem.add_hypothesis(new_hypothesis["description"], new_hypothesis["confidence"])
        pending.append(new_hypothesis)

    return PredictionResponse(
        predictions=predictions,
        confidences=confidences,
        top_features={
            "last_value": round(history[-1], 2) if history else None,
            "mean": round(sum(history) / len(history), 2) if history else None,
            "count": len(history),
            "models": model_info,
            "patterns": patterns
        },
        hypotheses={
            "validated": validated,
            "rejected": rejected,
            "pending": pending
        },
        warnings=warnings,
        patterns=patterns,
        new_hypothesis=new_hypothesis,
        timestamp=datetime.now().isoformat()
    )

@app.post("/actual_value")
async def log_actual_value(value: float):
    history = working_mem.get_all()
    if len(history) >= 3:
        last_predictions = {name: model.predict()[0] for name, model in fusion.models.items() if model}
        fusion.update_weights(value, last_predictions)
    return {"status": "success", "message": "Valeur réelle enregistrée, poids des modèles mis à jour"}

@app.post("/reset")
async def reset_memory():
    working_mem.reset()
    return {"status": "success", "message": "Mémoire de travail réinitialisée (mémoire d'apprentissage conservée)"}

@app.get("/history")
async def get_history():
    return {
        "history": [round(v, 2) for v in working_mem.get_all()],
        "count": working_mem.count()
    }

@app.get("/hypotheses")
async def get_hypotheses():
    return {
        "hypotheses": learning_mem.get_hypotheses()
    }

@app.post("/validate_hypothesis/{hypothesis_id}")
async def validate_hypothesis(hypothesis_id: int):
    learning_mem.update_hypothesis(hypothesis_id, "validated", 0.9)
    return {"status": "success", "message": f"Hypothèse {hypothesis_id} validée"}

@app.post("/reject_hypothesis/{hypothesis_id}")
async def reject_hypothesis(hypothesis_id: int):
    learning_mem.update_hypothesis(hypothesis_id, "rejected", 0.1)
    return {"status": "success", "message": f"Hypothèse {hypothesis_id} rejetée"}
