# ========== NOUVELLES IMPORTS ==========
from src.core.fusion import ModelFusion

# ========== INITIALISATION ==========
working_mem = WorkingMemory()
learning_mem = LearningMemory()
fusion = ModelFusion()  # <-- Ajoutez cette ligne

# ========== ENDPOINT /predict (MIS À JOUR) ==========
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MultiplicateurRequest):
    if request.values:
        working_mem.add_multiple(request.values)
        fusion.update_all(request.values)  # Met à jour tous les modèles

    history = working_mem.get_all()

    # Si assez de données, utiliser la fusion de modèles
    if len(history) >= 3:
        predictions, confidences, model_info = fusion.predict()
        warnings = [] if all(c >= 0.77 for c in confidences) else ["⚠️ Confiance < 77%"]
    else:
        # Sinon, utiliser la logique placeholder
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
            "models": model_info  # Ajoute les infos des modèles
        },
        hypotheses={"validated": [], "rejected": []},
        warnings=warnings
    )
