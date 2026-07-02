from fastapi import FastAPI

app = FastAPI(
    title="Moteur de Prédiction de Multiplicateurs",
    version="1.0.0"
)

@app.get("/health")
async def health():
    return {"status": "healthy"}