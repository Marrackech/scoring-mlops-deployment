import time
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import ClientData, PredictionResponse
from app.predict import predict_score, SEUIL
from app.logger import log_prediction


# Chargement du modèle une seule fois au démarrage
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et le scaler au démarrage de l'API."""
    try:
        ml_models["model"] = joblib.load("model/lgbm_best.joblib")
        ml_models["scaler"] = joblib.load("model/scaler.joblib")
        print("✅ Modèle et scaler chargés avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        raise e
    yield
    ml_models.clear()


# Initialisation de l'API
app = FastAPI(
    title="API Scoring Crédit — Prêt à Dépenser",
    description="API de prédiction du risque de défaut de remboursement.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/", tags=["Health"])
def root():
    """Route de santé — vérifie que l'API est en ligne."""
    return {"status": "ok", "message": "API Scoring Crédit opérationnelle"}


@app.get("/health", tags=["Health"])
def health():
    """Vérifie que le modèle est bien chargé."""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "model": "lgbm_best", "seuil": SEUIL}


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(data: ClientData):
    """
    Prédit le score de défaut d'un client.

    - **score** : probabilité de défaut entre 0 et 1
    - **decision** : ACCORD si score < seuil, REFUS sinon
    - **seuil** : seuil de décision optimisé selon coût métier
    """
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        start = time.time()
        input_data = data.model_dump()
        score, decision = predict_score(
            ml_models["model"],
            ml_models["scaler"],
            input_data
        )
        latence_ms = round((time.time() - start) * 1000, 3)

        # Log de la prédiction
        log_prediction(input_data, score, decision, latence_ms)

        return PredictionResponse(
            score=score,
            decision=decision,
            seuil=SEUIL,
            latence_ms=latence_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")
