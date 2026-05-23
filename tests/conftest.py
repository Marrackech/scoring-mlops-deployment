import pytest
import joblib
from app.main import ml_models


@pytest.fixture(autouse=True)
def load_models():
    """Charge le modèle et le scaler avant chaque test."""
    ml_models["model"] = joblib.load("model/lgbm_best.joblib")
    ml_models["scaler"] = joblib.load("model/scaler.joblib")
    yield
    ml_models.clear()
