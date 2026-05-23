import pytest
import joblib
from app.predict import predict_score, SEUIL


@pytest.fixture
def model_and_scaler():
    model = joblib.load("model/lgbm_best.joblib")
    scaler = joblib.load("model/scaler.joblib")
    return model, scaler


def test_seuil_value():
    """Seuil métier correctement défini."""
    assert SEUIL == 0.16


def test_predict_returns_float(model_and_scaler):
    """predict_score retourne un float entre 0 et 1."""
    model, scaler = model_and_scaler
    score, decision = predict_score(model, scaler, {})
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_decision_type(model_and_scaler):
    """La décision est ACCORD ou REFUS."""
    model, scaler = model_and_scaler
    _, decision = predict_score(model, scaler, {})
    assert decision in ["ACCORD", "REFUS"]


def test_predict_coherence(model_and_scaler):
    """Score et décision sont cohérents avec le seuil."""
    model, scaler = model_and_scaler
    score, decision = predict_score(model, scaler, {})
    if score >= SEUIL:
        assert decision == "REFUS"
    else:
        assert decision == "ACCORD"
