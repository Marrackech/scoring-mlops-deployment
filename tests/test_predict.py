import pytest
import numpy as np
from unittest.mock import MagicMock
from app.predict import predict_score, SEUIL


@pytest.fixture
def model_and_scaler():
    """Mock du modèle et scaler pour les tests."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.74, 0.26]])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 175))

    return mock_model, mock_scaler


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
