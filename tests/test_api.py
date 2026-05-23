import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ─── Health routes ────────────────────────────────────────────

def test_root():
    """Route racine répond 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    """Route /health répond 200 et modèle chargé."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == "lgbm_best"


# ─── Cas nominal ──────────────────────────────────────────────

def test_predict_nominal():
    """Prédiction avec des valeurs réalistes retourne 200."""
    payload = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.4,
        "AMT_CREDIT": 500000.0,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_ANNUITY": 25000.0,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "decision" in data
    assert 0.0 <= data["score"] <= 1.0
    assert data["decision"] in ["ACCORD", "REFUS"]
    assert data["seuil"] == 0.16


# ─── Données manquantes ───────────────────────────────────────

def test_predict_empty_payload():
    """Payload vide — doit retourner 200 (toutes valeurs à 0 par défaut)."""
    response = client.post("/predict", json={})
    assert response.status_code == 200
    data = response.json()
    assert "score" in data


# ─── Types incorrects ─────────────────────────────────────────

def test_predict_wrong_type():
    """Type incorrect (string au lieu de float) — doit retourner 422."""
    payload = {"EXT_SOURCE_1": "abc", "AMT_CREDIT": "pas_un_nombre"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ─── Valeurs extrêmes ─────────────────────────────────────────

def test_predict_extreme_values():
    """Valeurs extrêmes — doit retourner 200 sans planter."""
    payload = {
        "EXT_SOURCE_1": 999999.0,
        "DAYS_BIRTH": -99999,
        "AMT_CREDIT": -1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ─── Score et décision cohérents ─────────────────────────────

def test_decision_coherence():
    """La décision est cohérente avec le score et le seuil."""
    response = client.post("/predict", json={})
    assert response.status_code == 200
    data = response.json()
    if data["score"] >= data["seuil"]:
        assert data["decision"] == "REFUS"
    else:
        assert data["decision"] == "ACCORD"


# ─── Latence présente ─────────────────────────────────────────

def test_latence_presente():
    """La latence est retournée et positive."""
    response = client.post("/predict", json={})
    data = response.json()
    assert "latence_ms" in data
    assert data["latence_ms"] > 0


# ─── Cas modèle non chargé ────────────────────────────────────

def test_health_model_not_loaded():
    """503 si modèle absent."""
    from app.main import ml_models
    ml_models.clear()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Modèle non chargé"


def test_predict_model_not_loaded():
    """503 si modèle absent sur /predict."""
    from app.main import ml_models
    ml_models.clear()
    response = client.post("/predict", json={})
    assert response.status_code == 503


def test_predict_internal_error(monkeypatch):
    """500 si erreur interne dans predict_score."""
    from app import main
    def broken_predict(*args, **kwargs):
        raise ValueError("Erreur simulée")
    monkeypatch.setattr(main, "predict_score", broken_predict)
    response = client.post("/predict", json={})
    assert response.status_code == 500
    assert "Erreur de prédiction" in response.json()["detail"]


# ─── Lifespan ─────────────────────────────────────────────────

def test_lifespan_erreur_chargement(monkeypatch):
    """Vérifie que le lifespan lève une erreur si modèle introuvable."""
    import joblib
    from fastapi.testclient import TestClient
    from app.main import app, ml_models
    ml_models.clear()

    def broken_load(*args, **kwargs):
        raise FileNotFoundError("Modèle introuvable")

    monkeypatch.setattr(joblib, "load", broken_load)

    with pytest.raises(Exception):
        with TestClient(app) as c:
            pass


# ─── Lifespan avec mock ───────────────────────────────────────

def test_lifespan_charge_modele_mock(monkeypatch):
    """Vérifie que le lifespan charge bien le modèle via mock."""
    import numpy as np
    from unittest.mock import MagicMock
    import joblib

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.74, 0.26]])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 175))

    monkeypatch.setattr(joblib, "load", lambda path: mock_model if "lgbm" in path else mock_scaler)

    from fastapi.testclient import TestClient
    from app.main import app, ml_models
    ml_models.clear()

    with TestClient(app) as c:
        response = c.get("/health")
        assert response.status_code == 200
        assert "model" in ml_models
