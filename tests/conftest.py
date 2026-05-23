import pytest
import numpy as np
from unittest.mock import MagicMock
from app.main import ml_models


@pytest.fixture(autouse=True)
def load_models():
    """Mock le modèle et le scaler pour les tests CI/CD."""
    
    # Mock du modèle LightGBM
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.74, 0.26]])
    
    # Mock du scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 175))
    mock_scaler.feature_names_in_ = [f"feature_{i}" for i in range(175)]
    
    ml_models["model"] = mock_model
    ml_models["scaler"] = mock_scaler
    yield
    ml_models.clear()
