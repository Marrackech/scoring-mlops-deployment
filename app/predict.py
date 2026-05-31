import json
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Tuple

SEUIL = 0.16

with open("model/features.json", "r") as f:
    FEATURE_NAMES = json.load(f)

FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)

def safe_name(name: str) -> str:
    return (name.replace(" ", "_")
                .replace("/", "_")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_"))

SAFE_TO_REAL = {safe_name(f): f for f in FEATURE_NAMES}


def build_feature_array(input_data: dict) -> np.ndarray:
    """Construit un array numpy directement — plus rapide que DataFrame."""
    arr = np.zeros(N_FEATURES, dtype=np.float32)
    for safe, value in input_data.items():
        real = SAFE_TO_REAL.get(safe)
        if real and value is not None:
            idx = FEATURE_INDEX.get(real)
            if idx is not None:
                arr[idx] = float(value)
    return arr.reshape(1, -1)


@lru_cache(maxsize=1024)
def _cached_predict(model_id: int, scaler_id: int, features_tuple: tuple) -> Tuple[float, str]:
    """Cache LRU — évite de recalculer pour le même profil de features."""
    from app.main import ml_models
    arr = np.array(features_tuple, dtype=np.float32).reshape(1, -1)
    # DataFrame avec feature names pour éviter le warning sklearn
    df = pd.DataFrame(arr, columns=FEATURE_NAMES)
    arr_scaled = ml_models["scaler"].transform(df)
    score = ml_models["model"].predict_proba(arr_scaled)[0][1]
    decision = "REFUS" if score >= SEUIL else "ACCORD"
    return float(score), decision


def predict_score(model, scaler, input_data: dict) -> Tuple[float, str]:
    """
    Prédit le score de défaut d'un client.
    Optimisations : numpy array + cache LRU + DataFrame nommé pour scaler.

    Args:
        model: modèle LightGBM chargé
        scaler: StandardScaler chargé
        input_data: dictionnaire des features (noms safe Python)

    Returns:
        score (float): probabilité de défaut entre 0 et 1
        decision (str): ACCORD ou REFUS
    """
    arr = build_feature_array(input_data)
    features_tuple = tuple(arr.flatten().tolist())
    return _cached_predict(id(model), id(scaler), features_tuple)
