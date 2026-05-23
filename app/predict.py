import json
import pandas as pd
import joblib
from typing import Tuple

SEUIL = 0.16  # Seuil optimisé selon coût métier LightGBM (projet 6)

# Chargement des vrais noms de features
with open("model/features.json", "r") as f:
    FEATURE_NAMES = json.load(f)

# Mapping : nom safe Python -> vrai nom modèle
def safe_name(name: str) -> str:
    return (name.replace(" ", "_")
                .replace("/", "_")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_"))

SAFE_TO_REAL = {safe_name(f): f for f in FEATURE_NAMES}


def predict_score(model, scaler, input_data: dict) -> Tuple[float, str]:
    """
    Prédit le score de défaut d'un client.

    Args:
        model: modèle LightGBM chargé
        scaler: StandardScaler chargé
        input_data: dictionnaire des features (noms safe Python)

    Returns:
        score (float): probabilité de défaut entre 0 et 1
        decision (str): ACCORD ou REFUS
    """
    # Reconstruction avec les vrais noms de features
    row = {}
    for real_name in FEATURE_NAMES:
        safe = safe_name(real_name)
        row[real_name] = input_data.get(safe, None)

    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    df = df.fillna(0)

    # Application du scaler
    df_scaled = scaler.transform(df)

    # Prédiction
    score = model.predict_proba(df_scaled)[0][1]
    decision = "REFUS" if score >= SEUIL else "ACCORD"

    return float(score), decision
