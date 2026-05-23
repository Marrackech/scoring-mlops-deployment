import json
import os
from datetime import datetime


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "predictions.json")


def log_prediction(input_data: dict, score: float, decision: str, latence_ms: float):
    """Enregistre chaque prédiction dans un fichier JSON structuré."""

    os.makedirs(LOG_DIR, exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "score": score,
        "decision": decision,
        "latence_ms": latence_ms
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
