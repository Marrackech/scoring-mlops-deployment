# 🏦 Scoring Crédit — API MLOps Prêt à Dépenser

API de prédiction du risque de défaut de remboursement, déployée avec FastAPI et Docker.

## 📁 Structure du projet
scoring-mlops-deployment/
├── app/
│   ├── main.py          # API FastAPI
│   ├── predict.py       # Logique de prédiction
│   ├── schemas.py       # Modèles Pydantic
│   └── logger.py        # Logging JSON
├── model/
│   ├── lgbm_best.joblib # Modèle LightGBM
│   ├── scaler.joblib    # StandardScaler
│   └── features.json   # Noms des features
├── tests/               # Tests pytest (100% coverage)
├── monitoring/
│   ├── dashboard.py     # Dashboard Streamlit
│   └── drift_analysis.ipynb  # Analyse Evidently
├── logs/                # Logs de production (gitignore)
├── Dockerfile
└── .github/workflows/ci_cd.yml## 🚀 Lancer l'API

### En local

```bash
conda create -n scoring-mlops python=3.10
conda activate scoring-mlops
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API disponible sur : http://127.0.0.1:8000
Documentation Swagger : http://127.0.0.1:8000/docs

### Avec Docker

```bash
docker build -t scoring-api:v1.0 .
docker run -p 7860:7860 scoring-api:v1.0
```

## 📊 Routes API

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Statut du modèle |
| POST | `/predict` | Prédiction scoring |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.4}'
```

### Exemple de réponse

```json
{
  "score": 0.12,
  "decision": "ACCORD",
  "seuil": 0.16,
  "latence_ms": 30.5
}
```

## 🧪 Tests

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

Couverture : **100%** — 17 tests

## 📈 Monitoring

### Lancer le dashboard Streamlit

```bash
conda activate scoring-monitoring
streamlit run monitoring/dashboard.py
```

Dashboard disponible sur : http://localhost:8501

### Métriques surveillées
- Distribution des scores prédits
- Taux ACCORD / REFUS
- Latence de l'API (~30ms)
- Data Drift (Evidently AI)

### Résultats drift
- 9 features analysées
- 1 feature driftée : **EXT_SOURCE_1** (p-value 0.015)
- Action recommandée : surveiller et ré-entraîner si drift s'amplifie

## 🔄 Pipeline CI/CD

Déclenché sur push vers `main` et `develop` :

1. **Tests** — pytest avec couverture
2. **Build** — construction image Docker
3. **Deploy** — déploiement automatique

## 🏷️ Versions

| Tag | Description |
|---|---|
| v0.1.0 | Structure projet + artefacts modèle |
| v0.2.0 | API FastAPI opérationnelle |
| v0.3.0 | Tests pytest 100% coverage |
| v0.4.0 | Dockerfile |
| v0.5.0 | Pipeline CI/CD |
| v0.6.1 | Monitoring + Data Drift |

## ⚙️ Stack technique

- **API** : FastAPI + Uvicorn
- **Modèle** : LightGBM (seuil métier 0.16)
- **Tests** : pytest + httpx
- **Docker** : python:3.10-slim
- **CI/CD** : GitHub Actions
- **Monitoring** : Streamlit + Evidently AI
