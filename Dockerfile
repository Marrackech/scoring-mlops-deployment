# Image de base légère Python 3.10
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="Marrackech"
LABEL description="API Scoring Crédit — Prêt à Dépenser"
LABEL version="1.0.0"

# Installation des dépendances système pour LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copie des dépendances en premier (cache Docker)
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source uniquement
COPY app/ ./app/
COPY model/ ./model/

# Création du dossier logs
RUN mkdir -p logs

# Port exposé
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
