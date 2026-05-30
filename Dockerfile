FROM python:3.10-slim

LABEL maintainer="UserMarrakech"
LABEL description="API Scoring Crédit — Prêt à Dépenser"
LABEL version="1.0.0"

# Dépendances système pour LightGBM + wget
RUN apt-get update && apt-get install -y \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# User non-root requis par Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app/ ./app/

# Téléchargement des artefacts depuis GitHub Release
RUN mkdir -p model && \
    wget -q -O model/lgbm_best.joblib \
      https://github.com/Marrackech/scoring-mlops-deployment/releases/download/v0.1.0/lgbm_best.joblib && \
    wget -q -O model/scaler.joblib \
      https://github.com/Marrackech/scoring-mlops-deployment/releases/download/v0.1.0/scaler.joblib && \
    wget -q -O model/features.json \
      https://github.com/Marrackech/scoring-mlops-deployment/releases/download/v0.1.0/features.json

RUN mkdir -p logs

# Port 7860 requis par Hugging Face Spaces
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
