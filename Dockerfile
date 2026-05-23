FROM python:3.10-slim

LABEL maintainer="UserMarrakech"
LABEL description="API Scoring Crédit — Prêt à Dépenser"
LABEL version="1.0.0"

# Dépendances système pour LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# User non-root requis par Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app/ ./app/
COPY --chown=user model/ ./model/

RUN mkdir -p logs

# Port 7860 requis par Hugging Face Spaces
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
