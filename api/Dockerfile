# Image de l'API de prédiction (FastAPI + modèle scikit-learn).
# Construction depuis la RACINE du dépôt (le contexte doit contenir src/ et models/) :
#   docker build -f api/Dockerfile -t agritech-crop-api .
#   docker run --rm -p 8000:8000 agritech-crop-api      -> http://localhost:8000/docs

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    MODELS_DIR=/app/models

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas.
# Versions exportées de poetry.lock = celles de l'entraînement (indispensable pour
# recharger le .joblib : un modèle scikit-learn n'est fiable qu'avec la même version).
COPY api/requirements.txt api/requirements.txt
RUN pip install -r api/requirements.txt

# Code et modèle (le dataset et les notebooks ne sont PAS embarqués : cf. .dockerignore)
COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Pas de droits root à l'exécution
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Docker (et la CI) savent si l'API répond vraiment
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# PORT est fourni par certains hébergeurs (Render, Railway…) ; 8000 par défaut
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
