# Image de l'interface Streamlit (aucun modèle : elle appelle l'API).
# Construction depuis la RACINE du dépôt :
#   docker build -f frontend/Dockerfile -t agritech-crop-front .
#   docker run --rm -p 8501:8501 -e API_URL=http://host.docker.internal:8000 agritech-crop-front
# -> http://localhost:8501   (API_URL : adresse de l'API vue DEPUIS le conteneur)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_URL=http://api:8000 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dépendances d'abord : couche en cache tant que requirements.txt ne change pas
# (versions exportées de poetry.lock, groupe `front`).
COPY frontend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY frontend/ .

# Pas de droits root à l'exécution
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

# PORT est fourni par certains hébergeurs ; 8501 par défaut
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
