"""API Agritech : prédiction de rendement et recommandation de cultures.

Lancement local : ``uvicorn api.main:app --reload`` puis http://127.0.0.1:8000/docs
Le dossier du modèle peut être changé par la variable d'environnement ``MODELS_DIR``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request

from agritech_crop_reco import __version__
from agritech_crop_reco.serving import Conditions, YieldService
from api.schemas import (
    HealthResponse,
    ParcelConditions,
    PredictRequest,
    PredictResponse,
    RecommendResponse,
)

logger = logging.getLogger("agritech.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement unique au démarrage ; si le modèle est absent, l'API ne démarre pas
    # (mieux vaut un conteneur en erreur qu'un service qui répond n'importe quoi).
    models_dir = os.getenv("MODELS_DIR")
    app.state.service = YieldService.load(Path(models_dir) if models_dir else None)
    logger.info("Modèle chargé : %s", app.state.service.version)
    yield


app = FastAPI(
    title="Agritech Answers – API rendement & recommandation",
    version=__version__,
    description=(
        "Prédit le rendement (t/ha) d'une culture selon les conditions de la parcelle, "
        "et classe les 10 cultures connues pour ces conditions."
    ),
    lifespan=lifespan,
)


def _service(request: Request) -> YieldService:
    return request.app.state.service


def _conditions(payload: ParcelConditions) -> Conditions:
    fields = ParcelConditions.model_fields
    return Conditions(**{k: v for k, v in payload.model_dump().items() if k in fields})


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(status="ok", model_version=service.version, n_crops=len(service.crops))


@app.post("/predict", response_model=PredictResponse, tags=["prédiction"])
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Rendement prédit pour **une** culture et les conditions de la parcelle."""
    service = _service(request)
    conditions = _conditions(payload)
    result = service.predict(payload.crop.value, conditions)
    return PredictResponse(
        **result, warnings=service.warnings(conditions), model_version=service.version
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["recommandation"])
def recommend(payload: ParcelConditions, request: Request) -> RecommendResponse:
    """Toutes les cultures, classées par rendement prédit décroissant pour ces conditions."""
    service = _service(request)
    conditions = _conditions(payload)
    logger.info("recommend %s", asdict(conditions))
    return RecommendResponse(
        ranking=service.recommend(conditions),
        warnings=service.warnings(conditions),
        model_version=service.version,
    )
