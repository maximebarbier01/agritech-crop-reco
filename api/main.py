"""API Agritech : prédiction de rendement et recommandation de cultures.

Lancement local : ``uvicorn api.main:app --reload`` puis http://127.0.0.1:8000/docs
Variables d'environnement :
- ``MODELS_DIR`` : dossier du modèle (par défaut ``models/``) ;
- ``PREDICTION_DB_URL`` : base où enregistrer chaque prédiction (désactivé si absente).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from agritech_crop_reco import __version__
from agritech_crop_reco.serving import Conditions, YieldService
from api.prediction_store import PredictionStore
from api.schemas import (
    HealthResponse,
    MetadataResponse,
    MonitoringSummary,
    ParcelConditions,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
)


def _configure_logging() -> logging.Logger:
    """Journal de l'API sur la sortie standard (visible dans les logs Docker / Render).

    Sans cette configuration, Python n'affiche que les messages WARNING et plus graves :
    les messages INFO de l'API étaient perdus.
    """
    log = logging.getLogger("agritech.api")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


logger = _configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement unique au démarrage ; si le modèle est absent, l'API ne démarre pas
    # (mieux vaut un conteneur en erreur qu'un service qui répond n'importe quoi).
    models_dir = os.getenv("MODELS_DIR")
    app.state.service = YieldService.load(Path(models_dir) if models_dir else None)
    logger.info("Modèle chargé : %s", app.state.service.version)
    # Enregistrement des prédictions : optionnel, ne bloque jamais le démarrage
    app.state.store = PredictionStore.from_env()
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


def _custom_prices(payload) -> dict[str, float] | None:
    """Prix saisis : clés de l'énumération Crop -> noms de cultures du modèle."""
    if not payload.prices_usd_t:
        return None
    return {crop.value: price for crop, price in payload.prices_usd_t.items()}


def _unprocessable(exc: ValueError) -> HTTPException:
    # Pays inconnu, etc. : erreur de saisie (422), comme les erreurs de validation Pydantic
    return HTTPException(422, str(exc))


def _track(
    request: Request,
    background: BackgroundTasks,
    endpoint: str,
    conditions: Conditions,
    started: float,
    **result,
) -> None:
    """Journalise l'appel (une ligne JSON) et programme son enregistrement après la réponse."""
    event = {
        "endpoint": endpoint,
        **asdict(conditions),
        **result,
        "n_warnings": len(result.get("warnings") or []),
        "model_version": _service(request).version,
        "api_version": __version__,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    logger.info("prediction %s", json.dumps(event, ensure_ascii=False))
    store: PredictionStore | None = request.app.state.store
    if store is not None:
        background.add_task(store.record, event)


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    service = _service(request)
    return HealthResponse(status="ok", model_version=service.version, n_crops=len(service.crops))


@app.get("/metadata", response_model=MetadataResponse, tags=["service"])
def metadata(request: Request) -> MetadataResponse:
    """Cultures, plages d'entraînement et niveaux d'intensification (pour l'interface)."""
    service = _service(request)
    price_info = None
    if service.prices:
        price_info = {key: service.prices[key] for key in ("source", "period", "unit", "world")}
    return MetadataResponse(
        crops=service.crops,
        training_ranges=service.training_ranges,
        pesticide_levels=service.pesticide_levels,
        countries=service.countries,
        price_info=price_info,
        model_version=service.version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["prédiction"])
def predict(
    payload: PredictRequest, request: Request, background: BackgroundTasks
) -> PredictResponse:
    """Rendement prédit pour **une** culture et les conditions de la parcelle."""
    started = time.perf_counter()
    service = _service(request)
    conditions = _conditions(payload)
    try:
        result = service.predict(
            payload.crop.value, conditions, payload.country, _custom_prices(payload)
        )
    except ValueError as exc:
        raise _unprocessable(exc) from exc
    warnings = service.warnings(conditions)
    _track(
        request,
        background,
        "predict",
        conditions,
        started,
        crop=result["crop"],
        top_crop=result["crop"],
        yield_t_ha=result["yield_t_ha"],
        revenue_usd_ha=result["revenue_usd_ha"],
        country=payload.country,
        warnings=warnings,
    )
    return PredictResponse(**result, warnings=warnings, model_version=service.version)


@app.post("/recommend", response_model=RecommendResponse, tags=["recommandation"])
def recommend(
    payload: RecommendRequest, request: Request, background: BackgroundTasks
) -> RecommendResponse:
    """Toutes les cultures, classées par chiffre d'affaires estimé (défaut) ou par rendement."""
    started = time.perf_counter()
    service = _service(request)
    conditions = _conditions(payload)
    try:
        ranking = service.recommend(
            conditions, payload.country, _custom_prices(payload), payload.rank_by
        )
    except ValueError as exc:
        raise _unprocessable(exc) from exc
    warnings = service.warnings(conditions)
    _track(
        request,
        background,
        "recommend",
        conditions,
        started,
        top_crop=ranking[0]["crop"],
        yield_t_ha=ranking[0]["yield_t_ha"],
        revenue_usd_ha=ranking[0]["revenue_usd_ha"],
        country=payload.country,
        rank_by=payload.rank_by,
        ranking=[[r["crop"], round(r["yield_t_ha"], 3)] for r in ranking],
        warnings=warnings,
    )
    return RecommendResponse(
        ranking=ranking,
        rank_by=payload.rank_by,
        country=payload.country,
        warnings=warnings,
        model_version=service.version,
    )


@app.get("/monitoring/summary", response_model=MonitoringSummary, tags=["service"])
def monitoring_summary(request: Request) -> MonitoringSummary:
    """Synthèse des appels enregistrés : volume, temps de réponse, cultures, hors domaine.

    Uniquement des agrégats (aucune ligne brute). 503 si l'enregistrement est désactivé.
    """
    store: PredictionStore | None = request.app.state.store
    if store is None:
        raise HTTPException(503, "Enregistrement des prédictions désactivé (PREDICTION_DB_URL)")
    return MonitoringSummary(**store.summary())
