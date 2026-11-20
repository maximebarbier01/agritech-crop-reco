"""Tests de l'enregistrement des prédictions (base SQLite temporaire à la place de Neon)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.main import app
from api.prediction_store import ENV_VAR, PredictionStore, predictions

PARCEL = {"avg_temp_c": 18.5, "rainfall_mm": 900, "pesticides_tonnes": 12_000}


def _event(**overrides):
    event = {
        "endpoint": "predict",
        "crop": "Wheat",
        "top_crop": "Wheat",
        "yield_t_ha": 2.6,
        **PARCEL,
        "fertilizer_used": True,
        "irrigation_used": None,
        "n_warnings": 0,
        "warnings": [],
        "model_version": "test@abc",
        "api_version": "1.2.0",
        "latency_ms": 12.5,
    }
    return {**event, **overrides}


@pytest.fixture
def store(tmp_path):
    return PredictionStore(f"sqlite:///{tmp_path / 'predictions.db'}")


def test_record_and_summary(store):
    assert store.record(_event())
    assert store.record(_event(latency_ms=30.0, n_warnings=1, warnings=["avg_temp_c hors"]))
    assert store.record(
        _event(endpoint="recommend", crop=None, top_crop="Potatoes", ranking=[["Potatoes", 18.4]])
    )
    s = store.summary()
    assert s["total_calls"] == 3
    assert s["calls_by_endpoint"] == {"predict": 2, "recommend": 1}
    assert s["share_with_warnings"] == pytest.approx(0.333)  # arrondi à 3 décimales
    assert s["top_requested_crops"] == {"Wheat": 2}
    assert s["top_recommended_crops"] == {"Potatoes": 1}
    assert s["latency_ms_median"] == pytest.approx(12.5)


def test_empty_summary(store):
    assert store.summary() == {"total_calls": 0, "window": 0}


def test_record_never_raises_when_database_fails(store):
    store.engine.dispose()
    with store.engine.begin() as conn:  # on supprime la table : toute insertion échouera
        predictions.drop(conn)
    assert store.record(_event()) is False  # erreur journalisée, pas d'exception


def test_from_env_disabled_without_url(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert PredictionStore.from_env() is None


def test_from_env_disabled_when_database_unreachable(monkeypatch):
    # Hôte inexistant : l'API doit démarrer quand même, sans enregistrement
    monkeypatch.setenv(ENV_VAR, "postgresql://u:p@127.0.0.1:1/absente?connect_timeout=1")
    assert PredictionStore.from_env() is None


# ---------------------------------------------------------------- intégration avec l'API


def test_api_records_each_prediction(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'api.db'}"
    monkeypatch.setenv(ENV_VAR, url)
    with TestClient(app) as client:
        assert client.post("/predict", json={**PARCEL, "crop": "Wheat"}).status_code == 200
        assert client.post("/recommend", json={**PARCEL, "avg_temp_c": 35}).status_code == 200
        summary = client.get("/monitoring/summary").json()
    assert summary["total_calls"] == 2
    assert summary["calls_by_endpoint"] == {"predict": 1, "recommend": 1}
    assert summary["share_with_warnings"] == pytest.approx(0.5)  # 35 °C : hors domaine

    rows = PredictionStore(url).engine.connect().execute(select(predictions)).mappings().all()
    predict_row = next(r for r in rows if r["endpoint"] == "predict")
    assert predict_row["crop"] == "Wheat" and predict_row["yield_t_ha"] > 0
    recommend_row = next(r for r in rows if r["endpoint"] == "recommend")
    assert len(recommend_row["ranking"]) == 10 and recommend_row["n_warnings"] == 1


def test_monitoring_returns_503_when_disabled(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with TestClient(app) as client:
        assert client.post("/predict", json={**PARCEL, "crop": "Wheat"}).status_code == 200
        assert client.get("/monitoring/summary").status_code == 503
