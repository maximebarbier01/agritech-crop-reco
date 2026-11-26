"""Tests de l'API FastAPI (TestClient : le modèle réel est chargé au démarrage)."""

import pytest
from fastapi.testclient import TestClient

from agritech_crop_reco.serving import YieldService
from api.main import app
from api.schemas import Crop

PARCEL = {"avg_temp_c": 18.5, "rainfall_mm": 900, "pesticides_tonnes": 12_000}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # le "with" déclenche le chargement du modèle (lifespan)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["n_crops"] == 10


def test_crop_enum_matches_model():
    assert {c.value for c in Crop} == set(YieldService.load().crops)


def test_predict_ok(client):
    r = client.post("/predict", json={**PARCEL, "crop": "Wheat", "fertilizer_used": True})
    assert r.status_code == 200
    body = r.json()
    assert body["crop"] == "Wheat"
    assert body["yield_t_ha"] > body["yield_base_t_ha"] > 0
    assert body["warnings"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {**PARCEL, "crop": "Banana"},  # culture inconnue
        {**PARCEL, "crop": "Wheat", "avg_temp_c": 80},  # température impossible
        {**PARCEL, "crop": "Wheat", "rainfall_mm": -5},  # pluie négative
        {"crop": "Wheat", "avg_temp_c": 18.5},  # champs manquants
    ],
)
def test_predict_invalid_input_returns_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_out_of_training_range_warns(client):
    r = client.post("/predict", json={**PARCEL, "crop": "Wheat", "avg_temp_c": 35})
    assert r.status_code == 200
    assert any("avg_temp_c" in w for w in r.json()["warnings"])


def test_recommend_returns_sorted_ranking(client):
    r = client.post("/recommend", json=PARCEL)
    assert r.status_code == 200
    ranking = r.json()["ranking"]
    assert len(ranking) == 10
    revenues = [c["revenue_usd_ha"] for c in ranking]  # défaut : chiffre d'affaires
    assert revenues == sorted(revenues, reverse=True)
    assert ranking[0]["rank"] == 1 and r.json()["rank_by"] == "revenue"


def test_recommend_by_yield_on_request(client):
    r = client.post("/recommend", json={**PARCEL, "rank_by": "yield"})
    yields = [c["yield_t_ha"] for c in r.json()["ranking"]]
    assert yields == sorted(yields, reverse=True)


def test_country_prices_and_sources(client):
    countries = client.get("/metadata").json()["countries"]
    assert "France" in countries
    ranking = client.post("/recommend", json={**PARCEL, "country": "France"}).json()["ranking"]
    sources = {c["crop"]: c["price_source"] for c in ranking}
    assert sources["Wheat"] == "national"  # prix français connu
    assert sources["Yams"] == "mondial"  # pas de prix français pour l'igname


def test_custom_price_overrides_and_is_validated(client):
    body = {**PARCEL, "crop": "Wheat", "prices_usd_t": {"Wheat": 500}}
    result = client.post("/predict", json=body).json()
    assert result["price_source"] == "saisi"
    assert result["revenue_usd_ha"] == pytest.approx(result["yield_t_ha"] * 500)
    bad = {**PARCEL, "crop": "Wheat", "prices_usd_t": {"Wheat": -1}}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_country_returns_422(client):
    r = client.post("/recommend", json={**PARCEL, "country": "Atlantide"})
    assert r.status_code == 422 and "Atlantide" in r.json()["detail"]


def test_recommend_ignores_crop_field_if_sent(client):
    # /recommend ne prend que le contexte : un champ "crop" en trop est ignoré
    r = client.post("/recommend", json={**PARCEL, "crop": "Wheat"})
    assert r.status_code == 200 and len(r.json()["ranking"]) == 10


def test_metadata_exposes_ranges_and_levels(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert len(body["crops"]) == 10
    low, high = body["training_ranges"]["avg_temp_c"]
    assert low < high
    levels = [lvl["pesticides_tonnes"] for lvl in body["pesticide_levels"]]
    assert len(levels) == 5 and levels == sorted(levels)  # du plus faible au plus élevé
