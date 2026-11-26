"""Tests du front : client HTTP (API simulée) et parcours de l'interface Streamlit."""

from __future__ import annotations

from pathlib import Path

import api_client
import pytest
import requests
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "app.py")
CONDITIONS = api_client.conditions_payload(18.5, 900, 12_000, None, None)
HEALTH = {"status": "ok", "model_version": "test@abc", "n_crops": 10}
METADATA = {
    "crops": list(api_client.CROP_LABELS),
    "training_ranges": {
        "avg_temp_c": [1.3, 30.42],
        "rainfall_mm": [51.0, 3240.0],
        "pesticides_tonnes": [0.04, 367778.0],
    },
    "pesticide_levels": [
        {"label": lbl, "quantile": q, "pesticides_tonnes": t}
        for lbl, q, t in [
            ("Très faible", 0.1, 40.0),
            ("Faible", 0.3, 502.6),
            ("Moyen", 0.5, 2539.2),
            ("Élevé", 0.7, 9163.5),
            ("Très élevé", 0.9, 50891.9),
        ]
    ],
    "countries": ["France", "Kenya"],
    "price_info": {
        "period": [2018, 2023],
        "world": {"Potatoes": {"price_usd_t": 362.0}, "Wheat": {"price_usd_t": 260.0}},
    },
    "model_version": "test@abc",
}
PREDICTION = {
    "crop": "Wheat",
    "yield_t_ha": 2.5,
    "yield_base_t_ha": 2.5,
    "practice_factor": 1.0,
    "warnings": ["pluie hors de la plage d'entraînement"],
    "model_version": "test@abc",
}
PRICED = {"price_source": "mondial"}
RANKING = {
    "ranking": [
        {
            **PREDICTION,
            **PRICED,
            "crop": "Potatoes",
            "yield_t_ha": 21.0,
            "rank": 1,
            "price_usd_t": 362.0,
            "revenue_usd_ha": 7602.0,
        },
        {
            **PREDICTION,
            **PRICED,
            "crop": "Wheat",
            "rank": 2,
            "price_usd_t": 260.0,
            "revenue_usd_ha": 650.0,
        },
    ],
    "rank_by": "revenue",
    "country": None,
    "warnings": [],
    "model_version": "test@abc",
    "note": "Chiffre d'affaires estimé, hors coûts.",
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._payload is None:
            raise ValueError("pas de JSON")
        return self._payload


def fake_request(response):
    def _request(method, url, json=None, timeout=None):
        _request.calls.append((method, url, json))
        if isinstance(response, Exception):
            raise response
        return response

    _request.calls = []
    return _request


def test_predict_sends_crop_and_conditions(monkeypatch):
    fake = fake_request(FakeResponse(payload=PREDICTION))
    monkeypatch.setattr(requests, "request", fake)
    monkeypatch.setenv("API_URL", "http://api:8000/")
    assert api_client.predict("Wheat", CONDITIONS)["yield_t_ha"] == 2.5
    method, url, body = fake.calls[0]
    assert (method, url) == ("POST", "http://api:8000/predict")
    assert body["crop"] == "Wheat" and body["fertilizer_used"] is None


def test_unreachable_api_raises_readable_error(monkeypatch):
    monkeypatch.setattr(requests, "request", fake_request(requests.ConnectionError("boom")))
    with pytest.raises(api_client.ApiError, match="injoignable"):
        api_client.health()


def test_validation_error_is_explained(monkeypatch):
    detail = {"detail": [{"loc": ["body", "rainfall_mm"], "msg": "doit être >= 0"}]}
    monkeypatch.setattr(requests, "request", fake_request(FakeResponse(422, detail)))
    with pytest.raises(api_client.ApiError, match="rainfall_mm : doit être >= 0"):
        api_client.recommend(CONDITIONS)


def test_server_error_is_reported(monkeypatch):
    monkeypatch.setattr(requests, "request", fake_request(FakeResponse(500)))
    with pytest.raises(api_client.ApiError, match="HTTP 500"):
        api_client.health()


def test_crop_labels_match_api_crops():
    from api.schemas import Crop

    assert set(api_client.CROP_LABELS) == {c.value for c in Crop}


def run_app(monkeypatch, **api):
    monkeypatch.setattr(api_client, "health", api.get("health", lambda: HEALTH))
    monkeypatch.setattr(api_client, "metadata", api.get("metadata", lambda: METADATA))
    monkeypatch.setattr(api_client, "predict", api.get("predict", lambda c, p: PREDICTION))
    monkeypatch.setattr(api_client, "recommend", api.get("recommend", lambda p: RANKING))
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_predict_flow(monkeypatch):
    at = run_app(monkeypatch)
    assert not at.exception
    at.button[0].click().run()
    assert at.metric[0].value == "2,50 t/ha"
    assert "plage d'entraînement" in at.warning[0].value


def test_app_recommend_flow_shows_chart_and_table(monkeypatch):
    at = run_app(monkeypatch)
    at.button[1].click().run()
    assert not at.exception
    assert "Pommes de terre" in at.success[0].value
    assert len(at.get("plotly_chart")) == 1
    table = at.dataframe[0].value
    assert list(table["Culture"]) == ["Pommes de terre", "Blé"]
    assert list(table["Rang"]) == [1, 2]


def test_inputs_are_sliders_bounded_by_training_ranges(monkeypatch):
    at = run_app(monkeypatch)
    temp, rain = at.slider
    assert (temp.min, temp.max) == (1.0, 30.5)  # [1.3 ; 30.42] arrondi au pas de 0,5
    assert rain.min == 50.0 and rain.max == 3240.0
    assert at.select_slider[0].value == "Moyen"


def test_intensity_level_is_sent_as_tonnes(monkeypatch):
    sent = {}

    def fake_predict(crop, conditions):
        sent.update(conditions)
        return PREDICTION

    at = run_app(monkeypatch, predict=fake_predict)
    at.select_slider[0].set_value("Très élevé").run()
    at.button[0].click().run()
    assert sent["pesticides_tonnes"] == 50891.9


def test_result_flagged_as_stale_when_inputs_change(monkeypatch):
    at = run_app(monkeypatch)
    at.button[0].click().run()
    assert at.metric
    at.slider[0].set_value(25.0).run()
    assert not at.metric
    assert "relancez" in at.info[0].value


def test_app_shows_error_when_api_is_down(monkeypatch):
    def down():
        raise api_client.ApiError("API injoignable")

    at = run_app(monkeypatch, health=down)
    assert at.error[0].value == "API injoignable"
    assert not at.button


def test_foreign_api_is_reported_clearly(monkeypatch):
    # Cas réel : un autre service répondait à l'URL configurée (sous-domaine Render déjà pris)
    foreign = {"status": "ok", "model_loaded": True}
    monkeypatch.setattr(requests, "request", fake_request(FakeResponse(payload=foreign)))
    with pytest.raises(api_client.ApiError, match="ne correspond pas à l'API Agritech"):
        api_client.health()


def test_metadata_with_missing_fields_is_reported(monkeypatch):
    monkeypatch.setattr(requests, "request", fake_request(FakeResponse(payload={"crops": []})))
    with pytest.raises(api_client.ApiError, match="pesticide_levels"):
        api_client.metadata()


def test_recommend_shows_revenue_first(monkeypatch):
    at = run_app(monkeypatch)
    at.button[1].click().run()
    assert "7" in at.success[0].value and "$/ha" in at.success[0].value
    table = at.dataframe[0].value
    assert list(table["Origine du prix"]) == ["médiane mondiale", "médiane mondiale"]


def test_country_and_rank_choice_are_sent_to_api(monkeypatch):
    sent = {}

    def fake_recommend(payload):
        sent.update(payload)
        return RANKING

    at = run_app(monkeypatch, recommend=fake_recommend)
    country = next(sb for sb in at.selectbox if sb.label == "Pays de l'exploitation")
    country.set_value("France").run()
    at.segmented_control[0].set_value("Rendement").run()
    at.button[1].click().run()
    assert sent["country"] == "France" and sent["rank_by"] == "yield"


def test_custom_price_is_sent_only_when_filled(monkeypatch):
    sent = {}

    def fake_predict(crop, payload):
        sent.update(payload)
        return {
            **PREDICTION,
            "price_usd_t": 500.0,
            "price_source": "saisi",
            "revenue_usd_ha": 1250.0,
        }

    at = run_app(monkeypatch, predict=fake_predict)
    at.button[0].click().run()
    assert "prices_usd_t" not in sent  # rien de saisi
    at.number_input(key="price_Wheat").set_value(500.0).run()
    at.button[0].click().run()
    assert sent["prices_usd_t"] == {"Wheat": 500.0}
    assert any("$/ha" in m.value for m in at.metric)
