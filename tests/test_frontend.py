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
PREDICTION = {
    "crop": "Wheat",
    "yield_t_ha": 2.5,
    "yield_base_t_ha": 2.5,
    "practice_factor": 1.0,
    "warnings": ["pluie hors de la plage d'entraînement"],
    "model_version": "test@abc",
}
RANKING = {
    "ranking": [
        {**PREDICTION, "crop": "Potatoes", "yield_t_ha": 21.0, "rank": 1},
        {**PREDICTION, "crop": "Wheat", "rank": 2},
    ],
    "warnings": [],
    "model_version": "test@abc",
    "note": "Classement par rendement physique.",
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


def test_crop_labels_cover_the_ten_api_crops():
    assert len(api_client.CROP_LABELS) == 10


def run_app(monkeypatch, **api):
    monkeypatch.setattr(api_client, "health", api.get("health", lambda: HEALTH))
    monkeypatch.setattr(api_client, "predict", api.get("predict", lambda c, p: PREDICTION))
    monkeypatch.setattr(api_client, "recommend", api.get("recommend", lambda p: RANKING))
    return AppTest.from_file(APP, default_timeout=30).run()


def test_app_predict_flow(monkeypatch):
    at = run_app(monkeypatch)
    assert not at.exception
    at.button[0].click().run()
    assert at.metric[0].value == "2.50 t/ha"
    assert "plage d'entraînement" in at.warning[0].value


def test_app_recommend_flow(monkeypatch):
    at = run_app(monkeypatch)
    at.button[1].click().run()
    assert "Pommes de terre" in at.success[0].value


def test_app_shows_error_when_api_is_down(monkeypatch):
    def down():
        raise api_client.ApiError("API injoignable")

    at = run_app(monkeypatch, health=down)
    assert at.error[0].value == "API injoignable"
    assert not at.button
