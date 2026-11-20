"""Client HTTP de l'API Agritech. Aucune logique de ML ici : le front ne fait qu'appeler l'API."""

from __future__ import annotations

import os

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
# L'API hébergée (offre gratuite Render) se met en veille : le réveil prend 30 à 60 s.
TIMEOUT_S = 90

# Cultures connues du modèle (clé = nom envoyé à l'API) -> libellé affiché.
CROP_LABELS = {
    "Cassava": "Manioc",
    "Maize": "Maïs",
    "Plantains and others": "Plantains et autres",
    "Potatoes": "Pommes de terre",
    "Rice": "Riz",
    "Sorghum": "Sorgho",
    "Soybean": "Soja",
    "Sweet potatoes": "Patates douces",
    "Wheat": "Blé",
    "Yams": "Ignames",
}


class ApiError(Exception):
    """Erreur lisible par l'utilisateur (API injoignable, entrée refusée, erreur serveur)."""


def api_url() -> str:
    return os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")


def _describe_validation_error(response: requests.Response) -> str:
    try:
        details = response.json()["detail"]
        return " ; ".join(
            f"{'.'.join(str(p) for p in d['loc'] if p != 'body')} : {d['msg']}" for d in details
        )
    except (ValueError, KeyError, TypeError):
        return response.text[:200]


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{api_url()}{path}"
    try:
        response = requests.request(method, url, json=payload, timeout=TIMEOUT_S)
    except requests.RequestException as exc:
        raise ApiError(f"API injoignable ({url}). Est-elle démarrée ?") from exc
    if response.status_code == 422:
        raise ApiError(f"Valeurs refusées par l'API : {_describe_validation_error(response)}")
    if not response.ok:
        raise ApiError(f"Erreur de l'API (HTTP {response.status_code}).")
    return response.json()


def conditions_payload(
    avg_temp_c: float,
    rainfall_mm: float,
    pesticides_tonnes: float,
    fertilizer_used: bool | None,
    irrigation_used: bool | None,
) -> dict:
    return {
        "avg_temp_c": avg_temp_c,
        "rainfall_mm": rainfall_mm,
        "pesticides_tonnes": pesticides_tonnes,
        "fertilizer_used": fertilizer_used,
        "irrigation_used": irrigation_used,
    }


HEALTH_KEYS = {"status", "model_version", "n_crops"}
METADATA_KEYS = {"crops", "training_ranges", "pesticide_levels"}


def _check_keys(payload: dict, expected: set[str], path: str) -> dict:
    """Une autre API peut répondre à la même adresse (ex. sous-domaine déjà pris sur Render) :
    on le signale clairement plutôt que de planter plus loin sur un champ manquant."""
    missing = expected - set(payload) if isinstance(payload, dict) else expected
    if missing:
        raise ApiError(
            f"L'adresse configurée ({api_url()}) ne correspond pas à l'API Agritech : "
            f"réponse inattendue sur {path} (champs manquants : {', '.join(sorted(missing))}). "
            "Vérifiez la variable API_URL."
        )
    return payload


def health() -> dict:
    return _check_keys(_request("GET", "/health"), HEALTH_KEYS, "/health")


def metadata() -> dict:
    """Cultures, plages d'entraînement et niveaux d'intensification (pour borner les saisies)."""
    return _check_keys(_request("GET", "/metadata"), METADATA_KEYS, "/metadata")


def predict(crop: str, conditions: dict) -> dict:
    return _request("POST", "/predict", {"crop": crop, **conditions})


def recommend(conditions: dict) -> dict:
    return _request("POST", "/recommend", conditions)
