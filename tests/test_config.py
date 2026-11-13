"""Tests de la configuration : bascule SQLite local <-> URI distante (Neon)."""

import pytest

from agritech_crop_reco import config


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    # Isole les tests d'un éventuel .env local contenant l'URL Neon.
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)


def test_default_tracking_uri_is_local_sqlite(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    settings = config.load_settings()
    assert settings.mlflow_tracking_uri.startswith("sqlite:///")
    assert settings.mlflow_tracking_uri.endswith("mlflow.db")


def test_empty_env_var_falls_back_to_sqlite(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
    assert config.load_settings().mlflow_tracking_uri.startswith("sqlite:///")


def test_env_var_overrides_tracking_uri(monkeypatch):
    remote = "postgresql://user:pwd@host.neon.tech/mlflow?sslmode=require"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", remote)
    assert config.load_settings().mlflow_tracking_uri == remote


def test_paths_and_seed():
    settings = config.load_settings()
    assert settings.seed == 42
    assert settings.raw_dir == config.PROJECT_ROOT / "data" / "raw"
    assert settings.models_dir.name == "models"
