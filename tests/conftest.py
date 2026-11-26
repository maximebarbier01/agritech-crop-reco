"""Configuration commune des tests."""

import pytest

from agritech_crop_reco import config


@pytest.fixture(autouse=True)
def _ignore_local_env_file(monkeypatch):
    """Les tests ne lisent jamais le fichier .env du développeur.

    Sinon, une URL Neon (MLFLOW_TRACKING_URI, PREDICTION_DB_URL) présente dans .env serait
    chargée au démarrage de l'API et les tests écriraient dans la base de production.
    """
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
