"""Configuration centrale : chemins, seed et URI de suivi MLflow.

Les valeurs viennent de ``configs/default.toml``. Seule ``MLFLOW_TRACKING_URI``
peut être surchargée par l'environnement (ou le fichier ``.env``) :
- absente ou vide -> SQLite local (``mlflow.db``) ;
- définie -> par exemple la base PostgreSQL Neon utilisée par la CI.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"


@dataclass(frozen=True)
class Settings:
    seed: int
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Lit la configuration TOML et applique la surcharge d'environnement."""
    # override=False : une variable déjà exportée (ex. secret CI) prime sur .env.
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    paths = cfg["paths"]
    mlflow_cfg = cfg["mlflow"]
    local_uri = f"sqlite:///{PROJECT_ROOT / mlflow_cfg['local_db']}"

    return Settings(
        seed=cfg["project"]["seed"],
        raw_dir=PROJECT_ROOT / paths["raw"],
        processed_dir=PROJECT_ROOT / paths["processed"],
        models_dir=PROJECT_ROOT / paths["models"],
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI") or local_uri,
        mlflow_experiment=mlflow_cfg["experiment"],
    )
