"""Chargement *brut* des deux jeux de données (aucun nettoyage ici).

Le nettoyage et la fusion vivent dans des modules dédiés : ces fonctions
renvoient les données telles que livrées, avec seulement les types corrigés,
pour que l'EDA observe les vrais défauts des sources.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agritech_crop_reco.config import load_settings

# Noms de fichiers relatifs à data/raw/
CROP_YIELD_FILE = Path("agriculture_crop_yield/crop_yield.csv")
FAO_DIR = Path("crop_yield_prediction")

CROP_YIELD_CATEGORICAL = ["Region", "Soil_Type", "Crop", "Weather_Condition"]


def _raw_dir(raw_dir: Path | None) -> Path:
    return raw_dir if raw_dir is not None else load_settings().raw_dir


def load_crop_yield(raw_dir: Path | None = None) -> pd.DataFrame:
    """Agriculture Crop Yield (Kaggle) : 1 ligne = 1 parcelle simulée.

    Les colonnes texte sont typées ``category`` (mémoire divisée par ~5 sur
    1 million de lignes) ; Fertilizer_Used / Irrigation_Used restent booléens.
    """
    df = pd.read_csv(_raw_dir(raw_dir) / CROP_YIELD_FILE)
    return df.astype({col: "category" for col in CROP_YIELD_CATEGORICAL})


def load_fao_yield_df(raw_dir: Path | None = None) -> pd.DataFrame:
    """Crop Yield Prediction Dataset, fichier pré-fusionné ``yield_df.csv``.

    La première colonne du CSV est un ancien index pandas sans nom : on la retire.
    """
    df = pd.read_csv(_raw_dir(raw_dir) / FAO_DIR / "yield_df.csv")
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def load_fao_sources(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Les 4 fichiers sources FAO / Banque mondiale qui ont servi à construire yield_df.

    Noms de colonnes nettoyés des espaces parasites (ex. `` Area`` dans rainfall.csv).
    """
    base = _raw_dir(raw_dir) / FAO_DIR
    sources = {}
    for name in ("yield", "pesticides", "rainfall", "temp"):
        df = pd.read_csv(base / f"{name}.csv")
        df.columns = df.columns.str.strip()
        sources[name] = df
    return sources
