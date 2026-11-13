"""Nettoyage des deux jeux de données (décisions documentées dans notebooks/01_eda.ipynb)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Agriculture Crop Yield (Kaggle) -------------------------------------------------

KAGGLE_TARGET = "Yield_tons_per_hectare"
PRACTICES = {"Fertilizer_Used": "fertilizer_gain", "Irrigation_Used": "irrigation_gain"}
GLOBAL = "__global__"


def clean_crop_yield(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les rendements <= 0 t/ha, physiquement impossibles (artefact du simulateur)."""
    return df[df[KAGGLE_TARGET] > 0].reset_index(drop=True)


def _practice_gains(df: pd.DataFrame) -> dict[str, float]:
    """Régression rendement = b0 + b_engrais·E + b_irrigation·I, gains = b / b0.

    b0 est le rendement attendu **sans aucune pratique** : un gain de 0.45 signifie
    +45 % par rapport à ce rendement. Les effets s'additionnent (vérifié dans
    02_fusion : le modèle additif reproduit la combinaison des deux pratiques).
    """
    practices = list(PRACTICES)
    X = np.column_stack([np.ones(len(df)), df[practices].to_numpy(dtype=float)])
    coefs, *_ = np.linalg.lstsq(X, df[KAGGLE_TARGET].to_numpy(), rcond=None)
    b0, effects = coefs[0], coefs[1:]
    return {PRACTICES[p]: effect / b0 for p, effect in zip(practices, effects, strict=True)}


def compute_practice_gains(df: pd.DataFrame) -> pd.DataFrame:
    """Gain relatif de rendement dû à chaque pratique, par culture et global.

    Retour : une ligne par culture + une ligne ``__global__`` (toutes cultures).
    """
    groups = {str(crop): grp for crop, grp in df.groupby("Crop", observed=True)}
    groups[GLOBAL] = df
    return pd.DataFrame([{"crop": name, **_practice_gains(grp)} for name, grp in groups.items()])


# --- Crop Yield Prediction Dataset (FAO) ---------------------------------------------

FAO_KEYS = ["Area", "Item", "Year"]
CROP_NAME_MAP = {"Rice, paddy": "Rice", "Soybeans": "Soybean"}
HG_PER_TONNE = 10_000  # 1 t = 10 000 hg

FAO_RENAME = {
    "Area": "country",
    "Item": "crop",
    "Year": "year",
    "average_rain_fall_mm_per_year": "rainfall_mm",
    "pesticides_tonnes": "pesticides_tonnes",
    "avg_temp": "avg_temp_c",
}


def clean_fao(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par pays × culture × année, rendement en t/ha, noms harmonisés.

    - Doublons : yield_df.csv répète une observation pour chaque station de
      température du pays ; seules les températures diffèrent -> moyenne.
    - Unité : hg/ha -> t/ha.
    - Noms de cultures alignés sur le jeu Kaggle (Rice, Soybean).
    """
    constant = ["hg/ha_yield", "average_rain_fall_mm_per_year", "pesticides_tonnes"]
    varying = df.groupby(FAO_KEYS)[constant].nunique().gt(1).any()
    if varying.any():
        cols = list(varying[varying].index)
        raise ValueError(f"Colonnes non constantes au sein d'une clé : {cols}")

    out = df.groupby(FAO_KEYS, as_index=False).agg(
        {**{col: "first" for col in constant}, "avg_temp": "mean"}
    )
    out["yield_t_ha"] = out.pop("hg/ha_yield") / HG_PER_TONNE
    out["Item"] = out["Item"].replace(CROP_NAME_MAP)
    return out.rename(columns=FAO_RENAME)[
        ["country", "crop", "year", "rainfall_mm", "pesticides_tonnes", "avg_temp_c", "yield_t_ha"]
    ]
