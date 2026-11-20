"""Fusion des deux jeux (option A) et construction du dataset consolidé.

- Base : observations FAO (pays × culture × année) -> ce qui départage les cultures.
- Enrichissement : gain relatif engrais / irrigation mesuré sur Kaggle, joint **par culture**.
  Les cultures FAO absentes de Kaggle reçoivent le gain global (l'EDA montre que le gain
  ne dépend pas de la culture) ; ``gain_source`` trace l'origine de la valeur.

Usage : ``python -m agritech_crop_reco.data.merge`` écrit data/processed/crop_yield_consolidated.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agritech_crop_reco.config import load_settings
from agritech_crop_reco.data.clean import (
    GLOBAL,
    PRACTICES,
    clean_crop_yield,
    clean_fao,
    compute_practice_gains,
)
from agritech_crop_reco.data.load import load_crop_yield, load_fao_yield_df

CONSOLIDATED_FILE = "crop_yield_consolidated.csv"
KEY = ["country", "crop", "year"]
GAIN_COLS = list(PRACTICES.values())


def merge_datasets(fao: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    """Jointure gauche FAO <- gains par culture, repli sur le gain global."""
    per_crop = gains[gains["crop"] != GLOBAL]
    global_gain = gains.set_index("crop").loc[GLOBAL, GAIN_COLS]

    out = fao.merge(per_crop, on="crop", how="left", validate="many_to_one")
    missing = out[GAIN_COLS[0]].isna()
    out["gain_source"] = "crop"
    out.loc[missing, "gain_source"] = "global"
    for col in GAIN_COLS:
        out.loc[missing, col] = global_gain[col]
    return out


def validate_consolidated(df: pd.DataFrame) -> None:
    """Contrôles qualité bloquants du dataset consolidé."""
    errors = []
    if df.duplicated(KEY).any():
        errors.append(f"{df.duplicated(KEY).sum()} doublons sur la clé {KEY}")
    if df.isna().any().any():
        errors.append(f"valeurs manquantes : {df.isna().sum()[lambda s: s > 0].to_dict()}")
    if (df["yield_t_ha"] <= 0).any():
        errors.append("rendements <= 0")
    if not df[GAIN_COLS].apply(lambda s: s.between(0, 1)).all().all():
        errors.append("gains hors de [0, 1]")
    if errors:
        raise ValueError("Dataset consolidé invalide : " + " ; ".join(errors))


def build_consolidated(raw_dir: Path | None = None) -> pd.DataFrame:
    """Chaîne complète : chargement -> nettoyage -> fusion -> validation."""
    gains = compute_practice_gains(clean_crop_yield(load_crop_yield(raw_dir)))
    consolidated = merge_datasets(clean_fao(load_fao_yield_df(raw_dir)), gains)
    consolidated = consolidated.sort_values(KEY).reset_index(drop=True)
    validate_consolidated(consolidated)
    return consolidated


def main() -> None:
    settings = load_settings()
    df = build_consolidated(settings.raw_dir)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.processed_dir / CONSOLIDATED_FILE
    df.to_csv(out, index=False)
    print(f"{len(df):,} lignes × {df.shape[1]} colonnes -> {out}")


if __name__ == "__main__":
    main()
