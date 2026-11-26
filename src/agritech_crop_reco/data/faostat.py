"""Données FAOSTAT complémentaires : intrants agricoles par pays et par année.

Source : exports « bulk » de FAOSTAT (https://www.fao.org/faostat/), version du 1er au
9 octobre 2026, rangés dans ``data/raw/faostat/`` (non versionnés, empreintes dans
``data/raw/SHA256SUMS``) :

- ``fertilizers_nutrient.zip`` : azote (N), phosphate (P2O5), potasse (K2O),
  « Use per area of cropland » en kg/ha ;
- ``land_use.zip`` : « Land area equipped for irrigation », « Share in Cropland » en % ;
- ``pesticides_use.zip`` : « Pesticides (total) », « Use per area of cropland » en kg/ha.

Usage : ``python -m agritech_crop_reco.data.faostat`` écrit
``data/processed/faostat_inputs.csv`` (lignes pays × année du dataset consolidé),
versionné pour que l'expérience tourne sans les fichiers bruts (CI, livrables).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from agritech_crop_reco.config import load_settings
from agritech_crop_reco.data.merge import CONSOLIDATED_FILE

FAOSTAT_DIR = Path("faostat")
INPUTS_FILE = "faostat_inputs.csv"
AGGREGATE_AREA_CODE = 5000  # codes >= 5000 : régions et agrégats (Monde, Afrique…), pas des pays

# Noms du dataset consolidé (anciens noms FAO) -> noms FAOSTAT actuels
COUNTRY_RENAMES = {
    "Turkey": "Türkiye",
    "Netherlands": "Netherlands (Kingdom of the)",
    "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
}

# (fichier, item FAOSTAT, élément FAOSTAT) -> colonne
INPUT_SERIES = {
    "n_kg_ha": (
        "fertilizers_nutrient.zip",
        "Nutrient nitrogen N (total)",
        "Use per area of cropland",
    ),
    "p_kg_ha": (
        "fertilizers_nutrient.zip",
        "Nutrient phosphate P2O5 (total)",
        "Use per area of cropland",
    ),
    "k_kg_ha": (
        "fertilizers_nutrient.zip",
        "Nutrient potash K2O (total)",
        "Use per area of cropland",
    ),
    "irrig_share": ("land_use.zip", "Land area equipped for irrigation", "Share in Cropland"),
    "pest_kg_ha": ("pesticides_use.zip", "Pesticides (total)", "Use per area of cropland"),
}
INPUT_COLS = list(INPUT_SERIES)


def read_faostat_zip(path: Path, extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Lit le CSV principal d'un export FAOSTAT (format « Normalized »), pays seulement."""
    columns = ["Area Code", "Area", "Item", "Element", "Year", "Unit", "Value", *extra_columns]
    with zipfile.ZipFile(path) as archive:
        (member,) = [m for m in archive.namelist() if m.endswith("All_Data_(Normalized).csv")]
        df = pd.read_csv(archive.open(member), encoding="utf-8", usecols=columns, low_memory=False)
    return df[df["Area Code"] < AGGREGATE_AREA_CODE]


def load_inputs(raw_dir: Path) -> pd.DataFrame:
    """Une ligne par pays × année, une colonne par intrant (noms de pays du dataset consolidé)."""
    base = raw_dir / FAOSTAT_DIR
    cache: dict[str, pd.DataFrame] = {}
    series = []
    for col, (file, item, element) in INPUT_SERIES.items():
        df = cache.setdefault(file, read_faostat_zip(base / file))
        s = df[(df["Item"] == item) & (df["Element"] == element)]
        series.append(s.set_index(["Area", "Year"])["Value"].rename(col))
    wide = (
        pd.concat(series, axis=1).reset_index().rename(columns={"Area": "country", "Year": "year"})
    )
    back = {new: old for old, new in COUNTRY_RENAMES.items()}
    wide["country"] = wide["country"].replace(back)
    return wide


def build_inputs(raw_dir: Path, consolidated: pd.DataFrame) -> pd.DataFrame:
    """Intrants restreints aux couples pays × année du dataset consolidé."""
    keys = consolidated[["country", "year"]].drop_duplicates()
    out = keys.merge(load_inputs(raw_dir), on=["country", "year"], how="left")
    return out.sort_values(["country", "year"]).reset_index(drop=True)


def add_inputs(df: pd.DataFrame, inputs: pd.DataFrame) -> pd.DataFrame:
    """Jointure gauche des intrants sur pays × année (valeurs manquantes conservées)."""
    return df.merge(inputs, on=["country", "year"], how="left", validate="many_to_one")


def main() -> None:
    settings = load_settings()
    consolidated = pd.read_csv(settings.processed_dir / CONSOLIDATED_FILE)
    out = build_inputs(settings.raw_dir, consolidated)
    path = settings.processed_dir / INPUTS_FILE
    out.to_csv(path, index=False)
    coverage = out[INPUT_COLS].notna().mean().round(3).to_dict()
    print(f"{len(out):,} couples pays × année -> {path}")
    print(f"couverture : {coverage}")


if __name__ == "__main__":
    main()
