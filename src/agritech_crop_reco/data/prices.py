"""Prix à la production FAOSTAT : passer du rendement (t/ha) au chiffre d'affaires ($/ha).

Source : FAOSTAT « Producer Prices » (export du 9 octobre 2026,
``data/raw/faostat/producer_prices.zip``), élément « Producer Price (USD/tonne) »,
valeurs annuelles, **2018-2023**.

Deux niveaux de prix, écrits dans ``models/crop_prices.json`` (lu par l'API) :
- **national** : pour chaque pays du dataset consolidé et chaque culture, médiane de ses
  prix annuels sur la période (quand FAOSTAT en publie) ;
- **mondial** : pour chaque culture, médiane des prix nationaux (chaque pays compte une
  fois), avec le nombre de pays et l'intervalle interquartile (dispersion entre pays).

Limites : dollars courants (taux de change de chaque année), prix « départ exploitation »,
**aucun coût de production** : on obtient un chiffre d'affaires estimé, pas un profit.

Usage : ``python -m agritech_crop_reco.data.prices``
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agritech_crop_reco.config import load_settings
from agritech_crop_reco.data.faostat import COUNTRY_RENAMES, FAOSTAT_DIR, read_faostat_zip
from agritech_crop_reco.data.merge import CONSOLIDATED_FILE

PRICES_ZIP = "producer_prices.zip"
PRICES_FILE = "crop_prices.json"
PERIOD = (2018, 2023)
ELEMENT = "Producer Price (USD/tonne)"

# Libellés FAOSTAT actuels -> cultures du modèle
ITEM_TO_CROP = {
    "Cassava, fresh": "Cassava",
    "Maize (corn)": "Maize",
    "Plantains": "Plantains and others",
    "Plantains and cooking bananas": "Plantains and others",
    "Potatoes": "Potatoes",
    "Rice": "Rice",
    "Sorghum": "Sorghum",
    "Soya beans": "Soybean",
    "Sweet potatoes": "Sweet potatoes",
    "Wheat": "Wheat",
    "Yams": "Yams",
}


def load_annual_prices(raw_dir: Path) -> pd.DataFrame:
    """Prix annuels en USD/t par pays (noms du dataset consolidé) et culture, sur la période."""
    df = read_faostat_zip(raw_dir / FAOSTAT_DIR / PRICES_ZIP, extra_columns=("Months",))
    df = df[
        (df["Element"] == ELEMENT)
        & (df["Months"] == "Annual value")
        & (df["Item"].isin(ITEM_TO_CROP))
        & (df["Year"].between(*PERIOD))
        & (df["Value"] > 0)
    ]
    back = {new: old for old, new in COUNTRY_RENAMES.items()}
    return pd.DataFrame(
        {
            "country": df["Area"].replace(back),
            "crop": df["Item"].map(ITEM_TO_CROP),
            "year": df["Year"],
            "price_usd_t": df["Value"],
        }
    )


def build_price_table(annual: pd.DataFrame, countries: list[str], crops: list[str]) -> dict:
    """Prix nationaux (pays du modèle) et mondiaux (médiane entre pays) par culture."""
    per_country = annual.groupby(["country", "crop"])["price_usd_t"].median()
    world = per_country.groupby("crop").agg(
        price_usd_t="median",
        n_countries="size",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    )
    missing = sorted(set(crops) - set(world.index))
    if missing:
        raise ValueError(f"Aucun prix FAOSTAT pour : {missing}")
    national = per_country[per_country.index.get_level_values("country").isin(countries)]
    return {
        "source": "FAOSTAT Producer Prices (export du 9 octobre 2026)",
        "element": ELEMENT,
        "period": list(PERIOD),
        "unit": "USD/t",
        "world": {
            crop: {k: round(float(v), 1) for k, v in row.items()}
            for crop, row in world.loc[crops].iterrows()
        },
        "national": {
            country: {crop: round(float(p), 1) for crop, p in grp.droplevel(0).items()}
            for country, grp in national.groupby(level="country")
        },
    }


def main() -> None:
    settings = load_settings()
    consolidated = pd.read_csv(settings.processed_dir / CONSOLIDATED_FILE)
    table = build_price_table(
        load_annual_prices(settings.raw_dir),
        countries=sorted(consolidated["country"].unique()),
        crops=sorted(consolidated["crop"].unique()),
    )
    path = settings.models_dir / PRICES_FILE
    path.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    n_prices = sum(len(v) for v in table["national"].values())
    print(
        f"{path} : {len(table['world'])} prix mondiaux, {n_prices} prix nationaux "
        f"({len(table['national'])} pays)"
    )
    for crop, w in table["world"].items():
        print(
            f"  {crop:<22} {w['price_usd_t']:>7.0f} $/t  ({w['n_countries']:.0f} pays, "
            f"IQR {w['q25']:.0f}–{w['q75']:.0f})"
        )


if __name__ == "__main__":
    main()
