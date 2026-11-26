"""Service de prédiction : modèle de rendement + ajustement engrais / irrigation.

Indépendant de FastAPI pour être testé seul et réutilisé (API, notebooks, scripts).

Rendement prédit = rendement de base (modèle FAO) × facteur pratiques, avec
    facteur = (1 + g_E·E + g_I·I) / (1 + p·g_E + p·g_I)
- g_E, g_I : gains relatifs engrais / irrigation mesurés sur le jeu Kaggle (étape 1) ;
- E, I : 1 si la pratique est utilisée, 0 sinon, ``p`` si non renseignée ;
- p = 0,5 : part des parcelles avec la pratique dans le jeu Kaggle.
Le dénominateur « centre » le facteur : le rendement FAO est une moyenne nationale qui
mélange déjà des parcelles avec et sans pratiques (décision b de l'étape 3). Une
parcelle aux pratiques moyennes (ou non renseignées) garde donc la prédiction du modèle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from agritech_crop_reco.config import load_settings

MODEL_FILE = "yield_model.joblib"
MODEL_CARD_FILE = "model_card.json"
GAINS_FILE = "practice_gains.json"
PRICES_FILE = "crop_prices.json"
RANK_BY = ("revenue", "yield")
REFERENCE_PRACTICE_SHARE = 0.5
CONDITIONS = ("avg_temp_c", "rainfall_mm", "pesticides_tonnes")


@dataclass(frozen=True)
class Conditions:
    avg_temp_c: float
    rainfall_mm: float
    pesticides_tonnes: float
    fertilizer_used: bool | None = None
    irrigation_used: bool | None = None


class YieldService:
    def __init__(self, model, card: dict, gains: dict[str, dict], prices: dict | None = None):
        self.model = model
        self.card = card
        self.gains = gains
        self.prices = prices  # None : pas de prix -> classement par rendement uniquement
        self.features: list[str] = card["features"]
        self.crops: list[str] = card["crops"]
        missing = set(self.crops) - set(gains)
        if missing:
            raise ValueError(f"Gains pratiques absents pour : {sorted(missing)}")

    @classmethod
    def load(cls, models_dir: Path | None = None) -> YieldService:
        models_dir = models_dir or load_settings().models_dir
        card = json.loads((models_dir / MODEL_CARD_FILE).read_text())
        gains = json.loads((models_dir / GAINS_FILE).read_text())
        prices_path = models_dir / PRICES_FILE
        prices = json.loads(prices_path.read_text()) if prices_path.exists() else None
        return cls(joblib.load(models_dir / MODEL_FILE), card, gains, prices)

    @property
    def version(self) -> str:
        return f"{self.card['model']}@{self.card['git_sha']} ({self.card['trained_at']})"

    @property
    def training_ranges(self) -> dict[str, list[float]]:
        return self.card["training_ranges"]

    @property
    def pesticide_levels(self) -> list[dict]:
        return self.card["pesticide_levels"]

    @property
    def countries(self) -> list[str]:
        """Pays pour lesquels au moins un prix national est disponible."""
        return sorted(self.prices["national"]) if self.prices else []

    def price_for(
        self, crop: str, country: str | None = None, custom: dict[str, float] | None = None
    ) -> tuple[float | None, str | None]:
        """Prix (USD/t) et son origine : saisi > national > mondial (médiane entre pays).

        Le pays ne sert qu'au prix, jamais au modèle de rendement.
        """
        if custom and crop in custom:
            return float(custom[crop]), "saisi"
        if self.prices is None:
            return None, None
        national = self.prices["national"].get(country or "", {})
        if crop in national:
            return float(national[crop]), "national"
        return float(self.prices["world"][crop]["price_usd_t"]), "mondial"

    def _check_country(self, country: str | None) -> None:
        if country is not None and country not in self.countries:
            raise ValueError(f"Pays sans prix connus : {country!r}")

    def _with_revenue(self, row: dict, country: str | None, custom: dict | None) -> dict:
        price, source = self.price_for(row["crop"], country, custom)
        revenue = row["yield_t_ha"] * price if price is not None else None
        return {**row, "price_usd_t": price, "price_source": source, "revenue_usd_ha": revenue}

    def practice_factor(self, crop: str, fertilizer: bool | None, irrigation: bool | None) -> float:
        g_e = self.gains[crop]["fertilizer_gain"]
        g_i = self.gains[crop]["irrigation_gain"]
        p = REFERENCE_PRACTICE_SHARE
        e = p if fertilizer is None else float(fertilizer)
        i = p if irrigation is None else float(irrigation)
        return (1 + g_e * e + g_i * i) / (1 + p * g_e + p * g_i)

    def warnings(self, conditions: Conditions) -> list[str]:
        """Valeurs acceptées mais hors de la plage vue à l'entraînement (extrapolation)."""
        out = []
        for name, (low, high) in self.training_ranges.items():
            value = getattr(conditions, name)
            if not low <= value <= high:
                out.append(
                    f"{name}={value:g} hors de la plage d'entraînement [{low:g} ; {high:g}] : "
                    "prédiction extrapolée, moins fiable"
                )
        return out

    def _base_yields(self, conditions: Conditions, crops: list[str]) -> pd.Series:
        rows = [{"crop": crop, **{c: getattr(conditions, c) for c in CONDITIONS}} for crop in crops]
        X = pd.DataFrame(rows)[self.features]
        return pd.Series(self.model.predict(X), index=crops)

    def predict(
        self,
        crop: str,
        conditions: Conditions,
        country: str | None = None,
        custom_prices: dict[str, float] | None = None,
    ) -> dict:
        if crop not in self.crops:
            raise ValueError(f"Culture inconnue : {crop!r}")
        self._check_country(country)
        base = float(self._base_yields(conditions, [crop]).iloc[0])
        factor = self.practice_factor(crop, conditions.fertilizer_used, conditions.irrigation_used)
        row = {
            "crop": crop,
            "yield_base_t_ha": base,
            "practice_factor": factor,
            "yield_t_ha": base * factor,
        }
        return self._with_revenue(row, country, custom_prices)

    def recommend(
        self,
        conditions: Conditions,
        country: str | None = None,
        custom_prices: dict[str, float] | None = None,
        rank_by: str = "revenue",
    ) -> list[dict]:
        """Simule toutes les cultures et les classe par chiffre d'affaires estimé (défaut)
        ou par rendement. Sans prix disponibles, le classement se fait par rendement."""
        if rank_by not in RANK_BY:
            raise ValueError(f"rank_by doit valoir {RANK_BY}, reçu {rank_by!r}")
        self._check_country(country)
        bases = self._base_yields(conditions, self.crops)
        results = []
        for crop, base in bases.items():
            factor = self.practice_factor(
                crop, conditions.fertilizer_used, conditions.irrigation_used
            )
            row = {
                "crop": crop,
                "yield_base_t_ha": float(base),
                "practice_factor": factor,
                "yield_t_ha": float(base) * factor,
            }
            results.append(self._with_revenue(row, country, custom_prices))
        key = (
            "revenue_usd_ha"
            if rank_by == "revenue" and results[0]["revenue_usd_ha"]
            else "yield_t_ha"
        )
        results.sort(key=lambda r: r[key], reverse=True)
        return [{"rank": i, **r} for i, r in enumerate(results, start=1)]
