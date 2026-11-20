"""Schémas d'entrée / sortie de l'API (validation Pydantic, documentation /docs)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Crop(StrEnum):
    """Les 10 cultures connues du modèle (noms du jeu FAO, harmonisés)."""

    CASSAVA = "Cassava"
    MAIZE = "Maize"
    PLANTAINS = "Plantains and others"
    POTATOES = "Potatoes"
    RICE = "Rice"
    SORGHUM = "Sorghum"
    SOYBEAN = "Soybean"
    SWEET_POTATOES = "Sweet potatoes"
    WHEAT = "Wheat"
    YAMS = "Yams"


class ParcelConditions(BaseModel):
    """Conditions de la parcelle.

    Les bornes ci-dessous sont **physiques** (valeur impossible -> erreur 422). Une valeur
    plausible mais absente des données d'entraînement est acceptée avec un avertissement.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "avg_temp_c": 18.5,
                "rainfall_mm": 900,
                "pesticides_tonnes": 12000,
                "fertilizer_used": True,
                "irrigation_used": False,
            }
        }
    )

    avg_temp_c: float = Field(ge=-30, le=50, description="Température moyenne annuelle (°C)")
    rainfall_mm: float = Field(ge=0, le=12_000, description="Pluie annuelle (mm)")
    pesticides_tonnes: float = Field(
        ge=0,
        le=2_000_000,
        description="Pesticides utilisés dans la région (tonnes de matière active / an) : "
        "indicateur du niveau d'intensification agricole",
    )
    fertilizer_used: bool | None = Field(
        default=None, description="Engrais utilisé ? (non renseigné = pratique moyenne)"
    )
    irrigation_used: bool | None = Field(
        default=None, description="Parcelle irriguée ? (non renseigné = pratique moyenne)"
    )


class PredictRequest(ParcelConditions):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "crop": "Wheat",
                "avg_temp_c": 18.5,
                "rainfall_mm": 900,
                "pesticides_tonnes": 12000,
                "fertilizer_used": True,
                "irrigation_used": False,
            }
        }
    )

    crop: Crop


class CropYield(BaseModel):
    crop: str
    yield_t_ha: float = Field(description="Rendement prédit, pratiques comprises (t/ha)")
    yield_base_t_ha: float = Field(description="Rendement du modèle, pratiques moyennes (t/ha)")
    practice_factor: float = Field(description="Ajustement engrais / irrigation (1 = moyen)")


class PredictResponse(CropYield):
    warnings: list[str]
    model_version: str


class RankedCrop(CropYield):
    rank: int


class RecommendResponse(BaseModel):
    ranking: list[RankedCrop]
    warnings: list[str]
    model_version: str
    note: str = Field(
        default="Classement par rendement physique (t/ha) : il ne tient pas compte des prix "
        "de vente, qui varient fortement d'une culture à l'autre."
    )


class PesticideLevel(BaseModel):
    label: str = Field(description="Niveau d'intensification affiché à l'utilisateur")
    quantile: float = Field(description="Quantile des pays d'entraînement correspondant")
    pesticides_tonnes: float = Field(description="Valeur envoyée au modèle (tonnes / an)")


class MetadataResponse(BaseModel):
    """Ce dont une interface a besoin pour proposer des saisies dans le domaine du modèle."""

    crops: list[str]
    training_ranges: dict[str, list[float]] = Field(
        description="Plage [min, max] de chaque condition vue à l'entraînement"
    )
    pesticide_levels: list[PesticideLevel]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    n_crops: int
