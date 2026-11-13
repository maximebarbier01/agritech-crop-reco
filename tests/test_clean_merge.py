"""Tests du nettoyage et de la fusion sur des cas construits à la main."""

import pandas as pd
import pytest

from agritech_crop_reco.data.clean import (
    GLOBAL,
    clean_crop_yield,
    clean_fao,
    compute_practice_gains,
)
from agritech_crop_reco.data.merge import merge_datasets, validate_consolidated


@pytest.fixture
def kaggle():
    # Maïs, effets additifs : 4 t/ha sans pratique, +2 avec engrais (+50 %),
    # +1 avec irrigation (+25 %), 7 avec les deux. Blé : une ligne négative à supprimer.
    return pd.DataFrame(
        {
            "Crop": pd.Categorical(["Maize"] * 4 + ["Wheat"] * 3),
            "Fertilizer_Used": [False, True, False, True, False, False, True],
            "Irrigation_Used": [False, False, True, True, False, True, False],
            "Yield_tons_per_hectare": [4.0, 6.0, 5.0, 7.0, -0.5, 5.0, 6.0],
        }
    )


@pytest.fixture
def fao():
    # Une clé (Albania, Maize, 1990) répétée 2 fois avec 2 températures différentes
    return pd.DataFrame(
        {
            "Area": ["Albania", "Albania", "Albania"],
            "Item": ["Maize", "Maize", "Rice, paddy"],
            "Year": [1990, 1990, 1990],
            "hg/ha_yield": [36_613, 36_613, 23_333],
            "average_rain_fall_mm_per_year": [1485.0, 1485.0, 1485.0],
            "pesticides_tonnes": [121.0, 121.0, 121.0],
            "avg_temp": [16.0, 18.0, 16.37],
        }
    )


def test_clean_crop_yield_drops_non_positive(kaggle):
    out = clean_crop_yield(kaggle)
    assert len(out) == 6
    assert (out["Yield_tons_per_hectare"] > 0).all()


def test_practice_gains_relative_to_no_practice_yield(kaggle):
    gains = compute_practice_gains(clean_crop_yield(kaggle)).set_index("crop")
    # Maïs : rendement sans pratique 4 t/ha ; engrais +2 -> +50 % ; irrigation +1 -> +25 %
    assert gains.loc["Maize", "fertilizer_gain"] == pytest.approx(0.5)
    assert gains.loc["Maize", "irrigation_gain"] == pytest.approx(0.25)
    assert GLOBAL in gains.index


def test_clean_fao_dedup_units_and_names(fao):
    out = clean_fao(fao)
    assert len(out) == 2
    maize = out[out["crop"] == "Maize"].iloc[0]
    assert maize["avg_temp_c"] == pytest.approx(17.0)  # moyenne des 2 stations
    assert maize["yield_t_ha"] == pytest.approx(3.6613)  # hg/ha -> t/ha
    assert "Rice" in set(out["crop"])  # "Rice, paddy" harmonisé


def test_clean_fao_rejects_inconsistent_key(fao):
    fao.loc[1, "hg/ha_yield"] = 99_999
    with pytest.raises(ValueError, match="non constantes"):
        clean_fao(fao)


def _gains():
    return pd.DataFrame(
        {
            "crop": ["Maize", GLOBAL],
            "fertilizer_gain": [0.40, 0.38],
            "irrigation_gain": [0.30, 0.29],
        }
    )


def test_merge_uses_crop_gain_then_global_fallback(fao):
    out = merge_datasets(clean_fao(fao), _gains()).set_index("crop")
    assert out.loc["Maize", "fertilizer_gain"] == 0.40
    assert out.loc["Maize", "gain_source"] == "crop"
    assert out.loc["Rice", "fertilizer_gain"] == 0.38  # absente des gains -> global
    assert out.loc["Rice", "gain_source"] == "global"
    validate_consolidated(out.reset_index())


def test_validate_rejects_duplicate_key(fao):
    out = merge_datasets(clean_fao(fao), _gains())
    with pytest.raises(ValueError, match="doublons"):
        validate_consolidated(pd.concat([out, out.iloc[[0]]]))
