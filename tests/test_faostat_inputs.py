"""Tests du raccordement FAOSTAT et de la règle de décision de l'expérience 03."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from agritech_crop_reco.data.faostat import INPUT_COLS, add_inputs, build_inputs, read_faostat_zip
from agritech_crop_reco.modeling.experiment_inputs import (
    CONFIGS,
    REFERENCE,
    build_estimator,
    decide,
)

HEADER = "Area Code,Area,Item,Element,Year,Unit,Value\n"


def _zip(path, rows):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("X_E_All_Data_(Normalized).csv", HEADER + "".join(rows))


@pytest.fixture
def raw_dir(tmp_path):
    base = tmp_path / "faostat"
    base.mkdir()
    _zip(
        base / "fertilizers_nutrient.zip",
        [
            "223,Türkiye,Nutrient nitrogen N (total),Use per area of cropland,2000,kg/ha,60\n",
            "223,Türkiye,Nutrient phosphate P2O5 (total),Use per area of cropland,2000,kg/ha,25\n",
            "223,Türkiye,Nutrient potash K2O (total),Use per area of cropland,2000,kg/ha,3\n",
            "5000,World,Nutrient nitrogen N (total),Use per area of cropland,2000,kg/ha,70\n",
        ],
    )
    _zip(
        base / "land_use.zip",
        ["223,Türkiye,Land area equipped for irrigation,Share in Cropland,2000,%,20\n"],
    )
    _zip(
        base / "pesticides_use.zip",
        ["223,Türkiye,Pesticides (total),Use per area of cropland,2000,kg/ha,1.5\n"],
    )
    return tmp_path


def test_read_excludes_regional_aggregates(raw_dir):
    df = read_faostat_zip(raw_dir / "faostat" / "fertilizers_nutrient.zip")
    assert "World" not in set(df["Area"])


def test_build_inputs_maps_renamed_countries(raw_dir):
    consolidated = pd.DataFrame({"country": ["Turkey", "Turkey"], "year": [2000, 2001]})
    out = build_inputs(raw_dir, consolidated).set_index("year")
    assert out.loc[2000, "n_kg_ha"] == 60  # « Türkiye » dans FAOSTAT = « Turkey » chez nous
    assert out.loc[2000, "irrig_share"] == 20
    assert out.loc[2001, INPUT_COLS].isna().all()  # année absente -> manquant, pas d'erreur


def test_add_inputs_keeps_all_rows():
    df = pd.DataFrame(
        {"country": ["A", "A", "B"], "year": [2000, 2000, 2000], "crop": ["x", "y", "x"]}
    )
    inputs = pd.DataFrame({"country": ["A"], "year": [2000], **{c: [1.0] for c in INPUT_COLS}})
    out = add_inputs(df, inputs)
    assert len(out) == 3 and out.loc[2, INPUT_COLS].isna().all()


def _results(ref, ref_std, others):
    data = {REFERENCE: {"cv_rmse_mean": ref, "cv_rmse_std": ref_std}}
    data |= {k: {"cv_rmse_mean": v, "cv_rmse_std": 0.5} for k, v in others.items()}
    return pd.DataFrame(data).T


def test_decide_requires_gain_larger_than_one_std():
    res = decide(_results(5.0, 0.4, {"faible_gain": 4.9, "fort_gain": 4.5}))
    assert not res.loc["faible_gain", "retained"]  # 4,9 > 5,0 − 0,4
    assert res.loc["fort_gain", "retained"]  # 4,5 < 4,6
    assert not res.loc[REFERENCE, "retained"]


def test_every_configuration_fits_with_missing_inputs():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "crop": rng.choice(["Maize", "Wheat"], n),
            "avg_temp_c": rng.uniform(5, 30, n),
            "rainfall_mm": rng.uniform(200, 2000, n),
            "pesticides_tonnes": rng.uniform(10, 1e4, n),
            **{c: rng.uniform(0, 100, n) for c in INPUT_COLS},
            "yield_t_ha": rng.uniform(1, 10, n),
        }
    )
    df.loc[::7, "n_kg_ha"] = np.nan  # intrants manquants pour certains pays
    params = {"max_iter": 20}
    for numeric in CONFIGS.values():
        est = build_estimator(numeric, params, seed=0).fit(df[["crop", *numeric]], df["yield_t_ha"])
        assert (est.predict(df[["crop", *numeric]]) > 0).all()
