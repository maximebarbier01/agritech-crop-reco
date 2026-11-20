"""Tests des chargeurs bruts sur de mini-fichiers (pas besoin des vraies données)."""

import pandas as pd
import pytest

from agritech_crop_reco.data import load


@pytest.fixture
def raw_dir(tmp_path):
    (tmp_path / "agriculture_crop_yield").mkdir()
    (tmp_path / "agriculture_crop_yield" / "crop_yield.csv").write_text(
        "Region,Soil_Type,Crop,Rainfall_mm,Temperature_Celsius,Fertilizer_Used,"
        "Irrigation_Used,Weather_Condition,Days_to_Harvest,Yield_tons_per_hectare\n"
        "West,Sandy,Cotton,897.1,27.7,False,True,Cloudy,122,6.56\n"
        "South,Clay,Rice,992.7,18.0,True,True,Rainy,140,8.53\n"
    )
    fao = tmp_path / "crop_yield_prediction"
    fao.mkdir()
    (fao / "yield_df.csv").write_text(
        ",Area,Item,Year,hg/ha_yield,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp\n"
        "0,Albania,Maize,1990,36613,1485.0,121.0,16.37\n"
    )
    (fao / "yield.csv").write_text("Area,Item,Year,Value\nAlbania,Maize,1990,36613\n")
    (fao / "pesticides.csv").write_text("Area,Year,Value\nAlbania,1990,121\n")
    (fao / "rainfall.csv").write_text(
        " Area,Year,average_rain_fall_mm_per_year\nAlbania,1990,1485\n"
    )
    (fao / "temp.csv").write_text("year,country,avg_temp\n1990,Albania,16.37\n")
    return tmp_path


def test_load_crop_yield_types(raw_dir):
    df = load.load_crop_yield(raw_dir)
    assert len(df) == 2
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in load.CROP_YIELD_CATEGORICAL)
    assert df["Fertilizer_Used"].dtype == bool


def test_load_fao_yield_df_drops_unnamed_index(raw_dir):
    df = load.load_fao_yield_df(raw_dir)
    assert list(df.columns)[:3] == ["Area", "Item", "Year"]
    assert not any(c.startswith("Unnamed") for c in df.columns)


def test_load_fao_sources_strips_column_names(raw_dir):
    sources = load.load_fao_sources(raw_dir)
    assert set(sources) == {"yield", "pesticides", "rainfall", "temp"}
    assert "Area" in sources["rainfall"].columns
