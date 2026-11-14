"""Tests du découpage, de la référence naïve, des métriques et des modèles candidats."""

import numpy as np
import pandas as pd
import pytest

from agritech_crop_reco.modeling.baseline import CropMedianRegressor
from agritech_crop_reco.modeling.features import build_model, feature_columns
from agritech_crop_reco.modeling.metrics import ranking_spearman, regression_metrics
from agritech_crop_reco.modeling.split import make_test_countries, split_train_test
from agritech_crop_reco.modeling.train import candidate_configs, cross_validate


@pytest.fixture
def data():
    """Mini dataset consolidé : 10 pays × 3 cultures × 4 années, rendement dépendant
    de la culture et de la température."""
    rng = np.random.default_rng(0)
    rows = []
    crop_level = {"Maize": 3.0, "Potatoes": 16.0, "Wheat": 2.5}
    for c in range(10):
        temp = 10 + 2 * c
        for year in range(2000, 2004):
            for crop, level in crop_level.items():
                rows.append(
                    {
                        "country": f"C{c}",
                        "crop": crop,
                        "year": year,
                        "avg_temp_c": temp,
                        "rainfall_mm": 500 + 50 * c,
                        "pesticides_tonnes": 100.0 * (c + 1),
                        "yield_t_ha": level * (1 - 0.01 * temp) * rng.uniform(0.95, 1.05),
                    }
                )
    return pd.DataFrame(rows)


def test_split_is_by_country_and_deterministic(data):
    countries = make_test_countries(data, seed=42)
    assert countries == make_test_countries(data, seed=42)
    train, test = split_train_test(data, countries)
    assert set(train["country"]).isdisjoint(test["country"])
    assert len(train) + len(test) == len(data)


def test_baseline_predicts_crop_median(data):
    model = CropMedianRegressor().fit(data[["crop"]], data["yield_t_ha"])
    pred = model.predict(pd.DataFrame({"crop": ["Potatoes", "Unknown"]}))
    assert pred[0] == pytest.approx(data.loc[data["crop"] == "Potatoes", "yield_t_ha"].median())
    assert pred[1] == pytest.approx(data["yield_t_ha"].median())  # repli global


def test_ranking_spearman_perfect_and_inverse():
    meta = pd.DataFrame({"country": ["A"] * 3, "year": [2000] * 3})
    assert ranking_spearman(meta, [1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert ranking_spearman(meta, [1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    # moins de 3 cultures -> pas de score
    assert np.isnan(ranking_spearman(meta.iloc[:2], [1, 2], [1, 2]))


def test_regression_metrics_perfect_prediction():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1


@pytest.mark.parametrize("pesticides, expected", [("none", 3), ("raw", 4), ("log", 4)])
def test_feature_columns_variants(pesticides, expected):
    cols = feature_columns(pesticides)
    assert len(cols) == expected
    assert "country" not in cols and "year" not in cols  # non demandés par l'application


@pytest.mark.parametrize("model", ["ridge", "random_forest", "hist_gradient_boosting"])
@pytest.mark.parametrize("target", ["raw", "log"])
def test_every_candidate_fits_and_predicts_positive_t_ha(data, model, target):
    features = feature_columns("log")
    est = build_model(model, "log", target, seed=0).fit(data[features], data["yield_t_ha"])
    pred = est.predict(data[features])
    assert pred.shape == (len(data),)
    if target == "log":  # retour en t/ha via exp -> toujours positif
        assert (pred > 0).all()


def test_cross_validate_returns_one_row_per_fold(data):
    folds = cross_validate(build_model("ridge", "none", "raw", 0), data, feature_columns("none"))
    assert len(folds) == 5
    assert {"rmse", "mae", "r2", "ranking_spearman"} <= set(folds.columns)


def test_candidate_grid_size():
    # 1 référence + 3 modèles × 2 cibles × 3 variantes pesticides
    assert len(candidate_configs()) == 19


def _selection_row(model, rmse, std, rank, variety, pesticides="log"):
    return {
        "model": model,
        "target": "log",
        "pesticides": pesticides,
        "cv_rmse_mean": rmse,
        "cv_rmse_std": std,
        "cv_ranking_spearman_mean": rank,
        "ranking_variety": variety,
    }


def test_select_config_skips_frozen_ranking_within_one_std():
    from agritech_crop_reco.modeling.train import select_config

    results = pd.DataFrame(
        [
            _selection_row("baseline", 5.5, 0.3, 0.78, 1),
            _selection_row("ridge", 4.97, 0.46, 0.818, 1),  # meilleur RMSE mais classement figé
            _selection_row("hist_gradient_boosting", 5.14, 0.38, 0.828, 181, "raw"),
            _selection_row("hist_gradient_boosting", 5.14, 0.38, 0.828, 181, "log"),
            _selection_row("random_forest", 6.0, 0.2, 0.90, 200),  # hors d'un écart-type
        ]
    )
    chosen = results.loc[select_config(results)]
    assert chosen["model"] == "hist_gradient_boosting"
    assert chosen["pesticides"] == "log"  # égalité parfaite -> log


def test_ranking_variety_detects_additive_model(data):
    from agritech_crop_reco.modeling.train import probe_conditions, ranking_variety

    probe = probe_conditions(data, seed=0, n=50)
    features = feature_columns("log")
    ridge = build_model("ridge", "log", "log", seed=0).fit(data[features], data["yield_t_ha"])
    assert ranking_variety(ridge, features, probe) == 1
