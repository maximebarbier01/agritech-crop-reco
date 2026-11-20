"""Tests des outils d'ACP et du rapport de corrélation."""

import numpy as np
import pandas as pd
import pytest

from agritech_crop_reco.analysis import eta_squared, pca_correlations


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2_000)
    return pd.DataFrame(
        {"x": x, "x_copy": 2 * x + 1, "noise": rng.normal(size=2_000)}  # x_copy redondante
    )


def test_pca_variance_sums_to_one_and_detects_redundancy(data):
    explained, loadings = pca_correlations(data, ["x", "x_copy", "noise"])
    assert explained.sum() == pytest.approx(1.0)
    # x et x_copy forment un seul axe portant ~2/3 de la variance totale
    assert explained["F1"] == pytest.approx(2 / 3, abs=0.02)
    # ~1 et non 1 exactement : le bruit tiré au hasard est très faiblement corrélé à x
    assert abs(loadings.loc["x", "F1"]) == pytest.approx(1.0, abs=1e-3)


def test_pca_loadings_are_correlations(data):
    _, loadings = pca_correlations(data, ["x", "x_copy", "noise"])
    # Somme des carrés des corrélations d'une variable avec tous les axes = 1
    assert (loadings**2).sum(axis=1).to_numpy() == pytest.approx(np.ones(3))


def test_eta_squared_bounds():
    groups = pd.Series(["a", "a", "b", "b"])
    assert eta_squared(groups, pd.Series([1.0, 1.0, 5.0, 5.0])) == pytest.approx(1.0)
    assert eta_squared(groups, pd.Series([1.0, 5.0, 1.0, 5.0])) == pytest.approx(0.0)
