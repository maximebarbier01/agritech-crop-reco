"""Tests du garde-fou de performance du pipeline."""

from agritech_crop_reco.pipeline import quality_gate

REFERENCE = {"rmse": 4.67, "ranking_spearman": 0.794}


def test_gate_passes_when_metrics_are_stable_or_better():
    assert quality_gate(REFERENCE, {"rmse": 4.60, "ranking_spearman": 0.80}).passed
    # +4 % de RMSE et −0,01 de classement : dans les tolérances
    assert quality_gate(REFERENCE, {"rmse": 4.67 * 1.04, "ranking_spearman": 0.784}).passed


def test_gate_fails_on_rmse_regression():
    gate = quality_gate(REFERENCE, {"rmse": 4.67 * 1.06, "ranking_spearman": 0.80})
    assert not gate.passed
    assert not gate.rows[0]["ok"] and gate.rows[1]["ok"]


def test_gate_fails_on_ranking_regression():
    gate = quality_gate(REFERENCE, {"rmse": 4.60, "ranking_spearman": 0.76})
    assert not gate.passed
    assert "ÉCHEC" in gate.markdown()
