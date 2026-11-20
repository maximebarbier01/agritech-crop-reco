"""Tests du service de prédiction (facteur pratiques, avertissements, classement)."""

import pytest

from agritech_crop_reco.serving import Conditions, YieldService


@pytest.fixture(scope="module")
def service():
    return YieldService.load()  # modèle versionné dans models/


@pytest.fixture
def conditions():
    return Conditions(avg_temp_c=18.5, rainfall_mm=900, pesticides_tonnes=12_000)


def test_unknown_practices_keep_model_prediction(service):
    assert service.practice_factor("Wheat", None, None) == pytest.approx(1.0)


def test_practice_factor_centered_on_average(service):
    g = service.gains["Wheat"]
    g_e, g_i = g["fertilizer_gain"], g["irrigation_gain"]
    mean = 1 + 0.5 * g_e + 0.5 * g_i
    assert service.practice_factor("Wheat", True, True) == pytest.approx((1 + g_e + g_i) / mean)
    assert service.practice_factor("Wheat", False, False) == pytest.approx(1 / mean)
    # Ordres de grandeur annoncés au mentor : ≈ +29 % / −29 %
    assert service.practice_factor("Wheat", True, True) == pytest.approx(1.29, abs=0.01)
    assert service.practice_factor("Wheat", False, False) == pytest.approx(0.71, abs=0.01)


def test_practices_order(service):
    f = service.practice_factor
    assert f("Maize", False, False) < f("Maize", False, True) < f("Maize", True, False)
    assert f("Maize", True, False) < f("Maize", True, True)  # engrais > irrigation


def test_predict_applies_factor(service, conditions):
    with_practices = Conditions(**{**conditions.__dict__, "fertilizer_used": True})
    base = service.predict("Wheat", conditions)
    adjusted = service.predict("Wheat", with_practices)
    assert base["yield_t_ha"] == pytest.approx(base["yield_base_t_ha"])
    assert adjusted["yield_base_t_ha"] == pytest.approx(base["yield_base_t_ha"])
    assert adjusted["yield_t_ha"] > base["yield_t_ha"]


def test_predict_rejects_unknown_crop(service, conditions):
    with pytest.raises(ValueError, match="inconnue"):
        service.predict("Banana", conditions)


def test_recommend_ranks_all_crops_descending(service, conditions):
    ranking = service.recommend(conditions)
    assert [r["rank"] for r in ranking] == list(range(1, 11))
    assert {r["crop"] for r in ranking} == set(service.crops)
    yields = [r["yield_t_ha"] for r in ranking]
    assert yields == sorted(yields, reverse=True)


def test_recommend_consistent_with_predict(service, conditions):
    ranking = {r["crop"]: r["yield_t_ha"] for r in service.recommend(conditions)}
    assert ranking["Rice"] == pytest.approx(service.predict("Rice", conditions)["yield_t_ha"])


def test_warnings_only_outside_training_range(service, conditions):
    assert service.warnings(conditions) == []
    hot = Conditions(avg_temp_c=35, rainfall_mm=900, pesticides_tonnes=12_000)
    (warning,) = service.warnings(hot)
    assert "avg_temp_c" in warning
