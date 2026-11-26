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
    ranking = service.recommend(conditions, rank_by="yield")
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


# ---------------------------------------------------------------- prix et chiffre d'affaires


def test_price_priority_custom_then_national_then_world(service):
    world = service.prices["world"]["Wheat"]["price_usd_t"]
    assert service.price_for("Wheat") == (world, "mondial")
    country = next(c for c, p in service.prices["national"].items() if "Wheat" in p)
    assert service.price_for("Wheat", country) == (
        service.prices["national"][country]["Wheat"],
        "national",
    )
    assert service.price_for("Wheat", country, {"Wheat": 999.0}) == (999.0, "saisi")


def test_country_without_crop_price_falls_back_to_world(service):
    country, prices = next((c, p) for c, p in service.prices["national"].items() if "Yams" not in p)
    price, source = service.price_for("Yams", country)
    assert source == "mondial" and price == service.prices["world"]["Yams"]["price_usd_t"]


def test_recommend_by_revenue_is_sorted_and_consistent(service, conditions):
    ranking = service.recommend(conditions)  # défaut : chiffre d'affaires
    revenues = [r["revenue_usd_ha"] for r in ranking]
    assert revenues == sorted(revenues, reverse=True)
    for r in ranking:
        assert r["revenue_usd_ha"] == pytest.approx(r["yield_t_ha"] * r["price_usd_t"])


def test_custom_price_can_change_the_ranking(service, conditions):
    by_yield = service.recommend(conditions, rank_by="yield")
    last = by_yield[-1]["crop"]
    boosted = service.recommend(conditions, custom_prices={last: 1e6})
    assert boosted[0]["crop"] == last and boosted[0]["price_source"] == "saisi"


def test_unknown_country_and_rank_by_are_rejected(service, conditions):
    with pytest.raises(ValueError, match="Pays"):
        service.recommend(conditions, country="Atlantide")
    with pytest.raises(ValueError, match="rank_by"):
        service.recommend(conditions, rank_by="profit")


def test_without_price_table_ranking_falls_back_to_yield(service, conditions):
    no_prices = YieldService(service.model, service.card, service.gains, prices=None)
    ranking = no_prices.recommend(conditions)
    assert ranking[0]["revenue_usd_ha"] is None
    yields = [r["yield_t_ha"] for r in ranking]
    assert yields == sorted(yields, reverse=True)
