"""Figures du rapport métier (public non technique, impression).

Recalculées depuis le dataset consolidé, le modèle servi et les données brutes Kaggle
(pour l'effet des pratiques) : aucune valeur n'est recopiée à la main.

Palette (validée avec le validateur de la skill dataviz, mode clair) :
- une seule teinte (bleu) pour les grandeurs ; gris neutre pour la référence ;
- bleu / rouge pour un effet positif / négatif (divergent) ;
- rampe ordinale de bleus pour les 4 combinaisons de pratiques.

Usage : python docs/rapport/make_figures.py   ->  reports/figures/rapport_*.png
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.inspection import permutation_importance

from agritech_crop_reco.config import PROJECT_ROOT, load_settings
from agritech_crop_reco.data.clean import KAGGLE_TARGET, clean_crop_yield
from agritech_crop_reco.data.load import load_crop_yield
from agritech_crop_reco.data.merge import CONSOLIDATED_FILE
from agritech_crop_reco.modeling.metrics import all_metrics
from agritech_crop_reco.modeling.train import load_train_test

OUT = PROJECT_ROOT / "reports" / "figures"

# Jetons de couleur (palette de référence dataviz, mode clair)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
RED = "#e34948"
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#104281"]

CROP_FR = {
    "Cassava": "Manioc",
    "Maize": "Maïs",
    "Plantains and others": "Plantains",
    "Potatoes": "Pomme de terre",
    "Rice": "Riz",
    "Sorghum": "Sorgho",
    "Soybean": "Soja",
    "Sweet potatoes": "Patate douce",
    "Wheat": "Blé",
    "Yams": "Igname",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def fr(x: float, decimals: int = 1) -> str:
    """Nombre au format français (virgule décimale, espace des milliers)."""
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def style_hbar(ax, xlabel: str) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    # Graduations au format français (virgule décimale)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: fr(x, 0) if float(x).is_integer() else fr(x, 1))
    )


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"rapport_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("écrit :", OUT / f"rapport_{name}.png")


def hbar(ax, labels, values, colors, fmt) -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    span = max(abs(v) for v in values)
    for yi, v in zip(y, values, strict=True):
        ax.text(
            v + span * 0.015 * (1 if v >= 0 else -1),
            yi,
            fmt(v),
            va="center",
            ha="left" if v >= 0 else "right",
            color=INK,
            fontsize=9,
        )


# ---------------------------------------------------------------------------------------------


def fig_rendement_par_culture(df: pd.DataFrame) -> None:
    med = df.groupby("crop")["yield_t_ha"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hbar(ax, [CROP_FR[c] for c in med.index], med.values, BLUE, lambda v: f"{fr(v)} t/ha")
    ax.set_xlim(0, med.max() * 1.18)
    style_hbar(ax, "Rendement médian observé (tonnes par hectare)")
    save(fig, "01_rendement_culture")


def fig_pratiques(settings) -> None:
    crop = clean_crop_yield(load_crop_yield(settings.raw_dir))
    m = crop.groupby(["Fertilizer_Used", "Irrigation_Used"])[KAGGLE_TARGET].mean()
    combos = [
        ("Ni engrais ni irrigation", m[(False, False)]),
        ("Irrigation seule", m[(False, True)]),
        ("Engrais seul", m[(True, False)]),
        ("Engrais + irrigation", m[(True, True)]),
    ]
    base = combos[0][1]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    labels = [c[0] for c in combos]
    values = [c[1] for c in combos]
    hbar(
        ax,
        labels,
        values,
        RAMP,
        lambda v: f"{fr(v)} t/ha" + ("" if v == base else f"  (+{fr((v / base - 1) * 100, 0)} %)"),
    )
    ax.set_xlim(0, max(values) * 1.35)
    style_hbar(ax, "Rendement moyen (t/ha), 1 million de parcelles simulées")
    save(fig, "02_pratiques")


def fig_importance(model, test, features) -> None:
    perm = permutation_importance(
        model,
        test[features],
        test["yield_t_ha"],
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    names = {
        "crop": "Culture",
        "avg_temp_c": "Température",
        "pesticides_tonnes": "Intensification agricole",
        "rainfall_mm": "Pluie annuelle",
    }
    imp = pd.Series(perm.importances_mean, index=[names[f] for f in features]).sort_values(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(7, 2.4))
    hbar(ax, imp.index, imp.values, BLUE, lambda v: f"+{fr(v)} t/ha")
    ax.set_xlim(0, imp.max() * 1.2)
    style_hbar(ax, "Erreur supplémentaire si l'on brouille la variable (t/ha)")
    save(fig, "03_importance")


def fig_chaleur(model, train, features) -> dict:
    med = train[["avg_temp_c", "rainfall_mm", "pesticides_tonnes"]].median()
    grid = np.linspace(train["avg_temp_c"].quantile(0.02), train["avg_temp_c"].quantile(0.98), 40)
    effect = {}
    for crop in sorted(train["crop"].unique()):
        X = pd.DataFrame({**med.to_dict(), "avg_temp_c": grid, "crop": crop})[features]
        y = model.predict(X)
        effect[CROP_FR[crop]] = (y[-1] / y[0] - 1) * 100
    s = pd.Series(effect).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [BLUE if v >= 0 else RED for v in s.values]
    hbar(ax, s.index, s.values, colors, lambda v: f"{'+' if v >= 0 else '−'}{fr(abs(v), 0)} %")
    ax.axvline(0, color=INK_2, linewidth=1)
    lim = max(abs(s.min()), abs(s.max())) * 1.3
    ax.set_xlim(-lim, lim)
    style_hbar(
        ax, f"Variation du rendement prédit entre {fr(grid[0])} °C et {fr(grid[-1])} °C de moyenne"
    )
    save(fig, "04_chaleur")
    return {"t_min": grid[0], "t_max": grid[-1], **effect}


def fig_performance(card: dict) -> None:
    ref, mod = card["baseline_test_metrics"], card["test_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.2))
    panels = [
        ("Erreur moyenne (t/ha)\nplus bas = mieux", "mae", lambda v: f"{fr(v, 2)} t/ha"),
        (
            "Bon ordre des cultures (0 à 1)\nplus haut = mieux",
            "ranking_spearman",
            lambda v: fr(v, 2),
        ),
    ]
    for ax, (title, key, fmt) in zip(axes, panels, strict=True):
        vals = [ref[key], mod[key]]
        hbar(ax, ["Référence :\nmédiane de la culture", "Modèle"], vals, [MUTED, BLUE], fmt)
        ax.set_xlim(0, max(vals) * 1.45)
        style_hbar(ax, "")
        ax.set_title(title, fontsize=9.5, color=INK_2, loc="left")
    axes[1].set_yticklabels([])
    fig.tight_layout(w_pad=2)
    save(fig, "05_performance")


def fig_erreur_culture(model, test, features) -> pd.Series:
    t = test.assign(pred=model.predict(test[features]))
    t["abs_err"] = (t["pred"] - t["yield_t_ha"]).abs()
    rel = (
        t.groupby("crop")["abs_err"].mean() / t.groupby("crop")["yield_t_ha"].mean() * 100
    ).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hbar(ax, [CROP_FR[c] for c in rel.index], rel.values, BLUE, lambda v: f"{fr(v, 0)} %")
    ax.set_xlim(0, rel.max() * 1.18)
    style_hbar(ax, "Écart moyen entre prédiction et réalité, en % du rendement de la culture")
    save(fig, "06_erreur_culture")
    return rel


def fig_profils(model, features) -> dict:
    """Mêmes 10 cultures, même échelle pour les 3 profils : on compare d'un climat à l'autre."""
    profiles = {
        "Tempéré, très intensif\n11 °C · 800 mm/an": (11, 800, 50_000),
        "Tropical humide, peu intensif\n26 °C · 2 000 mm/an": (26, 2000, 500),
        "Chaud et sec, intensif moyen\n24 °C · 400 mm/an": (24, 400, 5_000),
    }
    crops = list(CROP_FR)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharex=True)
    ranks = {}
    for ax, (name, (t, r, p)) in zip(axes, profiles.items(), strict=True):
        X = pd.DataFrame(
            [{"crop": c, "avg_temp_c": t, "rainfall_mm": r, "pesticides_tonnes": p} for c in crops]
        )[features]
        s = pd.Series(model.predict(X), index=crops).sort_values(ascending=False)
        ranks[name.split("\n")[0]] = [CROP_FR[c] for c in s.index]
        colors = [BLUE] + [BLUE_LIGHT] * (len(s) - 1)
        hbar(ax, [CROP_FR[c] for c in s.index], s.values, colors, lambda v: fr(v))
        style_hbar(ax, "t/ha")
        ax.set_title(name, fontsize=9, color=INK_2, loc="left")
    axes[0].set_xlim(0, 32)
    fig.tight_layout(w_pad=1.0)
    save(fig, "07_profils")
    return ranks


def main() -> None:
    settings = load_settings()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(settings.processed_dir / CONSOLIDATED_FILE)
    card = json.loads((settings.models_dir / "model_card.json").read_text())
    model = joblib.load(settings.models_dir / "yield_model.joblib")
    features = card["features"]
    train, test, _ = load_train_test(settings)

    fig_rendement_par_culture(df)
    fig_pratiques(settings)
    fig_importance(model, test, features)
    heat = fig_chaleur(model, train, features)
    fig_performance(card)
    rel = fig_erreur_culture(model, test, features)
    tops = fig_profils(model, features)

    # Contrôle de cohérence : les métriques de la fiche sont-elles retrouvées ?
    check = all_metrics(test, test["yield_t_ha"], model.predict(test[features]))
    assert abs(check["rmse"] - card["test_metrics"]["rmse"]) < 1e-9
    print("chaleur :", {k: round(v, 1) for k, v in heat.items()})
    print("erreur relative :", rel.round(1).to_dict())
    print("profils :", tops)


if __name__ == "__main__":
    main()
