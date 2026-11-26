"""Interface Streamlit d'Agritech Answers. Tout le calcul est fait par l'API.

Lancement : ``streamlit run frontend/app.py`` (API sur http://127.0.0.1:8000,
modifiable avec la variable d'environnement ``API_URL``).
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import streamlit as st
from api_client import (
    CROP_LABELS,
    ApiError,
    api_url,
    conditions_payload,
    health,
    metadata,
    predict,
    recommend,
)

PRACTICE_CHOICES = {"Non renseigné (pratique moyenne)": None, "Oui": True, "Non": False}
NO_COUNTRY = "Non précisé (prix mondiaux)"
RANK_CHOICES = {"Chiffre d'affaires estimé": "revenue", "Rendement": "yield"}
SOURCE_LABELS = {"saisi": "saisi", "national": "national", "mondial": "médiane mondiale"}
DEFAULT_TEMP_C = 18.5
DEFAULT_RAIN_MM = 900.0


def fr(x: float, decimals: int = 2) -> str:
    """Nombre au format français : virgule décimale, espace fine pour les milliers."""
    return f"{x:,.{decimals}f}".replace(",", "\u202f").replace(".", ",")


def label(crop: str) -> str:
    return CROP_LABELS.get(crop, crop)


def _bounds(meta: dict, name: str, step: float) -> tuple[float, float]:
    """Plage d'entraînement arrondie au pas du slider (le curseur reste dans le domaine appris)."""
    low, high = meta["training_ranges"][name]
    return math.floor(low / step) * step, math.ceil(high / step) * step


def sidebar_conditions(meta: dict) -> dict:
    st.sidebar.header("Conditions de la parcelle")
    st.sidebar.caption("Curseurs limités aux conditions observées dans les données du modèle.")

    t_min, t_max = _bounds(meta, "avg_temp_c", 0.5)
    temp = st.sidebar.slider(
        "Température moyenne annuelle (°C)",
        min_value=t_min,
        max_value=t_max,
        value=min(max(DEFAULT_TEMP_C, t_min), t_max),
        step=0.5,
    )
    r_min, r_max = _bounds(meta, "rainfall_mm", 10.0)
    rain = st.sidebar.slider(
        "Pluie annuelle (mm)",
        min_value=r_min,
        max_value=r_max,
        value=min(max(DEFAULT_RAIN_MM, r_min), r_max),
        step=10.0,
    )

    levels = {lvl["label"]: lvl["pesticides_tonnes"] for lvl in meta["pesticide_levels"]}
    level = st.sidebar.select_slider(
        "Niveau d'intensification agricole de la région",
        options=list(levels),
        value="Moyen" if "Moyen" in levels else list(levels)[len(levels) // 2],
        help=(
            "Mesuré dans les données par l'usage de pesticides de la région. Les régions "
            "intensives obtiennent de meilleurs rendements, mais c'est l'ensemble des pratiques "
            "modernes qui joue : ce n'est pas une incitation à utiliser plus de pesticides."
        ),
    )
    pesticides = levels[level]
    if st.sidebar.toggle("Saisir une valeur précise (tonnes / an)"):
        pesticides = st.sidebar.number_input(
            "Pesticides de la région (tonnes de matière active / an)",
            min_value=0.0,
            max_value=2_000_000.0,
            value=float(pesticides),
            step=100.0,
        )
    else:
        st.sidebar.caption(f"Valeur utilisée par le modèle : {pesticides:,.0f} t / an")

    fertilizer = PRACTICE_CHOICES[st.sidebar.radio("Engrais utilisé ?", list(PRACTICE_CHOICES))]
    irrigation = PRACTICE_CHOICES[st.sidebar.radio("Parcelle irriguée ?", list(PRACTICE_CHOICES))]
    return conditions_payload(temp, rain, pesticides, fertilizer, irrigation)


def sidebar_pricing(meta: dict) -> dict:
    """Pays (pour les prix uniquement) et prix de vente saisis : options envoyées à l'API."""
    st.sidebar.header("Prix de vente")
    info = meta.get("price_info") or {}
    if info:
        period = "-".join(str(y) for y in info.get("period", []))
        st.sidebar.caption(f"FAOSTAT, prix à la production {period}, en dollars par tonne.")
    country = st.sidebar.selectbox(
        "Pays de l'exploitation",
        [NO_COUNTRY, *meta.get("countries", [])],
        help="Sert uniquement à choisir les prix nationaux ; il ne change pas le rendement prédit. "
        "Sans prix national pour une culture, la médiane mondiale est utilisée.",
    )
    custom = {}
    with st.sidebar.expander("Mes prix de vente (facultatif)"):
        st.caption("0 = utiliser le prix national ou mondial.")
        for crop in CROP_LABELS:
            world = (info.get("world") or {}).get(crop, {}).get("price_usd_t")
            value = st.number_input(
                f"{label(crop)} ($/t)",
                min_value=0.0,
                value=0.0,
                step=10.0,
                key=f"price_{crop}",
                help=f"Médiane mondiale : {fr(world, 0)} $/t" if world else None,
            )
            if value > 0:
                custom[crop] = value
    options: dict = {}
    if country != NO_COUNTRY:
        options["country"] = country
    if custom:
        options["prices_usd_t"] = custom
    return options


def show_warnings(warnings: list[str]) -> None:
    for message in warnings:
        st.warning(message)


def _last_result(key: str, inputs) -> dict | None:
    """Dernier résultat calculé ; signalé comme périmé si les saisies ont changé depuis."""
    stored = st.session_state.get(key)
    if stored is None:
        return None
    if stored["inputs"] != inputs:
        st.info("Les conditions ont changé : relancez le calcul pour mettre à jour le résultat.")
        return None
    return stored["result"]


def tab_predict(conditions: dict) -> None:
    crop = st.selectbox("Culture", list(CROP_LABELS), format_func=label)
    inputs = (crop, repr(sorted(conditions.items())))
    if st.button("Prédire le rendement", type="primary"):
        try:
            st.session_state["predict"] = {"inputs": inputs, "result": predict(crop, conditions)}
        except ApiError as exc:
            st.error(str(exc))
            return
    result = _last_result("predict", inputs)
    if result is None:
        return
    show_warnings(result["warnings"])
    col_yield, col_revenue = st.columns(2)
    col_yield.metric(f"Rendement estimé – {label(crop)}", f"{fr(result['yield_t_ha'])} t/ha")
    if result.get("revenue_usd_ha") is not None:
        col_revenue.metric("Chiffre d'affaires estimé", f"{fr(result['revenue_usd_ha'], 0)} $/ha")
    st.caption(
        f"Modèle seul (pratiques moyennes) : {fr(result['yield_base_t_ha'])} t/ha · "
        f"ajustement engrais / irrigation : × {fr(result['practice_factor'])}"
        + (
            f" · prix : {fr(result['price_usd_t'], 0)} $/t "
            f"({SOURCE_LABELS.get(result['price_source'], result['price_source'])})"
            if result.get("price_usd_t") is not None
            else ""
        )
    )


REVENUE_COL = "Chiffre d'affaires estimé ($/ha)"
YIELD_COL = "Rendement estimé (t/ha)"


def ranking_table(ranking: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Rang": [r["rank"] for r in ranking],
            "Culture": [label(r["crop"]) for r in ranking],
            REVENUE_COL: [r.get("revenue_usd_ha") for r in ranking],
            YIELD_COL: [r["yield_t_ha"] for r in ranking],
            "Prix ($/t)": [r.get("price_usd_t") for r in ranking],
            "Origine du prix": [SOURCE_LABELS.get(r.get("price_source"), "–") for r in ranking],
            "Ajustement pratiques": [r["practice_factor"] for r in ranking],
        }
    )


def tab_recommend(conditions: dict) -> None:
    choice = st.segmented_control(
        "Classer par", list(RANK_CHOICES), default=list(RANK_CHOICES)[0], required=True
    )
    rank_by = RANK_CHOICES[choice]
    request = {**conditions, "rank_by": rank_by}
    inputs = repr(sorted(request.items()))
    if st.button("Classer les cultures", type="primary"):
        try:
            st.session_state["recommend"] = {"inputs": inputs, "result": recommend(request)}
        except ApiError as exc:
            st.error(str(exc))
            return
    result = _last_result("recommend", inputs)
    if result is None:
        return
    show_warnings(result["warnings"])
    table = ranking_table(result["ranking"])
    by_revenue = result.get("rank_by") == "revenue" and table[REVENUE_COL].notna().all()
    best = table.iloc[0]
    headline = (
        f"{fr(best[REVENUE_COL], 0)} $/ha estimés, {fr(best[YIELD_COL])} t/ha"
        if by_revenue
        else f"{fr(best[YIELD_COL])} t/ha"
    )
    st.success(f"Meilleure culture : **{best['Culture']}** ({headline})")
    metric = REVENUE_COL if by_revenue else YIELD_COL
    fig = px.bar(table, x=metric, y="Culture", orientation="h")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, width="stretch")
    # Tableau d'affichage : valeurs formatées en français (le classement est déjà fait)
    money = lambda v: "–" if pd.isna(v) else fr(v, 0)  # noqa: E731
    display = table.assign(
        **{
            REVENUE_COL: table[REVENUE_COL].map(money),
            YIELD_COL: table[YIELD_COL].map(fr),
            "Prix ($/t)": table["Prix ($/t)"].map(money),
            "Ajustement pratiques": table["Ajustement pratiques"].map(lambda v: f"× {fr(v)}"),
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(result["note"])


def main() -> None:
    st.set_page_config(page_title="Agritech Answers", page_icon="🌾", layout="wide")
    st.title("🌾 Agritech Answers")
    st.write("Quelle culture choisir pour votre parcelle, et quel rendement en attendre ?")

    try:
        with st.spinner("Connexion à l'API… (jusqu'à une minute si le serveur se réveille)"):
            info = health()
            meta = metadata()
    except ApiError as exc:
        st.error(str(exc))
        st.stop()
    st.sidebar.caption(f"API : {api_url()}\n\nModèle : {info['model_version']}")

    conditions = {**sidebar_conditions(meta), **sidebar_pricing(meta)}
    predict_tab, recommend_tab = st.tabs(["Prédire pour une culture", "Recommander une culture"])
    with predict_tab:
        tab_predict(conditions)
    with recommend_tab:
        tab_recommend(conditions)


main()
