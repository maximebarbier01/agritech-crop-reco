"""Interface Streamlit d'Agritech Answers. Tout le calcul est fait par l'API.

Lancement : ``streamlit run frontend/app.py`` (API sur http://127.0.0.1:8000,
modifiable avec la variable d'environnement ``API_URL``).
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from api_client import (
    CROP_LABELS,
    ApiError,
    api_url,
    conditions_payload,
    health,
    predict,
    recommend,
)

PRACTICE_CHOICES = {"Non renseigné (pratique moyenne)": None, "Oui": True, "Non": False}


def label(crop: str) -> str:
    return CROP_LABELS.get(crop, crop)


def sidebar_conditions() -> dict:
    st.sidebar.header("Conditions de la parcelle")
    temp = st.sidebar.number_input(
        "Température moyenne annuelle (°C)", min_value=-30.0, max_value=50.0, value=18.5, step=0.5
    )
    rain = st.sidebar.number_input(
        "Pluie annuelle (mm)", min_value=0.0, max_value=12_000.0, value=900.0, step=50.0
    )
    pesticides = st.sidebar.number_input(
        "Pesticides de la région (tonnes / an)",
        min_value=0.0,
        max_value=2_000_000.0,
        value=12_000.0,
        step=1_000.0,
        help="Indicateur du niveau d'intensification agricole de la région.",
    )
    fertilizer = PRACTICE_CHOICES[
        st.sidebar.radio("Engrais utilisé ?", list(PRACTICE_CHOICES), horizontal=False)
    ]
    irrigation = PRACTICE_CHOICES[
        st.sidebar.radio("Parcelle irriguée ?", list(PRACTICE_CHOICES), horizontal=False)
    ]
    return conditions_payload(temp, rain, pesticides, fertilizer, irrigation)


def show_warnings(warnings: list[str]) -> None:
    for message in warnings:
        st.warning(message)


def tab_predict(conditions: dict) -> None:
    crop = st.selectbox("Culture", list(CROP_LABELS), format_func=label)
    if st.button("Prédire le rendement", type="primary"):
        try:
            result = predict(crop, conditions)
        except ApiError as exc:
            st.error(str(exc))
            return
        show_warnings(result["warnings"])
        st.metric(f"Rendement estimé – {label(crop)}", f"{result['yield_t_ha']:.2f} t/ha")
        st.caption(
            f"Modèle seul (pratiques moyennes) : {result['yield_base_t_ha']:.2f} t/ha · "
            f"facteur pratiques : × {result['practice_factor']:.2f}"
        )


def tab_recommend(conditions: dict) -> None:
    if st.button("Classer les cultures", type="primary"):
        try:
            result = recommend(conditions)
        except ApiError as exc:
            st.error(str(exc))
            return
        show_warnings(result["warnings"])
        ranking = result["ranking"]
        best = ranking[0]
        st.success(f"Meilleure culture : **{label(best['crop'])}** ({best['yield_t_ha']:.2f} t/ha)")
        rows = [{"Culture": label(r["crop"]), "Rendement (t/ha)": r["yield_t_ha"]} for r in ranking]
        fig = px.bar(rows, x="Rendement (t/ha)", y="Culture", orientation="h")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(result["note"])


def main() -> None:
    st.set_page_config(page_title="Agritech Answers", page_icon="🌾", layout="wide")
    st.title("🌾 Agritech Answers")
    st.write("Quelle culture choisir pour votre parcelle, et quel rendement en attendre ?")

    try:
        info = health()
    except ApiError as exc:
        st.error(str(exc))
        st.stop()
    st.sidebar.caption(f"API : {api_url()}\n\nModèle : {info['model_version']}")

    conditions = sidebar_conditions()
    predict_tab, recommend_tab = st.tabs(["Prédire pour une culture", "Recommander une culture"])
    with predict_tab:
        tab_predict(conditions)
    with recommend_tab:
        tab_recommend(conditions)


main()
