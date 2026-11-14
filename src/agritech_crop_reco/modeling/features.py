"""Variables d'entrée et construction des modèles candidats.

Entrées du modèle = ce que l'utilisateur saisit dans l'application :
culture + conditions (température, pesticides, pluie). Ni le pays ni l'année :
l'application ne les demande pas (cf. étape 2, décision 2).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from agritech_crop_reco.modeling.baseline import CropMedianRegressor

TARGET = "yield_t_ha"
CATEGORICAL = ["crop"]
NUMERIC = ["avg_temp_c", "rainfall_mm"]
PESTICIDES = "pesticides_tonnes"

PESTICIDE_VARIANTS = ("raw", "log", "none")
TARGET_VARIANTS = ("raw", "log")
MODELS = ("ridge", "random_forest", "hist_gradient_boosting")


def feature_columns(pesticides: str) -> list[str]:
    if pesticides not in PESTICIDE_VARIANTS:
        raise ValueError(f"pesticides doit valoir {PESTICIDE_VARIANTS}, reçu {pesticides!r}")
    return CATEGORICAL + NUMERIC + ([] if pesticides == "none" else [PESTICIDES])


def make_preprocessor(pesticides: str) -> ColumnTransformer:
    """One-hot de la culture, standardisation des numériques, pesticides selon la variante."""
    transformers = [
        ("crop", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ]
    if pesticides == "raw":
        transformers.append(("pest", StandardScaler(), [PESTICIDES]))
    elif pesticides == "log":
        log_scale = Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("pest", log_scale, [PESTICIDES]))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_regressor(name: str, seed: int) -> RegressorMixin:
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f"modèle inconnu : {name!r} (attendu : {MODELS})")


def build_model(name: str, pesticides: str, target: str, seed: int) -> RegressorMixin:
    """Pipeline complet : préparation des variables -> modèle (-> cible en log si demandé).

    Avec target="log", le modèle apprend log(rendement) mais ``predict`` renvoie des t/ha :
    les métriques restent donc comparables entre variantes.
    """
    if name == "baseline":
        return CropMedianRegressor()
    pipeline = Pipeline(
        [("prep", make_preprocessor(pesticides)), ("model", make_regressor(name, seed))]
    )
    if target == "raw":
        return pipeline
    if target == "log":
        return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
    raise ValueError(f"cible doit valoir {TARGET_VARIANTS}, reçu {target!r}")
