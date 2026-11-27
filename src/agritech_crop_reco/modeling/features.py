"""Variables d'entrée et construction des modèles candidats.

Entrées du modèle = ce que l'utilisateur saisit dans l'application :
culture + conditions (température, pesticides, pluie). Ni le pays ni l'année :
l'application ne les demande pas (cf. étape 2, décision 2).

XGBoost et CatBoost sont importés seulement à la construction du modèle : l'API, qui ne
fait que charger le modèle servi, n'en a besoin que si l'un d'eux est retenu.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
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
MODELS = ("ridge", "random_forest", "hist_gradient_boosting", "xgboost", "catboost")
BOOSTING = ("hist_gradient_boosting", "xgboost", "catboost")
# Petits jeux de données : au-delà de 4 fils, XGBoost passe plus de temps à se
# synchroniser qu'à calculer (mesuré : 0,2 s avec 4 fils contre 5 s avec 16).
N_THREADS = 4


def feature_columns(pesticides: str) -> list[str]:
    if pesticides not in PESTICIDE_VARIANTS:
        raise ValueError(f"pesticides doit valoir {PESTICIDE_VARIANTS}, reçu {pesticides!r}")
    return CATEGORICAL + NUMERIC + ([] if pesticides == "none" else [PESTICIDES])


def make_preprocessor(pesticides: str, native_categorical: bool = False) -> ColumnTransformer:
    """One-hot de la culture, standardisation des numériques, pesticides selon la variante.

    ``native_categorical=True`` (CatBoost) : la culture est transmise telle quelle, en
    texte, et la sortie reste un DataFrame pour que CatBoost la traite en catégorie.
    """
    crop = (
        "passthrough"
        if native_categorical
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    transformers = [
        ("crop", crop, CATEGORICAL),
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
    prep = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return prep.set_output(transform="pandas") if native_categorical else prep


class CatBoostCropRegressor(RegressorMixin, BaseEstimator):
    """Adaptateur scikit-learn de CatBoost, la culture traitée en catégorie native.

    ``CatBoostRegressor(cat_features=...)`` ne supporte pas ``sklearn.clone`` (il recopie
    la liste reçue), or la validation croisée et les recherches d'hyperparamètres clonent
    le modèle. L'adaptateur n'expose que des paramètres simples et crée CatBoost au ``fit``.
    Valeurs par défaut = celles de CatBoost (``learning_rate=None`` : choisi par CatBoost).
    """

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float | None = None,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 0,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            cat_features=[c for c in CATEGORICAL if c in X.columns],
            thread_count=N_THREADS,
            verbose=0,
            allow_writing_files=False,
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def make_regressor(name: str, seed: int) -> RegressorMixin:
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(tree_method="hist", n_jobs=N_THREADS, random_state=seed)
    if name == "catboost":
        return CatBoostCropRegressor(random_seed=seed)
    raise ValueError(f"modèle inconnu : {name!r} (attendu : {MODELS})")


def build_model(name: str, pesticides: str, target: str, seed: int) -> RegressorMixin:
    """Pipeline complet : préparation des variables -> modèle (-> cible en log si demandé).

    Avec target="log", le modèle apprend log(rendement) mais ``predict`` renvoie des t/ha :
    les métriques restent donc comparables entre variantes.
    """
    if name == "baseline":
        return CropMedianRegressor()
    pipeline = Pipeline(
        [
            ("prep", make_preprocessor(pesticides, native_categorical=name == "catboost")),
            ("model", make_regressor(name, seed)),
        ]
    )
    if target == "raw":
        return pipeline
    if target == "log":
        return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)
    raise ValueError(f"cible doit valoir {TARGET_VARIANTS}, reçu {target!r}")
