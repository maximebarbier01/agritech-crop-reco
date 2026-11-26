"""Expérience MLflow 03 : des intrants FAOSTAT réels améliorent-ils le modèle ?

Question : remplacer ou compléter les variables actuelles par des intrants mesurés par
pays et par année (azote, phosphate, potasse en kg/ha, part irriguée, pesticides en kg/ha,
cf. ``data/faostat.py``) améliore-t-il le modèle de rendement ?

Protocole (identique à l'étape 2) :
- jeu d'entraînement uniquement, validation croisée ``GroupKFold`` à 5 plis **par pays** ;
  les 21 pays de test ne sont pas utilisés ;
- même modèle que le modèle servi (HistGradientBoosting, cible en log), **mêmes
  hyperparamètres** (ceux de ``models/model_card.json``) : seules les variables changent ;
- règle fixée avant l'exécution (``decide``) : une configuration n'est retenue que si son
  RMSE moyen bat celui de la référence de plus d'**un écart-type** de la référence.

Usage : ``python -m agritech_crop_reco.modeling.experiment_inputs``
"""

from __future__ import annotations

import json
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from agritech_crop_reco.config import PROJECT_ROOT, Settings, load_settings
from agritech_crop_reco.data.faostat import INPUT_COLS, INPUTS_FILE, add_inputs
from agritech_crop_reco.modeling.features import PESTICIDES, TARGET
from agritech_crop_reco.modeling.train import (
    N_FOLDS,
    _common_tags,
    _setup_mlflow,
    cross_validate,
    load_train_test,
)
from agritech_crop_reco.serving import MODEL_CARD_FILE

RESULTS_FILE = PROJECT_ROOT / "reports" / "inputs_experiment.csv"
EXPERIMENT_SUFFIX = "03-intrants-faostat"
REFERENCE = "reference"

# Variables de chaque configuration (hors culture, toujours présente)
CONFIGS = {
    REFERENCE: ["avg_temp_c", "rainfall_mm", PESTICIDES],
    "avec_intrants": ["avg_temp_c", "rainfall_mm", PESTICIDES, *INPUT_COLS],
    "intrants_sans_pesticides_totaux": ["avg_temp_c", "rainfall_mm", *INPUT_COLS],
}
LABELS = {
    REFERENCE: "Référence (modèle servi : température, pluie, pesticides totaux)",
    "avec_intrants": "+ intrants FAOSTAT (N, P, K, part irriguée, pesticides/ha)",
    "intrants_sans_pesticides_totaux": "Intrants FAOSTAT à la place des pesticides totaux",
}


def build_estimator(numeric: list[str], params: dict, seed: int):
    """Même chaîne que le modèle servi ; pesticides totaux en log ; valeurs manquantes
    laissées telles quelles (gérées nativement par HistGradientBoosting)."""
    plain = [c for c in numeric if c != PESTICIDES]
    transformers = [
        ("crop", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["crop"]),
        ("num", StandardScaler(), plain),
    ]
    if PESTICIDES in numeric:
        log_scale = Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("pest", log_scale, [PESTICIDES]))
    model = HistGradientBoostingRegressor(random_state=seed, **params)
    pipeline = Pipeline(
        [
            ("prep", ColumnTransformer(transformers, verbose_feature_names_out=False)),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)


def decide(results: pd.DataFrame) -> pd.DataFrame:
    """Règle fixée à l'avance : retenue si RMSE moyen < RMSE réf. − écart-type réf."""
    ref = results.loc[REFERENCE]
    threshold = ref["cv_rmse_mean"] - ref["cv_rmse_std"]
    out = results.assign(threshold=threshold)
    out["retained"] = (out.index != REFERENCE) & (out["cv_rmse_mean"] < threshold)
    return out


def permutation_by_fold(
    estimator, train: pd.DataFrame, features: list[str], seed: int
) -> pd.DataFrame:
    """Importance par permutation (hausse du RMSE, t/ha) sur chaque pli de validation."""
    rows = []
    for fit_idx, val_idx in GroupKFold(n_splits=N_FOLDS).split(train, groups=train["country"]):
        fit_part, val_part = train.iloc[fit_idx], train.iloc[val_idx]
        model = clone(estimator).fit(fit_part[features], fit_part[TARGET])
        perm = permutation_importance(
            model,
            val_part[features],
            val_part[TARGET],
            scoring="neg_root_mean_squared_error",
            n_repeats=5,
            random_state=seed,
            n_jobs=-1,
        )
        rows.append(pd.Series(perm.importances_mean, index=features))
    return pd.DataFrame(rows)


def run(settings: Settings) -> pd.DataFrame:
    card = json.loads((settings.models_dir / MODEL_CARD_FILE).read_text())
    params = card["best_params"]
    inputs = pd.read_csv(settings.processed_dir / INPUTS_FILE)
    train, _, dataset_sha = load_train_test(settings)  # le jeu de test n'est pas utilisé
    train = add_inputs(train, inputs)

    _setup_mlflow(settings, EXPERIMENT_SUFFIX)
    rows, importances = {}, None
    for name, numeric in CONFIGS.items():
        features = ["crop", *numeric]
        estimator = build_estimator(numeric, params, settings.seed)
        start = time.perf_counter()
        folds = cross_validate(estimator, train, features)
        summary = {f"cv_{k}_mean": v for k, v in folds.mean().items()}
        summary |= {f"cv_{k}_std": v for k, v in folds.std().items()}
        summary["cv_duration_s"] = time.perf_counter() - start
        rows[name] = summary
        if name == "avec_intrants":
            importances = permutation_by_fold(estimator, train, features, settings.seed)
        with mlflow.start_run(run_name=LABELS[name]):
            mlflow.set_tags(
                {**_common_tags(dataset_sha, "intrants-faostat"), "configuration": name}
            )
            mlflow.log_params(
                {
                    "configuration": name,
                    "features": ",".join(features),
                    "model": card["model"],
                    "hyperparametres": "ceux du modèle servi (fixes)",
                    "cv": f"GroupKFold({N_FOLDS}) par pays",
                    "source_intrants": "FAOSTAT (oct. 2026)",
                    **{f"best_{k}": v for k, v in params.items()},
                }
            )
            mlflow.log_metrics(summary)
        print(
            f"{LABELS[name]:<70} RMSE {summary['cv_rmse_mean']:.3f} ± {summary['cv_rmse_std']:.3f}"
            f"  R² {summary['cv_r2_mean']:.3f}  rang {summary['cv_ranking_spearman_mean']:.3f}"
        )

    results = decide(pd.DataFrame(rows).T)
    imp = importances.agg(["mean", "std"]).T.sort_values("mean", ascending=False)
    # Décision et importances enregistrées dans un run de synthèse
    with mlflow.start_run(run_name="Décision (règle : gain > 1 écart-type)"):
        mlflow.set_tags(
            {**_common_tags(dataset_sha, "intrants-faostat"), "configuration": "decision"}
        )
        mlflow.log_metrics(
            {
                "threshold_rmse": float(results["threshold"].iloc[0]),
                "n_retained": int(results["retained"].sum()),
                **{f"imp_{k}_mean": float(v) for k, v in imp["mean"].items()},
                **{f"imp_{k}_std": float(v) for k, v in imp["std"].items()},
            }
        )
        mlflow.log_dict(
            {
                "decision": results[["cv_rmse_mean", "cv_rmse_std", "retained"]].to_dict("index"),
                "permutation_importance_t_ha": imp.round(4).to_dict("index"),
                "inputs_correlation_spearman": train[[PESTICIDES, *INPUT_COLS]]
                .corr("spearman")
                .round(3)
                .to_dict(),
            },
            "decision.json",
        )
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    results.assign(label=[LABELS[i] for i in results.index]).to_csv(RESULTS_FILE)
    imp.to_csv(RESULTS_FILE.with_name("inputs_experiment_importance.csv"))
    retained = results.index[results["retained"]].tolist()
    print(
        f"\nSeuil : RMSE < {results['threshold'].iloc[0]:.3f} | retenue(s) : {retained or 'aucune'}"
    )
    print("Importance par permutation (+ intrants) :")
    print(imp.round(3).to_string())
    return results


def main() -> None:
    run(load_settings())


if __name__ == "__main__":
    main()
