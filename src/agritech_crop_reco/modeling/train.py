"""Entraînement du modèle de rendement, suivi dans MLflow.

Deux expériences MLflow distinctes :
1. ``<prefix>-01-selection`` : référence naïve + 5 modèles (Ridge, Random Forest,
   HistGradientBoosting, XGBoost, CatBoost) × 2 cibles × 3 variantes pesticides,
   évalués avec leurs réglages par défaut en validation croisée **par pays**
   (GroupKFold) sur le jeu d'entraînement uniquement.
   Règle de choix : voir ``select_config`` (égalité statistique sur le RMSE, puis
   meilleur classement des cultures parmi les modèles dont ce classement varie).
2. ``<prefix>-02-optimisation`` : chacune des 3 familles de boosting (sa meilleure
   variante à l'étape 1) est optimisée par **deux méthodes à budget égal** :
   RandomizedSearchCV (tirage au hasard) et Optuna (TPE, qui apprend des essais
   précédents). Les 6 candidats optimisés sont départagés par la **même règle**
   ``select_config`` ; seul le gagnant est évalué, **une seule fois**, sur les pays de
   test figés, puis sauvegardé (.joblib + fiche JSON).

Usage :
    python -m agritech_crop_reco.modeling.train --stage all
    python -m agritech_crop_reco.modeling.train --stage selection
    python -m agritech_crop_reco.modeling.train --stage optimisation   # relit la sélection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from agritech_crop_reco.config import PROJECT_ROOT, Settings, load_settings
from agritech_crop_reco.data.merge import CONSOLIDATED_FILE, GAIN_COLS
from agritech_crop_reco.modeling.features import (
    BOOSTING,
    MODELS,
    PESTICIDE_VARIANTS,
    TARGET,
    TARGET_VARIANTS,
    build_model,
    feature_columns,
)
from agritech_crop_reco.modeling.metrics import all_metrics
from agritech_crop_reco.modeling.split import load_test_countries, split_train_test
from agritech_crop_reco.serving import GAINS_FILE, MODEL_CARD_FILE, MODEL_FILE

N_FOLDS = 5
# Budget identique pour les deux méthodes : la comparaison porte sur la façon de
# chercher, pas sur le nombre d'essais.
N_ITER_SEARCH = 30
N_TRIALS_OPTUNA = 30
SEARCH_METHODS = ("random_search", "optuna")
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
N_PROBE = 300  # parcelles fictives pour mesurer si le classement des cultures varie
# Niveaux d'intensification affichés dans l'interface (quantiles des pesticides à l'entraînement)
PESTICIDE_LEVELS = {
    "Très faible": 0.1,
    "Faible": 0.3,
    "Moyen": 0.5,
    "Élevé": 0.7,
    "Très élevé": 0.9,
}
REPORTS_DIR = PROJECT_ROOT / "reports"
SELECTION_FILE = REPORTS_DIR / "selection_results.csv"
OPTIMISATION_FILE = REPORTS_DIR / "optimisation_results.csv"

# Espaces de recherche, communs aux deux méthodes : (type, bas, haut), bornes incluses.
# « log » = tirage uniforme sur l'échelle logarithmique (taux d'apprentissage, régularisation).
# HistGradientBoosting : mêmes plages qu'en v1.0 (le tirage au hasard est reproduit à l'identique).
SEARCH_SPACES = {
    "hist_gradient_boosting": {
        "learning_rate": ("log", 0.01, 0.3),
        "max_iter": ("int", 100, 799),
        "max_leaf_nodes": ("int", 8, 63),
        "min_samples_leaf": ("int", 5, 59),
        "l2_regularization": ("log", 1e-4, 10),
    },
    "xgboost": {
        "n_estimators": ("int", 100, 800),
        "learning_rate": ("log", 0.01, 0.3),
        "max_depth": ("int", 3, 10),
        "min_child_weight": ("log", 1, 20),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_lambda": ("log", 1e-3, 10),
    },
    "catboost": {
        "iterations": ("int", 200, 1000),
        "learning_rate": ("log", 0.01, 0.3),
        "depth": ("int", 4, 8),
        "l2_leaf_reg": ("log", 1, 10),
    },
}


def scipy_space(name: str) -> dict:
    """Espace de recherche au format RandomizedSearchCV (lois scipy)."""
    laws = {
        "log": lambda lo, hi: loguniform(lo, hi),
        "int": lambda lo, hi: randint(lo, hi + 1),
        "float": lambda lo, hi: uniform(lo, hi - lo),
    }
    return {k: laws[kind](lo, hi) for k, (kind, lo, hi) in SEARCH_SPACES[name].items()}


def suggest_params(trial: optuna.Trial, name: str) -> dict:
    """Même espace de recherche, proposé par Optuna."""
    params = {}
    for k, (kind, lo, hi) in SEARCH_SPACES[name].items():
        if kind == "int":
            params[k] = trial.suggest_int(k, lo, hi)
        else:
            params[k] = trial.suggest_float(k, lo, hi, log=kind == "log")
    return params


# --------------------------------------------------------------------------- utilitaires


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train_test(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    path = settings.processed_dir / CONSOLIDATED_FILE
    df = pd.read_csv(path)
    train, test = split_train_test(df, load_test_countries())
    return train, test, _sha256(path)


def _common_tags(dataset_sha: str, stage: str) -> dict[str, str]:
    return {
        "stage": stage,
        "git_sha": _git_sha(),
        "dataset_sha256": dataset_sha[:12],
        # Distingue dans MLflow les runs de la CI (GitHub Actions) des runs locaux
        "source": "ci" if os.getenv("GITHUB_ACTIONS") == "true" else "local",
    }


def _setup_mlflow(settings: Settings, suffix: str) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(f"{settings.mlflow_experiment}-{suffix}")


def cross_validate(
    estimator, train: pd.DataFrame, features: list[str], n_splits: int = N_FOLDS
) -> pd.DataFrame:
    """Validation croisée groupée par pays : un pays n'est jamais à la fois en
    apprentissage et en validation. Retourne les métriques de chaque pli."""
    folds = []
    for train_idx, val_idx in GroupKFold(n_splits=n_splits).split(train, groups=train["country"]):
        fit_part, val_part = train.iloc[train_idx], train.iloc[val_idx]
        model = clone(estimator).fit(fit_part[features], fit_part[TARGET])
        folds.append(all_metrics(val_part, val_part[TARGET], model.predict(val_part[features])))
    return pd.DataFrame(folds)


# --------------------------------------------------------------------------- étape 1


def candidate_configs() -> list[dict[str, str]]:
    configs = [{"model": "baseline", "target": "raw", "pesticides": "none"}]
    configs += [
        {"model": m, "target": t, "pesticides": p}
        for m in MODELS
        for t in TARGET_VARIANTS
        for p in PESTICIDE_VARIANTS
    ]
    return configs


def run_selection(settings: Settings) -> pd.DataFrame:
    train, _, dataset_sha = load_train_test(settings)
    probe = probe_conditions(train, settings.seed)
    _setup_mlflow(settings, "01-selection")
    rows = []
    for cfg in candidate_configs():
        features = ["crop"] if cfg["model"] == "baseline" else feature_columns(cfg["pesticides"])
        estimator = build_model(cfg["model"], cfg["pesticides"], cfg["target"], settings.seed)
        run_name = f"{cfg['model']} | cible={cfg['target']} | pesticides={cfg['pesticides']}"
        start = time.perf_counter()
        folds = cross_validate(estimator, train, features)
        duration = time.perf_counter() - start

        summary = {f"cv_{k}_mean": v for k, v in folds.mean().items()}
        summary |= {f"cv_{k}_std": v for k, v in folds.std().items()}
        fitted = clone(estimator).fit(train[features], train[TARGET])
        summary["ranking_variety"] = ranking_variety(fitted, features, probe)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(_common_tags(dataset_sha, "selection"))
            mlflow.log_params(
                {
                    **cfg,
                    "features": ",".join(features),
                    "cv": f"GroupKFold({N_FOLDS}) par pays",
                    "n_train_rows": len(train),
                    "seed": settings.seed,
                }
            )
            mlflow.log_metrics({**summary, "cv_duration_s": duration})
        rows.append({**cfg, **summary, "cv_duration_s": duration})
        print(
            f"{run_name:<60} RMSE {summary['cv_rmse_mean']:.3f} ± {summary['cv_rmse_std']:.3f}"
            f"  R² {summary['cv_r2_mean']:.3f}  rang {summary['cv_ranking_spearman_mean']:.3f}"
            f"  variété {summary['ranking_variety']:.0f}"
        )

    results = pd.DataFrame(rows)
    results["selected"] = False
    results.loc[select_config(results), "selected"] = True
    results = results.sort_values("cv_rmse_mean").reset_index(drop=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(SELECTION_FILE, index=False)
    best = results[results["selected"]].iloc[0]
    print(f"\nRetenu : {best['model']} | cible={best['target']} | pesticides={best['pesticides']}")
    return results


def probe_conditions(train: pd.DataFrame, seed: int, n: int = N_PROBE) -> pd.DataFrame:
    """Parcelles fictives tirées dans les plages observées (pesticides en échelle log)."""
    rng = np.random.default_rng(seed)
    pest = train["pesticides_tonnes"].clip(lower=1)
    return pd.DataFrame(
        {
            "avg_temp_c": rng.uniform(train["avg_temp_c"].min(), train["avg_temp_c"].max(), n),
            "rainfall_mm": rng.uniform(train["rainfall_mm"].min(), train["rainfall_mm"].max(), n),
            "pesticides_tonnes": np.exp(rng.uniform(np.log(pest.min()), np.log(pest.max()), n)),
        }
    ).assign(_crops=[sorted(train["crop"].unique())] * n)


def ranking_variety(model, features: list[str], probe: pd.DataFrame) -> int:
    """Nombre de classements de cultures différents produits sur les parcelles fictives.

    1 = le modèle recommande toujours le même ordre, quelles que soient les conditions :
    il est alors inutilisable pour /recommend (cas d'un modèle additif comme Ridge).
    """
    grid = probe.reset_index(names="parcel").explode("_crops").rename(columns={"_crops": "crop"})
    grid["pred"] = model.predict(grid[features])
    orders = grid.sort_values(["parcel", "pred"], ascending=[True, False]).groupby("parcel")["crop"]
    return int(orders.agg(tuple).nunique())


def select_config(results: pd.DataFrame) -> int:
    """Règle de choix (complétée après la 1re sélection, cf. rapport) :

    1. candidats = modèles dont le RMSE CV est à moins d'un écart-type du meilleur
       (écarts plus petits que la variabilité entre plis = égalité statistique) ;
    2. on écarte ceux qui produisent toujours le même classement (ranking_variety == 1) ;
    3. on retient le meilleur ranking_spearman (métrique métier de /recommend) ;
       à égalité, le RMSE le plus bas, puis pesticides en log (identique pour les arbres,
       cohérent avec les modèles linéaires).
    Retourne l'index de la ligne retenue.
    """
    models = results[results["model"] != "baseline"]
    best = models.loc[models["cv_rmse_mean"].idxmin()]
    pool = models[models["cv_rmse_mean"] <= best["cv_rmse_mean"] + best["cv_rmse_std"]]
    pool = pool[pool["ranking_variety"] > 1]
    if pool.empty:
        raise ValueError(
            "Aucun modèle proche du meilleur ne fait varier le classement des cultures"
        )
    pest_order = pool["pesticides"].map({"log": 0, "raw": 1, "none": 2})
    ranked = pool.assign(_pest=pest_order).sort_values(
        ["cv_ranking_spearman_mean", "cv_rmse_mean", "_pest"], ascending=[False, True, True]
    )
    return ranked.index[0]


# --------------------------------------------------------------------------- étape 2


def pesticide_levels(train: pd.DataFrame) -> list[dict]:
    """Valeurs de pesticides (tonnes) correspondant à chaque niveau d'intensification.

    Un agriculteur ne connaît pas le tonnage national : l'interface propose 5 niveaux,
    traduits en tonnes par les quantiles observés sur les pays d'entraînement.
    """
    return [
        {
            "label": label,
            "quantile": q,
            "pesticides_tonnes": round(float(train["pesticides_tonnes"].quantile(q)), 1),
        }
        for label, q in PESTICIDE_LEVELS.items()
    ]


def write_practice_gains(settings: Settings) -> Path:
    """Exporte les gains engrais / irrigation par culture (issus du dataset consolidé)
    à côté du modèle : l'API n'a ainsi besoin d'aucun fichier de données."""
    df = pd.read_csv(settings.processed_dir / CONSOLIDATED_FILE)
    per_crop = df.groupby("crop")[GAIN_COLS + ["gain_source"]].first()
    payload = {crop: row.to_dict() for crop, row in per_crop.iterrows()}
    path = settings.models_dir / GAINS_FILE
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def _param_prefix(target: str) -> str:
    # Pipeline -> "model__" ; cible en log -> TransformedTargetRegressor(regressor=Pipeline)
    return "regressor__model__" if target == "log" else "model__"


def _plain(value):
    """Valeur numpy -> valeur Python (JSON, MLflow)."""
    return value.item() if isinstance(value, np.generic) else value


def family_starting_points(selection: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque famille de boosting, sa variante (cible, pesticides) retenue à l'étape 1
    par la même règle ``select_config`` appliquée à cette seule famille."""
    rows = []
    for name in BOOSTING:
        family = selection[selection["model"] == name]
        if family.empty:
            raise ValueError(f"Aucun résultat de sélection pour {name}")
        rows.append(family.loc[select_config(family)])
    return pd.DataFrame(rows).reset_index(drop=True)


def random_search(estimator, name: str, prefix: str, train: pd.DataFrame, features, seed: int):
    """RandomizedSearchCV : essais tirés au hasard, indépendants les uns des autres."""
    search = RandomizedSearchCV(
        estimator,
        {prefix + k: v for k, v in scipy_space(name).items()},
        n_iter=N_ITER_SEARCH,
        cv=GroupKFold(n_splits=N_FOLDS),
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        n_jobs=-1 if name == "hist_gradient_boosting" else 1,  # XGBoost/CatBoost : 4 fils
        refit=False,
    )
    search.fit(train[features], train[TARGET], groups=train["country"])
    cv = pd.DataFrame(search.cv_results_)
    trials = [
        {
            "params": {
                k.removeprefix("param_" + prefix): _plain(v)
                for k, v in row.items()
                if k.startswith("param_")
            },
            "cv_rmse_mean": -row["mean_test_score"],
            "cv_rmse_std": row["std_test_score"],
        }
        for _, row in cv.iterrows()
    ]
    best = {k.removeprefix(prefix): _plain(v) for k, v in search.best_params_.items()}
    return best, trials


def optuna_search(estimator, name: str, prefix: str, train: pd.DataFrame, features, seed: int):
    """Optuna (échantillonneur TPE) : chaque essai est choisi d'après les précédents."""
    trials = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, name)
        model = clone(estimator).set_params(**{prefix + k: v for k, v in params.items()})
        folds = cross_validate(model, train, features)
        rmse, std = float(folds["rmse"].mean()), float(folds["rmse"].std())
        trials.append({"params": params, "cv_rmse_mean": rmse, "cv_rmse_std": std})
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA)
    return study.best_params, trials


def run_optimisation(settings: Settings, log_artifacts: bool = True) -> dict:
    if not SELECTION_FILE.exists():
        raise FileNotFoundError(f"{SELECTION_FILE} absent : lancer d'abord --stage selection")
    starts = family_starting_points(pd.read_csv(SELECTION_FILE))
    train, test, dataset_sha = load_train_test(settings)
    probe = probe_conditions(train, settings.seed)
    searches = {"random_search": random_search, "optuna": optuna_search}
    budgets = {"random_search": N_ITER_SEARCH, "optuna": N_TRIALS_OPTUNA}

    _setup_mlflow(settings, "02-optimisation")
    candidates, fitted = [], {}
    for _, start_row in starts.iterrows():
        name, target, pesticides = start_row["model"], start_row["target"], start_row["pesticides"]
        features = feature_columns(pesticides)
        prefix = _param_prefix(target)
        base = build_model(name, pesticides, target, settings.seed)
        for method in SEARCH_METHODS:
            label = (
                f"{name} | cible={target} | pesticides={pesticides} | "
                f"{method}({budgets[method]})"
            )
            with mlflow.start_run(run_name=label):
                mlflow.set_tags(
                    {**_common_tags(dataset_sha, "optimisation"), "search_method": method}
                )
                start = time.perf_counter()
                best_params, trials = searches[method](
                    base, name, prefix, train, features, settings.seed
                )
                duration = time.perf_counter() - start
                for i, trial in enumerate(trials):
                    with mlflow.start_run(run_name=f"essai {i:02d}", nested=True):
                        mlflow.set_tags({"stage": "optimisation-essai", "search_method": method})
                        mlflow.log_params(trial["params"])
                        mlflow.log_metrics(
                            {
                                "cv_rmse_mean": trial["cv_rmse_mean"],
                                "cv_rmse_std": trial["cv_rmse_std"],
                            }
                        )

                # Le meilleur réglage est réévalué avec toutes les métriques (mêmes plis)
                model = clone(base).set_params(**{prefix + k: v for k, v in best_params.items()})
                folds = cross_validate(model, train, features)
                summary = {f"cv_{k}_mean": v for k, v in folds.mean().items()}
                summary |= {f"cv_{k}_std": v for k, v in folds.std().items()}
                model.fit(train[features], train[TARGET])
                summary["ranking_variety"] = ranking_variety(model, features, probe)
                mlflow.log_params(
                    {
                        "model": name,
                        "target": target,
                        "pesticides": pesticides,
                        "search_method": method,
                        "n_trials": budgets[method],
                        "cv": f"GroupKFold({N_FOLDS}) par pays",
                        **{f"best_{k}": v for k, v in best_params.items()},
                    }
                )
                mlflow.log_metrics(
                    {
                        **summary,
                        "cv_rmse_default": float(start_row["cv_rmse_mean"]),
                        "search_duration_s": duration,
                    }
                )
            key = len(candidates)
            fitted[key] = (model, features)
            candidates.append(
                {
                    "model": name,
                    "search_method": method,
                    "target": target,
                    "pesticides": pesticides,
                    "n_trials": budgets[method],
                    "cv_rmse_default": float(start_row["cv_rmse_mean"]),
                    **summary,
                    "search_duration_s": duration,
                    "best_params": json.dumps(best_params),
                }
            )
            print(
                f"{label:<75} RMSE {summary['cv_rmse_mean']:.3f} ± {summary['cv_rmse_std']:.3f}"
                f"  rang {summary['cv_ranking_spearman_mean']:.3f}"
                f"  variété {summary['ranking_variety']:.0f}  ({duration:.0f} s)"
            )

    # Même règle qu'à l'étape 1, appliquée aux 6 candidats optimisés
    results = pd.DataFrame(candidates)
    chosen_idx = select_config(results)
    results["selected"] = results.index == chosen_idx
    REPORTS_DIR.mkdir(exist_ok=True)
    results.sort_values("cv_rmse_mean").to_csv(OPTIMISATION_FILE, index=False)
    chosen = results.loc[chosen_idx]
    final_model, features = fitted[chosen_idx]
    name, target, pesticides = chosen["model"], chosen["target"], chosen["pesticides"]
    best_params = json.loads(chosen["best_params"])

    # Évaluation unique du gagnant sur les pays de test figés
    test_metrics = all_metrics(test, test[TARGET], final_model.predict(test[features]))
    baseline = build_model("baseline", "none", "raw", settings.seed).fit(
        train[["crop"]], train[TARGET]
    )
    baseline_test = all_metrics(test, test[TARGET], baseline.predict(test[["crop"]]))

    run_name = f"Décision : {name} | {chosen['search_method']} (règle select_config)"
    with mlflow.start_run(run_name=run_name) as decision:
        mlflow.set_tags({**_common_tags(dataset_sha, "optimisation"), "configuration": "decision"})
        mlflow.log_params(
            {
                "model": name,
                "target": target,
                "pesticides": pesticides,
                "search_method": chosen["search_method"],
                "n_trials": int(chosen["n_trials"]),
                "features": ",".join(features),
                "cv": f"GroupKFold({N_FOLDS}) par pays",
                "seed": settings.seed,
                "n_train_rows": len(train),
                "n_test_rows": len(test),
                **{f"best_{k}": v for k, v in best_params.items()},
            }
        )
        mlflow.log_metrics(
            {
                "cv_rmse_default": float(chosen["cv_rmse_default"]),
                "cv_rmse_tuned": float(chosen["cv_rmse_mean"]),
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"baseline_test_{k}": v for k, v in baseline_test.items()},
            }
        )
        mlflow.log_dict(
            results.drop(columns="best_params").round(4).to_dict("records"),
            "candidats_optimises.json",
        )

        card = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "dataset_sha256": dataset_sha,
            "mlflow_run_id": decision.info.run_id,
            "model": name,
            "target": target,
            "pesticides": pesticides,
            "features": features,
            "search_method": chosen["search_method"],
            "n_trials": int(chosen["n_trials"]),
            "best_params": best_params,
            "cv_rmse_default": float(chosen["cv_rmse_default"]),
            "cv_rmse_tuned": float(chosen["cv_rmse_mean"]),
            "optimisation_candidates": [
                {
                    "model": r["model"],
                    "search_method": r["search_method"],
                    "cv_rmse_mean": round(float(r["cv_rmse_mean"]), 4),
                    "cv_rmse_std": round(float(r["cv_rmse_std"]), 4),
                    "cv_ranking_spearman_mean": round(float(r["cv_ranking_spearman_mean"]), 4),
                    "ranking_variety": int(r["ranking_variety"]),
                }
                for _, r in results.sort_values("cv_rmse_mean").iterrows()
            ],
            "test_metrics": test_metrics,
            "baseline_test_metrics": baseline_test,
            "n_train_rows": len(train),
            "n_test_rows": len(test),
            "training_ranges": {
                col: [float(train[col].min()), float(train[col].max())]
                for col in features
                if col != "crop"
            },
            "crops": sorted(train["crop"].unique()),
            "pesticide_levels": pesticide_levels(train),
        }
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, settings.models_dir / MODEL_FILE)
        write_practice_gains(settings)
        (settings.models_dir / MODEL_CARD_FILE).write_text(
            json.dumps(card, indent=2, ensure_ascii=False)
        )

        if log_artifacts:
            mlflow.log_dict(card, MODEL_CARD_FILE)
            # skops n'accepte que des types déclarés sûrs : on n'autorise que l'arbre interne
            # de HistGradientBoosting (modèle entraîné ici, pas un fichier externe) ;
            # XGBoost / CatBoost : format cloudpickle.
            fmt = (
                {"skops_trusted_types": SKOPS_TRUSTED_TYPES}
                if name == "hist_gradient_boosting"
                else {"serialization_format": "cloudpickle"}
            )
            mlflow.sklearn.log_model(
                final_model, name="model", input_example=train[features].head(3), **fmt
            )

    print(f"\nRetenu : {name} | {chosen['search_method']} | CV RMSE {chosen['cv_rmse_mean']:.3f}")
    print(f"TEST    : {test_metrics}")
    print(f"Référence naïve (test) : {baseline_test}")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--stage", choices=["selection", "optimisation", "all"], default="all")
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="ne pas envoyer de fichiers à MLflow (CI : métadonnées seules)",
    )
    args = parser.parse_args()
    settings = load_settings()
    if args.stage in ("selection", "all"):
        run_selection(settings)
    if args.stage in ("optimisation", "all"):
        run_optimisation(settings, log_artifacts=not args.no_artifacts)


if __name__ == "__main__":
    main()
