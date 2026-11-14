"""Entraînement du modèle de rendement, suivi dans MLflow.

Deux expériences MLflow distinctes :
1. ``<prefix>-01-selection`` : référence naïve + 3 modèles × 2 cibles × 3 variantes
   pesticides, évalués en validation croisée **par pays** (GroupKFold) sur le jeu
   d'entraînement uniquement.
   Règle de choix : voir ``select_config`` (égalité statistique sur le RMSE, puis
   meilleur classement des cultures parmi les modèles dont ce classement varie).
2. ``<prefix>-02-optimisation`` : RandomizedSearchCV sur la configuration retenue
   (un run enfant par combinaison testée), puis **une seule** évaluation sur les pays
   de test figés, et sauvegarde du modèle final (.joblib + fiche JSON).

Usage :
    python -m agritech_crop_reco.modeling.train --stage all
    python -m agritech_crop_reco.modeling.train --stage selection
    python -m agritech_crop_reco.modeling.train --stage optimisation   # relit la sélection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from agritech_crop_reco.config import PROJECT_ROOT, Settings, load_settings
from agritech_crop_reco.data.merge import CONSOLIDATED_FILE
from agritech_crop_reco.modeling.features import (
    MODELS,
    PESTICIDE_VARIANTS,
    TARGET,
    TARGET_VARIANTS,
    build_model,
    feature_columns,
)
from agritech_crop_reco.modeling.metrics import all_metrics
from agritech_crop_reco.modeling.split import load_test_countries, split_train_test

N_FOLDS = 5
N_ITER_SEARCH = 30
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
N_PROBE = 300  # parcelles fictives pour mesurer si le classement des cultures varie
REPORTS_DIR = PROJECT_ROOT / "reports"
SELECTION_FILE = REPORTS_DIR / "selection_results.csv"
MODEL_FILE = "yield_model.joblib"
MODEL_CARD_FILE = "model_card.json"

PARAM_SPACES = {
    "ridge": {"alpha": loguniform(1e-3, 1e3)},
    "random_forest": {
        "n_estimators": randint(200, 700),
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": randint(1, 15),
        "max_features": [1.0, 0.5, "sqrt"],
    },
    "hist_gradient_boosting": {
        "learning_rate": loguniform(0.01, 0.3),
        "max_iter": randint(100, 800),
        "max_leaf_nodes": randint(8, 64),
        "min_samples_leaf": randint(5, 60),
        "l2_regularization": loguniform(1e-4, 10),
    },
}


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
    return {"stage": stage, "git_sha": _git_sha(), "dataset_sha256": dataset_sha[:12]}


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


def _param_prefix(target: str) -> str:
    # Pipeline -> "model__" ; cible en log -> TransformedTargetRegressor(regressor=Pipeline)
    return "regressor__model__" if target == "log" else "model__"


def run_optimisation(settings: Settings, log_artifacts: bool = True) -> dict:
    if not SELECTION_FILE.exists():
        raise FileNotFoundError(f"{SELECTION_FILE} absent : lancer d'abord --stage selection")
    selection = pd.read_csv(SELECTION_FILE)
    best = selection[selection["selected"]].iloc[0]
    name, target, pesticides = best["model"], best["target"], best["pesticides"]
    features = feature_columns(pesticides)

    train, test, dataset_sha = load_train_test(settings)
    prefix = _param_prefix(target)
    space = {prefix + k: v for k, v in PARAM_SPACES[name].items()}
    search = RandomizedSearchCV(
        build_model(name, pesticides, target, settings.seed),
        space,
        n_iter=N_ITER_SEARCH,
        cv=GroupKFold(n_splits=N_FOLDS),
        scoring="neg_root_mean_squared_error",
        random_state=settings.seed,
        n_jobs=1 if name == "random_forest" else -1,  # la forêt parallélise déjà ses arbres
        refit=True,
    )

    _setup_mlflow(settings, "02-optimisation")
    run_name = (
        f"{name} | cible={target} | pesticides={pesticides} | RandomizedSearch({N_ITER_SEARCH})"
    )
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.set_tags(_common_tags(dataset_sha, "optimisation"))
        start = time.perf_counter()
        search.fit(train[features], train[TARGET], groups=train["country"])
        duration = time.perf_counter() - start

        cv = pd.DataFrame(search.cv_results_)
        for i, row in cv.iterrows():
            with mlflow.start_run(run_name=f"essai {i:02d}", nested=True):
                mlflow.set_tag("stage", "optimisation-essai")
                mlflow.log_params(
                    {
                        k.removeprefix("param_" + prefix): v
                        for k, v in row.items()
                        if k.startswith("param_")
                    }
                )
                mlflow.log_metrics(
                    {"cv_rmse_mean": -row["mean_test_score"], "cv_rmse_std": row["std_test_score"]}
                )

        best_params = {k.removeprefix(prefix): v for k, v in search.best_params_.items()}
        cv_rmse_default = float(best["cv_rmse_mean"])
        cv_rmse_tuned = float(-search.best_score_)

        # Évaluation unique sur les pays de test figés
        test_metrics = all_metrics(test, test[TARGET], search.predict(test[features]))
        baseline = build_model("baseline", "none", "raw", settings.seed).fit(
            train[["crop"]], train[TARGET]
        )
        baseline_test = all_metrics(test, test[TARGET], baseline.predict(test[["crop"]]))

        mlflow.log_params(
            {
                "model": name,
                "target": target,
                "pesticides": pesticides,
                "features": ",".join(features),
                "n_iter": N_ITER_SEARCH,
                "cv": f"GroupKFold({N_FOLDS}) par pays",
                "seed": settings.seed,
                "n_train_rows": len(train),
                "n_test_rows": len(test),
                **{f"best_{k}": v for k, v in best_params.items()},
            }
        )
        mlflow.log_metrics(
            {
                "cv_rmse_default": cv_rmse_default,
                "cv_rmse_tuned": cv_rmse_tuned,
                "search_duration_s": duration,
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **{f"baseline_test_{k}": v for k, v in baseline_test.items()},
            }
        )

        card = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "dataset_sha256": dataset_sha,
            "mlflow_run_id": parent.info.run_id,
            "model": name,
            "target": target,
            "pesticides": pesticides,
            "features": features,
            "best_params": {
                k: (v.item() if isinstance(v, np.generic) else v) for k, v in best_params.items()
            },
            "cv_rmse_default": cv_rmse_default,
            "cv_rmse_tuned": cv_rmse_tuned,
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
        }
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(search.best_estimator_, settings.models_dir / MODEL_FILE)
        (settings.models_dir / MODEL_CARD_FILE).write_text(
            json.dumps(card, indent=2, ensure_ascii=False)
        )

        if log_artifacts:
            mlflow.log_dict(card, MODEL_CARD_FILE)
            mlflow.sklearn.log_model(
                search.best_estimator_,
                name="model",
                input_example=train[features].head(3),
                # Format skops : on n'autorise que ce type interne de HistGradientBoosting,
                # sûr car le modèle vient d'être entraîné ici (pas un fichier externe).
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )

    print(
        f"CV RMSE : défaut {cv_rmse_default:.3f} -> optimisé {cv_rmse_tuned:.3f} ({duration:.0f} s)"
    )
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
