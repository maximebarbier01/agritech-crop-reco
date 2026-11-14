"""Métriques de régression (t/ha) et métrique métier de classement des cultures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

MIN_CROPS_FOR_RANKING = 3


def ranking_spearman(
    meta: pd.DataFrame, y_true, y_pred, min_crops: int = MIN_CROPS_FOR_RANKING
) -> float:
    """Le modèle classe-t-il les cultures dans le bon ordre ?

    Pour chaque pays × année ayant au moins ``min_crops`` cultures, on compare le
    classement prédit des cultures au classement réel (corrélation de Spearman :
    1 = même ordre, 0 = aucun rapport, -1 = ordre inverse), puis on fait la moyenne.
    C'est exactement la question à laquelle répond /recommend.
    """
    frame = meta[["country", "year"]].assign(true=np.asarray(y_true), pred=np.asarray(y_pred))
    scores = [
        grp["true"].corr(grp["pred"], method="spearman")
        for _, grp in frame.groupby(["country", "year"])
        if len(grp) >= min_crops
    ]
    scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(scores)) if scores else float("nan")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def all_metrics(meta: pd.DataFrame, y_true, y_pred) -> dict[str, float]:
    return {
        **regression_metrics(y_true, y_pred),
        "ranking_spearman": ranking_spearman(meta, y_true, y_pred),
    }
