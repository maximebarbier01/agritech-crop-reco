"""Référence naïve : prédire le rendement médian de la culture, sans regarder le climat.

C'est la « situation de départ » : tout modèle doit faire mieux, sinon les
conditions de la parcelle n'apportent rien.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class CropMedianRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> CropMedianRegressor:
        y = pd.Series(y, index=X.index)
        self.medians_ = y.groupby(X["crop"]).median()
        self.global_median_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame):
        # Culture inconnue -> médiane globale (ne devrait pas arriver : 10 cultures fixes)
        return X["crop"].map(self.medians_).fillna(self.global_median_).to_numpy()
