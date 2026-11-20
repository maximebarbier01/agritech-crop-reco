"""Outils d'analyse : ACP normée et lien variable qualitative / rendement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_correlations(df: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """ACP normée (variables centrées-réduites).

    Retourne :
    - la part de variance expliquée par axe ;
    - la corrélation de chaque variable avec chaque axe (coordonnées du cercle des
      corrélations) : composante × racine(valeur propre).
    """
    z = StandardScaler().fit_transform(df[columns].astype(float))
    pca = PCA().fit(z)
    axes = [f"F{i + 1}" for i in range(len(columns))]
    explained = pd.Series(pca.explained_variance_ratio_, index=axes, name="variance expliquée")
    # explained_variance_ utilise n-1 ; StandardScaler divise par n -> correction pour
    # obtenir des corrélations exactes (|r| <= 1).
    eigen = pca.explained_variance_ * (len(z) - 1) / len(z)
    loadings = pd.DataFrame(pca.components_.T * np.sqrt(eigen), index=columns, columns=axes)
    return explained, loadings


def eta_squared(categories: pd.Series, values: pd.Series) -> float:
    """Rapport de corrélation η² : part de la variance de ``values`` expliquée par les groupes.

    0 = les moyennes par modalité sont identiques ; 1 = la modalité détermine la valeur.
    """
    group_means = values.groupby(categories, observed=True).transform("mean")
    total = ((values - values.mean()) ** 2).sum()
    return float(((group_means - values.mean()) ** 2).sum() / total)
