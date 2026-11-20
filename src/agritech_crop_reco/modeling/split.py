"""Découpage entraînement / test **par pays**, figé dans un fichier versionné.

Pourquoi par pays : le rendement d'un pays × culture varie peu d'une année à l'autre.
Un découpage aléatoire mettrait France-blé-2005 en entraînement et France-blé-2006 en
test : le modèle « reconnaîtrait » le pays au lieu de généraliser. L'application ne
demandant pas le pays, on évalue sur des **pays jamais vus**.

Pourquoi figé : le jeu de test ne doit jamais changer (ni servir à régler le modèle).
``python -m agritech_crop_reco.modeling.split`` le crée une seule fois ; l'entraînement
le relit et refuse de tourner s'il est absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from agritech_crop_reco.config import PROJECT_ROOT, load_settings

SPLIT_FILE = PROJECT_ROOT / "configs" / "test_countries.json"
TEST_SIZE = 0.2


def make_test_countries(df: pd.DataFrame, seed: int, test_size: float = TEST_SIZE) -> list[str]:
    """Tire ~test_size des pays (pas des lignes) pour le jeu de test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(splitter.split(df, groups=df["country"]))
    return sorted(df.iloc[test_idx]["country"].unique())


def load_test_countries(path: Path = SPLIT_FILE) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent : lancer une fois `python -m agritech_crop_reco.modeling.split`"
        )
    return json.loads(path.read_text())["test_countries"]


def split_train_test(
    df: pd.DataFrame, test_countries: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_test = df["country"].isin(test_countries)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def main() -> None:
    from agritech_crop_reco.data.merge import CONSOLIDATED_FILE

    if SPLIT_FILE.exists():
        raise SystemExit(f"{SPLIT_FILE} existe déjà : le jeu de test est figé, on n'y touche pas.")
    settings = load_settings()
    df = pd.read_csv(settings.processed_dir / CONSOLIDATED_FILE)
    countries = make_test_countries(df, settings.seed)
    _, test = split_train_test(df, countries)
    payload = {
        "seed": settings.seed,
        "test_size_countries": TEST_SIZE,
        "n_test_countries": len(countries),
        "n_test_rows": len(test),
        "test_countries": countries,
    }
    SPLIT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(countries)} pays / {len(test):,} lignes figés dans {SPLIT_FILE}")


if __name__ == "__main__":
    main()
