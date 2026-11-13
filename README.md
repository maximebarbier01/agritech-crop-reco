# agritech-crop-reco

Prototype **Agritech Answers** : aider un agriculteur à décider quoi cultiver.

- **Prédiction** : une culture + les conditions de la parcelle → rendement estimé (t/ha).
- **Recommandation** : les conditions de la parcelle seules → toutes les cultures classées par rendement simulé.

Architecture découplée : une API **FastAPI** (modèle de ML, conteneur Docker) interrogée par une interface **Streamlit** (aucune logique ML). Expériences suivies avec **MLflow**.

> Projet en cours. Étape terminée : **1 – exploration, fusion, ACP**. Prochaine : **2 – modèle + MLflow**.

## Étape 1 – données (notebooks)

| Notebook | Contenu |
|---|---|
| `01_eda.ipynb` | exploration séparée des deux jeux, défauts détectés, décisions de nettoyage |
| `02_fusion.ipynb` | nettoyage, mesure de l'effet engrais / irrigation, fusion, contrôles qualité, résumé des choix |
| `03_acp.ipynb` | ACP (Kaggle) validée sur FAO, variables clés |

Stratégie de fusion : les observations **FAO** (pays × culture × année) forment la base, car elles seules départagent les cultures ; le jeu **Kaggle** (simulé, sans effet culture) fournit le gain relatif dû à l'engrais (≈ +45 %) et à l'irrigation (≈ +36 %), joint par culture.

Régénérer le dataset consolidé (`data/processed/crop_yield_consolidated.csv`, versionné) :

```bash
python -m agritech_crop_reco.data.merge
```

## Structure

```
api/                    back-end FastAPI (main.py, Dockerfile, requirements.txt)
frontend/               front-end Streamlit (app.py, requirements.txt)
src/agritech_crop_reco/ logique réutilisable (données, features, entraînement, recommandation)
notebooks/              EDA, fusion, ACP, modélisation
configs/default.toml    seed, chemins, expérience MLflow
data/raw/               données sources (non versionnées, empreintes dans SHA256SUMS)
data/processed/         dataset consolidé (versionné)
models/                 modèle entraîné (.joblib)
reports/                figures et captures MLflow pour le rapport métier
docs/                   documentation (dont le pipeline CI/CD)
tests/                  tests pytest
```

## Installation (WSL2, conda + Poetry)

```bash
conda create -n agritech-crop-reco python=3.11
conda activate agritech-crop-reco
poetry install --with ml,api,front,eda,dev   # installe dans l'env conda (poetry.toml : pas de .venv)
cp .env.example .env
```

Groupes Poetry : `main` (cœur du modèle), `ml` (MLflow), `api`, `front`, `eda` (notebooks), `dev` (tests, lint).

## Données

Fichiers copiés dans `data/raw/` (non versionnés). Vérifier leur intégrité :

```bash
cd data/raw && sha256sum -c SHA256SUMS
```

| Jeu | Fichiers | Contenu |
|---|---|---|
| Agriculture Crop Yield (Kaggle) | `agriculture_crop_yield/crop_yield.csv` | 1 000 000 lignes : région, sol, culture, pluie, température, engrais, irrigation, météo, durée de culture, rendement (t/ha) |
| Crop Yield Prediction Dataset (FAO / Banque mondiale) | `crop_yield_prediction/*.csv` | rendement par pays × culture × année (hg/ha), pluie, pesticides, température |

## Suivi MLflow

Une seule variable choisit la base de suivi, `MLFLOW_TRACKING_URI` (voir `.env.example`) :

| Contexte | Valeur | Stockage des métadonnées | Artefacts |
|---|---|---|---|
| Local (défaut) | vide | `mlflow.db` (SQLite) | dossier local |
| CI (GitHub Actions) | secret `MLFLOW_TRACKING_URI` | PostgreSQL **Neon** | non conservés (le modèle de référence est entraîné en local) |

## Qualité

```bash
pytest
ruff check .
black --check .
```
