# agritech-crop-reco

Prototype **Agritech Answers** : aider un agriculteur à décider quoi cultiver.

- **Prédiction** : une culture + les conditions de la parcelle → rendement estimé (t/ha).
- **Recommandation** : les conditions de la parcelle seules → toutes les cultures classées par rendement simulé.

Architecture découplée : une API **FastAPI** (modèle de ML, conteneur Docker) interrogée par une interface **Streamlit** (aucune logique ML). Expériences suivies avec **MLflow**.

> Projet en cours. Étapes terminées : **1 – données**, **2 – modèle + MLflow**, **3 – API**. Prochaine : **4 – interface Streamlit**.

## Étape 1 – données (notebooks)

| Notebook | Contenu |
|---|---|
| `01_eda.ipynb` | exploration séparée des deux jeux, défauts détectés, décisions de nettoyage |
| `02_fusion.ipynb` | nettoyage, mesure de l'effet engrais / irrigation, fusion, contrôles qualité, résumé des choix |
| `03_acp.ipynb` | ACP (Kaggle) validée sur FAO, variables clés |
| `04_modelisation.ipynb` | sélection, optimisation, évaluation, interprétation du modèle |

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
models/                 modèle servi par l'API (.joblib + fiche + gains pratiques, versionnés)
reports/                figures et captures MLflow pour le rapport métier
docs/                   documentation (dont le pipeline CI/CD)
tests/                  tests pytest
```

## Étape 2 – modèle de rendement

| Élément | Choix |
|---|---|
| Entrées | culture, température, pluie, pesticides (ni pays ni année : non saisis dans l'application) |
| Test | 21 pays figés dans `configs/test_countries.json`, jamais utilisés pour régler |
| Validation | `GroupKFold` à 5 plis **par pays** |
| Sélection (MLflow `agritech-crop-yield-01-selection`) | référence naïve + 3 modèles × cible brute/log × pesticides brut/log/absent |
| Optimisation (MLflow `agritech-crop-yield-02-optimisation`) | `RandomizedSearchCV`, 30 essais |
| Modèle retenu | HistGradientBoosting, cible log, pesticides log |

Résultats sur les pays de test : RMSE 4,67 t/ha (référence 5,79), R² 0,70, classement des cultures 0,79 (référence 0,72). Détails et interprétation : `notebooks/04_modelisation.ipynb`.

```bash
python -m agritech_crop_reco.modeling.split    # une seule fois : fige les pays de test (déjà fait)
python -m agritech_crop_reco.modeling.train    # sélection + optimisation -> models/yield_model.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Étape 3 – API FastAPI

| Endpoint | Entrée | Sortie |
|---|---|---|
| `GET /health` | – | état du service et version du modèle |
| `POST /predict` | culture + conditions de la parcelle | rendement prédit (t/ha) |
| `POST /recommend` | conditions de la parcelle | les 10 cultures classées par rendement décroissant |

Conditions : `avg_temp_c`, `rainfall_mm`, `pesticides_tonnes` (obligatoires), `fertilizer_used`, `irrigation_used` (facultatifs, `null` = pratiques moyennes).

- **Rendement prédit** = rendement du modèle × facteur pratiques `(1 + g_E·E + g_I·I) / (1 + 0,5·g_E + 0,5·g_I)` : une parcelle aux pratiques moyennes garde la prédiction du modèle ; engrais + irrigation ≈ +29 %, aucune des deux ≈ −29 %.
- **Validation** : valeur physiquement impossible -> erreur 422 ; valeur plausible mais hors de la plage d'entraînement -> réponse 200 avec un avertissement (`warnings`).
- **Modèle servi** : `models/` (versionné, < 1 Mo) : `yield_model.joblib`, `model_card.json`, `practice_gains.json`.

```bash
uvicorn api.main:app --reload                                   # local -> http://127.0.0.1:8000/docs
docker build -f api/Dockerfile -t agritech-crop-api .           # depuis la racine du dépôt
docker run --rm -p 8000:8000 agritech-crop-api
```

`api/requirements.txt` est exporté de `poetry.lock` (mêmes versions qu'à l'entraînement) :

```bash
poetry export --only main,api --without-hashes -f requirements.txt -o api/requirements.txt
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
