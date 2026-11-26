# agritech-crop-reco

[![CI](https://github.com/maximebarbier01/agritech-crop-reco/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/maximebarbier01/agritech-crop-reco/actions/workflows/ci.yml)
[![CD](https://github.com/maximebarbier01/agritech-crop-reco/actions/workflows/cd.yml/badge.svg?branch=main)](https://github.com/maximebarbier01/agritech-crop-reco/actions/workflows/cd.yml)
[![Train](https://github.com/maximebarbier01/agritech-crop-reco/actions/workflows/train.yml/badge.svg?branch=main)](https://github.com/maximebarbier01/agritech-crop-reco/actions/workflows/train.yml)

Prototype **Agritech Answers** : aider un agriculteur à décider quoi cultiver.

- **Prédiction** : une culture + les conditions de la parcelle → rendement estimé (t/ha).
- **Recommandation** : les conditions de la parcelle seules → toutes les cultures classées par rendement simulé.

Architecture découplée : une API **FastAPI** (modèle de ML, conteneur Docker) interrogée par une interface **Streamlit** (aucune logique ML). Expériences suivies avec **MLflow**.

> Étapes : **1 – données**, **2 – modèle + MLflow**, **3 – API**, **4 – interface Streamlit**, **5 – CI/CD + pipeline MLOps**.
>
> En ligne : API https://agritech-crop-api-latest.onrender.com/docs · application https://agritech-crop-reco.streamlit.app (première requête lente : réveil de l'offre gratuite de Render, jusqu'à une minute).

## Étape 1 – données

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
docker-compose.yml      API + front en une commande
frontend/               front-end Streamlit (app.py, api_client.py, Dockerfile, requirements.txt)
src/agritech_crop_reco/ logique réutilisable (données, entraînement, service du modèle)
notebooks/              EDA, fusion, ACP, modélisation
configs/default.toml    seed, chemins, expérience MLflow
data/raw/               données sources (non versionnées, empreintes dans SHA256SUMS)
data/processed/         dataset consolidé (versionné)
models/                 modèle servi par l'API (.joblib + fiche + gains pratiques, versionnés)
reports/                figures et captures MLflow pour le rapport métier
docs/                   documentation (pipeline CI/CD : docs/ci-cd.md)
.github/workflows/      GitHub Actions : ci.yml, cd.yml, train.yml
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

Résultats sur les pays de test : RMSE 4,67 t/ha (référence 5,79), R² 0,70, classement des cultures 0,79 (référence 0,72). Détails et interprétation : `notebooks/04_modelisation.ipynb` (sélection, optimisation, évaluation, interprétation).

```bash
python -m agritech_crop_reco.modeling.split    # une seule fois : fige les pays de test (déjà fait)
python -m agritech_crop_reco.modeling.train    # sélection + optimisation -> models/yield_model.joblib
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Étape 3 – API FastAPI

| Endpoint | Entrée | Sortie |
|---|---|---|
| `GET /health` | – | état du service et version du modèle |
| `GET /metadata` | – | cultures, plages d'entraînement, niveaux d'intensification (utilisés par l'interface) |
| `GET /monitoring/summary` | – | synthèse des prédictions servies : volume par endpoint, temps de réponse, part hors domaine, cultures (503 si l'enregistrement est désactivé) |
| `POST /predict` | culture + conditions de la parcelle | rendement prédit (t/ha) |
| `POST /recommend` | conditions de la parcelle | les 10 cultures classées par rendement décroissant |

Conditions : `avg_temp_c`, `rainfall_mm`, `pesticides_tonnes` (obligatoires), `fertilizer_used`, `irrigation_used` (facultatifs, `null` = pratiques moyennes).

- **Rendement prédit** = rendement du modèle × facteur pratiques `(1 + g_E·E + g_I·I) / (1 + 0,5·g_E + 0,5·g_I)` : une parcelle aux pratiques moyennes garde la prédiction du modèle ; engrais + irrigation ≈ +29 %, aucune des deux ≈ −29 %.
- **Validation** : valeur physiquement impossible -> erreur 422 ; valeur plausible mais hors de la plage d'entraînement -> réponse 200 avec un avertissement (`warnings`).
- **Modèle servi** : `models/` (versionné, < 1 Mo) : `yield_model.joblib`, `model_card.json`, `practice_gains.json`.
- **Suivi en production** : chaque appel à `/predict` et `/recommend` est journalisé (une ligne JSON dans les logs) et, si `PREDICTION_DB_URL` est définie, enregistré dans une base dédiée (Neon `agritech_api`, table `predictions`) : conditions, culture, rendement prédit, avertissements, versions du modèle et de l'API, temps de réponse. Aucune donnée personnelle. L'écriture a lieu après la réponse et une panne de base ne fait jamais échouer une prédiction.

```bash
uvicorn api.main:app --reload                                   # local -> http://127.0.0.1:8000/docs
docker build -f api/Dockerfile -t agritech-crop-api .           # depuis la racine du dépôt
docker run --rm -p 8000:8000 agritech-crop-api
```

`api/requirements.txt` est exporté de `poetry.lock` (mêmes versions qu'à l'entraînement) :

```bash
poetry export --only main,api --without-hashes -f requirements.txt -o api/requirements.txt
```

## Étape 4 – interface Streamlit

L'interface n'embarque aucun modèle : elle appelle l'API (`frontend/api_client.py`).

- **Prédire** : choisir une culture + les conditions de la parcelle -> rendement estimé, avertissements éventuels.
- **Recommander** : conditions seules -> toutes les cultures classées (meilleure culture, graphique à barres et tableau).
- Conditions dans la barre latérale : **curseurs** de température et de pluie bornés à la plage d'entraînement (lue via `GET /metadata`, pour éviter l'extrapolation) ; **niveau d'intensification** de la région en 5 crans (très faible → très élevé), traduit en tonnes de pesticides par les quantiles 10/30/50/70/90 % des pays d'entraînement, avec saisie précise possible ; engrais / irrigation facultatifs (« non renseigné » = pratique moyenne).
- Le dernier résultat reste affiché ; si une condition change, il est signalé comme à relancer.
- API injoignable ou valeur refusée (422) : message d'erreur explicite, sans plantage.

```bash
uvicorn api.main:app --reload                 # terminal 1 -> http://127.0.0.1:8000
streamlit run frontend/app.py                 # terminal 2 -> http://localhost:8501
API_URL=https://mon-api.example streamlit run frontend/app.py   # autre adresse d'API
```

Toute l'application d'une commande (Docker Compose : l'API démarre d'abord, le front attend qu'elle soit prête) :

```bash
docker compose up --build        # front http://localhost:8501 · API http://localhost:8000/docs
docker compose down
```

Ou image par image (le front joint l'API par `API_URL`, `http://api:8000` par défaut) :

```bash
docker build -f frontend/Dockerfile -t agritech-crop-front .     # depuis la racine du dépôt
docker network create agri
docker run -d --name api   --network agri agritech-crop-api
docker run -d --name front --network agri -p 8501:8501 agritech-crop-front   # -> http://localhost:8501
```

`frontend/requirements.txt` est exporté de `poetry.lock` :

```bash
poetry export --only front --without-hashes -f requirements.txt -o frontend/requirements.txt
```

## Expérience complémentaire – intrants FAOSTAT réels

Test, sans modifier le modèle servi, de l'ajout d'intrants mesurés par pays et par année (FAOSTAT, oct. 2026) : azote, phosphate, potasse et pesticides en kg/ha de terres cultivées, part des terres équipées pour l'irrigation. Même protocole qu'à l'étape 2, mêmes hyperparamètres, règle fixée à l'avance : retenu seulement si le gain dépasse un écart-type.

| Configuration | RMSE CV (t/ha) | R² |
|---|---|---|
| Référence (modèle servi) | 4,97 ± 0,44 | 0,59 |
| + intrants FAOSTAT | 4,71 ± 0,36 | 0,63 |

Gain d'environ 5 % sous le seuil (4,53) : **non retenu**. Détails : `notebooks/05_intrants_faostat.ipynb`, expérience MLflow `agritech-crop-yield-03-intrants-faostat`.

```bash
python -m agritech_crop_reco.data.faostat               # data/raw/faostat/*.zip -> data/processed/faostat_inputs.csv
python -m agritech_crop_reco.modeling.experiment_inputs # expérience MLflow 03 (≈ 45 s)
```

## Étape 6 – rapport métier

[`reports/rapport_metier.pdf`](reports/rapport_metier.pdf) : résultats, variables clés et recommandations, pour un public non technique. Source : `docs/rapport/` (HTML + CSS) ; les figures sont recalculées depuis le modèle et les données, le PDF est imprimé par Chrome en mode sans affichage :

```bash
python docs/rapport/make_figures.py   # reports/figures/rapport_*.png
python docs/rapport/build_pdf.py      # reports/rapport_metier.pdf (+ liste des captures manquantes)
```

Captures d'écran attendues : `reports/mlflow/`, `reports/app/`, `reports/cicd/`.

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
| Distant (Neon) | `MLFLOW_TRACKING_URI` dans `.env` ou secret GitHub | PostgreSQL **Neon** | non conservés (le modèle de référence est entraîné en local) |

Le workflow `train.yml` l'utilise (secret GitHub) : ses runs portent le tag `source=ci`. Les tests, eux, n'écrivent pas dans MLflow.

## Qualité et CI

```bash
pytest
ruff check .
black --check .
```

| Workflow | Quand | Quoi |
|---|---|---|
| `ci.yml` | chaque PR, chaque poussée sur `develop` | ces trois contrôles, puis les images Docker (API + front) démarrées et testées |
| `cd.yml` | poussée sur `main`, tag `vX.Y.Z` | CI complète -> image de l'API sur Docker Hub -> déploiement Render -> test de l'API en ligne |
| `train.yml` | manuel, ou `main` si données / modèle modifiés | pipeline MLOps -> MLflow (Neon) -> garde-fou de performance |

Pipeline MLOps (prétraitement -> entraînement -> évaluation -> garde-fou), aussi en local :

```bash
python -m agritech_crop_reco.pipeline --from-raw    # reconstruit le dataset depuis data/raw et vérifie qu'il est identique
python -m agritech_crop_reco.pipeline --output-dir models   # remplace volontairement le modèle servi
```

Schéma, secrets à configurer, notifications : [`docs/ci-cd.md`](docs/ci-cd.md).
