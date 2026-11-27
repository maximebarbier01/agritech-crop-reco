# Journal des versions

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/), versions [SemVer](https://semver.org/lang/fr/).
Chaque version taguée `vX.Y.Z` sur `main` publie l'image `docker.io/maximebarbier01/agritech-crop-api:X.Y.Z`.

## [1.4.0] – 2026-11-27

### Ajouté
- **XGBoost et CatBoost** dans la sélection (MLflow `01-selection`) : 5 modèles × cible brute/log × pesticides brut/log/absent + référence naïve, soit 31 runs. CatBoost reçoit la culture en variable catégorielle native, via un adaptateur compatible scikit-learn (`CatBoostCropRegressor`).
- **Optuna** (échantillonneur TPE) comparé à `RandomizedSearchCV` dans l'optimisation (MLflow `02-optimisation`) : les 3 familles de boosting (HistGradientBoosting, XGBoost, CatBoost) sont réglées par les deux méthodes, avec le même espace de recherche (`SEARCH_SPACES`), le même budget (30 essais) et la même validation par pays. Chaque recherche est un run parent avec ses essais en runs imbriqués, puis un run « Décision » applique la règle `select_config` aux 6 candidats.
- `reports/optimisation_results.csv` (les 6 candidats) ; champs `search_method`, `n_trials` et `optimisation_candidates` dans `models/model_card.json`.
- Notebook 04 : comparaison des 6 candidats et courbes de convergence Optuna / RandomizedSearch (figures `04_optimisation_candidates.png`, `04_optimisation_convergence.png`).

### Résultat
- Les 6 candidats sont statistiquement à égalité (RMSE CV de 4,76 à 4,97 t/ha, tous à moins d'un écart-type du meilleur). La règle retient **HistGradientBoosting + RandomizedSearch**, au meilleur classement des cultures (0,843). CatBoost a le RMSE le plus bas, mais après optimisation il ne produit plus que 20 à 48 classements différents sur 300 parcelles, contre 136.
- **Modèle servi inchangé** : mêmes hyperparamètres et mêmes métriques de test qu'en 1.3.1 (RMSE 4,67 t/ha). L'API et ses dépendances ne changent pas : XGBoost, CatBoost et Optuna ne sont installés qu'avec le groupe Poetry `ml` (entraînement).

## [1.3.1] – 2026-11-26

### Documentation
- Rapport : captures de l'application v1.3.0 (chiffre d'affaires estimé, origine du prix, classement par chiffre d'affaires) ; numérotation des figures corrigée (les captures deviennent les figures 9 et 10).

## [1.3.0] – 2026-11-26

### Ajouté
- **Chiffre d'affaires estimé** : prix à la production FAOSTAT (2018-2023, USD/t) — médiane nationale pour 84 pays du modèle, médiane mondiale par culture — dans `models/crop_prices.json` (`python -m agritech_crop_reco.data.prices`). `/predict` et `/recommend` renvoient pour chaque culture le prix, son origine (saisi, national, mondial) et le chiffre d'affaires estimé ($/ha), hors coûts de production.
- **`/recommend` classe par chiffre d'affaires par défaut** (`rank_by="revenue"`, ou `"yield"` pour le rendement). Champs facultatifs `country` (prix nationaux ; jamais utilisé par le modèle de rendement) et `prices_usd_t` (prix saisis, prioritaires). `/metadata` liste les pays et les prix mondiaux.
- **Interface** : choix du pays, prix de vente saisis, classement par chiffre d'affaires ou par rendement, prix et origine du prix dans le tableau.
- **Expérience MLflow 03** (`agritech-crop-yield-03-intrants-faostat`, notebook 05) : intrants FAOSTAT réels testés — gain d'environ 5 %, sous le seuil fixé à l'avance, non retenus.
- **Rapport** : partie 6 réécrite (« Du rendement au chiffre d'affaires »), figure tonnes vs chiffre d'affaires, encadré et figures sur les intrants FAOSTAT.

### Modifié
- Suivi des prédictions : colonnes `country`, `rank_by` et `revenue_usd_ha`, ajoutées automatiquement à une table existante (migration douce au démarrage).

### Corrigé
- CI : une exécution sur `develop` ou `main` n'est plus annulée par une exécution plus récente (`cancel-in-progress` limité aux PR). Une annulation laissait un contrôle exigé « annulé » sur le commit et bloquait la PR de release.
- Rapport : la v1.2.0 affirmait que les tubercules se vendent moins cher à la tonne que les céréales ; les prix FAOSTAT montrent le contraire.

## [1.2.0] – 2026-11-20

### Ajouté
- **Suivi des prédictions en production** : chaque appel à `/predict` et `/recommend` est enregistré dans une base dédiée (Neon `agritech_api`, table `predictions`) si `PREDICTION_DB_URL` est définie : conditions, culture, rendement prédit, classement, avertissements, versions du modèle et de l'API, temps de réponse. Aucune donnée personnelle. Écriture après la réponse ; une panne de base est journalisée sans faire échouer la prédiction ; sans variable, l'enregistrement est désactivé (CI, docker compose).
- **`GET /monitoring/summary`** : synthèse agrégée (volume par endpoint, temps de réponse médian et 95ᵉ centile, part hors domaine d'entraînement, cultures les plus demandées et recommandées).

### Corrigé
- Le journal de l'API (`agritech.api`) n'était pas configuré : ses messages d'information étaient perdus. Chaque prédiction y est désormais tracée sur une ligne JSON.
- Les tests n'utilisent plus le fichier `.env` local (évite d'écrire dans une base de production pendant les tests).

## [1.1.0] – 2026-11-20

### Ajouté
- **Rapport métier** (`reports/rapport_metier.pdf`) : résultats, variables clés, recommandations agronomiques, du rendement au profit, captures MLflow, application et CI/CD. Sources dans `docs/rapport/` ; figures recalculées depuis le modèle et les données ; PDF imprimé par Chrome en mode sans affichage.
- **Garde-fou git flow** : le job CI « Branche cible » refuse toute *pull request* vers `main` qui ne vient pas d'une branche `release/…` ou `hotfix/…`.

### Modifié
- **Interface** : nombres affichés au format français (« 1,42 t/ha »).
- **GitHub Actions** : passage aux versions majeures fonctionnant sous Node 24 (`checkout` v7, `setup-python` v7, `upload-artifact` v7, `setup-buildx-action` v4, `login-action` v4, `metadata-action` v6, `build-push-action` v7), sans changement des paramètres utilisés.

## [1.0.1] – 2026-11-20

### Corrigé
- **CD** : un `/` final dans la variable `API_PUBLIC_URL` produisait `…//health` (404) et faisait échouer le test de l'API en ligne ; il est désormais retiré.
- **Interface** : si l'adresse `API_URL` désigne une autre API (cas réel : sous-domaine Render déjà pris par un autre service), un message explicite remplace l'erreur technique `KeyError`.

### Retiré
- Dossier `.devcontainer/` ajouté automatiquement par Streamlit Community Cloud directement sur `main` (inutilisé, et lançait Streamlit sans protections CORS / XSRF). `main` est désormais protégée par un ruleset (PR et CI obligatoires).

### Documentation
- URL publiques de l'API et de l'application ; protection de `main` ; déploiement Render par image (« Existing Image ») et piège des sous-domaines partagés.

## [1.0.0] – 2026-11-20

Première version déployée.

### Ajouté
- **Données** : fusion des jeux FAO (rendements réels par pays × culture × année) et Kaggle (effet engrais ≈ +45 % et irrigation ≈ +36 %) ; dataset consolidé de 13 130 observations ; ACP des variables clés.
- **Modèle** : HistGradientBoosting (cible et pesticides en log), choisi parmi 18 configurations en validation croisée par pays et optimisé (RandomizedSearchCV) ; suivi MLflow. Sur 21 pays jamais vus : RMSE 4,67 t/ha, soit −19 % par rapport à la référence « rendement médian de la culture ».
- **API FastAPI** : `/predict`, `/recommend`, `/metadata`, `/health` ; ajustement engrais / irrigation ; avertissement hors domaine d'entraînement ; image Docker.
- **Interface Streamlit** : prédiction pour une culture et classement des 10 cultures (graphique + tableau) ; curseurs bornés au domaine appris ; niveau d'intensification agricole.
- **CI/CD et MLOps** : `ci.yml` (qualité, tests, images Docker), `cd.yml` (Docker Hub puis Render), `train.yml` (pipeline prétraitement -> entraînement -> évaluation -> garde-fou, suivi MLflow sur Neon).

[1.3.1]: https://github.com/maximebarbier01/agritech-crop-reco/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/maximebarbier01/agritech-crop-reco/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/maximebarbier01/agritech-crop-reco/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/maximebarbier01/agritech-crop-reco/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/maximebarbier01/agritech-crop-reco/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/maximebarbier01/agritech-crop-reco/releases/tag/v1.0.0
