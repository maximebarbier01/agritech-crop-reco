# Pipeline CI/CD

Workflow : [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Déclenché sur chaque
*pull request* et sur chaque poussée vers `main`. Une nouvelle poussée annule l'exécution
précédente de la même branche.

| Job | Contenu | Dépend de |
|---|---|---|
| `quality` | venv Poetry sur Python 3.11, `poetry check --lock`, installation (`ml`, `api`, `front`, `dev`), `ruff check`, `black --check`, `pytest --cov` | – |
| `docker` | `docker compose up --build --wait` (API + front), puis appels `/health`, `/predict`, `/recommend` et liaison front → API | `quality` |

## Ce que la CI suppose

- **Environnement Python** : en local, `poetry.toml` (`virtualenvs.create = false`) installe dans l'env conda actif. La CI n'a pas de conda : la variable `POETRY_VIRTUALENVS_CREATE=true` y force un venv isolé basé sur le Python 3.11 de `setup-python` (sinon Poetry installerait dans le Python système du runner, en 3.14, et échouerait sur les paquets Debian).

- Le modèle (`models/`, < 1 Mo) et le dataset consolidé (`data/processed/`) sont **versionnés** :
  la CI n'a besoin d'aucune donnée brute ni d'aucun secret.
- Le modèle n'est **pas** réentraîné en CI : il est entraîné en local
  (`python -m agritech_crop_reco.modeling.train`) puis commité.
- Après un changement de dépendances : `poetry lock`, puis régénérer `api/requirements.txt` et
  `frontend/requirements.txt` (voir le README). `poetry check --lock` fait échouer la CI si `poetry.lock` est périmé.

## Reproduire la CI en local

```bash
poetry check --lock
ruff check . && black --check . && pytest --cov
docker compose up -d --build --wait   # puis http://127.0.0.1:8000/health et :8501
docker compose down
```

## Déploiement continu (CD)

Pas encore en place : l'hébergeur de l'API n'est pas choisi. Le `Dockerfile` lit `PORT`, donc
il est prêt pour Render, Railway ou équivalent. Prochaine étape : un job de déploiement sur
`main`, après `docker`, avec les secrets de l'hébergeur dans GitHub.
