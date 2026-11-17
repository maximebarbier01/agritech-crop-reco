# Pipeline CI/CD

Workflow : [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Déclenché sur chaque
*pull request* et sur chaque poussée vers `main`. Une nouvelle poussée annule l'exécution
précédente de la même branche.

| Job | Contenu | Dépend de |
|---|---|---|
| `quality` | `poetry check --lock`, installation (`ml`, `api`, `front`, `dev`), `ruff check`, `black --check`, `pytest --cov` | – |
| `docker` | `docker build -f api/Dockerfile`, démarrage du conteneur, appels `/health`, `/predict`, `/recommend` | `quality` |

## Ce que la CI suppose

- Le modèle (`models/`, < 1 Mo) et le dataset consolidé (`data/processed/`) sont **versionnés** :
  la CI n'a besoin d'aucune donnée brute ni d'aucun secret.
- Le modèle n'est **pas** réentraîné en CI : il est entraîné en local
  (`python -m agritech_crop_reco.modeling.train`) puis commité.
- Après un changement de dépendances : `poetry lock`, puis régénérer `api/requirements.txt`
  (voir le README). `poetry check --lock` fait échouer la CI si `poetry.lock` est périmé.

## Reproduire la CI en local

```bash
poetry check --lock
ruff check . && black --check . && pytest --cov
docker build -f api/Dockerfile -t agritech-crop-api .
docker run --rm -p 8000:8000 agritech-crop-api   # puis http://127.0.0.1:8000/health
```

## Déploiement continu (CD)

Pas encore en place : l'hébergeur de l'API n'est pas choisi. Le `Dockerfile` lit `PORT`, donc
il est prêt pour Render, Railway ou équivalent. Prochaine étape : un job de déploiement sur
`main`, après `docker`, avec les secrets de l'hébergeur dans GitHub.
