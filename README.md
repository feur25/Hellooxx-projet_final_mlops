# Diabetes Prediction — MLOps Pipeline

Prédiction du diabète (dataset Pima Indians) avec une pipeline MLOps complète : prétraitement, entraînement, évaluation, déploiement API, réentraînement automatique et CI/CD. Entièrement réaliser avec Seraplot : https://feur25.github.io/seraplot/introduction.html
PS : la predict et bancale, mais reste correcte je ne me suis guère attarder dessus


Partie claude lore explication, de l'archi, car en vrai il explique pas trop mal je trouve, et fait énormément moins de faute que moins en français, plus pratique pour ne pas finire, avec les yeux ensanglanter :

---

## Outils utilisés

| Rôle | Outil |
|---|---|
| ML (modèles, GridSearchCV) | [SeraPlot](https://pypi.org/project/seraplot/) (Rust, bindings Python) |
| Experiment tracking | MLflow |
| API REST | FastAPI + Uvicorn |
| Conteneurisation | Docker |
| CI/CD | GitHub Actions |
| Gestion dépendances | uv |

---

## Structure du projet
(pas update visiblement, il y arrivent pas mais des nouveaux directory ont été ajouter, plus lisible car je trouve les api python très peut maintenable contrairement, à une api C# par exemple, donc j'ai juste remastoriser le système sert à rien, sauf si plus tard j'ai besoin de refaire une api python, gain de temp++)

```
├── data/               # Données brutes et splits (train/val/test CSV)
│   └── incoming/       # Dépôt de nouvelles données pour le réentraînement
├── models/             # Configs de modèles versionnées (JSON)
├── mlruns/             # Tracking MLflow local
├── scripts/            # Points d'entrée CLI
│   ├── split_data.py
│   ├── run_training.py
│   └── run_retrain.py
├── src/                # Librairie custom
│   ├── config.py       # Constantes et hyperparamètres
│   ├── data_prep.py    # DataPipeline (nettoyage IQR, split, normalisation)
│   ├── train.py        # BasePipeline, TrainingPipeline (GridSearchCV + MLflow)
│   ├── evaluate.py     # ClassificationEvaluator
│   ├── model_store.py  # ModelRegistry (versionnage JSON)
│   └── retrain.py      # RetrainPipeline (détection nouvelles données)
├── api/
│   ├── app.py          # FastAPI — /health, /predict, /predict/batch, /retrain
│   └── schemas.py      # Pydantic v2 schemas
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Étapes

### 1. Collecte de données

Dataset public (il est sur kaggle, et aussi dans le repo, pas le analysis, le prediction (1).ipynb) : [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 observations, 8 features, cible binaire (`Outcome`).

### 2. Prétraitement

- Nettoyage : écrêtage IQR (facteur 1.5) sur `Insulin` et `DiabetesPedigreeFunction`
- Split stratifié : **70 % train / 15 % val / 15 % test** → 538 / 115 / 115 samples
- Normalisation : StandardScaler (fit sur train uniquement, appliqué à val/test)

```bash
python scripts/split_data.py
# Train: 538 | Val: 115 | Test: 115 | Total: 768
```

### 3. Entraînement

`RandomForestClassifier` via `seraplot.GridSearchCV` — parallélisme Rust natif, mon outil perso, j'écrit toujours la documentation en ce moment même donc pas complète.

Grille d'hyperparamètres (192 combinaisons, cv=5 = 960 fits) :

```python
GRID_PARAMS = {
    "n_estimators":      [10, 20, 50, 100],
    "max_depth":         [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
```

```bash
python scripts/run_training.py
```

Tous les paramètres et métriques sont loggés dans MLflow.

### 4. Résultats

| Métrique | Valeur |
|---|---|
| Best CV accuracy | **0.7900** |
| Train accuracy | 0.8643 |
| Val accuracy | 0.6957 |
| Test accuracy | **0.6957** |
| Test F1 | 0.4615 |
| Test precision | 0.5000 |
| Test recall | 0.4286 |

Meilleurs hyperparamètres (trouv"é) : `n_estimators=20, max_depth=5, min_samples_split=2, min_samples_leaf=2`

### 5. Déploiement

API FastAPI avec 4 endpoints :

| Endpoint | Description |
|---|---|
| `GET /health` | Statut + version du modèle chargé |
| `POST /predict` | Prédiction unitaire |
| `POST /predict/batch` | Prédiction batch |
| `POST /retrain` | Déclenche le réentraînement si nouvelles données présentes |

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t diabetes-prediction-api .
docker run -p 8000:8000 diabetes-prediction-api
```

### 6. Réentraînement

Pour faire un re entrainement facile, ont prend un fichier csv qui à les même labels, keys, nomenclature bref ont le send dans `data/incoming/` avec les mêmes "colonnes" que `diabetes.csv`, puis ont appele `POST /retrain` ou :

```bash
python scripts/run_retrain.py
```

La (ou le) pipeline, qui va permettre de fusionnée les nouvelles données dans `diabetes.csv`, réentraîne, et met à jour `models/model_latest.json` uniquement si il y a un gain dans les perf (une amélioration).

### 7. CI/CD (Bonus)

Pipeline GitHub Actions (`.github/workflows/ci.yml`), go checker jsutement le github action, du coup triple environement dev (moi en locale), test & prod pour le deploy :

- **test** : install → split → train (sur `push`/`PR` vers `main`)
- **deploy** : build image Docker → démarrage conteneur → health check (sur `push` vers `main` uniquement en tout cas à l'état actuel des choses, oui y a pas d'autre branch, je fessait tous en local, sauf pour jsutement le ci)

---

## Lancer en local

```bash
# Créer et activer l'environnement
uv venv .venv
uv pip install -r requirements.txt

# Pipeline complète
python scripts/split_data.py
python scripts/run_training.py

# API
uvicorn api.app:app --reload

# Réentraînement (simuler nouvelles données)
cp data/val.csv data/incoming/new_data.csv
python scripts/run_retrain.py
```

Je retiens une chose bah c'est pas avec moi, que vous serez sur si vous avez le diabète ou pas :D