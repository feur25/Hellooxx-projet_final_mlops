TAGS_METADATA = [
    {"name": "health", "description": "Liveness probe and service metadata."},
    {"name": "prediction", "description": "Diabetes prediction endpoints (single sample and batch)."},
    {"name": "model", "description": "Inspect or reload the model loaded in memory."},
    {"name": "lifecycle", "description": "Retraining pipeline and model rollout."},
]

DESCRIPTION = """
**Diabetes Prediction API** — production-ready REST service for the Pima Indians Diabetes dataset.

Exposes a `RandomForestClassifier` trained via `seraplot.GridSearchCV` (native Rust backend) and
tracked with MLflow. Reproducible end-to-end: same seed, same hyperparameter grid, same MLflow run.

### Capabilities
- single and batch prediction with full class probabilities
- health probe and model metadata
- on-demand retraining when new CSV data is dropped in `data/incoming/`
- listing of every persisted model version

Built with **FastAPI + Pydantic v2**, containerised with **Docker**, automated with **GitHub Actions**.
"""

CONTACT = {
    "name": "feur25",
    "email": "feur09@gmail.com",
    "url": "https://github.com/feur25",
}

LICENSE_INFO = {"name": "MIT", "url": "https://opensource.org/licenses/MIT"}

SERVERS = [
    {"url": "http://localhost:8000", "description": "Local development"},
    {"url": "http://0.0.0.0:8000", "description": "Docker container"},
]