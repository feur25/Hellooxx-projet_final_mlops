import numpy as np
from fastapi import HTTPException

from src.model_store import ModelRegistry
from src.data_prep import DataPipeline
from src.train import BasePipeline


class ModelServer:
    def __init__(self):
        self._model = None
        self._scaler_mean = None
        self._scaler_scale = None
        self._version = None
        self._name = None
        self._registry = ModelRegistry()

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._version

    @property
    def name(self) -> str:
        return self._name

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def load(self, version: str = "latest") -> None:
        config = self._registry.load(version)
        self._name, self._version = config["model_name"], config["version"]
        self._scaler_mean = np.array(config["scaler"]["mean"], dtype=np.float64)
        self._scaler_scale = np.array(config["scaler"]["scale"], dtype=np.float64)

        pipe = DataPipeline().clean()
        ds = pipe.split()
        scaled = DataPipeline.scale(ds.x_train)

        self._model = BasePipeline._build_model(config["params"])
        self._model.fit(scaled.x_train, ds.y_train)

    def require_ready(self) -> None:
        if not self.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Train it first via `python scripts/run_training.py`.",
            )

    def scale(self, raw: np.ndarray) -> np.ndarray:
        self.require_ready()
        return (raw - self._scaler_mean) / self._scaler_scale

    def predict(self, raw: np.ndarray):
        self.require_ready()
        scaled = self.scale(raw)
        return self._model.predict(scaled), self._model.predict_proba(scaled)

    def __repr__(self) -> str:
        return f"ModelServer(ready={self.is_ready}, version={self._version})"


server = ModelServer()
