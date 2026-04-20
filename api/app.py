import sys
import os
import numpy as np
import seraplot as sp
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
    HealthResponse, RetrainResponse,
)
from src.model_store import ModelRegistry
from src.data_prep import DataPipeline
from src.train import BasePipeline
from src.retrain import RetrainPipeline


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

    def load(self):
        config = self._registry.load("latest")
        self._name, self._version = config["model_name"], config["version"]
        self._scaler_mean = np.array(config["scaler"]["mean"], dtype=np.float64)
        self._scaler_scale = np.array(config["scaler"]["scale"], dtype=np.float64)

        pipe = DataPipeline().clean()
        ds = pipe.split()
        scaled = DataPipeline.scale(ds.x_train)

        self._model = BasePipeline._build_model(config["params"])
        self._model.fit(scaled.x_train, ds.y_train)

    def _require_ready(self):
        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Model not loaded")

    def _scale(self, raw: np.ndarray) -> np.ndarray:
        self._require_ready()
        return (raw - self._scaler_mean) / self._scaler_scale

    def predict_single(self, req: PredictionRequest) -> PredictionResponse:
        self._require_ready()
        scaled = self._scale(req.to_array())
        pred = self._model.predict(scaled)
        proba = self._model.predict_proba(scaled)
        return PredictionResponse.from_prediction(pred[0], proba[0])

    def predict_batch(self, req: BatchPredictionRequest) -> BatchPredictionResponse:
        self._require_ready()
        scaled = self._scale(req.to_array())
        preds = self._model.predict(scaled)
        probas = self._model.predict_proba(scaled)
        return BatchPredictionResponse(
            predictions=list(map(
                lambda pp: PredictionResponse.from_prediction(pp[0], pp[1]),
                zip(preds, probas),
            ))
        )

    def __repr__(self):
        return f"ModelServer(ready={self.is_ready}, version={self._version})"


server = ModelServer()


@asynccontextmanager
async def lifespan(_app):
    try:
        server.load()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(title="Diabetes Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    server._require_ready()
    return HealthResponse(status="healthy", model_version=server.version, model_name=server.name)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    return server.predict_single(req)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    return server.predict_batch(req)


@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    result = RetrainPipeline().run()
    if result["status"] == "retrained":
        server.load()
        return RetrainResponse(
            status="retrained",
            new_files_count=result["new_files_count"],
            improved=result["improved"],
            model_version=result["results"]["model_version"],
        )
    return RetrainResponse(status=result["status"])
