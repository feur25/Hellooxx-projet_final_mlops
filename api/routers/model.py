from fastapi import APIRouter

from api.schemas import (
    ModelInfoResponse, ModelVersionsResponse, ModelReloadResponse,
)
from api.server import server
from api.decorators import handle_errors, require_model

router = APIRouter(prefix="/model", tags=["model"])


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Loaded model information",
    description="Returns metadata of the currently loaded model: name, version, hyperparameters, stored metrics.",
    responses={
        200: {"description": "Model metadata"},
        503: {"description": "Model not loaded"},
    },
)
@handle_errors
@require_model
def info():
    config = server.registry.load(server.version)
    return ModelInfoResponse(
        name=config["model_name"],
        version=config["version"],
        params=config["params"],
        metrics=config.get("metrics", {}),
    )


@router.get(
    "/versions",
    response_model=ModelVersionsResponse,
    summary="List trained model versions",
    description="Returns every model version persisted under `models/` along with the currently loaded one.",
)
@handle_errors
def versions():
    return ModelVersionsResponse(
        versions=server.registry.versions,
        current=server.version,
    )


@router.post(
    "/reload",
    response_model=ModelReloadResponse,
    summary="Reload the model from disk",
    description="Reloads the latest model from the registry without retraining. Useful after a manual model swap.",
    responses={
        200: {"description": "Model reloaded"},
        404: {"description": "No model file found in registry"},
    },
)
@handle_errors
def reload():
    server.load()
    return ModelReloadResponse(
        status="reloaded", version=server.version, name=server.name,
    )
