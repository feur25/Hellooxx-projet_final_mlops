from fastapi import APIRouter

from api.schemas import HealthResponse, RootResponse
from api.server import server
from api.decorators import handle_errors, require_model

router = APIRouter(tags=["health"])


@router.get(
    "/",
    response_model=RootResponse,
    summary="Service root",
    description="Returns API metadata and the list of available endpoints.",
)
@handle_errors
def root():
    return RootResponse(
        name="Diabetes Prediction API",
        version="1.0.0",
        docs="/docs",
        redoc="/redoc",
        openapi="/openapi.json",
        endpoints=[
            "/health", "/predict", "/predict/batch",
            "/retrain", "/model/info", "/model/versions", "/model/reload",
        ],
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and the loaded model name + version. Returns 503 if no model is loaded.",
    responses={
        200: {"description": "Service is healthy and a model is loaded"},
        503: {"description": "Model not loaded"},
    },
)
@handle_errors
@require_model
def health():
    return HealthResponse(
        status="healthy",
        model_version=server.version,
        model_name=server.name,
    )
