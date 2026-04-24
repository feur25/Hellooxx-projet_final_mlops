from api.routers.health import router as health_router
from api.routers.predict import router as predict_router
from api.routers.retrain import router as retrain_router
from api.routers.model import router as model_router

__all__ = ("health_router", "predict_router", "retrain_router", "model_router")
