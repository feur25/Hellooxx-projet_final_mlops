import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import server
from api.routers import health_router, predict_router, retrain_router, model_router
from api.openapi import TAGS_METADATA, DESCRIPTION, CONTACT, LICENSE_INFO, SERVERS


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        server.load()
    except FileNotFoundError:
        pass
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Diabetes Prediction API",
        version="1.0.0",
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        contact=CONTACT,
        license_info=LICENSE_INFO,
        servers=SERVERS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (health_router, predict_router, retrain_router, model_router):
        app.include_router(router)

    return app


app = create_app()
