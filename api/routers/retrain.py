from fastapi import APIRouter

from api.schemas import RetrainResponse
from api.server import server
from api.decorators import handle_errors, timed

from src.retrain import RetrainPipeline

router = APIRouter(prefix="/retrain", tags=["lifecycle"])

@router.post(
    "",
    response_model=RetrainResponse,
    summary="Trigger model retraining",
    description=(
        "Detects new CSV files dropped in `data/incoming/`, merges them into the main dataset, "
        "retrains the model with the same GridSearch grid, and reloads the API model in memory "
        "if the new run is better than the previous one. Returns `no_new_data` if nothing was found."
    ),
    responses={
        200: {"description": "Retraining executed or skipped"},
        500: {"description": "Internal pipeline failure"},
    },
)
@handle_errors
@timed("retrain")
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