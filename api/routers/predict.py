from fastapi import APIRouter

from api.schemas import (
    PredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse,
)
from api.server import server
from api.decorators import handle_errors, require_model, timed

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict diabetes for one patient",
    description=(
        "Returns the predicted class (`0` non-diabetic, `1` diabetic) and the class "
        "probabilities `[P(0), P(1)]` for a single feature payload."
    ),
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Invalid feature payload"},
        503: {"description": "Model not loaded"},
    },
)
@handle_errors
@require_model
@timed("predict")
def predict(req: PredictionRequest):
    pred, proba = server.predict(req.to_array())
    return PredictionResponse.from_prediction(pred[0], proba[0])


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict diabetes for a batch of patients",
    description="Same as `/predict`, but accepts a list of feature payloads in a single request.",
    responses={
        200: {"description": "Successful batch prediction"},
        422: {"description": "Invalid batch payload"},
        503: {"description": "Model not loaded"},
    },
)
@handle_errors
@require_model
@timed("predict_batch")
def predict_batch(req: BatchPredictionRequest):
    preds, probas = server.predict(req.to_array())
    return BatchPredictionResponse(
        predictions=[
            PredictionResponse.from_prediction(p, pr)
            for p, pr in zip(preds, probas)
        ]
    )
