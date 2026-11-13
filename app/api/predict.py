import pandas as pd
from fastapi import APIRouter

from app.core.errors import ModelNotTrainedError, PredictionError
from app.core.logging_config import get_logger
from app.ml.model import get_global_model
from app.schemas.churn import FeatureVectorChurn, PredictionResponseChurn

logger = get_logger("churn.api.predict")
router = APIRouter(tags=["predict"])


def _predict_one(pipeline, features: FeatureVectorChurn) -> dict:
    try:
        row = pd.DataFrame([features.model_dump()])
        proba = pipeline.predict_proba(row)[0]
        pred = int(pipeline.predict(row)[0])
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise PredictionError(str(e))
    return {
        "churn": pred,
        "churn_probability": round(float(proba[1]), 4),
        "stay_probability": round(float(proba[0]), 4),
    }


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    summary="Predict churn for one client or a list of clients",
    responses={
        503: {"description": "Model is not trained yet"},
        422: {"description": "Validation error"},
        400: {"description": "Prediction error or empty list"},
    },
)
def predict(payload: FeatureVectorChurn | list[FeatureVectorChurn]):
    pipeline, _ = get_global_model()
    if pipeline is None:
        logger.warning("POST /predict called, but model is not trained")
        raise ModelNotTrainedError("model not trained")

    if isinstance(payload, list):
        if not payload:
            raise PredictionError("Empty list of clients")
        logger.info("POST /predict: batch size=%d", len(payload))
        return [_predict_one(pipeline, item) for item in payload]

    logger.info("POST /predict: single client")
    return _predict_one(pipeline, payload)