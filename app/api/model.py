from fastapi import APIRouter, Query

from app.core.errors import (
    DatasetEmptyError,
    DatasetNotLoadedError,
    ModelTrainingError,
)
from app.core.logging_config import get_logger
from app.ml.dataset_store import store
from app.ml.features import FEATURES, schema_response
from app.ml.model import (
    append_training_history,
    compare_models,
    get_global_model,
    get_training_history,
    save_churn_model,
    set_global_model,
    train_churn_model,
)
from app.schemas.churn import TrainingConfigChurn

logger = get_logger("churn.api.model")
router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train", summary="Train churn model")
def model_train(config: TrainingConfigChurn | None = None):
    config = config or TrainingConfigChurn()
    logger.info(
        "POST /model/train requested: model_type=%s hyperparameters=%s",
        config.model_type,
        config.hyperparameters,
    )

    try:
        df = store.df
    except FileNotFoundError as e:
        raise DatasetNotLoadedError(str(e))

    if df is None or df.empty:
        raise DatasetEmptyError("churn_dataset.csv contains no rows")

    try:
        result = train_churn_model(
            df, model_type=config.model_type, hyperparameters=config.hyperparameters
        )
    except ValueError as e:
        logger.error("Training failed: %s", e)
        raise ModelTrainingError(str(e))

    save_churn_model(result["pipeline"], result["metrics"])
    set_global_model(result["pipeline"], result["metrics"])
    append_training_history(result["metrics"])
    return result["metrics"]


@router.get("/status")
def model_status():
    pipeline, metrics = get_global_model()
    if pipeline is None:
        return {
            "trained": False,
            "trained_at": None,
            "model_type": None,
            "hyperparameters": None,
            "metrics": None,
        }
    return {
        "trained": True,
        "trained_at": metrics.get("trained_at") if metrics else None,
        "model_type": metrics.get("model_type") if metrics else None,
        "hyperparameters": metrics.get("hyperparameters") if metrics else None,
        "metrics": metrics,
    }


@router.get("/metrics")
def model_metrics(
    limit: int = Query(10, ge=1, le=100),
    model_type: str | None = Query(None, description="Фильтр по типу модели"),
):
    return get_training_history(limit=limit, model_type=model_type)


@router.get("/metrics/compare")
def model_metrics_compare():
    return compare_models()


@router.get("/schema")
def model_schema():
    return schema_response()


@router.get("/features-order")
def model_features_order():
    return {"features": FEATURES}