from fastapi import FastAPI

from app.api import dataset, health, model, predict
from app.core.errors import register_exception_handlers
from app.core.logging_config import get_logger, setup_logging
from app.ml.features import FEATURES
from app.ml.model import init_model_on_startup
from app.schemas.churn import FeatureVectorChurn

setup_logging()
logger = get_logger("churn.api")

app = FastAPI(title="ML Churn Service")


def _assert_feature_consistency() -> None:
    model_fields = list(FeatureVectorChurn.model_fields.keys())
    if model_fields != FEATURES:
        raise RuntimeError(
            "FeatureVectorChurn fields do not match FEATURES order:\n"
            f"  model: {model_fields}\n  features: {FEATURES}"
        )


_assert_feature_consistency()
register_exception_handlers(app)


@app.on_event("startup")
def startup_load_model():
    logger.info("Application startup: loading model if exists")
    init_model_on_startup()


app.include_router(health.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(predict.router)