from fastapi import APIRouter

from app.ml.dataset_store import store
from app.ml.model import is_model_loaded

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"message": "ml churn service is running"}


@router.get(
    "/health",
    summary="Service health",
    description="Возвращает состояние сервиса: доступность модели и датасета.",
)
def health():
    model_loaded = is_model_loaded()
    dataset_loaded = store.is_loaded()
    status = "ok" if (model_loaded and dataset_loaded) else "degraded"
    return {"status": status, "model_loaded": model_loaded, "dataset_loaded": dataset_loaded}