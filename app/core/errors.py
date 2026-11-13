"""Общие форматы ошибок и кастомные исключения."""

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging_config import get_logger

logger = get_logger("churn.errors")


def error_payload(code: str, message: str, details: Any = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}


class DatasetNotLoadedError(Exception):
    pass


class DatasetEmptyError(Exception):
    pass


class ModelNotTrainedError(Exception):
    pass


class ModelTrainingError(Exception):
    pass


class PredictionError(Exception):
    pass


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning("HTTPException at %s %s: %s", request.method, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(
            code=f"HTTP_{exc.status_code}",
            message=str(exc.detail) if exc.detail else "HTTP error",
        ),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "Validation error at %s %s: %d issue(s)",
        request.method,
        request.url.path,
        len(exc.errors()),
    )
    return JSONResponse(
        status_code=422,
        content=error_payload(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=exc.errors(),
        ),
    )


async def dataset_not_loaded_handler(request: Request, exc: DatasetNotLoadedError):
    logger.error("DatasetNotLoadedError at %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=error_payload(
            code="DATASET_NOT_LOADED",
            message="Dataset is not loaded or cannot be read",
            details=str(exc),
        ),
    )


async def dataset_empty_handler(request: Request, exc: DatasetEmptyError):
    logger.error("DatasetEmptyError at %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=error_payload(code="DATASET_EMPTY", message="Dataset is empty", details=str(exc)),
    )


async def model_not_trained_handler(request: Request, exc: ModelNotTrainedError):
    logger.warning("ModelNotTrainedError at %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=503,
        content=error_payload(
            code="MODEL_NOT_TRAINED",
            message="Model is not trained yet. Call POST /model/train first.",
            details=None,
        ),
    )


async def model_training_handler(request: Request, exc: ModelTrainingError):
    logger.error("ModelTrainingError at %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=error_payload(
            code="MODEL_TRAINING_ERROR", message="Model training failed", details=str(exc)
        ),
    )


async def prediction_handler(request: Request, exc: PredictionError):
    logger.error("PredictionError at %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content=error_payload(
            code="PREDICTION_ERROR", message="Prediction failed", details=str(exc)
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception at %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=error_payload(
            code="INTERNAL_ERROR", message="Internal server error", details=None
        ),
    )


def register_exception_handlers(app) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(DatasetNotLoadedError, dataset_not_loaded_handler)
    app.add_exception_handler(DatasetEmptyError, dataset_empty_handler)
    app.add_exception_handler(ModelNotTrainedError, model_not_trained_handler)
    app.add_exception_handler(ModelTrainingError, model_training_handler)
    app.add_exception_handler(PredictionError, prediction_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)