"""Pydantic-модели churn сервиса."""

from typing import Any

from pydantic import BaseModel, Field


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    autopay_enabled: int
    region: str
    device_type: str
    payment_method: str


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    churn: int = Field(..., description="Предсказанный класс: 1 — уйдёт, 0 — останется")
    churn_probability: float = Field(..., description="Вероятность класса 1 (уйдёт)")
    stay_probability: float = Field(..., description="Вероятность класса 0 (останется)")


class TrainingConfigChurn(BaseModel):
    model_type: str = Field(
        "logreg",
        description="Тип модели: 'logreg' или 'random_forest'",
        examples=["logreg", "random_forest"],
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Гиперпараметры модели, передаются в конструктор sklearn",
        examples=[{"max_iter": 1000}, {"n_estimators": 300, "max_depth": 10}],
    )