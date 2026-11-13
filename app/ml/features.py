"""Единый источник правды о признаках churn."""

from typing import Any

NUMERIC_FEATURES: list[str] = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES: list[str] = [
    "region",
    "device_type",
    "payment_method",
]

FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET: str = "churn"

FEATURE_SCHEMA: dict[str, dict[str, Any]] = {
    "monthly_fee": {"type": "float", "required": True, "example": 25.5},
    "usage_hours": {"type": "float", "required": True, "example": 120.0},
    "support_requests": {"type": "int", "required": True, "example": 2},
    "account_age_months": {"type": "int", "required": True, "example": 14},
    "failed_payments": {"type": "int", "required": True, "example": 0},
    "region": {
        "type": "str",
        "required": True,
        "allowed": ["europe", "asia", "america", "africa"],
    },
    "device_type": {
        "type": "str",
        "required": True,
        "allowed": ["mobile", "desktop", "tablet"],
    },
    "payment_method": {
        "type": "str",
        "required": True,
        "allowed": ["card", "paypal", "crypto"],
    },
    "autopay_enabled": {"type": "int", "required": True, "allowed": [0, 1]},
}


def schema_response() -> dict:
    return {
        "features": FEATURES,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "target": TARGET,
        "schema": FEATURE_SCHEMA,
    }