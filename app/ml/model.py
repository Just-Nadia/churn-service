"""ML pipeline churn: обучение, сохранение, история."""

import json
from datetime import datetime, timezone
from typing import Any

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import (
    DEFAULT_HYPERPARAMS,
    HISTORY_PATH,
    META_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    SUPPORTED_MODELS,
)
from app.core.logging_config import get_logger
from app.ml.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

logger = get_logger("churn.model")

_model: Pipeline | None = None
_meta: dict | None = None


def create_classifier(model_type: str, hyperparameters: dict[str, Any]):
    if model_type == "logreg":
        return LogisticRegression(**{**DEFAULT_HYPERPARAMS["logreg"], **hyperparameters})
    if model_type == "random_forest":
        return RandomForestClassifier(**{**DEFAULT_HYPERPARAMS["random_forest"], **hyperparameters})
    raise ValueError(
        f"Unsupported model_type '{model_type}'. Supported: {sorted(SUPPORTED_MODELS)}"
    )


def build_pipeline(model_type: str = "logreg", hyperparameters: dict | None = None) -> Pipeline:
    hyperparameters = hyperparameters or {}
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", create_classifier(model_type, hyperparameters)),
        ]
    )


def train_churn_model(df, model_type: str = "logreg", hyperparameters: dict | None = None) -> dict:
    if df is None or df.empty:
        logger.error("Training aborted: dataset is empty")
        raise ValueError("Dataset is empty or not loaded")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        logger.error("Training aborted: missing columns %s", missing)
        raise ValueError(f"Missing columns in dataset: {missing}")

    hyperparameters = hyperparameters or {}
    logger.info(
        "Training started: model_type=%s hyperparameters=%s rows=%d",
        model_type,
        hyperparameters,
        len(df),
    )

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline(model_type, hyperparameters)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    try:
        roc_auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    except ValueError:
        roc_auc = None

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": roc_auc,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "Training finished: model_type=%s accuracy=%.4f f1=%.4f roc_auc=%s",
        model_type,
        metrics["accuracy"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return {"pipeline": pipeline, "metrics": metrics}


def save_churn_model(pipeline: Pipeline, metrics: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    META_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Model saved: path=%s", MODEL_PATH)


def load_churn_model() -> tuple[Pipeline | None, dict | None]:
    if not MODEL_PATH.exists():
        logger.info("No saved model found at %s", MODEL_PATH)
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    metrics = None
    if META_PATH.exists():
        metrics = json.loads(META_PATH.read_text(encoding="utf-8"))
    logger.info("Model loaded: path=%s", MODEL_PATH)
    return pipeline, metrics


def _load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(history: list[dict]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def append_training_history(metrics: dict) -> dict:
    record = {
        "trained_at": metrics.get("trained_at"),
        "model_type": metrics.get("model_type"),
        "hyperparameters": metrics.get("hyperparameters", {}),
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "train_rows": metrics.get("train_rows"),
        "test_rows": metrics.get("test_rows"),
    }
    history = _load_history()
    history.append(record)
    _save_history(history)
    logger.info("Training history updated: total_records=%d", len(history))
    return record


def get_training_history(limit: int = 10, model_type: str | None = None) -> dict:
    history = _load_history()
    if model_type:
        history = [h for h in history if h.get("model_type") == model_type]
    total = len(history)
    last_n = history[-limit:] if limit > 0 else []
    return {
        "total_records": total,
        "limit": limit,
        "model_type_filter": model_type,
        "latest": history[-1] if history else None,
        "history": list(reversed(last_n)),
    }


def compare_models() -> dict:
    history = _load_history()
    if not history:
        return {"models": [], "total_records": 0}

    by_type: dict[str, list[dict]] = {}
    for record in history:
        by_type.setdefault(record.get("model_type") or "unknown", []).append(record)

    def best(records: list[dict], key: str) -> dict | None:
        values = [r for r in records if r.get(key) is not None]
        if not values:
            return None
        top = max(values, key=lambda r: r[key])
        return {
            "value": top[key],
            "trained_at": top.get("trained_at"),
            "hyperparameters": top.get("hyperparameters"),
        }

    summary = [
        {
            "model_type": mt,
            "runs": len(records),
            "best_accuracy": best(records, "accuracy"),
            "best_f1": best(records, "f1"),
            "best_roc_auc": best(records, "roc_auc"),
        }
        for mt, records in sorted(by_type.items())
    ]
    return {"models": summary, "total_records": len(history)}


def set_global_model(pipeline: Pipeline | None, metrics: dict | None) -> None:
    global _model, _meta
    _model = pipeline
    _meta = metrics


def get_global_model() -> tuple[Pipeline | None, dict | None]:
    return _model, _meta


def is_model_loaded() -> bool:
    return _model is not None


def init_model_on_startup() -> None:
    pipeline, metrics = load_churn_model()
    if pipeline is not None:
        set_global_model(pipeline, metrics)