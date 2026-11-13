import pandas as pd
import pytest

from app.ml.model import build_pipeline, train_churn_model


def test_build_pipeline_has_expected_steps():
    pipe = build_pipeline("logreg")
    assert list(pipe.named_steps.keys()) == ["preprocessor", "classifier"]


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError):
        build_pipeline("xgboost")


def test_train_churn_model_returns_metrics(sample_df):
    result = train_churn_model(sample_df, model_type="logreg")
    metrics = result["metrics"]
    pipeline = result["pipeline"]

    assert pipeline is not None
    for key in ("accuracy", "f1", "roc_auc", "train_rows", "test_rows", "model_type", "trained_at"):
        assert key in metrics
    assert metrics["model_type"] == "logreg"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["train_rows"] > 0
    assert metrics["test_rows"] > 0


def test_train_churn_model_empty_df_raises():
    with pytest.raises(ValueError):
        train_churn_model(pd.DataFrame(), model_type="logreg")


def test_train_churn_model_missing_columns_raises(sample_df):
    broken = sample_df.drop(columns=["region"])
    with pytest.raises(ValueError):
        train_churn_model(broken, model_type="logreg")