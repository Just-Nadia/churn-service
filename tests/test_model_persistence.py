import pytest

from app.ml.model import (
    append_training_history,
    get_training_history,
    load_churn_model,
    save_churn_model,
    train_churn_model,
)


@pytest.fixture(autouse=True)
def temp_model_dir(tmp_path, monkeypatch):
    import app.core.config as config
    import app.ml.model as model_module

    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(config, "META_PATH", tmp_path / "churn_model_meta.json")
    monkeypatch.setattr(config, "HISTORY_PATH", tmp_path / "training_history.json")

    monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(model_module, "META_PATH", tmp_path / "churn_model_meta.json")
    monkeypatch.setattr(model_module, "HISTORY_PATH", tmp_path / "training_history.json")
    yield tmp_path


def test_save_and_load_roundtrip(sample_df):
    result = train_churn_model(sample_df, model_type="logreg")
    save_churn_model(result["pipeline"], result["metrics"])

    loaded_pipeline, loaded_metrics = load_churn_model()
    assert loaded_pipeline is not None
    assert loaded_metrics is not None
    assert loaded_metrics["model_type"] == "logreg"


def test_load_without_files_returns_none():
    pipeline, metrics = load_churn_model()
    assert pipeline is None
    assert metrics is None


def test_history_append_and_read(sample_df, temp_model_dir):
    result = train_churn_model(sample_df, model_type="logreg")
    append_training_history(result["metrics"])

    history = get_training_history(limit=10)
    assert history["total_records"] == 1
    assert history["latest"]["model_type"] == "logreg"
    assert (temp_model_dir / "training_history.json").exists()


def test_history_filter_by_model_type(sample_df, temp_model_dir):
    r1 = train_churn_model(sample_df, model_type="logreg")
    r2 = train_churn_model(sample_df, model_type="random_forest")
    append_training_history(r1["metrics"])
    append_training_history(r2["metrics"])

    only_rf = get_training_history(limit=10, model_type="random_forest")
    assert only_rf["total_records"] == 1
    assert only_rf["latest"]["model_type"] == "random_forest"