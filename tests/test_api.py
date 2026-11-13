import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
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

    model_module.set_global_model(None, None)

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "ml churn service is running"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "dataset_loaded" in body


def test_model_schema(client):
    r = client.get("/model/schema")
    assert r.status_code == 200
    body = r.json()
    assert "features" in body
    assert "schema" in body


def test_predict_without_model_returns_503(client, valid_client_payload):
    r = client.post("/predict", json=valid_client_payload)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "MODEL_NOT_TRAINED"


def test_predict_validation_error(client):
    r = client.post("/predict", json={"monthly_fee": "abc"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(body["error"]["details"], list)


def test_train_with_unknown_model_returns_400(client):
    r = client.post("/model/train", json={"model_type": "xgboost"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "MODEL_TRAINING_ERROR"


def test_full_pipeline(client, sample_df, valid_client_payload, monkeypatch):
    import app.ml.dataset_store as dataset_store_module

    monkeypatch.setattr(dataset_store_module.store, "_df", sample_df)

    r = client.get("/dataset/info")
    assert r.status_code == 200
    assert r.json()["rows"] == len(sample_df)

    r = client.post("/model/train", json={"model_type": "logreg"})
    assert r.status_code == 200
    assert r.json()["model_type"] == "logreg"

    r = client.get("/model/status")
    assert r.status_code == 200
    assert r.json()["trained"] is True

    r = client.post("/predict", json=valid_client_payload)
    assert r.status_code == 200
    pred = r.json()
    assert pred["churn"] in (0, 1)
    assert 0.0 <= pred["churn_probability"] <= 1.0

    r = client.get("/model/metrics")
    assert r.status_code == 200
    assert r.json()["total_records"] >= 1