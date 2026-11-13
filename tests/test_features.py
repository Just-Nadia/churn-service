from app.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    schema_response,
)


def test_features_split_is_consistent():
    assert FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_feature_lists_not_empty():
    assert NUMERIC_FEATURES
    assert CATEGORICAL_FEATURES
    assert TARGET == "churn"


def test_schema_response_covers_all_features():
    payload = schema_response()
    assert payload["features"] == FEATURES
    assert payload["numeric"] == NUMERIC_FEATURES
    assert payload["categorical"] == CATEGORICAL_FEATURES
    assert payload["target"] == TARGET
    for name in FEATURES:
        assert name in payload["schema"]
        assert "type" in payload["schema"][name]