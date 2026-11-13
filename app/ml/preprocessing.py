"""Подготовка churn данных и разбиение на train/test."""

from sklearn.model_selection import train_test_split

from app.core.config import RANDOM_STATE, TEST_SIZE
from app.ml.dataset_store import store
from app.ml.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def prepare_xy() -> tuple:
    df = store.df.copy()
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("unknown")
    return df[FEATURES], df[TARGET]


def split_data() -> dict:
    X, y = prepare_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    def dist(series):
        counts = series.value_counts().to_dict()
        counts = {int(k): int(v) for k, v in counts.items()}
        total = sum(counts.values())
        return {
            "counts": counts,
            "shares": {k: round(v / total, 4) for k, v in counts.items()},
        }

    return {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": True,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "total_rows": int(len(y)),
        "train": {"rows": int(len(y_train)), "churn": dist(y_train)},
        "test": {"rows": int(len(y_test)), "churn": dist(y_test)},
    }