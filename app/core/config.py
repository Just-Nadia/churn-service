"""Константы и пути проекта."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "churn_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
META_PATH = MODEL_DIR / "churn_model_meta.json"
HISTORY_PATH = MODEL_DIR / "training_history.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

SUPPORTED_MODELS = {"logreg", "random_forest"}
DEFAULT_HYPERPARAMS = {
    "logreg": {"max_iter": 1000, "random_state": RANDOM_STATE},
    "random_forest": {"n_estimators": 200, "random_state": RANDOM_STATE},
}