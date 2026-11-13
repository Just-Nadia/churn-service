"""Хранилище тренировочного датасета churn."""

from pathlib import Path

import pandas as pd

from app.core.config import DATA_PATH
from app.core.logging_config import get_logger
from app.ml.features import FEATURES, TARGET
from app.schemas.churn import DatasetRowChurn

logger = get_logger("churn.dataset")


class DatasetStore:
    def __init__(self, path: Path = DATA_PATH):
        self.path = path
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            logger.error("Dataset file not found: %s", self.path)
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        df = pd.read_csv(self.path)
        missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
        if missing:
            logger.error("Dataset missing columns: %s", missing)
            raise ValueError(f"Missing columns in dataset: {missing}")
        logger.info(
            "Dataset loaded: path=%s rows=%d columns=%d churn_ratio=%.4f",
            self.path,
            len(df),
            df.shape[1],
            float(df[TARGET].mean()) if len(df) else 0.0,
        )
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def is_loaded(self) -> bool:
        return self._df is not None

    def preview(self, n: int = 5) -> list[dict]:
        return self.df.head(n).to_dict(orient="records")

    def info(self) -> dict:
        df = self.df
        churn_counts = df[TARGET].value_counts().to_dict()
        churn_counts = {int(k): int(v) for k, v in churn_counts.items()}
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "features": FEATURES,
            "target": TARGET,
            "churn_distribution": churn_counts,
        }

    def rows_as_models(self) -> list[DatasetRowChurn]:
        return [DatasetRowChurn(**row) for row in self.df.to_dict(orient="records")]


store = DatasetStore()