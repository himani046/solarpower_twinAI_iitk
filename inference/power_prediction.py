from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


class PowerPredictor:
    """Inference wrapper for saved Model 3 artifacts."""

    def __init__(self, model_path: str | Path, feature_columns_path: str | Path):
        self.model = joblib.load(model_path)
        self.feature_columns = json.loads(Path(feature_columns_path).read_text())

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return pd.Series(self.model.predict(frame[self.feature_columns]), index=frame.index, name="expected_power")


def performance_deviation(expected: pd.Series, actual: pd.Series) -> pd.Series:
    denominator = expected.abs().replace(0, pd.NA)
    return ((expected - actual) / denominator * 100).fillna(0.0)
