from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


class DegradationPredictor:
    """Inference wrapper for saved Model 2 artifacts."""

    def __init__(self, model_path: str | Path, feature_columns_path: str | Path):
        self.model = joblib.load(model_path)
        self.feature_columns = json.loads(Path(feature_columns_path).read_text())

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return pd.Series(self.model.predict(frame[self.feature_columns]), index=frame.index, name="degradation_prediction")
