from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class DegradationPredictor:
    """Inference wrapper for Model 2 electrical/thermal anomaly detection.

    Model 2 is intentionally unsupervised. It returns an anomaly risk from 0-100
    rather than pretending that an unvalidated degradation target exists.
    """

    def __init__(self, model_path: str | Path, scaler_path: str | Path, feature_columns_path: str | Path, preprocessing_path: str | Path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        payload = json.loads(Path(feature_columns_path).read_text())
        self.feature_columns = payload["features"] if isinstance(payload, dict) else payload
        self.preprocessing = json.loads(Path(preprocessing_path).read_text())
        mapping = self.preprocessing["score_mapping"]
        self.p1 = float(mapping["p1"])
        self.p99 = float(mapping["p99"])

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        x = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median()).fillna(0.0)
        scaled = self.scaler.transform(x)
        raw = -self.model.score_samples(scaled)
        risk = np.clip((raw - self.p1) / max(self.p99 - self.p1, 1e-9) * 100.0, 0, 100)
        return pd.DataFrame({"degradation_risk": risk}, index=frame.index)
