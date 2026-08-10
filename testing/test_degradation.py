from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run(csv_path: str, target: str, model_dir: str):
    root = Path(model_dir)
    model = joblib.load(root / "v1_xgboost_degradation.pkl")
    features = json.loads((root / "feature_columns.json").read_text())
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Ground-truth target '{target}' is missing.")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"External dataset is missing required features: {missing}")
    pred = model.predict(df[features])
    return {
        "MAE": mean_absolute_error(df[target], pred),
        "RMSE": mean_squared_error(df[target], pred) ** 0.5,
        "R2": r2_score(df[target], pred),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--model-dir", default="models/electrical_degradation")
    args = p.parse_args()
    print(run(args.csv, args.target, args.model_dir))
