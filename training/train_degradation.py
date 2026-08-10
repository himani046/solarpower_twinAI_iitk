"""Train Model 2 from a prepared feature CSV.

The dataset-specific target and feature extraction must be confirmed after inspecting
PV Mismatch files. This script deliberately accepts an explicit target and does not
invent a degradation label.
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from config import MODEL2_DIR


def train(csv_path: str, target: str):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in prepared features.")
    numeric = df.select_dtypes(include="number").dropna(subset=[target])
    features = [c for c in numeric.columns if c != target]
    if not features:
        raise ValueError("No numeric features available.")

    train_df, test_df = train_test_split(
        numeric, test_size=0.2, random_state=42,
        shuffle=True,
    )
    medians = train_df[features].median()
    model = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42,
    )
    model.fit(train_df[features].fillna(medians), train_df[target])
    pred = model.predict(test_df[features].fillna(medians))
    metrics = {
        "MAE": float(mean_absolute_error(test_df[target], pred)),
        "RMSE": float(mean_squared_error(test_df[target], pred) ** 0.5),
        "R2": float(r2_score(test_df[target], pred)),
        "split": "80:20",
        "target": target,
    }
    MODEL2_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL2_DIR / "v1_xgboost_degradation.pkl")
    (MODEL2_DIR / "feature_columns.json").write_text(json.dumps(features, indent=2))
    (MODEL2_DIR / "preprocessing.json").write_text(json.dumps({"numeric_medians": medians.to_dict()}, indent=2))
    (MODEL2_DIR / "model_metadata.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    train(args.csv, args.target)
