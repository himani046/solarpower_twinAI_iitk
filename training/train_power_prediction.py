"""Train Model 3 with a chronological 80:20 split.

Usage:
    python training/train_power_prediction.py --csv path/to/data.csv --target AC_POWER

The script intentionally requires the user to identify the target column from the
actual dataset rather than assuming a schema that may not exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import MODEL3_DIR


def train(csv_path: str, target: str, test_fraction: float = 0.2):
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    timestamp_candidates = [c for c in df.columns if c.lower() in {"timestamp", "date_time", "datetime", "date"}]
    if timestamp_candidates:
        ts = timestamp_candidates[0]
        df[ts] = pd.to_datetime(df[ts], errors="coerce")
        df = df.dropna(subset=[ts]).sort_values(ts)
        df["hour"] = df[ts].dt.hour
        df["day"] = df[ts].dt.day
        df["month"] = df[ts].dt.month
        df["day_of_year"] = df[ts].dt.dayofyear
        df["day_of_week"] = df[ts].dt.dayofweek
        df = df.drop(columns=[ts])

    numeric = df.select_dtypes(include="number").copy()
    numeric = numeric.dropna(subset=[target])
    features = [c for c in numeric.columns if c != target]
    if not features:
        raise ValueError("No numeric predictors remain after preprocessing.")

    split = int(len(numeric) * (1 - test_fraction))
    train_df, test_df = numeric.iloc[:split], numeric.iloc[split:]
    medians = train_df[features].median()
    X_train = train_df[features].fillna(medians)
    X_test = test_df[features].fillna(medians)
    y_train, y_test = train_df[target], test_df[target]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "R2": float(r2_score(y_test, pred)),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "split": "chronological 80:20",
        "target": target,
    }

    MODEL3_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL3_DIR / "v1_xgboost_power.pkl")
    (MODEL3_DIR / "feature_columns.json").write_text(json.dumps(features, indent=2))
    (MODEL3_DIR / "preprocessing.json").write_text(json.dumps({"numeric_medians": medians.to_dict()}, indent=2))
    (MODEL3_DIR / "model_metadata.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    train(args.csv, args.target)
