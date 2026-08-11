"""Train SolarTwin AI Models 2 and 3 in one Colab run.

Model 2: unsupervised electrical/thermal anomaly detection with Isolation Forest.
It deliberately does not invent a degradation target when the PV-Mismatch dataset
has no validated target. It produces an anomaly score/risk that can feed the
Digital Twin risk engine.

Model 3: expected AC power prediction from environmental, electrical and temporal
features using XGBoost. The split is chronological to avoid future leakage.

The script downloads both configured Kaggle datasets automatically, discovers the
actual files, writes schema reports, trains both models, and saves all artifacts.
No raw dataset is committed to GitHub.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
RESULTS = REPO / "results"
MODEL2_DIR = REPO / "models" / "electrical_degradation"
MODEL3_DIR = REPO / "models" / "power_prediction"

MODEL2_KAGGLE = "himani04012007/pv-mismatch"
MODEL3_KAGGLE = "anikannal/solar-power-generation-data"


def download_dataset(dataset_id: str, destination: Path) -> Path:
    import kagglehub
    path = Path(kagglehub.dataset_download(dataset_id))
    destination.mkdir(parents=True, exist_ok=True)
    # Keep the Kaggle cache as the source of truth; copying large raw files into
    # the repo is unnecessary and data/raw is gitignored.
    return path


def csv_inventory(root: Path) -> list[dict]:
    items = []
    for p in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(p, nrows=5)
            items.append({"path": str(p), "columns": [str(c) for c in df.columns]})
        except Exception as exc:
            items.append({"path": str(p), "error": str(exc)})
    return items


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def train_model2(root: Path) -> dict:
    print("\n================ MODEL 2: ELECTRICAL ANOMALY DETECTOR ================")
    files = list(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    frames = []
    schema = []
    for p in files:
        try:
            df = pd.read_csv(p)
            schema.append({"path": str(p), "shape": list(df.shape), "columns": [str(c) for c in df.columns]})
            if len(df):
                frames.append(df)
        except Exception as exc:
            schema.append({"path": str(p), "error": str(exc)})

    write_json(RESULTS / "model2" / "schema_report.json", {"dataset": MODEL2_KAGGLE, "files": schema})
    if not frames:
        raise RuntimeError("Model 2 dataset contains no readable CSV data.")

    # Use only numeric measurements. Identifier-like columns are removed.
    df = pd.concat(frames, ignore_index=True, sort=False)
    numeric = df.select_dtypes(include="number").copy()
    if numeric.empty:
        raise RuntimeError("Model 2 dataset has no numeric measurements.")

    drop = []
    for c in numeric.columns:
        name = str(c).lower()
        nunique = numeric[c].nunique(dropna=True)
        if nunique <= 1 or "id" in name or "index" in name:
            drop.append(c)
    numeric = numeric.drop(columns=drop, errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.dropna(axis=1, how="any")

    if numeric.shape[1] < 2:
        raise RuntimeError("Not enough usable numeric features for Model 2.")

    # Fit on the first 80%; hold out the final 20% for an untouched anomaly-rate check.
    split = max(1, int(len(numeric) * 0.80))
    train = numeric.iloc[:split]
    test = numeric.iloc[split:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train)
    x_test = scaler.transform(test)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train)

    train_raw = -model.score_samples(x_train)
    test_raw = -model.score_samples(x_test)
    lo, hi = np.percentile(train_raw, 1), np.percentile(train_raw, 99)
    denom = max(hi - lo, 1e-9)
    train_risk = np.clip((train_raw - lo) / denom * 100.0, 0, 100)
    test_risk = np.clip((test_raw - lo) / denom * 100.0, 0, 100)

    MODEL2_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL2_DIR / "v1_isolation_forest.pkl")
    joblib.dump(scaler, MODEL2_DIR / "scaler.pkl")
    write_json(MODEL2_DIR / "feature_columns.json", {"features": list(numeric.columns)})
    write_json(MODEL2_DIR / "preprocessing.json", {
        "median_imputation": True,
        "scaler": "StandardScaler",
        "score_mapping": {"formula": "clip((raw - p1)/(p99-p1)*100, 0, 100)", "p1": float(lo), "p99": float(hi)},
        "contamination": 0.05,
    })

    metrics = {
        "task": "unsupervised electrical_thermal_anomaly_detection",
        "dataset": MODEL2_KAGGLE,
        "rows": int(len(numeric)),
        "features": list(numeric.columns),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_anomaly_rate_at_0_5": float(np.mean(train_risk >= 50.0)),
        "test_anomaly_rate_at_0_5": float(np.mean(test_risk >= 50.0)) if len(test) else None,
        "test_mean_risk": float(np.mean(test_risk)) if len(test) else None,
        "test_p95_risk": float(np.percentile(test_risk, 95)) if len(test) else None,
        "ground_truth": False,
        "note": "No accuracy/F1 is claimed because no validated anomaly/degradation target was invented.",
    }
    write_json(RESULTS / "model2" / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def find_power_files(root: Path) -> tuple[Path, Path]:
    csvs = list(root.rglob("*.csv"))
    generation = None
    weather = None
    for p in csvs:
        try:
            cols = {str(c).upper() for c in pd.read_csv(p, nrows=2).columns}
        except Exception:
            continue
        if "AC_POWER" in cols and "DATE_TIME" in cols:
            generation = p
        if "IRRADIATION" in cols and "DATE_TIME" in cols:
            weather = p
    if generation is None:
        raise FileNotFoundError("Could not find the Solar Power Generation CSV containing AC_POWER and DATE_TIME.")
    return generation, weather


def train_model3(root: Path) -> dict:
    print("\n================ MODEL 3: EXPECTED POWER PREDICTOR ================")
    generation_path, weather_path = find_power_files(root)
    generation = pd.read_csv(generation_path)
    generation.columns = [str(c).strip().upper() for c in generation.columns]
    generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"], errors="coerce")
    generation = generation.dropna(subset=["DATE_TIME", "AC_POWER"]).copy()

    if weather_path is not None:
        weather = pd.read_csv(weather_path)
        weather.columns = [str(c).strip().upper() for c in weather.columns]
        weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce")
        weather = weather.dropna(subset=["DATE_TIME"]).copy()
        merge_keys = ["DATE_TIME"]
        if "PLANT_ID" in generation.columns and "PLANT_ID" in weather.columns:
            merge_keys.append("PLANT_ID")
        weather_cols = [c for c in ["DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"] if c in weather.columns]
        weather = weather[weather_cols].drop_duplicates(merge_keys)
        df = generation.merge(weather, on=merge_keys, how="left")
    else:
        df = generation.copy()

    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    df["hour"] = df["DATE_TIME"].dt.hour
    df["minute"] = df["DATE_TIME"].dt.minute
    df["day_of_week"] = df["DATE_TIME"].dt.dayofweek
    df["day_of_year"] = df["DATE_TIME"].dt.dayofyear
    df["month"] = df["DATE_TIME"].dt.month
    df["hour_sin"] = np.sin(2 * math.pi * (df["hour"] + df["minute"] / 60) / 24)
    df["hour_cos"] = np.cos(2 * math.pi * (df["hour"] + df["minute"] / 60) / 24)
    df["day_sin"] = np.sin(2 * math.pi * df["day_of_year"] / 365.25)
    df["day_cos"] = np.cos(2 * math.pi * df["day_of_year"] / 365.25)

    # DC_POWER is an allowed operational predictor. AC_POWER itself and cumulative
    # yields are excluded to prevent target leakage.
    preferred = [
        "DC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
        "hour", "minute", "day_of_week", "day_of_year", "month",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
    ]
    features = [c for c in preferred if c in df.columns and c != "AC_POWER"]
    if not features:
        raise RuntimeError("No validated predictor columns available for Model 3.")

    model_df = df[features + ["AC_POWER"]].copy()
    for c in features:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
    model_df = model_df.dropna(subset=["AC_POWER"])
    medians = model_df[features].median()
    model_df[features] = model_df[features].fillna(medians)

    split = max(1, int(len(model_df) * 0.80))
    train = model_df.iloc[:split]
    test = model_df.iloc[split:]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(train[features], train["AC_POWER"])
    pred = model.predict(test[features])
    actual = test["AC_POWER"].to_numpy()

    nonzero = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if np.any(nonzero) else None
    metrics = {
        "task": "expected_ac_power_prediction",
        "dataset": MODEL3_KAGGLE,
        "generation_file": str(generation_path),
        "weather_file": str(weather_path) if weather_path else None,
        "features": features,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
        "R2": float(r2_score(actual, pred)),
        "MAPE_percent_nonzero_actual": mape,
        "split": "chronological 80:20",
    }

    MODEL3_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL3_DIR / "v1_xgboost_power.pkl")
    write_json(MODEL3_DIR / "feature_columns.json", {"features": features})
    write_json(MODEL3_DIR / "preprocessing.json", {"numeric_medians": medians.to_dict()})
    write_json(MODEL3_DIR / "model_metadata.json", metrics)

    # Save a small, reproducible test prediction file rather than raw data.
    prediction = pd.DataFrame({"actual_power": actual, "expected_power": pred})
    prediction["deviation_percent"] = ((prediction["expected_power"] - prediction["actual_power"]) / prediction["expected_power"].abs().replace(0, np.nan) * 100).fillna(0)
    RESULTS_MODEL3 = RESULTS / "model3"
    RESULTS_MODEL3.mkdir(parents=True, exist_ok=True)
    prediction.head(5000).to_csv(RESULTS_MODEL3 / "test_predictions.csv", index=False)
    write_json(RESULTS_MODEL3 / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    print("SolarTwin AI — Models 2 + 3")
    model2_root = download_dataset(MODEL2_KAGGLE, RAW / "model2")
    model3_root = download_dataset(MODEL3_KAGGLE, RAW / "model3")
    print("Model 2 dataset:", model2_root)
    print("Model 3 dataset:", model3_root)
    train_model2(model2_root)
    train_model3(model3_root)
    print("\nModels 2 and 3 complete. Raw datasets remain outside Git tracking.")


if __name__ == "__main__":
    main()
