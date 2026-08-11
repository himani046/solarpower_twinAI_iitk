"""Train SolarTwin AI Models 2 and 3 in one Colab run.

Model 2
-------
Electrical/thermal anomaly detection. The PV-Mismatch CSV is often headerless,
so the loader detects and preserves numeric first rows instead of accidentally
turning the first observation into column names. If a validated target column is
present, a supervised classifier is trained and accuracy/precision/recall/F1,
ROC-AUC, PR-AUC and a confusion matrix are reported. Otherwise the model remains
Isolation Forest and reports only valid unsupervised diagnostics (no invented
accuracy/F1).

Model 3
-------
Expected AC-power regression with a chronological split. DC_POWER is deliberately
excluded from the final predictor set so the model does not get an almost-direct
proxy for AC_POWER. Complete regression metrics and diagnostic plots are saved.

Raw datasets are downloaded from Kaggle but never committed to Git.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    max_error,
)
from sklearn.model_selection import train_test_split
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


def download_dataset(dataset_id: str) -> Path:
    import kagglehub
    return Path(kagglehub.dataset_download(dataset_id))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read normal CSVs and headerless numeric CSVs without losing row 1."""
    normal = pd.read_csv(path)
    # If every header is numeric-looking (e.g. 0.0, 0.0.1, ...), the source is
    # probably headerless and pandas consumed the first observation as a header.
    numeric_header_count = 0
    for c in normal.columns:
        try:
            float(str(c).replace(".1", ""))
            numeric_header_count += 1
        except Exception:
            pass
    if len(normal.columns) >= 2 and numeric_header_count / len(normal.columns) > 0.8:
        raw = pd.read_csv(path, header=None)
        raw.columns = [f"feature_{i}" for i in range(raw.shape[1])]
        return raw
    return normal


def load_all_csvs(root: Path) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    frames, schema = [], []
    for p in files:
        try:
            df = read_csv_robust(p)
            schema.append({"path": str(p), "shape": list(df.shape), "columns": [str(c) for c in df.columns]})
            if len(df):
                frames.append(df)
        except Exception as exc:
            schema.append({"path": str(p), "error": str(exc)})
    if not frames:
        raise RuntimeError("No readable CSV data found.")
    return pd.concat(frames, ignore_index=True, sort=False), schema


def find_label_column(df: pd.DataFrame) -> str | None:
    names = {str(c).strip().lower(): c for c in df.columns}
    exact = ["label", "target", "class", "fault", "fault_label", "anomaly", "status", "condition", "healthy", "mismatch"]
    for key in exact:
        if key in names:
            c = names[key]
            if 2 <= df[c].nunique(dropna=True) <= max(20, int(len(df) * 0.1)):
                return c
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ["label", "target", "class", "fault", "anomaly", "condition"]):
            if 2 <= df[c].nunique(dropna=True) <= max(20, int(len(df) * 0.1)):
                return c
    return None


def train_model2(root: Path) -> dict:
    print("\n================ MODEL 2: ELECTRICAL ANOMALY DETECTOR ================")
    df, schema = load_all_csvs(root)
    write_json(RESULTS / "model2" / "schema_report.json", {"dataset": MODEL2_KAGGLE, "files": schema})
    print(f"Rows loaded: {len(df)} | Columns: {len(df.columns)}")

    label_col = find_label_column(df)
    if label_col is not None:
        return train_model2_supervised(df, label_col)

    print("No validated target column detected -> using Isolation Forest.")
    numeric = df.select_dtypes(include="number").copy()
    if numeric.empty:
        raise RuntimeError("Model 2 dataset has no numeric measurements.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.dropna(axis=1, how="any")
    if numeric.shape[1] < 2:
        raise RuntimeError("Not enough usable numeric features for Model 2.")

    # Random holdout is used here because this dataset is not a chronological
    # time-series target; it avoids the artificial distribution jump caused by
    # putting all late observations into the test set.
    train, test = train_test_split(numeric, test_size=0.20, random_state=SEED)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train)
    x_test = scaler.transform(test)
    model = IsolationForest(n_estimators=300, contamination=0.05, random_state=SEED, n_jobs=-1)
    model.fit(x_train)

    train_raw = -model.score_samples(x_train)
    test_raw = -model.score_samples(x_test)
    lo, hi = np.percentile(train_raw, 1), np.percentile(train_raw, 99)
    denom = max(hi - lo, 1e-9)
    train_risk = np.clip((train_raw - lo) / denom * 100.0, 0, 100)
    test_risk = np.clip((test_raw - lo) / denom * 100.0, 0, 100)
    train_flag = (train_risk >= 50).astype(int)
    test_flag = (test_risk >= 50).astype(int)

    MODEL2_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL2_DIR / "v2_isolation_forest.pkl")
    joblib.dump(scaler, MODEL2_DIR / "scaler.pkl")
    write_json(MODEL2_DIR / "feature_columns.json", {"features": list(numeric.columns)})
    write_json(MODEL2_DIR / "preprocessing.json", {
        "imputation": "training median",
        "scaler": "StandardScaler",
        "score_mapping": {"formula": "clip((raw-p1)/(p99-p1)*100,0,100)", "p1": float(lo), "p99": float(hi)},
        "contamination": 0.05,
        "threshold_risk": 50.0,
    })

    metrics = {
        "task": "unsupervised electrical_thermal_anomaly_detection",
        "dataset": MODEL2_KAGGLE,
        "rows": int(len(numeric)),
        "features": list(numeric.columns),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_flag_rate": float(train_flag.mean()),
        "test_flag_rate": float(test_flag.mean()),
        "train_mean_risk": float(train_risk.mean()),
        "test_mean_risk": float(test_risk.mean()),
        "test_median_risk": float(np.median(test_risk)),
        "test_p95_risk": float(np.percentile(test_risk, 95)),
        "ground_truth_available": False,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "pr_auc": None,
        "confusion_matrix": None,
        "note": "No validated target exists in this dataset; classification metrics are intentionally null rather than fabricated.",
    }
    write_json(RESULTS / "model2" / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def train_model2_supervised(df: pd.DataFrame, label_col: str) -> dict:
    """Fallback supervised branch when a real target is present."""
    print(f"Validated-looking target detected: {label_col}")
    y = df[label_col].copy()
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid]
    if y.dtype == object:
        classes, y_encoded = np.unique(y.astype(str), return_inverse=True)
    else:
        classes = np.unique(y)
        mapping = {v: i for i, v in enumerate(classes)}
        y_encoded = y.map(mapping).to_numpy()
    if len(classes) != 2:
        raise RuntimeError(f"Detected target '{label_col}' is not binary; refusing to invent a binary anomaly target.")
    X = df.drop(columns=[label_col]).select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all").fillna(X.median(numeric_only=True)).dropna(axis=1, how="any")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=SEED, stratify=y_encoded)
    model = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    metrics = {
        "task": "supervised electrical_thermal_anomaly_detection",
        "dataset": MODEL2_KAGGLE,
        "target": str(label_col),
        "classes": [str(x) for x in classes],
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    MODEL2_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL2_DIR / "v2_supervised_random_forest.pkl")
    write_json(MODEL2_DIR / "feature_columns.json", {"features": list(X.columns)})
    write_json(MODEL2_DIR / "metrics.json", metrics)
    write_json(RESULTS / "model2" / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def find_power_files(root: Path) -> tuple[Path, Path | None]:
    generation = weather = None
    for p in sorted(root.rglob("*.csv")):
        try:
            cols = {str(c).strip().upper() for c in pd.read_csv(p, nrows=2).columns}
        except Exception:
            continue
        if "AC_POWER" in cols and "DATE_TIME" in cols:
            generation = p
        if "IRRADIATION" in cols and "DATE_TIME" in cols:
            weather = p
    if generation is None:
        raise FileNotFoundError("Could not find generation CSV containing AC_POWER and DATE_TIME.")
    return generation, weather


def train_model3(root: Path) -> dict:
    print("\n================ MODEL 3: EXPECTED POWER PREDICTOR ================")
    generation_path, weather_path = find_power_files(root)
    generation = pd.read_csv(generation_path)
    generation.columns = [str(c).strip().upper() for c in generation.columns]
    generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"], dayfirst=True, errors="coerce")
    generation = generation.dropna(subset=["DATE_TIME", "AC_POWER"]).copy()

    if weather_path is not None:
        weather = pd.read_csv(weather_path)
        weather.columns = [str(c).strip().upper() for c in weather.columns]
        weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], dayfirst=True, errors="coerce")
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

    # IMPORTANT: DC_POWER is excluded. It is too close to the AC_POWER target and
    # can make a power-forecast model look unrealistically perfect.
    features = [c for c in [
        "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
        "hour", "minute", "day_of_week", "day_of_year", "month",
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ] if c in df.columns]
    if not features:
        raise RuntimeError("No valid non-leaky Model 3 predictor columns found.")

    model_df = df[features + ["AC_POWER"]].copy()
    for c in features:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
    model_df["AC_POWER"] = pd.to_numeric(model_df["AC_POWER"], errors="coerce")
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

    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = float(np.sqrt(mse))
    nonzero = np.abs(actual) > 1e-6
    mape = float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100) if np.any(nonzero) else None

    metrics = {
        "task": "expected_ac_power_prediction",
        "dataset": MODEL3_KAGGLE,
        "generation_file": str(generation_path),
        "weather_file": str(weather_path) if weather_path else None,
        "features": features,
        "dc_power_excluded": True,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2_score(actual, pred)),
        "MAPE_percent_nonzero_actual": mape,
        "MedianAbsoluteError": float(median_absolute_error(actual, pred)),
        "ExplainedVariance": float(explained_variance_score(actual, pred)),
        "MaxError": float(max_error(actual, pred)),
        "split": "chronological 80:20",
    }

    MODEL3_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_MODEL3 = RESULTS / "model3"
    RESULTS_MODEL3.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL3_DIR / "v2_xgboost_power_no_dc.pkl")
    write_json(MODEL3_DIR / "feature_columns.json", {"features": features})
    write_json(MODEL3_DIR / "preprocessing.json", {"numeric_medians": medians.to_dict(), "dc_power_excluded": True})
    write_json(MODEL3_DIR / "model_metadata.json", metrics)
    write_json(RESULTS_MODEL3 / "metrics.json", metrics)

    prediction = pd.DataFrame({"actual_power": actual, "expected_power": pred})
    prediction["residual"] = prediction["actual_power"] - prediction["expected_power"]
    prediction["absolute_error"] = prediction["residual"].abs()
    prediction["deviation_percent"] = (
        prediction["residual"] / prediction["expected_power"].abs().replace(0, np.nan) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    prediction.head(5000).to_csv(RESULTS_MODEL3 / "test_predictions.csv", index=False)

    # Diagnostic plots
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, pred, s=8, alpha=0.35)
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual AC Power")
    plt.ylabel("Expected AC Power")
    plt.title("Model 3 — Actual vs Expected AC Power")
    plt.tight_layout()
    plt.savefig(RESULTS_MODEL3 / "actual_vs_expected.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(prediction["residual"], bins=50)
    plt.xlabel("Residual (Actual - Expected)")
    plt.ylabel("Count")
    plt.title("Model 3 — Residual Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS_MODEL3 / "residual_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(prediction["actual_power"].head(1000).to_numpy(), label="Actual")
    plt.plot(prediction["expected_power"].head(1000).to_numpy(), label="Expected")
    plt.xlabel("Test observation")
    plt.ylabel("AC Power")
    plt.title("Model 3 — Actual vs Expected Power (first 1000 test points)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_MODEL3 / "actual_vs_expected_timeseries.png", dpi=150)
    plt.close()

    print("\nMODEL 3 COMPLETE METRICS")
    print("----------------------------------------")
    for key in ["MAE", "MSE", "RMSE", "R2", "MAPE_percent_nonzero_actual", "MedianAbsoluteError", "ExplainedVariance", "MaxError"]:
        print(f"{key:30s}: {metrics[key]}")
    return metrics


def main() -> None:
    print("SolarTwin AI — Models 2 + 3")
    model2_root = download_dataset(MODEL2_KAGGLE)
    model3_root = download_dataset(MODEL3_KAGGLE)
    print("Model 2 dataset:", model2_root)
    print("Model 3 dataset:", model3_root)
    train_model2(model2_root)
    train_model3(model3_root)
    print("\nModels 2 and 3 complete. Raw datasets remain outside Git tracking.")


if __name__ == "__main__":
    main()
