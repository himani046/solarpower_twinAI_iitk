"""Train Model 3: expected solar power prediction with XGBoost.

Uses a chronological 80/20 split. The script searches data/raw/model3 for CSV files,
normalizes common column names, selects AC power as the target when present, and
saves the model plus feature metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SEED = 42


def norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def load_csvs(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df):
                frames.append(df)
                print(f"loaded {f}: {df.shape}")
        except Exception as exc:
            print(f"skipping {f}: {exc}")
    if not frames:
        raise RuntimeError("No readable CSV files found")
    return pd.concat(frames, ignore_index=True, sort=False)


def choose_column(columns, candidates):
    by_norm = {norm(c): c for c in columns}
    for candidate in candidates:
        if norm(candidate) in by_norm:
            return by_norm[norm(candidate)]
    for c in columns:
        nc = norm(c)
        if any(norm(x) in nc for x in candidates):
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/model3")
    parser.add_argument("--output", default="models/power_prediction/v1_xgboost_power.pkl")
    args = parser.parse_args()

    df = load_csvs(Path(args.data))
    time_col = choose_column(df.columns, ["DATE_TIME", "Timestamp", "DateTime", "Date"])
    target = choose_column(df.columns, ["AC_POWER", "AC Power", "Power"])
    if target is None:
        raise ValueError(f"Could not identify a power target. Columns: {list(df.columns)}")

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        time_features = pd.DataFrame({
            "hour": df[time_col].dt.hour,
            "minute": df[time_col].dt.minute,
            "dayofweek": df[time_col].dt.dayofweek,
            "dayofyear": df[time_col].dt.dayofyear,
            "month": df[time_col].dt.month,
        }, index=df.index)
        df = pd.concat([df, time_features], axis=1)

    numeric = df.select_dtypes(include=[np.number]).copy()
    target = target if target in numeric.columns else choose_column(numeric.columns, [target])
    if target is None:
        raise ValueError("Power target is not numeric after parsing")

    # Remove target and obvious identifier columns. Do not use future target values.
    drop_cols = {target}
    for c in numeric.columns:
        nc = norm(c)
        if nc in {"plantid", "sourcekey", "id", "unnamed0"} or nc.startswith("unnamed"):
            drop_cols.add(c)
    X = numeric.drop(columns=list(drop_cols), errors="ignore")
    y = numeric[target].astype(float)
    valid = y.notna() & np.isfinite(y)
    X, y = X.loc[valid], y.loc[valid]
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    split = int(len(X) * 0.80)
    if split <= 0 or split >= len(X):
        raise ValueError("Dataset is too small for an 80:20 split")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }
    denom = np.where(np.abs(y_test.to_numpy()) < 1e-8, np.nan, np.abs(y_test.to_numpy()))
    metrics["mape_percent"] = float(np.nanmean(np.abs((y_test.to_numpy() - pred) / denom)) * 100)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    Path("models/power_prediction/feature_columns.json").write_text(json.dumps(list(X.columns), indent=2), encoding="utf-8")
    Path("models/power_prediction/model_metadata.json").write_text(json.dumps({"dataset": "anikannal/solar-power-generation-data", "target": target, "split": "chronological 80:20", "random_state": SEED, "metrics": metrics}, indent=2), encoding="utf-8")
    Path("results/model3").mkdir(parents=True, exist_ok=True)
    Path("results/model3/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved model to {out}")


if __name__ == "__main__":
    main()
