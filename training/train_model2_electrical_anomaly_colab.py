"""Train SolarTwin AI Model 2 from PV-Mismatch thermal measurements.

Improves the previous version by explicitly decoding UTF-8/UTF-16/BOM/legacy
text thermal CSVs, preserving their source-file labels (Clean/Dirt/Shadow),
and using leakage-safe source-file group splits. Both a healthy-vs-defective
classifier and a 3-class fault-type classifier are evaluated. An Isolation
Forest risk score is retained as a secondary unsupervised signal.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "model2"
MODEL_DIR = REPO / "models" / "electrical_degradation"
DATASET = "himani04012007/pv-mismatch"
LABELS = {"clean": 0, "dirt": 1, "shadow": 2}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def decode_csv(path: Path) -> pd.DataFrame:
    """Decode numeric thermal matrices stored with common text encodings."""
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin1"]
    errors = []
    for enc in encodings:
        try:
            raw = pd.read_csv(path, header=None, encoding=enc)
            num = raw.apply(pd.to_numeric, errors="coerce")
            # Require meaningful numeric content; do not accept a text-only parse.
            if num.notna().sum().sum() >= max(4, int(num.size * 0.25)):
                num = num.dropna(axis=0, how="all").dropna(axis=1, how="all")
                if num.shape[0] >= 2 and num.shape[1] >= 2:
                    return num
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise ValueError("No usable numeric matrix: " + " | ".join(errors))


def thermal_features(mat: pd.DataFrame, tile: int = 16) -> list[list[float]]:
    """Generate compact thermal statistics for non-overlapping tiles."""
    a = np.asarray(mat, dtype=float)
    if a.ndim != 2 or min(a.shape) < 2:
        return []
    finite = np.isfinite(a)
    fill = float(np.nanmedian(a)) if finite.any() else 0.0
    a = np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)
    h, w = a.shape
    # Adaptive tile size: for tiny matrices use a single full-image tile.
    ts = min(tile, h, w)
    if ts < 2:
        return []
    rows = list(range(0, max(1, h - ts + 1), ts))
    cols = list(range(0, max(1, w - ts + 1), ts))
    if not rows:
        rows = [0]
    if not cols:
        cols = [0]
    out = []
    for r in rows:
        for c in cols:
            z = a[r:min(r + ts, h), c:min(c + ts, w)]
            gx = np.diff(z, axis=1) if z.shape[1] > 1 else np.zeros_like(z)
            gy = np.diff(z, axis=0) if z.shape[0] > 1 else np.zeros_like(z)
            q5, q25, q50, q75, q95 = np.percentile(z, [5, 25, 50, 75, 95])
            mean = float(z.mean())
            std = float(z.std()) + 1e-8
            out.append([
                mean,
                std,
                float(z.min()),
                float(z.max()),
                float(q5), float(q25), float(q50), float(q75), float(q95),
                float(np.mean(np.abs(gx))),
                float(np.mean(np.abs(gy))),
                float(np.mean(gx ** 2)),
                float(np.mean(gy ** 2)),
                float(np.mean(z > mean + 2 * std)),
                float(np.mean(z > q95)),
                float(r / max(h, 1)),
                float(c / max(w, 1)),
            ])
    return out


def label_from_filename(path: Path) -> str | None:
    stem = path.stem.lower()
    for key in LABELS:
        if re.search(rf"(^|[_ -]){key}($|[_ -])", stem):
            return key
    return None


def load_records(root: Path):
    records = []
    skipped = []
    files = sorted(root.rglob("*.csv"))
    for path in files:
        label = label_from_filename(path)
        if label is None:
            continue
        try:
            matrix = decode_csv(path)
            feats = thermal_features(matrix)
            if not feats:
                raise ValueError(f"No usable thermal features; shape={matrix.shape}")
            for tile_id, feature in enumerate(feats):
                records.append({
                    "features": feature,
                    "label": label,
                    "source": str(path),
                    "tile": tile_id,
                })
        except Exception as exc:
            skipped.append({"file": str(path), "label": label, "error": str(exc)})
    return records, skipped, len(files)


def binary_metrics(y_true, y_pred, prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Healthy", "Defective"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    print("SolarTwin AI — Model 2 (thermal supervised + anomaly risk)")
    import kagglehub

    root = Path(kagglehub.dataset_download(DATASET))
    print("Dataset:", root)

    records, skipped, total_csvs = load_records(root)
    if not records:
        raise RuntimeError("No labelled thermal CSVs could be decoded.")

    frame = pd.DataFrame(records)
    X = np.asarray(frame["features"].tolist(), dtype=float)
    y_type = frame["label"].map(LABELS).to_numpy()
    y_binary = (y_type > 0).astype(int)
    groups = frame["source"].to_numpy()

    # ------------------------------------------------------------
    # Group-aware 70/15/15 split by source thermal CSV.
    # No tiles from one source file can cross train/val/test.
    # ------------------------------------------------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y_binary, groups))
    train_groups = groups[train_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2143, random_state=SEED)
    tr_rel, val_rel = next(inner.split(X[train_idx], y_binary[train_idx], train_groups))
    tr_idx = train_idx[tr_rel]
    val_idx = train_idx[val_rel]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[tr_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    # ------------------------------------------------------------
    # Model 2A: Healthy vs Defective
    # ------------------------------------------------------------
    binary_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    binary_model.fit(X_train, y_binary[tr_idx])
    binary_prob = binary_model.predict_proba(X_test)[:, 1]
    binary_pred = (binary_prob >= 0.50).astype(int)
    healthy_defective = binary_metrics(y_binary[test_idx], binary_pred, binary_prob)

    # ------------------------------------------------------------
    # Model 2B: Clean / Dirt / Shadow
    # ------------------------------------------------------------
    type_model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    type_model.fit(X_train, y_type[tr_idx])
    type_pred = type_model.predict(X_test)
    fault_type = {
        "accuracy": float(accuracy_score(y_type[test_idx], type_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_type[test_idx], type_pred)),
        "precision_macro": float(precision_score(y_type[test_idx], type_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_type[test_idx], type_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_type[test_idx], type_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_type[test_idx], type_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_type[test_idx],
            type_pred,
            target_names=["Clean", "Dirt", "Shadow"],
            output_dict=True,
            zero_division=0,
        ),
    }

    # ------------------------------------------------------------
    # Secondary anomaly risk
    # ------------------------------------------------------------
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=SEED,
        n_jobs=-1,
    ).fit(X_train)
    train_raw = -iso.score_samples(X_train)
    test_raw = -iso.score_samples(X_test)
    p1, p99 = np.percentile(train_raw, 1), np.percentile(train_raw, 99)
    risk = np.clip((test_raw - p1) / max(p99 - p1, 1e-9) * 100.0, 0, 100)
    risk_metrics = {
        "mean": float(risk.mean()),
        "median": float(np.median(risk)),
        "p95": float(np.percentile(risk, 95)),
        "flag_rate_at_50": float((risk >= 50).mean()),
    }

    metrics = {
        "task": "thermal_pv_anomaly_detection",
        "dataset": DATASET,
        "total_csv_files_found": int(total_csvs),
        "decoded_labelled_files": int(frame["source"].nunique()),
        "skipped_files": int(len(skipped)),
        "usable_tiles": int(len(frame)),
        "split": "group-aware 70/15/15 by source thermal file",
        "train_tiles": int(len(tr_idx)),
        "validation_tiles": int(len(val_idx)),
        "test_tiles": int(len(test_idx)),
        "healthy_defective": healthy_defective,
        "fault_type": fault_type,
        "risk": risk_metrics,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(binary_model, MODEL_DIR / "v3_random_forest_healthy_defective.pkl")
    joblib.dump(type_model, MODEL_DIR / "v3_random_forest_fault_type.pkl")
    joblib.dump(iso, MODEL_DIR / "v3_isolation_forest.pkl")
    joblib.dump(scaler, MODEL_DIR / "thermal_scaler.pkl")

    write_json(MODEL_DIR / "metadata.json", {
        "feature_count": int(X.shape[1]),
        "labels": LABELS,
        "threshold": 0.50,
        "split": "group-aware by source thermal file",
        "risk_threshold": 50.0,
    })
    write_json(RESULTS / "metrics.json", metrics)
    write_json(RESULTS / "skipped_files.json", {"skipped": skipped})

    prediction_df = pd.DataFrame({
        "source": groups[test_idx],
        "true_label": [["Clean", "Dirt", "Shadow"][i] for i in y_type[test_idx]],
        "healthy_defective_probability": binary_prob,
        "predicted_healthy_defective": ["Defective" if x else "Healthy" for x in binary_pred],
        "predicted_fault_type": [["Clean", "Dirt", "Shadow"][i] for i in type_pred],
        "risk": risk,
    })
    prediction_df.to_csv(RESULTS / "test_predictions.csv", index=False)

    print("\nMODEL 2 COMPLETE")
    print(f"Decoded labelled files: {frame['source'].nunique()} | Tiles: {len(frame)} | Skipped: {len(skipped)}")
    print("\nHEALTHY vs DEFECTIVE")
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        value = healthy_defective[key]
        print(f"{key:22s}: {value:.4f}" if value is not None else f"{key:22s}: N/A")
    print("Confusion Matrix:")
    print(np.asarray(healthy_defective["confusion_matrix"]))

    print("\nFAULT TYPE (Clean/Dirt/Shadow)")
    for key in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"{key:22s}: {fault_type[key]:.4f}")
    print("Confusion Matrix:")
    print(np.asarray(fault_type["confusion_matrix"]))

    print(
        "\nRisk: mean={:.2f}, median={:.2f}, P95={:.2f}, flag@50={:.2%}".format(
            risk_metrics["mean"], risk_metrics["median"], risk_metrics["p95"], risk_metrics["flag_rate_at_50"]
        )
    )


if __name__ == "__main__":
    main()
