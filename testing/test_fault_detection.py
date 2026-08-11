"""External testing for the saved multi-label PV anomaly model.

The external dataset must provide multi-label annotations using either:
- one subdirectory per anomaly, with exact duplicate images grouped across
  directories, or
- a manifest JSON with {"path": ..., "labels": [...]} records.

If the external dataset has no ground truth, this script can still perform
inference but must report that quantitative validation is unavailable.

By default the repository's validation-selected per-anomaly thresholds are
used. A scalar --threshold can still be supplied for backward-compatible
experiments where the same threshold is intentionally applied to every class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, hamming_loss
from torchvision import models, transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_manifest(root: Path, classes: list[str]) -> list[dict]:
    groups = defaultdict(lambda: {"path": None, "labels": set()})
    for class_name in classes:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                groups[digest]["path"] = str(path)
                groups[digest]["labels"].add(class_name)
    return [
        {"path": v["path"], "labels": sorted(v["labels"])}
        for v in groups.values()
    ]


def load_thresholds(model_dir: Path, classes: list[str], scalar: float | None):
    checkpoint = torch.load(
        model_dir / "v2_convnext_pvmd_multilabel.pth",
        map_location="cpu",
        weights_only=False,
    )
    saved_threshold = float(checkpoint.get("threshold", 0.5))

    if scalar is not None:
        return {name: float(scalar) for name in classes}

    config_path = Path("config/model1_thresholds.json")
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text())
            configured = payload.get("thresholds", {})
            return {
                name: float(configured.get(name, saved_threshold))
                for name in classes
            }
        except (OSError, json.JSONDecodeError):
            pass

    return {name: saved_threshold for name in classes}


def run(dataset_root: str, model_dir: str, threshold: float | None = None):
    model_dir = Path(model_dir)
    classes = json.loads((model_dir / "class_names.json").read_text())
    checkpoint = torch.load(
        model_dir / "v2_convnext_pvmd_multilabel.pth",
        map_location="cpu",
        weights_only=False,
    )
    thresholds = load_thresholds(model_dir, classes, threshold)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = torch.nn.Linear(
        model.classifier[2].in_features,
        len(classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    records = build_manifest(Path(dataset_root), classes)
    predictions = []
    y_true, y_pred = [], []

    threshold_array = np.array([
        thresholds[name] for name in classes
    ])

    for record in records:
        image = Image.open(record["path"]).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(
                model(tensor)
            )[0].cpu().numpy()

        pred = (probs >= threshold_array).astype(int)

        predictions.append({
            "path": record["path"],
            "detected_anomalies": [
                classes[i]
                for i, value in enumerate(pred)
                if value
            ],
            "probabilities": {
                classes[i]: float(probs[i])
                for i in range(len(classes))
            },
            "thresholds": thresholds,
        })

        truth = np.zeros(len(classes), dtype=int)
        for label in record["labels"]:
            if label in classes:
                truth[classes.index(label)] = 1

        y_true.append(truth)
        y_pred.append(pred)

    result = {
        "samples": len(records),
        "thresholds": thresholds,
        "predictions": predictions,
    }

    if records:
        y_true = np.stack(y_true)
        y_pred = np.stack(y_pred)
        result["ground_truth_metrics"] = {
            "micro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="micro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "hamming_loss": float(
                hamming_loss(y_true, y_pred)
            ),
        }
    else:
        result["ground_truth_metrics"] = None
        result["note"] = (
            "No compatible labelled external records were found; "
            "predictions are inference only."
        )

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--model-dir", default="models/fault_detection")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    print(
        json.dumps(
            run(args.dataset_root, args.model_dir, args.threshold),
            indent=2,
        )
    )
