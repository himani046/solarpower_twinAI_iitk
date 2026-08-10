"""Train Model 1 as a multi-label PV anomaly detector.

The PVMD source is organized as one folder per anomaly, but the audit of the
Kaggle copy shows many exact duplicate images occurring under multiple anomaly
folders. Instead of treating those duplicates as contradictory single-class
samples, this trainer collapses identical image bytes into one sample and takes
the union of all folder labels as its multi-label target.

The model therefore answers: which of Crack, Hotspot and Shading are present?
A panel is considered "defective" at inference time when at least one anomaly
probability exceeds the configured threshold.

Important: the supplied PVMD dataset contains only anomalous examples; it does
not contain a healthy/normal class. Therefore this script does NOT claim to
train a validated healthy-vs-defective classifier. A healthy class can be added
later through a documented healthy dataset and retraining.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
DEFAULT_THRESHOLD = 0.50
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_image_root(base: Path) -> Path:
    for p in [base, *base.rglob("*")]:
        if not p.is_dir():
            continue
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if len(subdirs) >= 2 and all(
            any(f.suffix.lower() in IMAGE_EXTS for f in d.rglob("*")) for d in subdirs
        ):
            return p
    raise FileNotFoundError(f"Could not find PVMD class folders under {base}")


def build_manifest(root: Path) -> list[dict]:
    groups: dict[str, dict] = {}
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        label = class_dir.name
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            item = groups.setdefault(digest, {"hash": digest, "path": str(path), "labels": set()})
            item["labels"].add(label)

    manifest = []
    for item in groups.values():
        manifest.append({"hash": item["hash"], "path": item["path"], "labels": sorted(item["labels"])})
    return sorted(manifest, key=lambda x: x["hash"])


def split_groups(manifest: list[dict], test_size: float, seed: int):
    # Stratify by the complete label signature when possible. If a signature is
    # too rare for stratification, fall back to the first label so the split is
    # still reproducible and all anomaly families remain represented.
    signatures = ["|".join(x["labels"]) for x in manifest]
    counts = {s: signatures.count(s) for s in set(signatures)}
    stratify = signatures if min(counts.values()) >= 2 else [x["labels"][0] for x in manifest]
    try:
        train, test = train_test_split(
            np.arange(len(manifest)), test_size=test_size, random_state=seed, stratify=stratify
        )
    except ValueError:
        train, test = train_test_split(np.arange(len(manifest)), test_size=test_size, random_state=seed)
    return train.tolist(), test.tolist()


class PVMDMultiLabelDataset(Dataset):
    def __init__(self, records: list[dict], class_names: list[str], transform):
        self.records = records
        self.class_names = class_names
        self.class_to_idx = {n: i for i, n in enumerate(class_names)}
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        target = torch.zeros(len(self.class_names), dtype=torch.float32)
        for label in record["labels"]:
            target[self.class_to_idx[label]] = 1.0
        return self.transform(image), target


def evaluate(model, loader, device, threshold):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            y_prob.append(probs)
            y_true.append(y.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "mean_average_precision": float(average_precision_score(y_true, y_prob, average="macro")),
    }
    try:
        metrics["macro_roc_auc"] = float(roc_auc_score(y_true, y_prob, average="macro"))
    except ValueError:
        metrics["macro_roc_auc"] = None
    return metrics, y_true, y_prob, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/model1")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", default="models/fault_detection/v2_convnext_pvmd_multilabel.pth")
    args = parser.parse_args()
    seed_everything()

    root = find_image_root(Path(args.data))
    manifest = build_manifest(root)
    class_names = sorted({label for r in manifest for label in r["labels"]})
    train_idx, test_idx = split_groups(manifest, 0.20, SEED)
    train_records = [manifest[i] for i in train_idx]
    test_records = [manifest[i] for i in test_idx]
    train_idx2, val_idx = split_groups(train_records, 0.20, SEED)
    actual_train = [train_records[i] for i in train_idx2]
    val_records = [train_records[i] for i in val_idx]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = PVMDMultiLabelDataset(actual_train, class_names, train_tf)
    val_ds = PVMDMultiLabelDataset(val_records, class_names, eval_tf)
    test_ds = PVMDMultiLabelDataset(test_records, class_names, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
    model.to(device)

    # Start with the backbone frozen; the dataset has only 198 unique exact-image
    # groups after duplicate collapse. This is safer than full-network fine-tuning.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    train_positive = np.zeros(len(class_names), dtype=np.float32)
    for r in actual_train:
        for label in r["labels"]:
            train_positive[class_names.index(label)] += 1
    train_count = max(1, len(actual_train))
    pos_weight = torch.tensor((train_count - train_positive) / np.maximum(train_positive, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    best_score = -1.0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        val_metrics, _, _, _ = evaluate(model, val_loader, device, args.threshold)
        epoch_info = {"epoch": epoch + 1, "train_loss": running_loss / len(train_ds), **val_metrics}
        history.append(epoch_info)
        print(epoch_info)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "threshold": args.threshold,
                "architecture": "convnext_tiny",
            }, out)

    checkpoint = torch.load(out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, y_true, y_prob, y_pred = evaluate(model, test_loader, device, args.threshold)

    results = Path("results/model1")
    results.mkdir(parents=True, exist_ok=True)
    (results / "metrics.json").write_text(json.dumps({
        "task": "multi-label anomaly detection",
        "classes": class_names,
        "raw_files": sum(len(list((root / label).rglob("*"))) for label in class_names),
        "unique_exact_image_groups": len(manifest),
        "train_groups": len(actual_train),
        "validation_groups": len(val_records),
        "test_groups": len(test_records),
        "split": "80:20 group split; validation only inside training portion",
        "test_metrics": test_metrics,
        "threshold": args.threshold,
        "healthy_class_available": False,
    }, indent=2), encoding="utf-8")
    (results / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (results / "test_predictions.json").write_text(json.dumps([
        {"labels": [class_names[j] for j, v in enumerate(row) if v], "probabilities": {class_names[j]: float(p) for j, p in enumerate(prob)}}
        for row, prob in zip(y_true, y_prob)
    ], indent=2), encoding="utf-8")
    (results / "multilabel_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (Path("models/fault_detection") / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (Path("models/fault_detection") / "preprocessing.json").write_text(json.dumps({
        "image_size": IMG_SIZE,
        "normalization": "ImageNet",
        "task": "multi-label anomaly detection",
        "threshold": args.threshold,
        "train_split": 0.8,
        "test_split": 0.2,
        "validation_fraction_of_training": 0.2,
        "duplicate_policy": "collapse exact image hashes and union folder labels",
        "healthy_class": "not available in PVMD",
    }, indent=2), encoding="utf-8")
    print(f"saved {out}; test_metrics={test_metrics}")


if __name__ == "__main__":
    main()
