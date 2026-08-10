"""Train Model 1: PV image fault detection with ConvNeXt-Tiny.

The script discovers class folders under data/raw/model1. It creates a stratified
80/20 split and saves the best model plus class/preprocessing metadata.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_root(base: Path) -> Path:
    candidates = [p for p in [base, *base.rglob("*")] if p.is_dir()]
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in candidates:
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if len(subdirs) >= 2 and any(f.suffix.lower() in image_exts for d in subdirs for f in d.rglob("*")):
            return p
    raise FileNotFoundError(f"Could not find a class-folder image dataset under {base}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/model1")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", default="models/fault_detection/v1_convnext_pvmd.pth")
    args = parser.parse_args()

    seed_everything()
    root = find_image_root(Path(args.data))
    base_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = datasets.ImageFolder(root=root, transform=base_tf)
    targets = np.array(full.targets)
    indices = np.arange(len(full))
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED, stratify=targets)

    train_ds_full = datasets.ImageFolder(root=root, transform=train_tf)
    test_ds = full
    train_ds = Subset(train_ds_full, train_idx)
    test_ds = Subset(test_ds, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(full.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best_acc = -1.0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(y.numpy().tolist())
        acc = accuracy_score(y_true, y_pred)
        print(f"epoch={epoch+1}/{args.epochs} test_accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(), "num_classes": len(full.classes)}, out)

    report = classification_report(y_true, y_pred, target_names=full.classes, output_dict=True, zero_division=0)
    result_dir = Path("results/model1")
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (result_dir / "confusion_matrix.json").write_text(json.dumps(confusion_matrix(y_true, y_pred).tolist()), encoding="utf-8")
    Path("models/fault_detection/class_names.json").write_text(json.dumps(full.classes, indent=2), encoding="utf-8")
    Path("models/fault_detection/preprocessing.json").write_text(json.dumps({"image_size": IMG_SIZE, "normalization": "ImageNet", "train_split": 0.8, "test_split": 0.2, "random_state": SEED}, indent=2), encoding="utf-8")
    print(f"Saved {out}; best_test_accuracy={best_acc:.4f}")


if __name__ == "__main__":
    main()
