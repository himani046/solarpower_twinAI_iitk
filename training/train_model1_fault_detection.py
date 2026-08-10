"""Train Model 1: PV image fault detection with ConvNeXt-Tiny.

Creates a strict 80/20 train-test split. A validation split is taken only from
the 80% training portion, keeping the final 20% test set untouched until the
end of training/model selection.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED, IMG_SIZE, BATCH_SIZE, EPOCHS = 42, 224, 32, 10


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def find_image_root(base: Path) -> Path:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in [base, *base.rglob("*")]:
        if not p.is_dir(): continue
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if len(subdirs) >= 2 and any(f.suffix.lower() in image_exts for d in subdirs for f in d.rglob("*")):
            return p
    raise FileNotFoundError(f"Could not find class-folder images under {base}")


def evaluate(model, loader, device):
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            y_pred.extend(model(x.to(device)).argmax(1).cpu().tolist()); y_true.extend(y.tolist())
    return y_true, y_pred, accuracy_score(y_true, y_pred)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/model1")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--output", default="models/fault_detection/v1_convnext_pvmd.pth")
    args = p.parse_args(); seed_everything()

    root = find_image_root(Path(args.data))
    base_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(8), transforms.ColorJitter(brightness=.15, contrast=.15), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    full = datasets.ImageFolder(root=root, transform=base_tf)
    targets = np.array(full.targets); indices = np.arange(len(full))
    train80, test20 = train_test_split(indices, test_size=.20, random_state=SEED, stratify=targets)
    train_idx, val_idx = train_test_split(train80, test_size=.20, random_state=SEED, stratify=targets[train80])
    train_ds = Subset(datasets.ImageFolder(root=root, transform=train_tf), train_idx)
    val_ds, test_ds = Subset(full, val_idx), Subset(full, test20)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(full.classes)); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); criterion = nn.CrossEntropyLoss()
    best_val = -1.0; out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        _, _, val_acc = evaluate(model, val_loader, device); print(f"epoch={epoch+1}/{args.epochs} val_accuracy={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc; torch.save({"model_state_dict": model.state_dict(), "num_classes": len(full.classes)}, out)

    checkpoint = torch.load(out, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred, test_acc = evaluate(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=full.classes, output_dict=True, zero_division=0)
    Path("results/model1").mkdir(parents=True, exist_ok=True)
    Path("results/model1/classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("results/model1/confusion_matrix.json").write_text(json.dumps(confusion_matrix(y_true, y_pred).tolist()), encoding="utf-8")
    Path("results/model1/metrics.json").write_text(json.dumps({"test_accuracy": test_acc, "best_validation_accuracy": best_val, "split": "80:20 train:test; validation only inside training"}, indent=2), encoding="utf-8")
    Path("models/fault_detection/class_names.json").write_text(json.dumps(full.classes, indent=2), encoding="utf-8")
    Path("models/fault_detection/preprocessing.json").write_text(json.dumps({"image_size": IMG_SIZE, "normalization": "ImageNet", "train_split": .8, "test_split": .2, "validation_fraction_of_training": .2, "random_state": SEED}, indent=2), encoding="utf-8")
    print(f"saved {out}; untouched_test_accuracy={test_acc:.4f}")


if __name__ == "__main__": main()
