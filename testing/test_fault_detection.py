from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch import nn


def run(dataset_root: str, model_dir: str):
    root = Path(model_dir)
    classes = json.loads((root / "class_names.json").read_text())
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(dataset_root, transform=transform)
    if dataset.classes != classes:
        raise ValueError(f"External class mapping {dataset.classes} does not match training classes {classes}")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(classes))
    model.load_state_dict(torch.load(root / "v1_convnext_pvmd.pth", map_location=device))
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(pred)
    return classification_report(y_true, y_pred, target_names=classes, output_dict=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--model-dir", default="models/fault_detection")
    args = p.parse_args()
    print(json.dumps(run(args.dataset_root, args.model_dir), indent=2))
