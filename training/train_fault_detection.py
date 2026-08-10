"""Train Model 1 using an image-folder dataset with an 80:20 split.

Expected structure:
    dataset_root/
        class_a/
        class_b/
        ...

The script discovers the actual classes from the dataset and saves the checkpoint
and class mapping. For small datasets, transfer learning with ImageNet weights is
preferred; the pretrained weights are downloaded by torchvision at first run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from config import MODEL1_DIR


def train(dataset_root: str, epochs: int = 10, batch_size: int = 16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(dataset_root, transform=transform)
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT
    model = models.convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch + 1}/{epochs} complete")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            output = model(images.to(device))
            correct += (output.argmax(1).cpu() == targets).sum().item()
            total += len(targets)
    accuracy = correct / total if total else 0.0

    MODEL1_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL1_DIR / "v1_convnext_pvmd.pth")
    (MODEL1_DIR / "class_names.json").write_text(json.dumps(dataset.classes, indent=2))
    (MODEL1_DIR / "model_metadata.json").write_text(json.dumps({
        "split": "80:20 stratified",
        "epochs": epochs,
        "test_accuracy": accuracy,
        "classes": dataset.classes,
    }, indent=2))
    print(f"test_accuracy={accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.dataset_root, args.epochs, args.batch_size)
