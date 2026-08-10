from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


class FaultDetector:
    """ConvNeXt inference wrapper. Loads a trained checkpoint if available."""

    def __init__(self, checkpoint: str | Path, class_names: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = json.loads(Path(class_names).read_text())
        self.model = models.convnext_tiny(weights=None)
        self.model.classifier[2] = torch.nn.Linear(self.model.classifier[2].in_features, len(self.class_names))
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(probabilities.argmax())
        return {
            "class": self.class_names[index],
            "confidence": float(probabilities[index] * 100),
            "probabilities": {name: float(probabilities[i] * 100) for i, name in enumerate(self.class_names)},
        }
