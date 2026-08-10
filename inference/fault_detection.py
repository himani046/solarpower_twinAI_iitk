from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


class FaultDetector:
    """Multi-label ConvNeXt PV anomaly detector.

    The model independently estimates Crack, Hotspot and Shading probabilities.
    Any anomaly above the configured threshold makes the image "defective".

    PVMD does not contain healthy examples, so a low-probability result must not
    be presented as a validated healthy classification. The UI should describe
    it as "No trained anomaly detected" until a healthy dataset is added.
    """

    def __init__(self, checkpoint: str | Path, class_names: str | Path, device: str | None = None, threshold: float | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = json.loads(Path(class_names).read_text())
        self.model = models.convnext_tiny(weights=None)
        self.model.classifier[2] = torch.nn.Linear(self.model.classifier[2].in_features, len(self.class_names))
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            saved_threshold = state.get("threshold", 0.5)
        else:
            self.model.load_state_dict(state)
            saved_threshold = 0.5
        self.threshold = float(threshold if threshold is not None else saved_threshold)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(tensor))[0].cpu().tolist()

        scores = {name: float(prob * 100.0) for name, prob in zip(self.class_names, probabilities)}
        detected = [
            {"name": name, "confidence": scores[name]}
            for name in self.class_names
            if scores[name] >= self.threshold * 100.0
        ]

        return {
            "status": "DEFECTIVE" if detected else "NO_TRAINED_ANOMALY_DETECTED",
            "detected_anomalies": detected,
            "probabilities": scores,
            "threshold": self.threshold * 100.0,
            "healthy_class_available": False,
            "note": "PVMD contains anomaly classes only; healthy-vs-defective performance is not validated without healthy reference data.",
        }
