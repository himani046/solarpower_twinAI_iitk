from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


class FaultDetector:
    """Multi-label ConvNeXt PV anomaly detector.

    The model independently estimates Crack, Hotspot and Shading probabilities.
    Each anomaly has its own validation-selected threshold, so one image can
    report multiple anomalies. PVMD does not contain healthy examples; therefore
    a low-probability result is reported as "NO_TRAINED_ANOMALY_DETECTED" rather
    than a validated healthy classification.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        class_names: str | Path,
        device: str | None = None,
        threshold: float | dict[str, float] | None = None,
        thresholds_path: str | Path | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.class_names = json.loads(Path(class_names).read_text())
        self.model = models.convnext_tiny(weights=None)
        self.model.classifier[2] = torch.nn.Linear(
            self.model.classifier[2].in_features,
            len(self.class_names),
        )

        state = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            saved_threshold = state.get("threshold", 0.5)
        else:
            self.model.load_state_dict(state)
            saved_threshold = 0.5

        # Prefer explicit thresholds, then the repository calibration file,
        # then the checkpoint's legacy scalar threshold.
        calibrated_path = Path(thresholds_path) if thresholds_path else Path("config/model1_thresholds.json")
        calibrated = None
        if calibrated_path.exists():
            try:
                payload = json.loads(calibrated_path.read_text())
                calibrated = payload.get("thresholds")
            except (OSError, json.JSONDecodeError):
                calibrated = None

        if isinstance(threshold, dict):
            self.thresholds = {
                name: float(threshold.get(name, saved_threshold))
                for name in self.class_names
            }
        elif isinstance(threshold, (int, float)):
            value = float(threshold)
            self.thresholds = {name: value for name in self.class_names}
        elif isinstance(calibrated, dict):
            self.thresholds = {
                name: float(calibrated.get(name, saved_threshold))
                for name in self.class_names
            }
        else:
            value = float(saved_threshold)
            self.thresholds = {name: value for name in self.class_names}

        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(
            image.convert("RGB")
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.sigmoid(
                self.model(tensor)
            )[0].cpu().tolist()

        scores = {
            name: float(prob * 100.0)
            for name, prob in zip(self.class_names, probabilities)
        }

        detected = [
            {
                "name": name,
                "confidence": scores[name],
                "threshold": self.thresholds[name] * 100.0,
            }
            for name in self.class_names
            if scores[name] >= self.thresholds[name] * 100.0
        ]

        return {
            "status": "DEFECTIVE" if detected else "NO_TRAINED_ANOMALY_DETECTED",
            "detected_anomalies": detected,
            "probabilities": scores,
            "thresholds": {
                name: value * 100.0
                for name, value in self.thresholds.items()
            },
            "healthy_class_available": False,
            "note": "PVMD contains anomaly classes only; healthy-vs-defective performance is not validated without healthy reference data.",
        }
