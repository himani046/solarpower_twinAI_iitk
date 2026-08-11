from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"models/fault_detection/v2_convnext_pvmd_multilabel.pth"
CLASS_PATH=ROOT/"models/fault_detection/class_names.json"
THRESH_PATH=ROOT/"config/model1_thresholds.json"

class Model1VisualFault:
    def __init__(self, device=None):
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        raw=json.loads(CLASS_PATH.read_text())
        self.classes=raw.get("classes",raw.get("class_names",raw)) if isinstance(raw,dict) else raw
        cfg=json.loads(THRESH_PATH.read_text())
        self.thresholds=cfg.get("thresholds",cfg)
        self.model=models.convnext_tiny(weights=None)
        self.model.classifier[2]=nn.Linear(self.model.classifier[2].in_features,len(self.classes))
        ckpt=torch.load(MODEL_PATH,map_location=self.device,weights_only=False)
        state=ckpt.get("model_state_dict",ckpt.get("state_dict",ckpt)) if isinstance(ckpt,dict) else ckpt
        state={k.replace("module.","",1) if k.startswith("module.") else k:v for k,v in state.items()}
        self.model.load_state_dict(state,strict=False)
        self.model.to(self.device).eval()
        self.tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    @torch.no_grad()
    def predict(self,image_path):
        image=Image.open(image_path).convert("RGB")
        p=torch.sigmoid(self.model(self.tf(image).unsqueeze(0).to(self.device))).cpu().numpy()[0]
        probs={c:float(x) for c,x in zip(self.classes,p)}
        labels=[c for c,x in probs.items() if x>=float(self.thresholds.get(c,0.5))]
        return {"status":"Defective" if labels else "No trained anomaly label detected","anomalies":labels,"probabilities":probs}
