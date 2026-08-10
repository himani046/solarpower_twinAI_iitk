# SolarTwin AI methodology

## 1. Independent model development

The project deliberately keeps the three datasets and models independent because their records are not paired.

### Model 1 — Visual fault detection

PVMD images → image preprocessing → ConvNeXt-Tiny → fault class/confidence → optional Grad-CAM.

### Model 2 — Electrical performance degradation

PV Mismatch electrical/thermal measurements → validated feature extraction → XGBoost → degradation/risk output → SHAP.

The exact target must be established from the real dataset schema; no synthetic degradation label should be invented.

### Model 3 — Solar power prediction

Solar time series → timestamp features and validated environmental/operational features → chronological 80:20 split → XGBoost Regressor → expected power.

Expected and actual power can be compared using:

`deviation_percent = (expected - actual) / abs(expected) * 100`

## 2. Evaluation

Every model has an untouched 20% internal test set. Model 3 uses chronological splitting. External datasets are evaluated only after the saved model has been tested internally, and external evaluation is not considered quantitative validation without ground truth.

## 3. Application integration

The model outputs are combined only after inference by a configurable Alert/Risk Engine. The prototype risk score is:

`0.30 * fault_risk + 0.40 * degradation_risk + 0.30 * power_deviation_risk`

The result is converted to a 0–100 health score and alert level. These thresholds are engineering-prototype rules and must be calibrated with real site data before operational deployment.

## 4. Digital Twin

The Digital Twin is the application representation of a PV asset. It stores/display model-derived state such as fault status, degradation, expected power, actual power, deviation, risk, health and maintenance status. It is not a fourth ML model.
