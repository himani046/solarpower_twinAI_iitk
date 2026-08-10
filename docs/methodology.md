# SolarTwin AI methodology

## 1. Independent model development

The project deliberately keeps the three datasets and models independent because their records are not paired.

### Model 1 — Multi-label visual PV anomaly detection

PVMD images → exact-image audit/grouping → multi-label target construction → image preprocessing → ConvNeXt-Tiny → independent sigmoid anomaly probabilities → detected anomaly names/confidence → Alert Engine.

The current PVMD copy contains anomaly folders only (`Cracks`, `Hotspots`, `Shadings`) and no healthy class. Therefore Model 1 can identify trained anomaly types, but it does **not** claim validated healthy-vs-defective performance. A future healthy reference dataset is required for that binary capability.

The project audit found exact duplicate image bytes across multiple anomaly folders. The current research pipeline collapses identical image bytes into one sample and takes the union of the folder labels. This prevents the same image from being treated as three contradictory single-class training examples and allows the model to represent multiple anomalies for one image. This is a dataset-derived engineering convention and should be verified against the dataset's original annotation metadata if available.

The Model 1 test split is performed at the unique-image-group level to prevent exact duplicate leakage between train and test.

### Model 2 — Electrical performance degradation

PV Mismatch electrical/thermal measurements → validated feature extraction → XGBoost → degradation/risk output → SHAP.

The exact target must be established from the real dataset schema; no synthetic degradation label should be invented.

### Model 3 — Solar power prediction

Solar time series → timestamp features and validated environmental/operational features → chronological 80:20 split → XGBoost Regressor → expected power.

Expected and actual power can be compared using:

`deviation_percent = (expected - actual) / abs(expected) * 100`

## 2. Evaluation

Every model has an untouched 20% internal test set. Model 1 uses unique exact-image groups, Model 3 uses chronological splitting, and validation is performed only inside the 80% training portion. External datasets are evaluated only after the saved model has been tested internally, and external evaluation is not considered quantitative validation without ground truth.

## 3. Application integration

The model outputs are combined only after inference by a configurable Alert/Risk Engine. The prototype risk score is:

`0.30 * fault_risk + 0.40 * degradation_risk + 0.30 * power_deviation_risk`

The result is converted to a 0–100 health score and alert level. These thresholds are engineering-prototype rules and must be calibrated with real site data before operational deployment.

For Model 1, multiple detected anomalies can contribute to fault severity; the UI displays each detected anomaly and its confidence rather than selecting only one class.

## 4. Digital Twin

The Digital Twin is the application representation of a PV asset. It stores/displays model-derived state such as multiple fault/anomaly names, degradation, expected power, actual power, deviation, risk, health and maintenance status. It is not a fourth ML model.
