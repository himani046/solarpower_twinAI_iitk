# SolarTwin AI methodology

## 1. Independent model development

The project keeps the three datasets and models independent because their records are not paired.

### Model 1 — Multi-label visual PV anomaly detection

PVMD images → exact-image audit/grouping → multi-label target construction → image preprocessing → ConvNeXt-Tiny → independent sigmoid anomaly probabilities → detected anomaly names/confidence → Alert Engine.

The current production Model 1 remains the validated PVMD ConvNeXt-Tiny checkpoint already stored in Git LFS. It predicts the trained anomaly labels `Cracks`, `Hotspots`, and `Shadings`; multiple labels may be returned for one image.

### Model 2 — Electrical / thermal anomaly and degradation-risk detection

PV Mismatch electrical/thermal measurements → numeric feature cleaning → StandardScaler → Isolation Forest → anomaly score → normalized 0–100 degradation risk → Alert Engine.

Model 2 is intentionally unsupervised because the configured PV Mismatch dataset does not provide a validated degradation target in the repository. The pipeline therefore does not invent a target or report classification accuracy/F1 without ground truth. A future labeled site dataset can be used to calibrate and evaluate the risk score.

### Model 3 — Expected solar power prediction

Solar generation + weather time series → timestamp features + validated environmental/electrical predictors → chronological 80:20 split → XGBoost Regressor → expected AC power.

The current Model 3 target is `AC_POWER`. `DC_POWER`, irradiation, module temperature, ambient temperature, and temporal features are eligible predictors. `AC_POWER` itself and cumulative yield fields are excluded from the predictors to prevent direct target leakage.

Expected and actual power are compared using:

`deviation_percent = (expected - actual) / abs(expected) * 100`

### Combined system

Model 1 supplies visual fault evidence, Model 2 supplies electrical/thermal anomaly risk, and Model 3 supplies expected-power deviation. These outputs are combined only after independent inference by the configurable Alert/Risk Engine.

The prototype risk score remains:

`0.30 * fault_risk + 0.40 * degradation_risk + 0.30 * power_deviation_risk`

These weights and alert thresholds are engineering-prototype rules and must be calibrated with real site data before operational deployment.

## 2. Evaluation

Model 1 uses its unique-image-group test protocol. Model 2 is unsupervised and is evaluated through held-out anomaly-score/risk distributions unless validated labels become available. Model 3 uses a chronological 80:20 split so future observations are not used to train the expected-power model.

## 3. Digital Twin

The Digital Twin is the application representation of a PV asset. It stores/displays model-derived state such as anomaly names, degradation risk, expected power, actual power, deviation, health, risk, alert level, and maintenance status. It is not a fourth ML model.
