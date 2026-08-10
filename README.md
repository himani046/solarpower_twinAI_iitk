# SolarTwin AI

AI-driven solar PV fault intelligence, performance monitoring, Digital Twin, alerting, and predictive maintenance platform.

## Official datasets

| Model | Dataset | Purpose |
|---|---|---|
| Model 1 | [PVMD](https://www.kaggle.com/datasets/himani04012007/pvmd-dataset1) | Multi-label PV anomaly detection from thermal images |
| Model 2 | [PV Mismatch](https://www.kaggle.com/datasets/himani04012007/pv-mismatch) | Electrical/thermal performance degradation |
| Model 3 | [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) | Expected solar power prediction |

The datasets are **not paired**. Each model is trained independently. Their outputs are combined only at the application-level Alert/Risk Engine and Digital Twin layer.

## Model 1 design

The goal of Model 1 is **not** to force every image into exactly one anomaly class. A PV image may contain more than one anomaly, so Model 1 is a **multi-label anomaly detector**.

PVMD provides three anomaly folders:

```text
PVMD/
├── Cracks/       (350 images)
├── Hotspots/     (350 images)
└── Shadings/     (300 images)
```

The project audit of the downloaded Kaggle copy found many exact duplicate image bytes across different anomaly folders. To avoid treating identical images as contradictory single-class samples, Model 1 collapses exact image hashes into one sample and **unions the folder labels**. Thus an image appearing under `Cracks` and `Hotspots` becomes:

```text
Cracks = 1
Hotspots = 1
Shadings = 0
```

The model uses independent sigmoid outputs rather than softmax/argmax:

```text
PV image
   ↓
ConvNeXt-Tiny
   ↓
Sigmoid probabilities
   ├── Crack: 91%
   ├── Hotspot: 96%
   └── Shading: 12%
   ↓
Threshold
   ↓
Detected anomalies: Crack + Hotspot
```

A panel is reported as **DEFECTIVE / ANOMALY DETECTED** when at least one anomaly crosses the configured threshold. If no anomaly crosses the threshold, the UI reports **No trained anomaly detected** rather than claiming the panel is healthy.

### Healthy-panel limitation

PVMD contains anomaly examples only; it does **not** contain a healthy/normal class. Therefore Model 1 does **not** claim validated healthy-vs-defective accuracy. A healthy dataset must be added later to train and validate a genuine normal-vs-defective classifier.

### Duplicate-aware 80:20 evaluation

The Model 1 split is performed at the **unique exact-image group level**, not at the individual duplicate-file level. This prevents the same image bytes from appearing in both training and test sets.

- 80% unique image groups → training portion
- 20% unique image groups → untouched internal test
- validation is taken only from the 80% training portion
- duplicate labels are unioned before splitting

## Evaluation policy

- Model 1 uses an 80:20 unique-image-group train-test split.
- Model 2 uses an 80:20 train-test split where the data structure permits it.
- Model 3 uses a chronological 80:20 train-test split because it is time-series data.
- The final test set is kept untouched for final internal evaluation.
- Model selection/validation occurs only inside the training portion.
- Learned preprocessing is fitted only on training data.
- External datasets are evaluated with saved models before any retraining.

## Current repository structure

```text
SolarTwin-AI/
├── app.py
├── config.py
├── config/
│   └── datasets.yaml
├── requirements.txt
├── data/
├── docs/
├── inference/
├── models/
├── results/
├── testing/
├── training/
└── utils/
```

## Dataset download

The three datasets are public Kaggle datasets. Raw data is intentionally excluded from GitHub.

Install dependencies:

```bash
pip install -r requirements.txt
```

Download all datasets:

```bash
python -m utils.dataset_download --all
```

Or download one:

```bash
python -m utils.dataset_download --dataset model1
python -m utils.dataset_download --dataset model2
python -m utils.dataset_download --dataset model3
```

If Kaggle asks for authentication in your environment, configure Kaggle credentials locally or as a CI secret. Never commit credentials.

## Model 1 — Visual PV Anomaly Detection

```bash
python training/train_model1_fault_detection.py --data data/raw/model1
```

The trainer:

1. Finds the nested PVMD class folders.
2. Hashes image bytes.
3. Collapses exact duplicate images.
4. Unions anomaly labels for duplicate images.
5. Creates a unique-image-group 80:20 split.
6. Uses validation only inside the 80% training portion.
7. Trains ConvNeXt-Tiny with a multi-label sigmoid head and `BCEWithLogitsLoss`.
8. Evaluates the untouched 20% test groups.
9. Saves the model and reproducibility metadata.

The saved artifact is:

```text
models/fault_detection/v2_convnext_pvmd_multilabel.pth
```

## Model 2 — Electrical Performance Degradation

First inspect the real PV Mismatch structure:

```bash
python training/inspect_model2_data.py
```

This step is intentional: the project does **not** invent a degradation target. After inspecting the schema, prepare a feature CSV with a scientifically supported target and train:

```bash
python training/train_degradation.py --csv <prepared_features.csv> --target <target_column>
```

## Model 3 — Solar Power Prediction

```bash
python training/train_model3_power_prediction.py --data data/raw/model3
```

The pipeline sorts the data chronologically and trains on the first 80%, testing on the final 20%.

## External testing

Saved models are tested on compatible external data without retraining. Ground truth is required for quantitative validation. For Model 1, external image folders are converted to multi-label records using the same exact-hash label-union policy.

## Website

Run locally:

```bash
streamlit run app.py
```

The dashboard contains:

- Dashboard
- Fault Detection
- Electrical Analysis
- Power Prediction
- Alert Center
- Digital Twin
- Model Testing Lab
- Reports

The Fault Detection page displays **all detected anomalies for the uploaded image**, not just the single highest-probability class.

## Alert engine

The three model outputs are converted into configurable risk and health scores at the application layer. Prototype thresholds are not universal engineering standards and the system does not automatically control physical equipment.

## Model artifacts

Large `.pth`/`.pkl` artifacts are ignored by ordinary Git commits. For model storage in GitHub, use Git LFS or another artifact store; do not commit files over GitHub's normal file-size limits.

## Public deployment

The intended deployment target is Streamlit Community Cloud connected to this repository.

## Development rule

Do not merge unrelated datasets into one training table. The project is deliberately a **three-model architecture**:

```text
PVMD → Model 1 → Multi-label Visual Anomaly Intelligence
PV Mismatch → Model 2 → Electrical Degradation Intelligence
Solar Generation → Model 3 → Expected Power Intelligence
                                      ↓
                              Alert/Risk Engine
                                      ↓
                                 Digital Twin
                                      ↓
                                  Dashboard
```
