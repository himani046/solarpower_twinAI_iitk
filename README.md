# SolarTwin AI

AI-driven solar PV fault intelligence, performance monitoring, Digital Twin, alerting, and predictive maintenance platform.

## Official datasets

| Model | Dataset | Purpose |
|---|---|---|
| Model 1 | [PVMD](https://www.kaggle.com/datasets/himani04012007/pvmd-dataset1) | PV image fault detection |
| Model 2 | [PV Mismatch](https://www.kaggle.com/datasets/himani04012007/pv-mismatch) | Electrical/thermal performance degradation |
| Model 3 | [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) | Expected solar power prediction |

The datasets are **not paired**. Each model is trained independently. Their outputs are combined only at the application-level Alert/Risk Engine and Digital Twin layer.

## Evaluation policy

- Models 1 and 2 use an 80:20 train-test split where the data structure permits it.
- Model 3 uses a chronological 80:20 train-test split because it is time-series data.
- The final 20% test set is kept untouched for final internal evaluation.
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
├── pages/
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

## Model 1 — Visual PV Fault Detection

Dataset: PVMD

```bash
python training/train_model1_fault_detection.py --data data/raw/model1
```

Model: ConvNeXt-Tiny. The script creates an 80:20 split, uses validation only inside the 80% training portion, evaluates the untouched 20% test set, and saves class/preprocessing metadata.

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

Saved models are tested on feature-compatible external data without retraining. Ground truth is required for quantitative validation.

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

## Alert engine

The three model outputs are converted into configurable risk and health scores at the application layer. Prototype thresholds are not universal engineering standards and the system does not automatically control physical equipment.

## Model artifacts

Large `.pth`/`.pkl` artifacts are ignored by ordinary Git commits. For model storage in GitHub, use Git LFS or another artifact store; do not commit files over GitHub's normal file-size limits.

## Public deployment

The intended deployment target is Streamlit Community Cloud connected to this repository.

## Development rule

Do not merge unrelated datasets into one training table. The project is deliberately a **three-model architecture**:

```text
PVMD → Model 1 → Visual Fault Intelligence
PV Mismatch → Model 2 → Electrical Degradation Intelligence
Solar Generation → Model 3 → Expected Power Intelligence
                                  ↓
                         Alert/Risk Engine
                                  ↓
                            Digital Twin
                                  ↓
                             Dashboard
```
