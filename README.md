# SolarTwin AI

AI-driven solar PV fault intelligence, performance monitoring, digital twin, and predictive maintenance platform.

## Architecture

SolarTwin AI contains three independently trained models:

1. **Model 1 — Visual PV Fault Detection**
   - Dataset: [PVMD Dataset](https://www.kaggle.com/datasets/himani04012007/pvmd-dataset1)
   - Model: ConvNeXt-Tiny
   - Task: PV image fault classification

2. **Model 2 — Electrical Performance Degradation**
   - Dataset: [PV Mismatch](https://www.kaggle.com/datasets/himani04012007/pv-mismatch)
   - Model: XGBoost
   - Task: electrical/thermal degradation estimation or supported fault-risk prediction

3. **Model 3 — Solar Power Prediction**
   - Dataset: [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
   - Model: XGBoost Regressor
   - Task: expected power prediction and performance deviation

The datasets are not paired. Models are trained and evaluated independently and their **outputs** are combined only by the application-level Alert/Risk Engine and Digital Twin layer.

## Evaluation policy

Each model uses an **80:20 train-test split**.

- Models 1 and 2: stratified/group-aware splitting where appropriate.
- Model 3: chronological 80:20 split for time-series forecasting.
- The 20% test set is kept untouched for final internal evaluation.
- External datasets are evaluated after model saving, without retraining first.
- Learned preprocessing is fitted only on training data.

## Project structure

```text
SolarTwin-AI/
├── app.py
├── requirements.txt
├── config.py
├── pages/
├── models/
├── inference/
├── training/
├── testing/
├── utils/
├── results/
├── data/README.md
└── docs/methodology.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The website uses saved model artifacts when they exist. Training is not performed on application startup.

## Public deployment

The intended deployment target is Streamlit Community Cloud connected to this GitHub repository.

## Important scientific constraints

- Do not artificially pair records across the three unrelated datasets.
- Do not fit scalers/encoders/PCA/feature-selection transforms on the test set.
- Do not use the 20% internal test set for tuning.
- Do not call inference without ground truth "validation".
- Do not make automatic physical shutdown decisions from the prototype alert system.
