from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL1_DIR = MODEL_DIR / "fault_detection"
MODEL2_DIR = MODEL_DIR / "electrical_degradation"
MODEL3_DIR = MODEL_DIR / "power_prediction"

DATASET1_URL = "https://www.kaggle.com/datasets/himani04012007/pvmd-dataset1"
DATASET2_URL = "https://www.kaggle.com/datasets/himani04012007/pv-mismatch"
DATASET3_URL = "https://www.kaggle.com/datasets/anikannal/solar-power-generation-data"

for directory in (MODEL1_DIR, MODEL2_DIR, MODEL3_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
