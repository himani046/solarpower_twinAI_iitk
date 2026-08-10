"""Download the three public Kaggle datasets.

Usage:
    python -m utils.dataset_download --all

Kaggle credentials are intentionally never stored in this repository. If the
Kaggle CLI requires authentication in your environment, configure it locally
or through the CI secret KAGGLE_API_TOKEN.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DATASETS = {
    "model1": "himani04012007/pvmd-dataset1",
    "model2": "himani04012007/pv-mismatch",
    "model3": "anikannal/solar-power-generation-data",
}


def download(name: str) -> Path:
    dataset_id = DATASETS[name]
    out = Path("data/raw") / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(out), "--unzip"]
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()
    names = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name in names:
        print(f"Downloading {name}: {DATASETS[name]}")
        print(f"Saved under data/raw/{name}")
        download(name)


if __name__ == "__main__":
    main()
