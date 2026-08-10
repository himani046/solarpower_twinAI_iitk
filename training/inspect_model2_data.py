"""Inspect PV Mismatch data before defining a scientifically valid target.

This intentionally does not invent a degradation target. Run it after downloading
Dataset 2 to produce results/model2/schema_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.io import loadmat

ROOT = Path("data/raw/model2")
OUT = Path("results/model2/schema_report.json")


def inspect() -> dict:
    report = {"root": str(ROOT), "files": []}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        item = {"path": str(p), "suffix": p.suffix.lower(), "size_bytes": p.stat().st_size}
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                item.update({"shape": list(df.shape), "columns": [str(c) for c in df.columns], "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}, "head": df.head(3).to_dict(orient="records")})
            elif p.suffix.lower() == ".mat":
                mat = loadmat(p, simplify_cells=True)
                item["mat_keys"] = [k for k in mat.keys() if not k.startswith("__")]
            elif p.suffix.lower() in {".txt", ".json"}:
                item["preview"] = p.read_text(errors="ignore")[:2000]
        except Exception as exc:
            item["error"] = str(exc)
        report["files"].append(item)
    return report


if __name__ == "__main__":
    if not ROOT.exists():
        raise SystemExit(f"Missing {ROOT}. Run: python -m utils.dataset_download --dataset model2")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = inspect()
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
