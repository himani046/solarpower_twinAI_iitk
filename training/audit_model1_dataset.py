"""Audit PVMD duplicates and the resulting multi-label combinations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_root(base: Path) -> Path:
    for p in [base, *base.rglob("*")]:
        if not p.is_dir():
            continue
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if len(subdirs) >= 2 and all(any(f.suffix.lower() in IMAGE_EXTS for f in d.rglob("*")) for d in subdirs):
            return p
    raise FileNotFoundError(f"No PVMD class root found below {base}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/model1")
    parser.add_argument("--output", default="results/model1/dataset_audit.json")
    args = parser.parse_args()

    root = find_root(Path(args.data))
    groups = defaultdict(set)
    counts = Counter()
    raw_files = 0

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for path in class_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            raw_files += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups[digest].add(class_dir.name)
            counts[digest] += 1

    combinations = Counter(" + ".join(sorted(labels)) for labels in groups.values())
    result = {
        "root": str(root),
        "raw_files": raw_files,
        "unique_exact_images": len(groups),
        "duplicate_groups": sum(1 for n in counts.values() if n > 1),
        "files_in_duplicate_groups": sum(n for n in counts.values() if n > 1),
        "single_label_groups": sum(len(labels) == 1 for labels in groups.values()),
        "multi_label_groups": sum(len(labels) > 1 for labels in groups.values()),
        "label_combinations": dict(combinations),
        "classes": sorted({label for labels in groups.values() for label in labels}),
        "healthy_class_available": False,
        "interpretation": "Exact duplicate image hashes are collapsed and labels are unioned for multi-label training.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
