"""Summarise the failure modes of the trained detector from per-image records.

Produces results/error_analysis.json, consumed by the paper:

* missed lesions  (FN): how large are the annotated boxes the model missed?
* false positives (FP): how confident are they, and how many boxes per image?
* which test images contain hardware-like high-contrast material cannot be
  derived automatically, so that part stays a qualitative observation.

Usage
-----
python scripts/error_analysis.py --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def label_boxes(path: Path) -> list[tuple]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            boxes.append(tuple(float(v) for v in p[1:5]))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=str(REPO / "data" / "fracatlas_yolo"))
    args = ap.parse_args()

    root = Path(args.data) / args.split
    rows = list(csv.DictReader(open(REPO / "results" / f"per_image_{args.split}.csv",
                                    encoding="utf-8-sig")))

    fn_rows = [r for r in rows if r["outcome"] == "FN"]
    fp_rows = [r for r in rows if r["outcome"] == "FP"]
    tp_rows = [r for r in rows if r["outcome"] == "TP"]
    tn_rows = [r for r in rows if r["outcome"] == "TN"]

    # --- size of the lesions that were missed -------------------------
    fn_areas, tp_areas = [], []
    for bucket, store in ((fn_rows, fn_areas), (tp_rows, tp_areas)):
        for r in bucket:
            name = r["image"]
            lbl = root / "labels" / (Path(name).stem + ".txt")
            if not (root / "images" / name).exists():
                continue
            for _cx, _cy, cw, ch in label_boxes(lbl):
                store.append(cw * ch)  # normalised area fraction

    def stats(v):
        if not v:
            return {}
        v = sorted(v)
        return {
            "n": len(v),
            "median_pct": round(statistics.median(v) * 100, 3),
            "mean_pct": round(statistics.mean(v) * 100, 3),
            "min_pct": round(v[0] * 100, 3),
            "max_pct": round(v[-1] * 100, 3),
            "share_below_1pct": round(sum(x < 0.01 for x in v) / len(v), 3),
        }

    fp_confs = [float(r["max_conf"]) for r in fp_rows if r["max_conf"]]
    fp_boxes = [int(float(r["n_det"])) for r in fp_rows]

    out = {
        "split": args.split,
        "counts": {"TP": len(tp_rows), "FP": len(fp_rows),
                   "FN": len(fn_rows), "TN": len(tn_rows)},
        "missed_lesion_area": stats(fn_areas),
        "detected_lesion_area": stats(tp_areas),
        "false_positive_confidence": {
            "n": len(fp_confs),
            "median": round(statistics.median(fp_confs), 4) if fp_confs else None,
            "mean": round(statistics.mean(fp_confs), 4) if fp_confs else None,
            "share_below_0.5": round(sum(c < 0.5 for c in fp_confs) / len(fp_confs), 3)
            if fp_confs else None,
        },
        "false_positive_boxes_per_image": {
            "median": statistics.median(fp_boxes) if fp_boxes else None,
            "max": max(fp_boxes) if fp_boxes else None,
        },
    }

    path = REPO / "results" / "error_analysis.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
