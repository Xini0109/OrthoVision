"""Fair latency benchmark for several checkpoints under identical conditions.

The per-image latency recorded by evaluate.py is a wall-clock measurement taken
while other work may be running on the GPU. For the model-size comparison we
need the two checkpoints measured back-to-back, after warm-up, with explicit
CUDA synchronisation. This script does exactly that.

Usage:
    python scripts/benchmark_latency.py --n 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from orthovision.detector import FractureDetector  # noqa: E402
from orthovision.image_ops import load_image  # noqa: E402

MODELS = [
    ("YOLO11n", REPO / "weights" / "orthovision_yolo11n.pt"),
    ("YOLO11s", REPO / "weights" / "orthovision_yolo11s.pt"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)

    img_dir = REPO / "data" / "fracatlas_yolo" / "test" / "images"
    paths = sorted(img_dir.glob("*.jpg"))[: args.n]
    images = []
    for p in paths:
        try:
            images.append(load_image(p))
        except ValueError:
            continue
    print(f"benchmark images: {len(images)}")

    results = {}
    for name, weights in MODELS:
        if not weights.exists():
            print(f"[skip] {name}: {weights} 不存在")
            continue
        det = FractureDetector(weights, device=args.device)
        # warm-up
        for im in images[:10]:
            det.predict(im, conf=args.conf, imgsz=args.imgsz, device=args.device)
        sync()
        times = []
        for im in images:
            sync()
            t0 = time.perf_counter()
            det.predict(im, conf=args.conf, imgsz=args.imgsz, device=args.device)
            sync()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        results[name] = {
            "n": len(times),
            "mean_ms": round(statistics.fmean(times), 2),
            "p50_ms": round(statistics.median(times), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "min_ms": round(times[0], 2),
        }
        print(name, results[name])

    out = {"imgsz": args.imgsz, "conf": args.conf, "device": args.device,
           "note": "back-to-back, warm-up 10 images, CUDA synchronised",
           "models": results}
    path = REPO / "results" / "latency_benchmark.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
