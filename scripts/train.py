"""Train the OrthoVision detector on FracAtlas.

Starting from the COCO-pretrained YOLO11-S checkpoint and fine-tuning on a
single-class fracture dataset keeps the run inside a laptop GPU budget
(~1 h on an RTX 4060 8 GB) while still producing a genuinely trained model.

Usage
-----
python scripts/train.py --epochs 100 --batch 16 --imgsz 640
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data" / "fracatlas_yolo" / "data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt", help="COCO 预训练检查点")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="orthovision_yolo11s")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        project=str(REPO / "results" / "train"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        # FracAtlas has only 717 positive images; keep light geometric
        # augmentation so the model still sees realistic projections.
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.0,          # left/right is clinically meaningful on radiographs
        flipud=0.0,
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.0,           # X-ray is greyscale; colour jitter is meaningless
        hsv_s=0.0,
        hsv_v=0.1,
        plots=True,
        verbose=True,
    )

    run_dir = REPO / "results" / "train" / args.name
    best = run_dir / "weights" / "best.pt"
    out_dir = REPO / "weights"
    out_dir.mkdir(exist_ok=True)
    target = out_dir / "orthovision_yolo11s.pt"
    if best.exists():
        import shutil
        shutil.copy2(best, target)
        print(f"best.pt -> {target}")

    (run_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    # --- compact summary for the paper ---------------------------------
    summary = {"args": vars(args), "epochs_completed": None, "best_epoch": None,
               "best_val_mAP50": None, "final": {}}
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        import csv as _csv
        rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
        if rows:
            key = "metrics/mAP50(B)"
            vals = [(i, float(r[key])) for i, r in enumerate(rows)
                    if r.get(key) not in (None, "", "0")]
            if vals:
                bi, bv = max(vals, key=lambda t: t[1])
                summary["best_epoch"] = bi + 1
                summary["best_val_mAP50"] = round(bv, 4)
            summary["epochs_completed"] = len(rows)
            last = rows[-1]
            summary["final"] = {k: last[k] for k in last
                                if k and ("metrics/" in k or "loss" in k)}
    (REPO / "results" / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig_src = run_dir / "results.png"
    if fig_src.exists():
        (REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(fig_src, REPO / "results" / "figures" / "train_curves.png")

    print("训练完成")


if __name__ == "__main__":
    main()
