"""Evaluate a trained checkpoint on a split and dump per-image records.

Outputs (in results/):
  eval_<split>.json      aggregate metrics + run metadata
  per_image_<split>.csv  one row per image (used for error-case selection)
  figures/               PR curve, confusion matrix and predictions from
                         ultralytics.val(), copied for the paper

Usage
-----
python scripts/evaluate.py --weights weights/orthovision_yolo11s.pt --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from orthovision.detector import FractureDetector  # noqa: E402
from orthovision.image_ops import load_image  # noqa: E402
from orthovision.metrics import (  # noqa: E402
    ImageResult, aggregate, load_yolo_labels, match_boxes,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights" / "orthovision_yolo11s.pt"))
    ap.add_argument("--data", default=str(REPO / "data" / "fracatlas_yolo"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.5, help="IoU 判定 TP")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-map", action="store_true", help="跳过 ultralytics val（只算逐图指标）")
    args = ap.parse_args()

    data_root = Path(args.data)
    img_dir = data_root / args.split / "images"
    lbl_dir = data_root / args.split / "labels"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"没有找到图像：{img_dir}")

    det = FractureDetector(args.weights, device=args.device)

    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    tag = args.tag or args.split

    # --- 1) COCO mAP via the official evaluator -------------------------
    map_metrics = {}
    if not args.skip_map:
        print("[1/2] ultralytics.val() 计算 mAP ...")
        val = det.model.val(
            data=str(data_root / "data.yaml"),
            split=args.split,
            imgsz=args.imgsz,
            conf=0.001,
            iou=args.iou,
            device=args.device,
            plots=True,
            project=str(out_dir / "val"),
            name=tag,
            exist_ok=True,
            verbose=False,
        )
        map_metrics = {
            "mAP50": round(float(val.box.map50), 4),
            "mAP50_95": round(float(val.box.map), 4),
            "precision": round(float(val.box.mp), 4),
            "recall": round(float(val.box.mr), 4),
        }
        print("   ", map_metrics)
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        val_dir = out_dir / "val" / tag
        for name in ("BoxPR_curve.png", "BoxF1_curve.png", "confusion_matrix.png",
                     "confusion_matrix_normalized.png", "val_batch0_pred.jpg"):
            src = val_dir / name
            if src.exists():
                shutil.copy2(src, fig_dir / f"{tag}_{name}")

    # --- 2) per-image records -------------------------------------------
    print("[2/2] 逐图推理 ...")
    results = []
    skipped = []
    for i, img_path in enumerate(images, 1):
        try:
            img = load_image(img_path)
        except ValueError:
            # A handful of FracAtlas JPEGs are truncated at the source and
            # cannot be decoded by OpenCV or PIL; the COCO evaluator skips the
            # same files, so we skip them here to keep both reports aligned.
            skipped.append(img_path.name)
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(lbl_dir / f"{img_path.stem}.txt", w, h)
        pred = det.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                           device=args.device)
        preds = [(d.xyxy, d.conf) for d in pred.detections]
        tp, fp, fn, _ = match_boxes(preds, gts, iou_thr=args.match_iou)
        results.append(ImageResult(
            image=img_path.name, split=args.split, gt_boxes=gts,
            gt_positive=bool(gts), det_boxes=[list(d.xyxy) for d in pred.detections],
            det_confs=[d.conf for d in pred.detections],
            tp=tp, fp=fp, fn=fn, latency_ms=pred.latency_ms,
        ))
        if i % 100 == 0:
            print(f"    {i}/{len(images)}")

    agg = aggregate(results)
    payload = {
        "weights": str(args.weights),
        "split": args.split,
        "conf": args.conf,
        "nms_iou": args.iou,
        "match_iou": args.match_iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "coco_map": map_metrics,
        "box_level": agg["boxes"],
        "image_level": agg["image_level"],
        "latency_ms": agg["latency_ms"],
        "n_images": agg["n_images"],
        "n_positive": agg["n_positive"],
        "n_negative": agg["n_negative"],
        "n_skipped_undecodable": len(skipped),
        "skipped_files": skipped,
    }
    (out_dir / f"eval_{tag}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_dir / f"per_image_{tag}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].to_dict().keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
