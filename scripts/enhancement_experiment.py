"""图像增强对比实验：原图 vs CLAHE vs CLAHE+锐化。

同一模型权重、同一测试集、同一置信度阈值下，只改变送入模型的预处理，
观察增强是否真的改善检测。这是“预处理是否有帮助”的直接对照，
不需要重新训练。

Usage:
    python scripts/enhancement_experiment.py --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from orthovision.detector import FractureDetector  # noqa: E402
from orthovision.image_ops import (  # noqa: E402
    apply_clahe, apply_sharpen, load_image, side_by_side, to_pil, save_image,
)
from orthovision.metrics import load_yolo_labels, match_boxes, prf  # noqa: E402

CONDITIONS = {
    "original": lambda im: im,
    "clahe": lambda im: apply_clahe(im, clip_limit=2.0),
    "clahe_sharpen": lambda im: apply_sharpen(apply_clahe(im, clip_limit=2.0), 0.6),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights" / "orthovision_yolo11s.pt"))
    ap.add_argument("--data", default=str(REPO / "data" / "fracatlas_yolo"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path(args.data) / args.split
    images = sorted((root / "images").glob("*.jpg"))
    det = FractureDetector(args.weights, device=args.device)

    # cache decoded images once; the enhanced variants are generated per condition
    decoded = []
    skipped = []
    for p in images:
        try:
            decoded.append((p, load_image(p)))
        except ValueError:
            skipped.append(p.name)
    n_pos = 0
    totals = {k: {"tp": 0, "fp": 0, "fn": 0, "img_tp": 0, "img_fp": 0,
                  "lat": []} for k in CONDITIONS}

    for i, (p, img) in enumerate(decoded, 1):
        h, w = img.shape[:2]
        gts = load_yolo_labels(root / "labels" / f"{p.stem}.txt", w, h)
        if gts:
            n_pos += 1
        for name, fn in CONDITIONS.items():
            view = fn(img)
            res = det.predict(view, conf=args.conf, iou=0.45, imgsz=args.imgsz,
                              device=args.device)
            preds = [(d.xyxy, d.conf) for d in res.detections]
            tp, fp, fnn, _ = match_boxes(preds, gts, iou_thr=args.match_iou)
            t = totals[name]
            t["tp"] += tp
            t["fp"] += fp
            t["fn"] += fnn
            if gts and preds:
                t["img_tp"] += 1
            elif not gts and preds:
                t["img_fp"] += 1
            t["lat"].append(res.latency_ms)
        if i % 100 == 0:
            print(f"  {i}/{len(decoded)}")

    n_neg = len(decoded) - n_pos
    out = {
        "weights": str(args.weights),
        "split": args.split,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "match_iou": args.match_iou,
        "n_images": len(decoded),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_skipped_undecodable": len(skipped),
        "conditions": {},
    }
    for name, t in totals.items():
        sens = t["img_tp"] / n_pos if n_pos else 0.0
        spec = (n_neg - t["img_fp"]) / n_neg if n_neg else 0.0
        ppv = t["img_tp"] / (t["img_tp"] + t["img_fp"]) if (t["img_tp"] + t["img_fp"]) else 0.0
        lat = sorted(t["lat"])
        out["conditions"][name] = {
            "box": {"tp": t["tp"], "fp": t["fp"], "fn": t["fn"],
                    **prf(t["tp"], t["fp"], t["fn"])},
            "image_level": {
                "sensitivity": round(sens, 4), "specificity": round(spec, 4),
                "accuracy": round((t["img_tp"] + n_neg - t["img_fp"]) / len(decoded), 4),
                "precision_ppv": round(ppv, 4),
            },
            "latency_ms": {
                "mean": round(float(np.mean(lat)), 2),
                "p50": round(float(np.median(lat)), 2),
            },
        }

    path = REPO / "results" / f"enhancement_{args.split}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["conditions"], indent=2, ensure_ascii=False))

    # one visual example: original / CLAHE / CLAHE+sharpen
    if decoded:
        p, img = decoded[0]
        panels = [img,
                  apply_clahe(img, 2.0),
                  apply_sharpen(apply_clahe(img, 2.0), 0.6)]
        save_image(REPO / "results" / "figures" / "fig_enhancement_example.jpg",
                   side_by_side(panels, ["original", "CLAHE", "CLAHE+sharpen"]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.family": "sans-serif", "axes.unicode_minus": False,
        })
        zh = {"original": "原图", "clahe": "CLAHE", "clahe_sharpen": "CLAHE+锐化"}
        keys = list(CONDITIONS)
        fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=220)
        width = 0.26
        x = np.arange(len(keys))
        for j, (key, label, color) in enumerate((
                ("precision", "框级 Precision", "#2a78d6"),
                ("recall", "框级 Recall", "#eb6834"),
                ("f1", "框级 F1", "#6250d6"))):
            vals = [out["conditions"][k]["box"][key] for k in keys]
            bars = ax.bar(x + (j - 1) * width, vals, width, label=label, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                        ha="center", fontsize=8, color="#334155")
        ax.set_xticks(x, [zh[k] for k in keys])
        ax.set_ylim(0, 0.62)
        ax.set_ylabel("得分")
        ax.set_title(f"图像增强对检测指标的影响（{args.split}，conf={args.conf}）",
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(REPO / "results" / "figures" / "fig_enhancement.png",
                    facecolor="white")
        plt.close(fig)
        print("figure saved")
    except Exception as exc:
        print(f"[warn] 绘图失败：{exc}")


if __name__ == "__main__":
    main()
