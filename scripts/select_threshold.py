"""Choose the operating threshold on the validation split, not on the test set.

Runs the detector once at a very low confidence, keeps every candidate box,
then sweeps the threshold and reports image-level sensitivity / specificity /
F1. The threshold maximising F1 on **validation** is written to
results/threshold_sweep.json and used for the reported test-set operating
point, so the test split is never used for tuning.

Usage
-----
python scripts/select_threshold.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orthovision.detector import FractureDetector  # noqa: E402
from orthovision.image_ops import load_image  # noqa: E402
from orthovision.metrics import load_yolo_labels  # noqa: E402

THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(18)]  # 0.05 .. 0.90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights" / "orthovision_yolo11s.pt"))
    ap.add_argument("--data", default=str(REPO / "data" / "fracatlas_yolo"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = Path(args.data) / args.split
    images = sorted((root / "images").glob("*.jpg"))
    det = FractureDetector(args.weights, device=args.device)

    records = []  # (has_gt, [conf, ...])
    skipped = 0
    for i, p in enumerate(images, 1):
        try:
            img = load_image(p)
        except ValueError:
            skipped += 1  # truncated JPEG in the source release
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(root / "labels" / f"{p.stem}.txt", w, h)
        res = det.predict(img, conf=0.01, device=args.device, imgsz=args.imgsz)
        records.append((bool(gts), [d.conf for d in res.detections]))
        if i % 200 == 0:
            print(f"  {i}/{len(images)}")

    n_pos = sum(1 for g, _ in records if g)
    n_neg = len(records) - n_pos
    sweep = []
    for t in THRESHOLDS:
        tp = sum(1 for g, c in records if g and any(x >= t for x in c))
        fp = sum(1 for g, c in records if not g and any(x >= t for x in c))
        fn = n_pos - tp
        tn = n_neg - fp
        sens = tp / n_pos if n_pos else 0.0
        spec = tn / n_neg if n_neg else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        sweep.append({"threshold": t, "sensitivity": round(sens, 4),
                      "specificity": round(spec, 4), "precision": round(prec, 4),
                      "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn})

    best = max(sweep, key=lambda r: (r["f1"], r["sensitivity"]))
    out = {
        "split": args.split, "n_images": len(records),
        "n_skipped_undecodable": skipped,
        "n_positive": n_pos, "n_negative": n_neg,
        "selection_rule": "max image-level F1 on the validation split",
        "selected_threshold": best["threshold"],
        "selected": best,
        "sweep": sweep,
    }
    path = REPO / "results" / "threshold_sweep.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"选定阈值 = {best['threshold']}  (val F1={best['f1']}, "
          f"sens={best['sensitivity']}, spec={best['specificity']})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.family": "sans-serif", "axes.unicode_minus": False,
        })
        xs = [r["threshold"] for r in sweep]
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
        ax.plot(xs, [r["sensitivity"] for r in sweep], marker="o", ms=4,
                label="敏感度", color="#2a78d6", lw=2)
        ax.plot(xs, [r["specificity"] for r in sweep], marker="s", ms=4,
                label="特异度", color="#eb6834", lw=2)
        ax.plot(xs, [r["f1"] for r in sweep], marker="^", ms=4,
                label="图级 F1", color="#6250d6", lw=2)
        ax.axvline(best["threshold"], ls="--", lw=1.2, color="#898781")
        ax.set_xlabel("置信度阈值")
        ax.set_ylabel("得分")
        ax.set_ylim(0, 1.02)
        ax.set_title("验证集上的工作点选择", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        (REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(REPO / "results" / "figures" / "threshold_sweep.png")
    except Exception as exc:
        print(f"[warn] 绘图失败：{exc}")


if __name__ == "__main__":
    main()
