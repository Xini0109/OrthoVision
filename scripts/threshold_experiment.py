"""置信度阈值实验：一次低阈值推理，离线扫描全部阈值。

与 select_threshold.py 的区别：
    select_threshold.py 只在**验证集**上按图级 F1 选出一个工作点；
    本脚本在**测试集**上报告不同阈值下的框级 Precision / Recall / F1
    与图级敏感度 / 特异度，用于论文的“阈值影响”实验。

why one pass:
    置信度阈值只是对同一批预测框做过滤。以 conf=0.01 推理一次并缓存
    全部候选框，即可离线计算任意阈值下的指标，避免重复推理。

Usage:
    python scripts/threshold_experiment.py --split test
    python scripts/threshold_experiment.py --split val --tag val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from orthovision.detector import FractureDetector  # noqa: E402
from orthovision.image_ops import load_image  # noqa: E402
from orthovision.metrics import load_yolo_labels, match_boxes  # noqa: E402

THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(18)]  # 0.05 .. 0.90
REPORT_POINTS = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75]


def draw_figure(sweep: list, split: str, n: int, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "font.family": "sans-serif", "axes.unicode_minus": False,
    })
    xs = [r["threshold"] for r in sweep]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=220)
    series = {
        "box": (("precision", "Precision", "#2a78d6"),
                ("recall", "Recall", "#eb6834"),
                ("f1", "F1", "#6250d6")),
        "image": (("sensitivity", "敏感度", "#2a78d6"),
                  ("specificity", "特异度", "#eb6834"),
                  ("f1", "图级 F1", "#6250d6")),
    }
    for ax, prefix, title in ((axes[0], "box", "框级指标"),
                              (axes[1], "image", "图级指标")):
        for key, name, color in series[prefix]:
            ax.plot(xs, [r[prefix][key] for r in sweep], marker="o", ms=3.5,
                    lw=1.9, color=color, label=name)
        ax.axvline(0.15, ls="--", lw=1.1, color="#94a3b8")
        ax.set_xlabel("置信度阈值")
        ax.set_ylabel("得分")
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"置信度阈值对检测指标的影响（{split}，n={n}）", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPO / "results" / "figures" / f"fig_threshold_{tag}.png",
                facecolor="white")
    plt.close(fig)
    print("figure saved")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights" / "orthovision_yolo11s.pt"))
    ap.add_argument("--data", default=str(REPO / "data" / "fracatlas_yolo"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--figure-only", action="store_true",
                    help="从已有 JSON 重新绘图，不重新推理")
    args = ap.parse_args()

    if args.figure_only:
        tag = args.tag or args.split
        src = REPO / "results" / f"threshold_experiment_{tag}.json"
        data = json.loads(src.read_text(encoding="utf-8"))
        draw_figure(data["sweep"], data["split"], data["n_images"], tag)
        return

    root = Path(args.data) / args.split
    images = sorted((root / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"没有找到图像：{root / 'images'}")

    det = FractureDetector(args.weights, device=args.device)
    records = []  # {"gt": [...], "preds": [(box, conf), ...], "latency": ms}
    skipped = []
    for i, p in enumerate(images, 1):
        try:
            img = load_image(p)
        except ValueError:
            skipped.append(p.name)
            continue
        h, w = img.shape[:2]
        gts = load_yolo_labels(root / "labels" / f"{p.stem}.txt", w, h)
        res = det.predict(img, conf=0.01, iou=0.45, imgsz=args.imgsz, device=args.device)
        preds = [(tuple(d.xyxy), float(d.conf)) for d in res.detections]
        preds.sort(key=lambda t: t[1], reverse=True)
        records.append({"image": p.name, "gt": gts, "preds": preds,
                        "latency": res.latency_ms})
        if i % 100 == 0:
            print(f"  {i}/{len(images)}")

    n_pos = sum(1 for r in records if r["gt"])
    n_neg = len(records) - n_pos
    sweep = []
    for t in THRESHOLDS:
        tp = fp = fn = 0
        img_tp = img_fp = 0
        for r in records:
            preds = [(b, c) for b, c in r["preds"] if c >= t]
            btp, bfp, bfn, _ = match_boxes(preds, r["gt"], iou_thr=args.match_iou)
            tp += btp
            fp += bfp
            fn += bfn
            if r["gt"] and preds:
                img_tp += 1
            elif not r["gt"] and preds:
                img_fp += 1
        img_fn = n_pos - img_tp
        img_tn = n_neg - img_fp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
        sens = img_tp / n_pos if n_pos else 0.0
        spec = img_tn / n_neg if n_neg else 0.0
        img_f1 = (2 * sens * (img_tp / (img_tp + img_fp))
                  / (sens + img_tp / (img_tp + img_fp))
                  if (img_tp + img_fp) and (sens + img_tp / (img_tp + img_fp)) else 0.0)
        sweep.append({
            "threshold": t,
            "box": {"tp": tp, "fp": fp, "fn": fn,
                    "precision": round(p, 4), "recall": round(r_, 4),
                    "f1": round(f1, 4)},
            "image": {"tp": img_tp, "fp": img_fp, "fn": img_fn, "tn": img_tn,
                      "sensitivity": round(sens, 4), "specificity": round(spec, 4),
                      "f1": round(img_f1, 4)},
        })

    # 论文报告点：0.25 / 0.40 / 0.50 / 0.60 / 0.75（外加 val 选出的 0.15）
    def get(t):
        for row in sweep:
            if abs(row["threshold"] - t) < 1e-9:
                return row
        return None

    report = [get(t) for t in sorted(set([0.15] + REPORT_POINTS))
              if get(t) is not None]
    lat = sorted(r["latency"] for r in records)
    out = {
        "weights": str(args.weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": args.device,
        "match_iou": args.match_iou,
        "n_images": len(records),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_skipped_undecodable": len(skipped),
        "skipped_files": skipped,
        "latency_ms": {
            "mean": round(float(np.mean(lat)), 2) if lat else 0.0,
            "p50": round(float(np.median(lat)), 2) if lat else 0.0,
            "p95": round(float(np.percentile(lat, 95)), 2) if lat else 0.0,
        },
        "report_points": report,
        "sweep": sweep,
    }
    tag = args.tag or args.split
    path = REPO / "results" / f"threshold_experiment_{tag}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(REPO / "results" / f"threshold_experiment_{tag}.csv", "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "box_precision", "box_recall", "box_f1",
                    "image_sensitivity", "image_specificity", "image_f1"])
        for row in sweep:
            w.writerow([row["threshold"], row["box"]["precision"], row["box"]["recall"],
                        row["box"]["f1"], row["image"]["sensitivity"],
                        row["image"]["specificity"], row["image"]["f1"]])
    print(f"saved {path}")

    try:
        draw_figure(sweep, args.split, len(records), tag)
    except Exception as exc:
        print(f"[warn] 绘图失败：{exc}")


if __name__ == "__main__":
    main()
