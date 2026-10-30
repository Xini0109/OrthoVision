"""Generate the figures used in the paper from real experiment artefacts.

Figures produced in results/figures/:
  fig_architecture.png   system pipeline diagram
  fig_dataset.png        split composition
  fig_qualitative.png    正确 / 误检 / 漏检 cases with detections and CAM
  fig_robustness.png     already produced by perturbation_experiment.py

Usage
-----
python scripts/make_figures.py
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
import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

FIG = REPO / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#0f172a"
ACCENT = "#0284c7"
ACCENT2 = "#6366f1"
MUTED = "#64748b"
# Windows ships Microsoft YaHei / SimHei; fall back to DejaVu Sans elsewhere.
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "axes.edgecolor": "#cbd5e1",
})


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    stages = [
        ("输入\nX 光影像", 0.3, "#e0f2fe"),
        ("影像工作台\n增强 / 质量检查", 2.15, "#e0f2fe"),
        ("YOLO11-S\n骨折检测", 4.0, "#dbeafe"),
        ("可解释性\nEigen-CAM / Grad-CAM", 5.85, "#eef2ff"),
        ("评价中心\n指标 / 错误案例", 7.7, "#f5f3ff"),
    ]
    for text, x, color in stages:
        box = FancyBboxPatch((x, 1.55), 1.65, 1.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.2, edgecolor="#94a3b8", facecolor=color)
        ax.add_patch(box)
        ax.text(x + 0.825, 2.05, text, ha="center", va="center", fontsize=9, color=INK)

    for i in range(len(stages) - 1):
        x0 = stages[i][1] + 1.73
        ax.annotate("", xy=(x0 + 0.2, 2.05), xytext=(x0 - 0.02, 2.05),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))

    # feedback loop: routed below the stage boxes so it never crosses them
    ax.annotate("", xy=(4.82, 1.28), xytext=(8.52, 1.28),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT2, lw=1.2,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(6.67, 0.78, "测试结果反馈：阈值调整与局限分析", ha="center",
            fontsize=8.5, color=ACCENT2)

    ax.text(5.0, 3.18, "OrthoVision 系统流程", ha="center", fontsize=13,
            color=INK, fontweight="bold")
    ax.text(5.0, 0.42, "研究原型，非临床诊断工具", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "fig_architecture.png", facecolor="white")
    plt.close(fig)


def fig_dataset() -> None:
    stats_path = REPO / "data" / "fracatlas_yolo" / "dataset_stats.json"
    if not stats_path.exists():
        print("[skip] 缺少 dataset_stats.json")
        return
    s = json.loads(stats_path.read_text(encoding="utf-8"))
    splits = ["train", "val", "test"]
    fx = [s["splits"][k]["fractured"] for k in splits]
    nfx = [s["splits"][k]["images"] - s["splits"][k]["fractured"] for k in splits]

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    ax.bar(splits, fx, label="含骨折", color=ACCENT)
    ax.bar(splits, nfx, bottom=fx, label="无骨折", color="#cbd5e1")
    for i, k in enumerate(splits):
        total = fx[i] + nfx[i]
        ax.text(i, total + 30, f"{total}", ha="center", fontsize=9, color=INK)
        ax.text(i, fx[i] / 2, f"{fx[i]}", ha="center", fontsize=8, color="white")
    ax.set_ylabel("图像数")
    ax.set_title(f"FracAtlas 划分（共 {s['total_images']} 张，{s['fractured_images']} 张含骨折）",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dataset.png", facecolor="white")
    plt.close(fig)


def fig_confusion() -> None:
    """2x2 image-level confusion matrix at the reported operating point.

    The frame-level matrix printed by the COCO evaluator is computed at
    conf=0.001 and is dominated by background predictions, which makes it
    unreadable; the image-level matrix matches the sensitivity/specificity
    actually reported in the paper.
    """
    per_image = REPO / "results" / "per_image_test.csv"
    if not per_image.exists():
        print("[skip] 缺少 per_image_test.csv")
        return
    rows = list(csv.DictReader(open(per_image, encoding="utf-8-sig")))
    c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in rows:
        c[r["outcome"]] = c.get(r["outcome"], 0) + 1
    mat = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]])
    row_tot = mat.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=200)
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.15)
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            share = v / row_tot[i, 0] * 100 if row_tot[i, 0] else 0
            color = "white" if v > mat.max() * 0.55 else "#0f172a"
            ax.text(j, i - 0.08, f"{v}", ha="center", va="center",
                    fontsize=20, color=color, fontweight="bold")
            ax.text(j, i + 0.22, f"{share:.1f}%", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_xticks([0, 1], ["检出阳性", "未检出"])
    ax.set_yticks([0, 1], ["实际有骨折", "实际无骨折"])
    ax.set_xlabel("模型判定")
    ax.set_ylabel("金标准")
    ax.set_title("图级混淆矩阵（测试集）", fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion_image_level.png", facecolor="white")
    plt.close(fig)
    print("已生成 fig_confusion_image_level.png")


def fig_qualitative(weights: str, device: str, n_each: int = 1) -> None:
    """TP / FP / FN examples from the test split with detection + CAM."""
    from orthovision import explain as explain_mod
    from orthovision.detector import FractureDetector
    from orthovision.image_ops import draw_detections, load_image, overlay_heatmap

    per_image = REPO / "results" / "per_image_test.csv"
    if not per_image.exists():
        print("[skip] 缺少 per_image_test.csv，先运行 evaluate.py")
        return
    rows = list(csv.DictReader(open(per_image, encoding="utf-8-sig")))
    det = FractureDetector(weights, device=device)
    img_root = REPO / "data" / "fracatlas_yolo" / "test" / "images"

    panels, labels = [], []
    for outcome in ("TP", "FP", "FN"):
        picks = [r for r in rows if r["outcome"] == outcome][:n_each]
        for r in picks:
            p = img_root / r["image"]
            if not p.exists():
                continue
            img = load_image(p)
            res = det.predict(img, conf=0.15, device=device)
            panels.append(draw_detections(img, res.detections))
            labels.append(f"{outcome}  {r['image']}")
            for name, fn in (("Eigen-CAM", explain_mod.eigen_cam),
                             ("Grad-CAM", explain_mod.grad_cam)):
                ex = fn(det, img, device=device)
                if ex.cam is not None:
                    panels.append(overlay_heatmap(img, ex.cam, alpha=0.45))
                    labels.append(f"{outcome}  {name}")

    if not panels:
        print("[skip] 没有可用的定性样例")
        return

    cols = 3
    rows_n = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.2, 3.4 * rows_n), dpi=180)
    axes = axes.ravel() if rows_n * cols > 1 else [axes]
    for ax_, im, lab in zip(axes, panels, labels):
        ax_.imshow(im)
        ax_.set_title(lab, fontsize=8, color=INK)
        ax_.axis("off")
    for ax_ in axes[len(panels):]:
        ax_.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig_qualitative.png", facecolor="white")
    plt.close(fig)
    print("已生成 fig_qualitative.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights" / "orthovision_yolo11s.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-qualitative", action="store_true")
    args = ap.parse_args()

    fig_architecture()
    fig_dataset()
    fig_confusion()
    print("已生成 fig_architecture.png / fig_dataset.png / 图级混淆矩阵")
    if not args.skip_qualitative:
        fig_qualitative(args.weights, args.device)


if __name__ == "__main__":
    main()
