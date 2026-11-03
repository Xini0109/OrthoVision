"""Redraw every paper figure from the saved experiment artefacts.

This script does not re-run detection (except for the qualitative panel,
which needs a forward pass to obtain CAMs). It exists so the paper's visual
language can be changed without re-running the experiments.

Run:
    python scripts/redraw_figures.py
    python scripts/redraw_figures.py --skip-qualitative
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from orthovision import plotstyle as ps  # noqa: E402
from orthovision.image_ops import draw_detections, load_image, overlay_heatmap  # noqa: E402

FIG = REPO / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    p = REPO / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------
def fig_architecture() -> None:
    ps.setup()
    fig, ax = plt.subplots(figsize=(8.8, 3.2), dpi=300)
    fig.patch.set_facecolor("#F9FDFF")
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    ps.gradient_background(ax, "#F9FDFF", "#EDF8F4")

    stages = [
        ("输入", "X 光影像", 0.30, ps.C_SKY),
        ("预处理", "显示增强", 1.98, ps.C_MINT),
        ("检测", "YOLO 检测", 3.66, ps.C_LAVENDER),
        ("归因", "CAM 归因", 5.34, ps.C_PEACH),
        ("评价", "指标导出", 7.02, ps.C_CORAL),
    ]
    for title, sub, x, colors in stages:
        ps.gradient_box(ax, x, 1.48, 1.50, 0.98, colors, title, sub,
                        fontsize=16, subsize=12)

    for i in range(len(stages) - 1):
        x0 = stages[i][2] + 1.54
        ax.add_patch(FancyArrowPatch((x0, 1.97), (x0 + 0.12, 1.97),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color="#9DB8D6", lw=1.1, zorder=4))

    ax.add_patch(FancyArrowPatch((7.80, 1.38), (4.40, 1.38),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color="#B6A6E8", lw=1.0,
                                 connectionstyle="arc3,rad=-0.16", zorder=4))
    ax.text(6.10, 0.98, "结果反馈：阈值调整与失败案例分析", ha="center",
            fontsize=11, color="#7C6BB0")

    ax.text(4.40, 2.92, "OrthoVision 系统流程", ha="center", va="center",
            fontsize=22, color=ps.INK)
    ax.text(4.40, 0.42, "课程研究原型 · 不用于临床诊断", ha="center",
            fontsize=10.5, color=ps.MUTED)
    save(fig, "fig_architecture.png")


# --------------------------------------------------------------------------
# dataset composition
# --------------------------------------------------------------------------
def fig_dataset() -> None:
    ps.setup()
    stats = load_json("../data/fracatlas_yolo/dataset_stats.json")
    if not stats:
        stats = json.loads((REPO / "data" / "fracatlas_yolo" /
                            "dataset_stats.json").read_text(encoding="utf-8"))
    splits = ["train", "val", "test"]
    zh = {"train": "训练集", "val": "验证集", "test": "测试集"}
    fx = [stats["splits"][k]["fractured"] for k in splits]
    nfx = [stats["splits"][k]["images"] - stats["splits"][k]["fractured"]
           for k in splits]

    fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    ps.gradient_background(ax, "#FBFEFF", "#F0F9F6")
    ys = np.arange(len(splits))[::-1]
    for i, k in enumerate(splits):
        ps.gradient_barh(ax, ys[i], fx[i], 0.46, 0, ps.C_SKY)
        ps.gradient_barh(ax, ys[i], nfx[i], 0.46, fx[i], ps.C_MINT)
        ax.text(fx[i] + nfx[i] + 35, ys[i], f"{fx[i] + nfx[i]}", va="center",
                fontsize=10.5, color=ps.INK)
        if fx[i] >= 120:
            ax.text(fx[i] / 2, ys[i], f"{fx[i]}", va="center",
                    ha="center", fontsize=9.5, color="white")
        else:
            ax.text(fx[i] + 14, ys[i], f"{fx[i]}", va="center",
                    ha="left", fontsize=9.5, color=ps.INK)
    ax.set_yticks(ys, [zh[k] for k in splits])
    ax.set_xlabel("图像数")
    ax.set_xlim(0, max(np.array(fx) + np.array(nfx)) * 1.16)
    ax.set_title(f"FracAtlas 数据划分（共 {stats['total_images']} 张，"
                 f"{stats['fractured_images']} 张含骨折）", pad=8)
    ax.plot([], [], color=ps.SKY, lw=7, label="含骨折")
    ax.plot([], [], color=ps.MINT, lw=7, label="无骨折")
    ax.legend(loc="lower right", ncol=1)
    ps.soft_axes(ax, "x")
    save(fig, "fig_dataset.png")


# --------------------------------------------------------------------------
# PR curve from the threshold sweep
# --------------------------------------------------------------------------
def fig_pr() -> None:
    ps.setup()
    d = load_json("threshold_experiment_test.json")
    rows = d.get("sweep", [])
    if not rows:
        return
    pts = list(reversed(rows))
    x = [r["box"]["recall"] for r in pts]
    y = [r["box"]["precision"] for r in pts]

    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    ps.gradient_background(ax, "#FBFEFF", "#F1F8FF")
    ax.fill_between(x, 0, y, color=ps.SKY, alpha=0.16, linewidth=0)
    ax.plot(x, y, color=ps.SKY, lw=2.4, marker="o", ms=3.2,
            markerfacecolor="white", markeredgecolor=ps.SKY, markeredgewidth=1.0)
    for t, color in ((0.15, ps.MINT), (0.25, ps.LAVENDER), (0.50, ps.PEACH)):
        r = next((r for r in rows if abs(r["threshold"] - t) < 1e-9), None)
        if r:
            ax.scatter([r["box"]["recall"]], [r["box"]["precision"]],
                       s=48, color=color, zorder=5, edgecolor="white", linewidth=1.2)
            ax.annotate(f"conf={t:.2f}", (r["box"]["recall"], r["box"]["precision"]),
                        textcoords="offset points", xytext=(7, 7),
                        fontsize=10, color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall 曲线（测试集，逐点阈值扫描）", pad=10)
    ps.soft_axes(ax, "both")
    save(fig, "fig_pr_curve.png")


# --------------------------------------------------------------------------
# confusion matrix
# --------------------------------------------------------------------------
def fig_confusion(tag: str = "test") -> None:
    ps.setup()
    p = REPO / "results" / f"per_image_{tag}.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    c = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in rows:
        c[r["outcome"]] = c.get(r["outcome"], 0) + 1
    mat = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]])
    row_tot = mat.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    ps.gradient_matrix(ax, mat, "#FBFEFF", "#6FA8F5")
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            share = v / row_tot[i, 0] * 100 if row_tot[i, 0] else 0
            color = "white" if v > mat.max() * 0.55 else ps.INK
            ax.text(j, i - 0.08, f"{v}", ha="center", va="center",
                    fontsize=26, color=color)
            ax.text(j, i + 0.22, f"{share:.1f}%", ha="center", va="center",
                    fontsize=11, color=color)
    ax.set_xticks([0, 1], ["检出阳性", "未检出"])
    ax.set_yticks([0, 1], ["实际有骨折", "实际无骨折"])
    ax.set_xlabel("模型判定")
    ax.set_ylabel("金标准")
    ax.set_title("图级混淆矩阵（测试集）", pad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    fig.savefig(FIG / f"fig_confusion_image_level_{tag}.png", dpi=300,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    if tag == "test":
        fig.savefig(FIG / "fig_confusion_image_level.png", dpi=300,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", f"fig_confusion_image_level_{tag}.png")


# --------------------------------------------------------------------------
# validation threshold sweep
# --------------------------------------------------------------------------
def fig_threshold_sweep() -> None:
    ps.setup()
    d = load_json("threshold_sweep.json")
    rows = d.get("sweep", [])
    if not rows:
        return
    xs = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    ps.gradient_background(ax, "#FBFEFF", "#F1F8FF")
    series = [("sensitivity", "敏感度", ps.C_SKY),
              ("specificity", "特异度", ps.C_MINT),
              ("f1", "图级 F1", ps.C_LAVENDER)]
    for key, name, color in series:
        ys = [r[key] for r in rows]
        ax.fill_between(xs, 0, ys, color=color[0], alpha=0.08, linewidth=0)
        ax.plot(xs, ys, color=color[0], lw=2.0, label=name)
    sel = d.get("selected_threshold")
    if sel is not None:
        ax.axvline(sel, ls="--", lw=1.1, color=ps.MUTED)
        ax.annotate(f"选定 {sel}", (sel, 1.02), xytext=(4, 0),
                    textcoords="offset points", fontsize=10, color=ps.MUTED)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("置信度阈值")
    ax.set_ylabel("得分")
    ax.set_title("验证集上的工作点选择", pad=10)
    ax.legend(loc="center right")
    ps.soft_axes(ax, "both")
    save(fig, "threshold_sweep.png")


# --------------------------------------------------------------------------
# model comparison
# --------------------------------------------------------------------------
def fig_model_comparison() -> None:
    ps.setup()
    d = load_json("model_comparison.json")
    rows = d.get("models", [])
    if not rows:
        return
    labels = [r["模型"] for r in rows]
    panels = [("mAP@0.5", "mAP@0.5", ps.C_SKY, "{:.4f}"),
              ("p50_ms", "p50 延迟 (ms)", ps.C_MINT, "{:.1f}"),
              ("参数量(M)", "参数量 (M)", ps.C_LAVENDER, "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    for ax, (key, title, colors, fmt) in zip(axes, panels):
        vals = [r.get(key) or 0 for r in rows]
        ps.gradient_background(ax, "#FBFEFF", "#F1F8FF")
        for i, v in enumerate(vals):
            ps.gradient_bar(ax, i, v, 0.48, 0, colors, text=fmt.format(v))
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(title, pad=8, fontsize=16)
        ax.set_ylim(0, max(vals) * 1.35 if max(vals) else 1)
        if key == "mAP@0.5":
            ax.set_yticks([0, 0.1, 0.2, 0.3])
        ps.soft_axes(ax)
    fig.suptitle("YOLO11n 与 YOLO11s：精度、速度与参数量", fontsize=16,
                 color=ps.INK, y=1.05)
    fig.tight_layout()
    save(fig, "fig_model_comparison.png")


# --------------------------------------------------------------------------
# test-set threshold curves
# --------------------------------------------------------------------------
def fig_threshold_test() -> None:
    ps.setup()
    d = load_json("threshold_experiment_test.json")
    rows = d.get("sweep", [])
    if not rows:
        return
    xs = [r["threshold"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    series = {
        "box": [("precision", "Precision", ps.C_SKY),
                ("recall", "Recall", ps.C_PEACH),
                ("f1", "F1", ps.C_LAVENDER)],
        "image": [("sensitivity", "敏感度", ps.C_SKY),
                  ("specificity", "特异度", ps.C_MINT),
                  ("f1", "图级 F1", ps.C_LAVENDER)],
    }
    for ax, prefix, title in ((axes[0], "box", "框级指标"),
                              (axes[1], "image", "图级指标")):
        ps.gradient_background(ax, "#FBFEFF", "#F3F9FF")
        for key, name, color in series[prefix]:
            ys = [r[prefix][key] for r in rows]
            ax.fill_between(xs, 0, ys, color=color[0], alpha=0.08, linewidth=0)
            ax.plot(xs, ys, color=color[0], lw=2.0, marker="o", ms=3.0,
                    markerfacecolor="white", markeredgecolor=color[0],
                    markeredgewidth=0.9, label=name)
        ax.axvline(0.15, ls="--", lw=1.0, color=ps.MUTED)
        ax.set_xlabel("置信度阈值")
        ax.set_ylabel("得分")
        ax.set_ylim(0, 1.06)
        ax.set_title(title, pad=8, fontsize=13)
        ax.legend(loc="lower left", fontsize=9, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.85)
        ps.soft_axes(ax, "both")
    fig.suptitle(f"置信度阈值对检测指标的影响（测试集，n={d.get('n_images')}）",
                 fontsize=16, color=ps.INK, y=1.04)
    fig.tight_layout()
    save(fig, "fig_threshold_test.png")


# --------------------------------------------------------------------------
# enhancement comparison
# --------------------------------------------------------------------------
def fig_enhancement() -> None:
    ps.setup()
    d = load_json("enhancement_test.json")
    conds = d.get("conditions", {})
    if not conds:
        return
    zh = {"original": "原图", "clahe": "CLAHE", "clahe_sharpen": "CLAHE+锐化"}
    keys = list(conds)
    metrics = [("precision", "Precision", ps.C_SKY),
               ("recall", "Recall", ps.C_PEACH),
               ("f1", "F1", ps.C_LAVENDER)]
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    ps.gradient_background(ax, "#FBFEFF", "#F1F8FF")
    width = 0.24
    for j, (key, name, colors) in enumerate(metrics):
        for i, k in enumerate(keys):
            v = conds[k]["box"][key]
            ps.gradient_bar(ax, i + (j - 1) * width, v, width, 0, colors,
                            text=f"{v:.3f}" if j == 2 else None)
        ax.plot([], [], color=colors[0], lw=6, label=name)
    ax.set_xticks(range(len(keys)), [zh.get(k, k) for k in keys])
    ax.set_ylim(0, max(conds[k]["box"]["precision"] for k in keys) * 1.35)
    ax.set_ylabel("得分")
    ax.set_title(f"图像增强对检测指标的影响（测试集，conf={d.get('conf')}）", pad=10)
    ax.legend(loc="upper right", ncol=3)
    ps.soft_axes(ax)
    save(fig, "fig_enhancement.png")


# --------------------------------------------------------------------------
# perturbation
# --------------------------------------------------------------------------
def fig_perturbation() -> None:
    ps.setup()
    d = load_json("perturbation.json")
    conds = d.get("conditions", {})
    if not conds:
        return
    order = ["original", "blur_mild", "blur_moderate", "brightness_-30",
             "brightness_+30", "contrast_0.7", "noise_sigma12"]
    keys = [k for k in order if k in conds]
    zh = {"original": "原始", "blur_mild": "轻度模糊", "blur_moderate": "中度模糊",
          "brightness_-30": "亮度-30", "brightness_+30": "亮度+30",
          "contrast_0.7": "对比度×0.7", "noise_sigma12": "噪声σ12"}
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=300)
    fig.patch.set_facecolor("#FBFEFF")
    ps.gradient_background(ax, "#FBFEFF", "#F1F8FF")
    width = 0.34
    for j, (getter, name, colors) in enumerate((
            (lambda v: v["box"]["f1"], "框级 F1", ps.C_SKY),
            (lambda v: v["image_level"]["sensitivity"], "图级敏感度", ps.C_MINT))):
        for i, k in enumerate(keys):
            ps.gradient_bar(ax, i + (j - 0.5) * width,
                            getter(conds[k]), width, 0, colors)
        ax.plot([], [], color=colors[0], lw=6, label=name)
    ax.set_xticks(range(len(keys)), [zh.get(k, k) for k in keys])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("得分")
    ax.set_title(f"图像扰动对检测性能的影响（{d.get('split', 'test')}）", pad=10)
    ax.legend(loc="upper right", ncol=2)
    ps.soft_axes(ax)
    save(fig, "perturbation.png")


# --------------------------------------------------------------------------
# qualitative TP / FP / FN panel
# --------------------------------------------------------------------------
def fig_qualitative(weights: str, device: str) -> None:
    from orthovision import explain as ex
    from orthovision.detector import FractureDetector

    ps.setup()
    rows = list(csv.DictReader(open(REPO / "results" / "per_image_test.csv",
                                    encoding="utf-8-sig")))
    det = FractureDetector(weights, device=device)
    img_root = REPO / "data" / "fracatlas_yolo" / "test" / "images"
    outcomes = [("TP", "正确检出"), ("FP", "误检"), ("FN", "漏检")]

    fig, axes = plt.subplots(3, 3, figsize=(8.8, 10.0), dpi=300)
    fig.patch.set_facecolor("#F8FCFF")
    for r, (outcome, zh) in enumerate(outcomes):
        pick = next((x for x in rows if x["outcome"] == outcome), None)
        if pick is None:
            continue
        p = img_root / pick["image"]
        if not p.exists():
            continue
        img = load_image(p)
        res = det.predict(img, conf=0.15, device=device)
        boxes = draw_detections(img, res.detections)
        e1 = ex.eigen_cam(det, img, device=device)
        e2 = ex.grad_cam(det, img, device=device)
        panels = [boxes,
                  overlay_heatmap(img, e1.cam, alpha=0.45) if e1.cam is not None else img,
                  overlay_heatmap(img, e2.cam, alpha=0.45) if e2.cam is not None else img]
        titles = [f"{zh} · 检测框", f"{zh} · Eigen-CAM", f"{zh} · Grad-CAM"]
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(panels[c])
            ax.set_title(titles[c], fontsize=14, pad=6, color=ps.INK)
            ax.axis("off")
            ps.add_panel_border(ax, "#DCE9F5", 1.0)
    fig.suptitle("正确检出、误检与漏检案例及其特征响应热力图", fontsize=17,
                 color=ps.INK, y=0.995)
    fig.text(0.5, 0.012,
             "热力图仅表示检测头依赖的特征响应区域，不是骨折分割结果。",
             ha="center", fontsize=10.5, color=ps.MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.985])
    save(fig, "fig_qualitative.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-qualitative", action="store_true")
    ap.add_argument("--weights", default=str(REPO / "weights" / "orthovision_yolo11s.pt"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    fig_architecture()
    fig_dataset()
    fig_pr()
    fig_confusion("test")
    fig_confusion("test_yolo11n")
    fig_threshold_sweep()
    fig_model_comparison()
    fig_threshold_test()
    fig_enhancement()
    fig_perturbation()
    if not args.skip_qualitative:
        fig_qualitative(args.weights, args.device)


if __name__ == "__main__":
    main()
