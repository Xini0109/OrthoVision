"""汇总 YOLO11n / YOLO11s 的测试结果，生成 results/model_comparison.json。

结果全部读取自已有的 eval_*.json 与训练日志，不手工填数。
若 YOLO11n 尚未评测，脚本只输出 YOLO11s 并给出提示。

Usage:
    python scripts/build_comparison.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODELS = [
    {
        "label": "YOLO11n",
        "eval": REPO / "results" / "eval_test_yolo11n.json",
        "weights": REPO / "weights" / "orthovision_yolo11n.pt",
        "run": REPO / "results" / "train" / "orthovision_yolo11n",
        "summary": REPO / "results" / "train_summary_orthovision_yolo11n.json",
    },
    {
        "label": "YOLO11s",
        "eval": REPO / "results" / "eval_test.json",
        "weights": REPO / "weights" / "orthovision_yolo11s.pt",
        "run": REPO / "results" / "train" / "orthovision_yolo11s",
        "summary": REPO / "results" / "train_summary.json",
    },
]


def n_params(weights: Path) -> float | None:
    if not weights.exists():
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        return round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    except Exception:
        return None


def train_seconds(run: Path) -> float | None:
    csv_path = run / "results.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return None
    try:
        return round(float(rows[-1]["time"]), 1)
    except (KeyError, ValueError):
        return None


def main() -> None:
    bench_path = REPO / "results" / "latency_benchmark.json"
    bench = (json.loads(bench_path.read_text(encoding="utf-8")).get("models", {})
             if bench_path.exists() else {})

    rows = []
    for m in MODELS:
        if not m["eval"].exists():
            print(f"[skip] 缺少 {m['eval'].name}")
            continue
        e = json.loads(m["eval"].read_text(encoding="utf-8"))
        cm = e.get("coco_map", {})
        bl = e.get("box_level", {})
        il = e.get("image_level", {})
        lat = e.get("latency_ms", {})
        b = bench.get(m["label"], {})
        summary = {}
        if m["summary"].exists():
            summary = json.loads(m["summary"].read_text(encoding="utf-8"))
        secs = train_seconds(m["run"])
        rows.append({
            "模型": m["label"],
            "参数量(M)": n_params(m["weights"]),
            "mAP@0.5": cm.get("mAP50"),
            "mAP@0.5:0.95": cm.get("mAP50_95"),
            "Precision": bl.get("precision"),
            "Recall": bl.get("recall"),
            "F1": bl.get("f1"),
            "图级敏感度": il.get("sensitivity"),
            "图级特异度": il.get("specificity"),
            # fair back-to-back benchmark preferred; eval latency kept for audit
            "p50_ms": b.get("p50_ms", lat.get("p50")),
            "p95_ms": b.get("p95_ms", lat.get("p95")),
            "eval_p50_ms": lat.get("p50"),
            "eval_p95_ms": lat.get("p95"),
            "最佳epoch": summary.get("best_epoch"),
            "验证集最佳mAP@0.5": summary.get("best_val_mAP50"),
            "训练时长(min)": round(secs / 60, 1) if secs else None,
            "权重(MB)": round(m["weights"].stat().st_size / 1e6, 1)
            if m["weights"].exists() else None,
        })

    if not rows:
        raise SystemExit("没有任何可用的评测结果")

    rows.sort(key=lambda r: r["参数量(M)"] or 0)
    out = {
        "note": "两个模型使用完全相同的划分、增强策略、训练轮数与测试流程。"
                "p50_ms/p95_ms 来自 scripts/benchmark_latency.py 的同条件连续测量"
                "（同一批 200 张图像、预热后、CUDA 同步）；"
                "eval_p50_ms/eval_p95_ms 为评测脚本当时的墙钟记录，仅用于审计。",
        "models": rows,
    }
    path = REPO / "results" / "model_comparison.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- comparison figure -------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.family": "sans-serif", "axes.unicode_minus": False,
        })
        labels = [r["模型"] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), dpi=220)
        panels = [
            ("mAP@0.5", "mAP@0.5", "#2a78d6", "{:.4f}"),
            ("p50_ms", "p50 延迟 (ms)", "#0f9f8f", "{:.1f}"),
            ("参数量(M)", "参数量 (M)", "#7c5cd6", "{:.2f}"),
        ]
        for ax, (key, title, color, fmt) in zip(axes, panels):
            vals = [r.get(key) or 0 for r in rows]
            bars = ax.bar(labels, vals, width=0.5, color=color)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.03,
                        fmt.format(v), ha="center", fontsize=9, color="#334155")
            ax.set_title(title, fontsize=10)
            ax.set_ylim(0, max(vals) * 1.22 if max(vals) else 1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.22)
        fig.suptitle("YOLO11n 与 YOLO11s：精度、速度与参数量", fontsize=11)
        fig.tight_layout()
        (REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(REPO / "results" / "figures" / "fig_model_comparison.png",
                    facecolor="white")
        plt.close(fig)
        print("figure saved: fig_model_comparison.png")
    except Exception as exc:
        print(f"[warn] 绘图失败：{exc}")


if __name__ == "__main__":
    main()
