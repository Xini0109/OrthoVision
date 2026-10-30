"""Experiment 3: sensitivity of the detector to controlled image degradations.

Each perturbation is applied to the *test* images only; the ground-truth labels
are copied unchanged, so the labels remain valid bounding boxes for the
original anatomy. Every condition is evaluated with the same checkpoint,
thresholds and IoU rules as the baseline.

Important interpretation caveat (also stated in the paper): these are
synthetic corruptions applied to already-acquired images. They quantify
sensitivity to that transformation, not the effect of real-world variation in
acquisition protocol or scanner quality.

Usage
-----
python scripts/perturbation_experiment.py --max-images 0     # 0 = all test images
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from orthovision import image_ops as ops  # noqa: E402
from orthovision.detector import FractureDetector  # noqa: E402
from orthovision.metrics import ImageResult, aggregate, load_yolo_labels, match_boxes  # noqa: E402


def build_temp_dataset(src_images: Path, src_labels: Path, transform, out: Path,
                       limit: int = 0) -> tuple[Path, Path]:
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(src_images.glob("*.jpg"))
    if limit:
        images = images[:limit]
    for p in images:
        try:
            rgb = ops.load_image(p)
        except ValueError:
            continue  # truncated source JPEG; excluded from every condition
        out_rgb = transform(rgb) if transform else rgb
        ops.save_image(img_dir / p.name, out_rgb)
        lbl = src_labels / f"{p.stem}.txt"
        (lbl_dir / f"{p.stem}.txt").write_text(
            lbl.read_text(encoding="utf-8") if lbl.exists() else "", encoding="utf-8")
    # Ultralytics requires both keys even when only `val` is evaluated.
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.as_posix()}\ntrain: images\nval: images\n"
                    f"names:\n  0: fractured\n", encoding="utf-8")
    return yaml, img_dir


def run_condition(det: FractureDetector, yaml: Path, img_dir: Path, lbl_dir: Path,
                  conf: float, iou: float, imgsz: int, device: str,
                  want_map: bool) -> dict:
    results = []
    for p in sorted(img_dir.glob("*.jpg")):
        img = ops.load_image(p)
        h, w = img.shape[:2]
        gts = load_yolo_labels(lbl_dir / f"{p.stem}.txt", w, h)
        pred = det.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device)
        preds = [(d.xyxy, d.conf) for d in pred.detections]
        tp, fp, fn, _ = match_boxes(preds, gts, iou_thr=0.5)
        results.append(ImageResult(
            image=p.name, split="perturbed", gt_boxes=gts, gt_positive=bool(gts),
            det_boxes=[list(d.xyxy) for d in pred.detections],
            det_confs=[d.conf for d in pred.detections],
            tp=tp, fp=fp, fn=fn, latency_ms=pred.latency_ms))

    agg = aggregate(results)
    out = {
        "box": agg["boxes"],
        "image_level": agg["image_level"],
        "latency_ms": agg["latency_ms"],
        "mean_max_conf_positive": None,
    }
    pos = [r for r in results if r.gt_positive and r.det_confs]
    if pos:
        out["mean_max_conf_positive"] = round(
            sum(max(r.det_confs) for r in pos) / len(pos), 4)

    if want_map:
        val = det.model.val(data=str(yaml), split="val", imgsz=imgsz, conf=0.001,
                            iou=iou, device=device, plots=False, verbose=False)
        out["mAP50"] = round(float(val.box.map50), 4)
        out["mAP50_95"] = round(float(val.box.map), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights" / "orthovision_yolo11s.pt"))
    ap.add_argument("--data", default=str(REPO / "data" / "fracatlas_yolo"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-images", type=int, default=0,
                    help="限制每个条件的图像数（0=全部测试集）")
    ap.add_argument("--no-map", action="store_true", help="只算阈值下的 P/R，不算 mAP")
    args = ap.parse_args()

    src_images = Path(args.data) / args.split / "images"
    src_labels = Path(args.data) / args.split / "labels"
    det = FractureDetector(args.weights, device=args.device)

    conditions = [("original", None)] + list(ops.PERTURBATIONS.items())
    out = {
        "weights": args.weights, "split": args.split, "conf": args.conf,
        "iou": args.iou, "imgsz": args.imgsz, "device": args.device,
        "max_images": args.max_images or "all", "conditions": {},
    }

    tmp_root = Path(tempfile.mkdtemp(prefix="orthovision_pert_"))
    try:
        for name, fn in conditions:
            print(f"[{name}] 生成扰动数据 …")
            d = tmp_root / name
            yaml, img_dir = build_temp_dataset(src_images, src_labels, fn, d,
                                               limit=args.max_images)
            print(f"[{name}] 推理与评测 …")
            out["conditions"][name] = run_condition(
                det, yaml, img_dir, src_labels, args.conf, args.iou, args.imgsz,
                args.device, want_map=not args.no_map)
            c = out["conditions"][name]
            print(f"    P={c['box']['precision']:.3f} R={c['box']['recall']:.3f} "
                  f"F1={c['box']['f1']:.3f} "
                  f"sens={c['image_level']['sensitivity']:.3f} "
                  f"spec={c['image_level']['specificity']:.3f} "
                  f"mAP50={c.get('mAP50', 'n/a')}")
            shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    res_dir = REPO / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "perturbation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "font.family": "sans-serif", "axes.unicode_minus": False,
        })
        names = list(out["conditions"])
        zh = {"original": "原始", "blur_mild": "轻度模糊", "blur_moderate": "中度模糊",
              "brightness_-30": "亮度−30", "brightness_+30": "亮度+30",
              "contrast_0.7": "对比度×0.7", "noise_sigma12": "噪声σ12"}
        labels = [zh.get(n, n) for n in names]
        f1 = [out["conditions"][n]["box"]["f1"] for n in names]
        sens = [out["conditions"][n]["image_level"]["sensitivity"] for n in names]
        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
        x = list(range(len(names)))
        ax.bar([i - 0.19 for i in x], f1, width=0.36, label="框级 F1", color="#3987e5")
        ax.bar([i + 0.19 for i in x], sens, width=0.36, label="图级敏感度", color="#9085e9")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("得分")
        ax.set_title("图像质量扰动下的检测性能（FracAtlas 测试集）", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        (res_dir / "figures").mkdir(exist_ok=True)
        fig.savefig(res_dir / "figures" / "perturbation.png")
    except Exception as exc:
        print(f"[warn] 绘图失败：{exc}")

    print("已写入", res_dir / "perturbation.json")


if __name__ == "__main__":
    main()
