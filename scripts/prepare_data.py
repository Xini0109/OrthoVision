"""Convert the FracAtlas release into a YOLO detection dataset with a fixed split.

FracAtlas (Abedeen et al., 2023, Data in Brief; CC BY 4.0) ships 4,083
musculoskeletal radiographs, of which 717 contain at least one fracture with a
bounding-box annotation. The release does not define train/val/test splits, so
this script creates a **stratified, seeded** split and writes it to disk once;
every later experiment reuses the exact same files.

Usage
-----
python scripts/prepare_data.py --src "D:/.../FractureAtlas" --out data/fracatlas_yolo
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASS_NAMES = ["fractured"]


def collect(src: Path) -> list[dict]:
    """Return one record per image: {stem, image, label, has_fracture}."""
    records = []
    for folder, has_fx in (("Fractured", True), ("Non_fractured", False)):
        img_dir = src / "images" / folder
        if not img_dir.is_dir():
            raise FileNotFoundError(f"缺少图像目录：{img_dir}")
        for img in sorted(img_dir.glob("*.jpg")):
            label = src / "Annotations" / "YOLO" / f"{img.stem}.txt"
            records.append({
                "stem": img.stem,
                "image": str(img),
                "label": str(label) if label.exists() else None,
                "has_fracture": has_fx,
                "label_lines": _read_boxes(label),
            })
    return records


def _read_boxes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def stratified_split(records: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    splits = {k: [] for k in SPLIT_RATIOS}
    for cls in (True, False):
        group = [r for r in records if r["has_fracture"] is cls]
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        splits["train"] += group[:n_train]
        splits["val"] += group[n_train:n_train + n_val]
        splits["test"] += group[n_train + n_val:]
    for k in splits:
        splits[k].sort(key=lambda r: r["stem"])
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\医学人工智能\data\FracAtlas\FractureAtlas")
    ap.add_argument("--out", default=r"D:\医学人工智能\project\OrthoVision\data\fracatlas_yolo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的数据集目录")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        if not args.force:
            print(f"[skip] 已存在：{out}（使用 --force 重建）")
            return
        shutil.rmtree(out)

    records = collect(src)
    n_pos = sum(r["has_fracture"] for r in records)
    print(f"图像总数 {len(records)}，含骨折 {n_pos}，无骨折 {len(records) - n_pos}")
    if len(records) != 4083:
        print(f"[warn] 期望 4083 张，实际 {len(records)} 张，请核对数据集版本")

    missing = [r["stem"] for r in records if r["has_fracture"] and r["label_lines"] == 0]
    if missing:
        print(f"[warn] {len(missing)} 张骨折图像缺少边界框标注，例如 {missing[:5]}")

    splits = stratified_split(records, seed=args.seed)
    for split, items in splits.items():
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for r in items:
            shutil.copy2(r["image"], img_dir / f"{r['stem']}.jpg")
            dst = lbl_dir / f"{r['stem']}.txt"
            if r["label"]:
                shutil.copy2(r["label"], dst)
            else:
                dst.write_text("", encoding="utf-8")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\n"
        f"test: test/images\nnames:\n  0: fractured\n",
        encoding="utf-8",
    )

    stats = {
        "source": str(src),
        "seed": args.seed,
        "total_images": len(records),
        "fractured_images": n_pos,
        "non_fractured_images": len(records) - n_pos,
        "total_boxes": sum(r["label_lines"] for r in records),
        "splits": {
            k: {
                "images": len(v),
                "fractured": sum(r["has_fracture"] for r in v),
                "boxes": sum(r["label_lines"] for r in v),
            } for k, v in splits.items()
        },
    }
    (out / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    for k, v in stats["splits"].items():
        print(f"  {k}: {v['images']} 张 (骨折 {v['fractured']}, 框 {v['boxes']})")
    print(f"已写入 {data_yaml}")


if __name__ == "__main__":
    main()
