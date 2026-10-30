"""Detection and image-level metrics used by the evaluation scripts and the UI.

Two families of metrics are reported, and the paper keeps them strictly apart:

* **Box-level**  - precision / recall / F1 / mAP at a fixed IoU, counting every
  predicted box as a decision. Computed with a greedy IoU matcher here, and
  cross-checked against the COCO mAP reported by ``ultralytics.val()``.
* **Image-level** - does the study contain *any* fracture? This is what a
  clinician would use the system for as a triage tool, so sensitivity and
  specificity at the image level are reported separately.

Nothing in this module infers a population-level metric from a single image.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(preds, gts, iou_thr: float = 0.5):
    """Greedy one-to-one matching, highest confidence first.

    preds: list of (xyxy, conf) sorted by conf descending
    gts:   list of xyxy
    returns (tp, fp, fn, matches) where matches maps pred index -> gt index
    """
    used = set()
    tp, fp = 0, 0
    matches: dict[int, int] = {}
    for pi, (pbox, _conf) in enumerate(preds):
        best_iou, best_gi = 0.0, -1
        for gi, gbox in enumerate(gts):
            if gi in used:
                continue
            v = iou(pbox, gbox)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= iou_thr and best_gi >= 0:
            used.add(best_gi)
            tp += 1
            matches[pi] = best_gi
        else:
            fp += 1
    fn = len(gts) - len(used)
    return tp, fp, fn, matches


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


@dataclass
class ImageResult:
    """Everything recorded for one test image."""
    image: str
    split: str
    gt_boxes: list
    gt_positive: bool
    det_boxes: list          # [[x1,y1,x2,y2], ...]
    det_confs: list
    tp: int
    fp: int
    fn: int
    latency_ms: float

    @property
    def pred_positive(self) -> bool:
        return len(self.det_boxes) > 0

    @property
    def outcome(self) -> str:
        """Image-level outcome, used to pick qualitative error cases."""
        if self.gt_positive and self.pred_positive:
            return "TP"
        if not self.gt_positive and self.pred_positive:
            return "FP"
        if self.gt_positive and not self.pred_positive:
            return "FN"
        return "TN"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome
        d["n_gt"] = len(self.gt_boxes)
        d["n_det"] = len(self.det_boxes)
        d["max_conf"] = max(self.det_confs) if self.det_confs else 0.0
        d.pop("gt_boxes")
        d.pop("det_boxes")
        d.pop("det_confs")
        return d


def aggregate(results: list[ImageResult]) -> dict:
    """Box-level and image-level summaries over a whole split."""
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    box = prf(tp, fp, fn)

    pos = [r for r in results if r.gt_positive]
    neg = [r for r in results if not r.gt_positive]
    img_tp = sum(1 for r in pos if r.pred_positive)
    img_fn = len(pos) - img_tp
    img_fp = sum(1 for r in neg if r.pred_positive)
    img_tn = len(neg) - img_fp

    sens = img_tp / len(pos) if pos else 0.0
    spec = img_tn / len(neg) if neg else 0.0
    acc = (img_tp + img_tn) / len(results) if results else 0.0
    ppv = img_tp / (img_tp + img_fp) if (img_tp + img_fp) else 0.0
    npv = img_tn / (img_tn + img_fn) if (img_tn + img_fn) else 0.0
    f1_img = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0

    lat = sorted(r.latency_ms for r in results)
    return {
        "n_images": len(results),
        "n_positive": len(pos),
        "n_negative": len(neg),
        "boxes": {"tp": tp, "fp": fp, "fn": fn, **box},
        "image_level": {
            "tp": img_tp, "fp": img_fp, "fn": img_fn, "tn": img_tn,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "accuracy": round(acc, 4),
            "precision_ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "f1": round(f1_img, 4),
        },
        "latency_ms": {
            "mean": round(float(np.mean(lat)), 2) if lat else 0.0,
            "p50": round(float(np.median(lat)), 2) if lat else 0.0,
            "p95": round(float(np.percentile(lat, 95)), 2) if lat else 0.0,
        },
    }


# --------------------------------------------------------------------------
# ground truth loading
# --------------------------------------------------------------------------
def load_yolo_labels(label_path: Path, width: int, height: int) -> list[list[float]]:
    """YOLO txt -> list of absolute xyxy boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, w, h = parts[:5]
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes
