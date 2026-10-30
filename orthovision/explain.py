"""Feature-map based explanations for the YOLO detector.

Two CAM variants are provided:

* ``EigenCAM``  - first principal component of the feature map, gradient-free
  (Muhammet et al., 2020).  Stable for detectors because it does not rely on a
  single scalar class score.
* ``GradCAM``   - gradient of the summed objectness at the *predicted*
  locations with respect to the backbone/neck features.  This is the detector
  analogue of the usual classification Grad-CAM.

Both operate on the P5 feature map produced by the last neck block, i.e. the
tensor the detection head consumes.  The resulting map is upsampled back to
the original image and is **not** a segmentation of the fracture: it shows
which regions the feature extractor responds to, which is exactly the caveat
discussed in the paper.

If the feature hook fails for any reason the caller receives ``ExplainResult``
with ``cam=None`` and an ``error`` message - the UI then hides the heat-map tab
instead of showing a fabricated image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ExplainResult:
    method: str
    cam: np.ndarray | None
    error: str | None = None
    feature_shape: tuple | None = None


def _normalise(cam: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Percentile-stretched normalisation.

    A plain min-max stretch leaves detector CAMs almost flat: the activated
    region occupies a tiny fraction of the map, so nearly everything collapses
    to the colormap's lowest step. Clipping at the 1st/99th percentile makes
    the structure visible without inventing activation where there is none.
    """
    if cam.size == 0:
        return cam
    lo, hi = np.percentile(cam, [lo_pct, hi_pct])
    if hi - lo < 1e-12:
        cam = cam - cam.min()
        denom = cam.max()
        return cam / denom if denom > 1e-12 else np.zeros_like(cam)
    return np.clip((cam - lo) / (hi - lo), 0.0, 1.0)


def _letterbox_batch(image_rgb: np.ndarray, imgsz: int = 640):
    """Letterbox an RGB image exactly like Ultralytics does at inference time.

    Returns the normalised NCHW tensor and the geometry needed to map a CAM
    computed on the padded 640x640 canvas back to the original image.
    """
    import torch
    from ultralytics.data.augment import LetterBox

    h, w = image_rgb.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    dw, dh = (imgsz - new_w) // 2, (imgsz - new_h) // 2

    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    padded = lb(image=image_rgb)
    tensor = torch.from_numpy(np.ascontiguousarray(padded)).permute(2, 0, 1)
    return tensor.float().unsqueeze(0) / 255.0, (r, dw, dh, new_w, new_h, imgsz)


def _crop_padding(cam: np.ndarray, geom, orig_hw: tuple[int, int]) -> np.ndarray:
    """Remove letterbox padding and resize the CAM to the original resolution.

    The CAM lives on a coarse feature map (stride 32), so the padding box must
    be scaled from the 640x640 network input into feature-map coordinates.
    """
    _r, dw, dh, new_w, new_h, imgsz = geom
    fy, fx = cam.shape[0] / imgsz, cam.shape[1] / imgsz
    y0, y1 = int(round(dh * fy)), int(round((dh + new_h) * fy))
    x0, x1 = int(round(dw * fx)), int(round((dw + new_w) * fx))
    y1, x1 = max(y0 + 1, y1), max(x0 + 1, x1)
    crop = cam[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError("热力图裁剪失败：letterbox 几何信息不匹配")
    return cv2.resize(crop, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)


def _head_features(detector, image_rgb: np.ndarray, imgsz: int, device: str):
    """Run one forward pass and return (feature_maps, predictions, geom).

    ``feature_maps`` is the list the Detect head consumes (P3, P4, P5, coarse
    to fine->coarse order); ``predictions`` is the decoded (4 + nc, N) tensor.
    """
    import torch

    model = detector.model.model.to(device).eval()
    tensor, geom = _letterbox_batch(image_rgb, imgsz)
    tensor = tensor.to(device)

    store: dict[str, object] = {}
    head = model.model[-1]

    def hook(_m, _inp):
        store["x"] = _inp[0]

    handle = head.register_forward_pre_hook(hook)
    try:
        with torch.enable_grad():
            out = model(tensor)
    finally:
        handle.remove()

    feats = store.get("x")
    if feats is None:
        raise RuntimeError("未能捕获特征图：模型结构与该版本 Ultralytics 不兼容")
    if torch.is_tensor(feats):
        feats = [feats]
    preds = out[0] if isinstance(out, (list, tuple)) else out
    if preds.ndim == 3:
        preds = preds[0]
    return list(feats), preds, geom


def _p5_feature(detector, image_rgb: np.ndarray, imgsz: int, device: str):
    feats, preds, geom = _head_features(detector, image_rgb, imgsz, device)
    return feats[-1], preds, geom


def eigen_cam(detector, image_rgb: np.ndarray, imgsz: int = 640,
              device: str = "cuda:0") -> ExplainResult:
    import torch

    try:
        with torch.no_grad():
            feat, _out, geom = _p5_feature(detector, image_rgb, imgsz, device)
        f = feat.detach()[0]                      # (C, H, W)
        c, h, w = f.shape
        mat = f.reshape(c, h * w).float().cpu().numpy()
        mat = mat - mat.mean(axis=1, keepdims=True)
        # First principal component without the full SVD cost on tiny matrices.
        u, s, vt = np.linalg.svd(mat, full_matrices=False)
        cam = vt[0].reshape(h, w)
        if cam.sum() < 0:
            cam = -cam
        cam = _normalise(cam)
        cam = _crop_padding(cam, geom, image_rgb.shape[:2])
        return ExplainResult("EigenCAM", cam, feature_shape=(c, h, w))
    except Exception as exc:  # pragma: no cover - depends on torch/ultralytics version
        return ExplainResult("EigenCAM", None, error=f"EigenCAM 不可用：{exc}")


def grad_cam(detector, image_rgb: np.ndarray, imgsz: int = 640,
             device: str = "cuda:0", level: int = -1) -> ExplainResult:
    """Grad-CAM on one pyramid level, scored by that level's own predictions.

    A naive implementation back-propagates the globally highest scores, which
    in practice all come from the finest level (P3); the coarser maps then
    receive a zero gradient and the heat-map is blank. Here the scalar score
    is the maximum class score *within the chosen level's slice* of the
    prediction tensor, so the gradient always reaches the map being visualised.
    """
    import torch

    backbone = detector.model.model
    prev_flags = [p.requires_grad for p in backbone.parameters()]
    try:
        # Ultralytics disables parameter gradients after loading a checkpoint;
        # Grad-CAM needs them back on for a single forward/backward pass.
        for p in backbone.parameters():
            p.requires_grad_(True)
        feats, preds, geom = _head_features(detector, image_rgb, imgsz, device)
        if not feats:
            raise RuntimeError("未捕获到检测头输入")

        n_anchors = [f.shape[2] * f.shape[3] for f in feats]
        offset = 0
        slices = []
        for n in n_anchors:
            slices.append((offset, offset + n))
            offset += n

        idx = level % len(feats)
        starts, ends = slices[idx]
        if ends > preds.shape[1]:
            idx = 0
            starts, ends = slices[0]

        score = preds[4:, starts:ends].max()
        grads = torch.autograd.grad(score, feats, allow_unused=True, retain_graph=False)

        # Prefer the requested level; if its CAM is degenerate (all-zero after
        # ReLU, which happens when the level carries no positive evidence),
        # fall back to the level with the strongest gradient.
        order = [idx] + sorted((j for j in range(len(feats)) if j != idx),
                               key=lambda j: 0.0 if grads[j] is None
                               else float(grads[j].abs().max()), reverse=True)
        cam, chosen = None, None
        for j in order:
            g = grads[j]
            if g is None or float(g.abs().max()) == 0.0:
                continue
            f = feats[j].detach()[0]
            raw = torch.relu((g[0].mean(dim=(1, 2), keepdim=True) * f).sum(dim=0)).cpu().numpy()
            norm = _normalise(raw)
            if float(norm.max()) <= 1e-3:      # nothing visible after normalisation
                continue
            cam = norm
            chosen = j
            break
        if cam is None:
            raise RuntimeError("各特征层的激活均为零，无法生成 Grad-CAM")

        cam = _crop_padding(cam, geom, image_rgb.shape[:2])
        name = f"Grad-CAM (P{chosen + 3})"
        return ExplainResult(name, cam, feature_shape=tuple(feats[chosen].shape[1:]))
    except Exception as exc:  # pragma: no cover
        return ExplainResult("Grad-CAM", None, error=f"Grad-CAM 不可用：{exc}")
    finally:
        for p, flag in zip(backbone.parameters(), prev_flags):
            p.requires_grad_(flag)


METHODS = {
    "EigenCAM（无梯度）": eigen_cam,
    "Grad-CAM（梯度）": grad_cam,
}


def cam_in_boxes(cam: np.ndarray, detections, pad: int = 8) -> float | None:
    """Fraction of total CAM mass that falls inside the predicted boxes.

    Used in the paper as a coarse sanity check ("did the model look where it
    said it looked"), NOT as evidence that the model is correct.
    """
    if cam is None or not detections:
        return None
    h, w = cam.shape
    mask = np.zeros((h, w), np.uint8)
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
        cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)),
                      (min(w - 1, x2 + pad), min(h - 1, y2 + pad)), 1, -1)
    total = float(cam.sum())
    if total <= 1e-8:
        return None
    return float(cam[mask == 1].sum() / total)
