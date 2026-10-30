"""Thin, explicit wrapper around the Ultralytics YOLO detector.

The wrapper exists for three reasons:
1. The UI must never silently fall back to a fake/random prediction - if the
   weights are missing we raise a descriptive error instead.
2. Inference latency is measured here with a wall-clock timer so that the
   number shown in the UI is reproducible and documented.
3. Everything that the explainability module needs (raw input tensor, feature
   maps) is produced in one place.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CLASS_NAMES_ZH, DEFAULT_CONF, DEFAULT_IOU, IMG_SIZE


class WeightsNotFoundError(RuntimeError):
    pass


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str

    @property
    def label(self) -> str:
        """ASCII label used for OpenCV drawing."""
        return self.cls_name

    @property
    def label_zh(self) -> str:
        return CLASS_NAMES_ZH.get(self.cls_name, self.cls_name)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    def as_dict(self) -> dict:
        x1, y1, x2, y2 = (round(float(v), 1) for v in self.xyxy)
        return {
            "类别": self.label_zh,
            "置信度": round(float(self.conf), 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "宽(px)": round(self.width, 1), "高(px)": round(self.height, 1),
        }


@dataclass
class InferenceResult:
    detections: list[Detection]
    latency_ms: float
    model_names: dict[int, str] = field(default_factory=dict)
    image_shape: tuple[int, int] = (0, 0)
    conf_threshold: float = DEFAULT_CONF
    raw_result: object | None = None

    @property
    def n_detections(self) -> int:
        return len(self.detections)

    @property
    def max_conf(self) -> float:
        return max((d.conf for d in self.detections), default=0.0)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.label_zh] = counts.get(d.label_zh, 0) + 1
        return {
            "检出目标数": self.n_detections,
            "类别分布": counts,
            "最高置信度": round(self.max_conf, 4),
            "推理耗时(ms)": round(self.latency_ms, 2),
        }


class FractureDetector:
    """Loads a YOLO checkpoint once and exposes predict()."""

    def __init__(self, weights: str | Path, device: str = "cuda:0", warmup: bool = True):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise WeightsNotFoundError(
                f"未找到模型权重：{self.weights}\n"
                "请先运行 scripts/train.py 训练模型，或从 Release 页面下载 "
                "orthovision_yolo11s.pt 并放入 weights/ 目录。"
            )
        from ultralytics import YOLO  # imported lazily so `--help` stays fast

        self.model = YOLO(str(self.weights))
        self.device = device
        self.names = self.model.names
        # Ultralytics keeps the checkpoint on CPU until something moves it;
        # passing `device=` to predict() only routes the input. Move it once
        # here so single-image inference actually runs on the GPU.
        if device and device != "cpu":
            try:
                self.model.to(device)
            except Exception:
                pass
        if warmup:
            self._warmup()

    # ------------------------------------------------------------------
    def _warmup(self) -> None:
        try:
            dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=IMG_SIZE, device=self.device, verbose=False)
        except Exception:
            # A failed warm-up must not prevent the app from starting; the
            # first real inference will surface the error with context.
            pass

    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray, conf: float = DEFAULT_CONF,
                iou: float = DEFAULT_IOU, imgsz: int = IMG_SIZE,
                device: str | None = None) -> InferenceResult:
        device = device or self.device
        t0 = time.perf_counter()
        results = self.model.predict(image, imgsz=imgsz, conf=conf, iou=iou,
                                     device=device, verbose=False)
        latency = (time.perf_counter() - t0) * 1000.0
        r = results[0]

        dets: list[Detection] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for box, c, k in zip(xyxy, confs, clss):
                dets.append(Detection(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    conf=float(c), cls_id=int(k),
                    cls_name=str(self.names.get(int(k), f"class_{k}")),
                ))
        dets.sort(key=lambda d: d.conf, reverse=True)
        return InferenceResult(
            detections=dets,
            latency_ms=latency,
            model_names=dict(self.names),
            image_shape=(int(image.shape[0]), int(image.shape[1])),
            conf_threshold=conf,
            raw_result=r,
        )

    # ------------------------------------------------------------------
    @property
    def class_names(self) -> list[str]:
        return [str(v) for _, v in sorted(self.names.items())]

    def describe(self) -> dict:
        n_params = sum(p.numel() for p in self.model.model.parameters())
        return {
            "权重文件": self.weights.name,
            "权重大小(MB)": round(self.weights.stat().st_size / 1e6, 1),
            "类别": ", ".join(self.class_names),
            "参数量(M)": round(n_params / 1e6, 2),
            "设备": self.device,
        }


def load_detector(weights, device: str = "cuda:0") -> FractureDetector:
    return FractureDetector(weights, device=device)
