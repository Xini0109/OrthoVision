"""Image loading, display-time enhancement, controlled perturbations and drawing.

The enhancement functions are *display-only* helpers: they never change the
image that is fed to the model unless the user explicitly enables
"以增强图像推理" in the workbench. The perturbation functions are used by the
robustness experiment (scripts/perturbation_experiment.py) and are documented
as synthetic corruptions, not as clinical degradation.
"""
from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# BGR palette used consistently for every figure in the project
COLOR_BOX = (79, 209, 255)      # cyan-ish
COLOR_TEXT_BG = (20, 24, 36)
COLOR_TEXT = (255, 255, 255)
COLOR_WARN = (80, 80, 255)


def load_image(data: bytes | str) -> np.ndarray:
    """Load an image as an RGB uint8 array, raising ValueError on bad input.

    Uses imdecode/fromfile rather than cv2.imread so that paths containing
    non-ASCII characters (e.g. a Chinese project directory on Windows) work.
    """
    if isinstance(data, bytes):
        arr = np.frombuffer(data, np.uint8)
    else:
        try:
            arr = np.fromfile(str(data), dtype=np.uint8)
        except OSError:
            arr = np.empty(0, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    if img is None:
        # OpenCV 5.x rejects a small subset of valid baseline JPEGs that PIL
        # reads without trouble, so fall back to PIL before giving up.
        try:
            pil = Image.open(io.BytesIO(arr.tobytes())).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    if img is None:
        raise ValueError("无法解析图像文件，请上传 JPG/PNG/BMP/TIFF 格式的影像")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path, img: np.ndarray, quality: int = 95) -> None:
    """Write an RGB array to disk; unicode-safe counterpart of cv2.imwrite."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    buf.tofile(str(path))


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def to_jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("图像编码失败")
    return io.BytesIO(buf.tobytes()).getvalue()


def image_info(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return {
        "width": int(w),
        "height": int(h),
        "megapixels": round(w * h / 1e6, 2),
        "mean_intensity": round(float(gray.mean()), 1),
        "contrast_std": round(float(gray.std()), 1),
        "is_grayscale": bool(np.allclose(img[..., 0], img[..., 1], atol=2)),
    }


# --------------------------------------------------------------------------
# display-time enhancement
# --------------------------------------------------------------------------
def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0,
                               contrast: float = 1.0) -> np.ndarray:
    """brightness in [-100, 100], contrast as a multiplicative factor."""
    out = img.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the luminance channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)


def apply_sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)


# --------------------------------------------------------------------------
# controlled perturbations (used by the robustness experiment)
# --------------------------------------------------------------------------
def perturb_blur(img: np.ndarray, level: str = "mild") -> np.ndarray:
    """Gaussian blur. mild: sigma=1.0, moderate: sigma=2.0, severe: sigma=3.5."""
    sigma = {"mild": 1.0, "moderate": 2.0, "severe": 3.5}[level]
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def perturb_brightness(img: np.ndarray, delta: int) -> np.ndarray:
    """Additive brightness shift in 8-bit intensity units."""
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def perturb_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale intensities around the mean."""
    mean = img.mean()
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def perturb_noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


PERTURBATIONS = {
    "blur_mild": lambda im: perturb_blur(im, "mild"),
    "blur_moderate": lambda im: perturb_blur(im, "moderate"),
    "brightness_-30": lambda im: perturb_brightness(im, -30),
    "brightness_+30": lambda im: perturb_brightness(im, +30),
    "contrast_0.7": lambda im: perturb_contrast(im, 0.7),
    "noise_sigma12": lambda im: perturb_noise(im, 12.0),
}


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def draw_detections(img: np.ndarray, detections, show_conf: bool = True) -> np.ndarray:
    """Draw boxes on an RGB image. `detections` is a list of objects exposing
    xyxy (x1, y1, x2, y2), conf and label (str)."""
    out = img.copy()
    thickness = max(2, int(round(min(out.shape[:2]) / 320)))
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_BOX, thickness)
        if show_conf:
            text = f"{d.label} {d.conf:.2f}"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y0 = max(0, y1 - th - base - 4)
            cv2.rectangle(out, (x1, y0), (x1 + tw + 6, y0 + th + base + 4), COLOR_TEXT_BG, -1)
            cv2.putText(out, text, (x1 + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    return out


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Blend a [0,1] activation map over an RGB image."""
    cam = np.ascontiguousarray(np.clip(cam, 0, 1), dtype=np.float32)
    if cam.shape[:2] != img.shape[:2]:
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    cam_u8 = np.ascontiguousarray(cam * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u8, colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * img.astype(np.float32) + alpha * heat.astype(np.float32)).astype(np.uint8)


def side_by_side(images: list[np.ndarray], labels: list[str] | None = None,
                 gap: int = 12, bg: int = 255) -> np.ndarray:
    """Concatenate RGB images of different sizes (resized to a common height)."""
    h = min(im.shape[0] for im in images)
    resized = []
    for im in images:
        scale = h / im.shape[0]
        resized.append(cv2.resize(im, (int(im.shape[1] * scale), h),
                                  interpolation=cv2.INTER_AREA))
    total_w = sum(im.shape[1] for im in resized) + gap * (len(resized) - 1)
    canvas = np.full((h, total_w, 3), bg, np.uint8)
    x = 0
    for im in resized:
        canvas[:, x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    if labels:
        overlay = canvas.copy()
        for im, text, xx in zip(resized, labels, _label_positions(resized, gap)):
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(overlay, (xx, 0), (xx + tw + 20, th + base + 12),
                          (18, 24, 38), -1)
        canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)
        for im, text, xx in zip(resized, labels, _label_positions(resized, gap)):
            cv2.putText(canvas, text, (xx + 10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _label_positions(resized, gap):
    xs, x = [], 0
    for im in resized:
        xs.append(x)
        x += im.shape[1] + gap
    return xs
