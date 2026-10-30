"""Global paths and defaults for OrthoVision.

All paths are resolved relative to the repository root so the app can be moved
around freely. Environment variables can override the two external assets
(model weights and the experiment results directory).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- external assets -------------------------------------------------------
# Model weights are not committed to git (see README "模型权重" section).
DEFAULT_WEIGHTS = Path(
    os.environ.get("ORTHOVISION_WEIGHTS", REPO_ROOT / "weights" / "orthovision_yolo11s.pt")
)

# Directory produced by scripts/evaluate.py and scripts/perturbation_experiment.py
RESULTS_DIR = Path(os.environ.get("ORTHOVISION_RESULTS", REPO_ROOT / "results"))

FIGURES_DIR = RESULTS_DIR / "figures"

# --- inference defaults ----------------------------------------------------
IMG_SIZE = 640
# Operating point selected on the validation split by scripts/select_threshold.py
# (max image-level F1). Kept in sync with results/threshold_sweep.json.
DEFAULT_CONF = 0.15
DEFAULT_IOU = 0.45
DEFAULT_DEVICE = os.environ.get("ORTHOVISION_DEVICE", "cuda:0")

CLASS_NAMES_ZH = {
    "fracture": "骨折",
    "hardware": "内固定物",
}

APP_TITLE = "OrthoVision · 骨科 X 光骨折辅助检测与可解释性分析"
APP_SUBTITLE = "课程研究原型 · 仅用于教学与研究，不用于临床诊断"

DISCLAIMER = (
    "本系统为《医学人工智能导论》课程研究原型，模型输出仅供研究与教学演示，"
    "不能替代放射科医师的诊断意见。"
)
