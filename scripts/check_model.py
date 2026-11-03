"""Check that the checkpoint loads and print its basic metadata.

Usage:
    python scripts/check_model.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from orthovision.config import DEFAULT_WEIGHTS  # noqa: E402
from orthovision.detector import FractureDetector  # noqa: E402


def main() -> None:
    print("权重路径：", DEFAULT_WEIGHTS)
    print("文件是否存在：", DEFAULT_WEIGHTS.exists())
    try:
        det = FractureDetector(DEFAULT_WEIGHTS, device="cuda:0")
        print("检测器初始化成功")
        for k, v in det.describe().items():
            print(f"  {k}: {v}")
    except Exception:
        print("检测器初始化失败：")
        traceback.print_exc()


if __name__ == "__main__":
    main()
