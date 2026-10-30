"""Inject the real test-set metrics into README.md.

Keeps the README's results table in sync with results/eval_test.json so the
documentation can never quote numbers that the code did not produce.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MARKER = "<!-- RESULTS_TABLE -->"


def main() -> None:
    readme = REPO / "README.md"
    eval_path = REPO / "results" / "eval_test.json"
    if not eval_path.exists():
        raise SystemExit("缺少 results/eval_test.json，请先运行 scripts/evaluate.py")

    d = json.loads(eval_path.read_text(encoding="utf-8"))
    cm = d.get("coco_map", {})
    bl = d.get("box_level", {})
    il = d.get("image_level", {})
    lat = d.get("latency_ms", {})

    table = f"""| 指标 | 数值 | 说明 |
| --- | --- | --- |
| mAP@0.5 | **{cm.get('mAP50')}** | COCO 协议 |
| mAP@0.5:0.95 | {cm.get('mAP50_95')} | 定位精度更严格 |
| 框级 Precision | {bl.get('precision')} | conf={d.get('conf')}, IoU={d.get('match_iou')} |
| 框级 Recall | {bl.get('recall')} | 同上 |
| 框级 F1 | {bl.get('f1')} | 同上 |
| 图级敏感度 | {il.get('sensitivity')} | 含骨折图像被检出的比例 |
| 图级特异度 | {il.get('specificity')} | 无骨折图像被正确排除的比例 |
| 图级准确率 | {il.get('accuracy')} | 样本不平衡下参考价值有限 |
| 推理耗时 p50 | {lat.get('p50')} ms | {d.get('gpu')}, {d.get('imgsz')}px |
| 推理耗时 p95 | {lat.get('p95')} ms | 同上 |

测试集规模：{d.get('n_images')} 张（含骨折 {d.get('n_positive')} / 无骨折 {d.get('n_negative')}）。
运行环境：PyTorch {d.get('torch')}，{d.get('gpu')}。
"""

    text = readme.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit("README 中找不到结果表标记")
    head, tail = text.split(MARKER, 1)
    readme.write_text(head + table + tail, encoding="utf-8")
    print("README 结果表已更新")


if __name__ == "__main__":
    main()
