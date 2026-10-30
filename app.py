"""OrthoVision - Streamlit front-end.

Run:
    streamlit run app.py

The app is a thin presentation layer: every number it shows comes from the
trained checkpoint or from the JSON/CSV files produced by scripts/evaluate.py.
It never fabricates a result when the model or the evaluation files are
missing - it explains what is missing and how to produce it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from orthovision.config import (  # noqa: E402
    APP_SUBTITLE, APP_TITLE, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_WEIGHTS,
    DISCLAIMER, FIGURES_DIR, IMG_SIZE, RESULTS_DIR,
)
from orthovision.detector import FractureDetector, WeightsNotFoundError  # noqa: E402
from orthovision import explain as explain_mod  # noqa: E402
from orthovision.image_ops import (  # noqa: E402
    adjust_brightness_contrast, apply_clahe, apply_sharpen, draw_detections,
    image_info, load_image, overlay_heatmap, side_by_side, to_pil,
)

st.set_page_config(page_title="OrthoVision · 骨科 X 光骨折辅助检测",
                   page_icon="🦴", layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
:root{
  --bg:#0a1020; --panel:#111a2e; --panel2:#16223a; --line:#22304d;
  --ink:#f0efec; --muted:#c3c2b7; --accent:#3987e5; --accent2:#9085e9;
  --ok:#0ca30c; --warn:#fab219; --bad:#d03b3b;
}
.stApp{background:radial-gradient(1200px 600px at 10% -10%, #14213c 0%, var(--bg) 55%) fixed;}
section[data-testid="stSidebar"]{background:var(--panel); border-right:1px solid var(--line);}
h1,h2,h3,h4{color:var(--ink)!important; letter-spacing:.2px;}
p, span, label, div{color:var(--ink);}
.block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:1500px;}
.ov-header{display:flex; align-items:center; gap:18px; padding:22px 26px;
  background:linear-gradient(120deg, #12203a 0%, #0f1a30 60%, #0d1728 100%);
  border:1px solid var(--line); border-radius:18px; margin-bottom:6px;}
.ov-logo{width:54px;height:54px;border-radius:14px;display:grid;place-items:center;
  background:linear-gradient(135deg,var(--accent),#6366f1); font-size:28px;}
.ov-title{font-size:26px;font-weight:700;line-height:1.2;margin:0;}
.ov-sub{color:var(--muted);font-size:13px;margin-top:4px;}
.ov-badge{margin-left:auto;font-size:12px;color:var(--muted);border:1px solid var(--line);
  padding:6px 12px;border-radius:999px;white-space:nowrap;}
.ov-card{background:var(--panel); border:1px solid var(--line); border-radius:16px;
  padding:18px 20px; margin-bottom:14px;}
.ov-kpi{background:linear-gradient(160deg,var(--panel2),var(--panel));
  border:1px solid var(--line); border-radius:14px; padding:14px 16px;}
.ov-kpi .k{color:var(--muted); font-size:12px; letter-spacing:.4px;}
.ov-kpi .v{font-size:26px; font-weight:700; color:var(--ink); margin-top:2px;}
.ov-kpi .d{color:var(--muted); font-size:12px;}
.ov-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;
  border:1px solid var(--line);color:var(--muted);margin-right:6px;margin-bottom:4px;}
.ov-note{border-left:3px solid var(--accent); background:#0e1830; padding:10px 14px;
  border-radius:10px; color:var(--muted); font-size:13px;}
.ov-summary{margin-top:10px; padding:14px 16px; border-radius:12px; font-size:14px;
  color:#e8eefc; background:linear-gradient(100deg,#132340,#101a30 70%);
  border:1px solid var(--line); border-left:3px solid var(--accent2);}
.ov-footer{margin-top:34px; padding-top:14px; border-top:1px solid var(--line);
  color:var(--muted); font-size:12px; display:flex; justify-content:space-between;}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{background:var(--panel); border:1px solid var(--line);
  border-radius:12px; padding:8px 18px; color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--ink)!important; border-color:var(--accent)!important;}
div[data-testid="stMetricValue"]{color:var(--ink);}
.stButton>button, .stDownloadButton>button{
  background:linear-gradient(135deg,var(--accent),#6366f1); color:#04101f; border:0;
  border-radius:10px; font-weight:700; padding:.5rem 1.1rem;}
.stDataFrame{border:1px solid var(--line); border-radius:12px;}
footer{visibility:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Validated categorical / status slots for the dark surface (#111a2e).
# Contrast on the panel: blue 4.77, orange 4.46, aqua 5.09, yellow 5.65,
# violet 5.55, red 5.37, status 5.17 / 9.45 / 3.61 -> all clear 3:1.
C_BLUE = "#3987e5"
C_ORANGE = "#d95926"
C_AQUA = "#199e70"
C_YELLOW = "#c98500"
C_VIOLET = "#9085e9"
C_RED = "#e66767"
S_GOOD = "#0ca30c"
S_WARN = "#fab219"
S_CRIT = "#d03b3b"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def show_image(img, caption: str = "", width=None):
    """st.image with version-tolerant arguments."""
    try:
        if width is None:
            st.image(img, caption=caption)
        else:
            st.image(img, caption=caption, width=width)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


@st.cache_resource(show_spinner=False)
def get_detector(weights: str, device: str):
    return FractureDetector(weights, device=device)


@st.cache_data(show_spinner=False)
def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv(path: str, mtime: float):
    return pd.read_csv(path)


def png_bytes(img) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def style_fig(fig, height=280, show_legend=False, y_title=""):
    """Apply the shared chart chrome: recessive grid, muted ink, no clutter."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_MUTED, size=12, family="system-ui, Segoe UI, sans-serif"),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=INK_MUTED)),
        hoverlabel=dict(bgcolor="#16223a", font=dict(color="#f0efec", size=12),
                        bordercolor=AXIS),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=AXIS,
                     tickfont=dict(color=INK_MUTED, size=11))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS,
                     tickfont=dict(color=INK_MUTED, size=11), title_text=y_title,
                     title_font=dict(size=11, color=INK_MUTED))
    return fig


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def kpi(col, label, value, delta=""):
    col.markdown(
        f'<div class="ov-kpi"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="d">{delta}</div></div>',
        unsafe_allow_html=True)


def header():
    st.markdown(
        f'<div class="ov-header"><div class="ov-logo">🦴</div>'
        f'<div><div class="ov-title">{APP_TITLE}</div>'
        f'<div class="ov-sub">{APP_SUBTITLE}</div></div>'
        f'<div class="ov-badge">FracAtlas · YOLO11 · CAM</div></div>',
        unsafe_allow_html=True)


def model_status():
    """Load (or report) the detector once per session."""
    if "detector" not in st.session_state:
        try:
            st.session_state.detector = get_detector(
                str(DEFAULT_WEIGHTS), st.session_state.get("device", "cuda:0"))
            st.session_state.model_error = None
        except Exception as exc:
            st.session_state.detector = None
            st.session_state.model_error = str(exc)
    return st.session_state.detector


def build_report(payload: dict, note: str) -> str:
    """Render a Markdown report from one evaluation record."""
    cm = payload.get("coco_map", {})
    il = payload.get("image_level", {})
    bl = payload.get("box_level", {})
    lines = [
        "# OrthoVision 实验报告", "",
        f"- 权重：`{payload.get('weights')}`",
        f"- 测试集：{payload.get('n_images')} 张（骨折 {payload.get('n_positive')} / "
        f"无骨折 {payload.get('n_negative')}）",
        f"- 置信度阈值：{payload.get('conf')}，NMS IoU：{payload.get('nms_iou')}，"
        f"判定 IoU：{payload.get('match_iou')}",
        f"- 运行环境：torch {payload.get('torch')} · {payload.get('gpu')}", "",
        "## 指标", "",
        "| 指标 | 数值 |", "| --- | --- |",
        f"| mAP@0.5 | {cm.get('mAP50')} |",
        f"| mAP@0.5:0.95 | {cm.get('mAP50_95')} |",
        f"| 框级 Precision | {bl.get('precision')} |",
        f"| 框级 Recall | {bl.get('recall')} |",
        f"| 框级 F1 | {bl.get('f1')} |",
        f"| 图级敏感度 | {il.get('sensitivity')} |",
        f"| 图级特异度 | {il.get('specificity')} |",
        f"| 图级准确率 | {il.get('accuracy')} |",
        f"| 耗时 p50 / p95 (ms) | {payload.get('latency_ms', {}).get('p50')} / "
        f"{payload.get('latency_ms', {}).get('p95')} |", "",
    ]
    if note.strip():
        lines += ["## 人工分析备注", "", note.strip(), ""]
    lines += ["---", "", "> " + DISCLAIMER]
    return "\n".join(lines)


def headline(payload: dict) -> str:
    """One-paragraph, data-driven reading of the current evaluation record."""
    il = payload.get("image_level", {})
    cm = payload.get("coco_map", {})
    bl = payload.get("box_level", {})
    try:
        sens = float(il.get("sensitivity", 0.0))
        spec = float(il.get("specificity", 0.0))
    except (TypeError, ValueError):
        return "该记录缺少图级指标。"
    if spec - sens >= 0.15:
        reading = (f"模型更擅长排除无骨折影像（特异度 {spec:.2f}），"
                   f"发现骨折的能力偏弱（敏感度 {sens:.2f}）——漏检是主要风险。")
    elif sens - spec >= 0.15:
        reading = (f"模型倾向于多报（敏感度 {sens:.2f}），"
                   f"代价是无骨折影像上的误报偏多（特异度 {spec:.2f}）。")
    else:
        reading = (f"敏感度 {sens:.2f} 与特异度 {spec:.2f} 较为接近，"
                   "误检与漏检的数量相对均衡，但绝对值仍有提升空间。")
    return (f"<b>怎么读这组数字：</b>{reading} "
            f"mAP@0.5 = {cm.get('mAP50')}，框级 F1 = {bl.get('f1')}；"
            "这些指标来自固定测试集，不代表真实临床人群的表现。")


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
header()

with st.sidebar:
    st.markdown("### 运行配置")
    device = st.selectbox("推理设备", ["cuda:0", "cpu"],
                          index=0 if st.session_state.get("device", "cuda:0") == "cuda:0" else 1)
    st.session_state["device"] = device

    conf = st.slider("置信度阈值 conf", 0.05, 0.95, DEFAULT_CONF, 0.05)
    iou = st.slider("NMS IoU", 0.10, 0.90, DEFAULT_IOU, 0.05)
    imgsz = st.select_slider("推理分辨率", options=[416, 512, 640, 768, 896], value=IMG_SIZE)

    st.markdown("---")
    st.markdown("### 模型状态")
    det = model_status()
    if det is None:
        st.error("模型未就绪")
        st.code(str(DEFAULT_WEIGHTS), language="text")
        st.caption("先运行 scripts/train.py，或把权重放到 weights/ 目录")
    else:
        for k, v in det.describe().items():
            st.markdown(f'<div class="ov-pill">{k}：{v}</div>', unsafe_allow_html=True)

    stats_path = REPO / "data" / "fracatlas_yolo" / "dataset_stats.json"
    if stats_path.exists():
        s = read_json(str(stats_path))
        st.markdown("---")
        st.markdown("### 数据集")
        st.caption(f"FracAtlas · 共 {s['total_images']} 张 · 含骨折 {s['fractured_images']} 张")
        for k, zh in (("train", "训练"), ("val", "验证"), ("test", "测试")):
            v = s.get("splits", {}).get(k, {})
            st.markdown(
                f'<div class="ov-pill">{zh} {v.get("images","—")} 张 / 阳性 {v.get("fractured","—")}</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.caption(DISCLAIMER)


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------
tab_work, tab_detect, tab_explain, tab_eval, tab_about = st.tabs(
    ["影像工作台", "AI 骨折检测", "可解释性分析", "实验评价中心", "数据与来源"])


# ==========================================================================
# 1. workbench
# ==========================================================================
with tab_work:
    st.markdown("### 影像工作台")
    st.markdown('<div class="ov-note">① 选择或上传一张 X 光影像 → ② 切到「AI 骨折检测」点击开始检测 '
                '→ ③ 在「可解释性分析」查看特征归因。图像增强默认只影响显示，'
                '除非勾选“以增强图像推理”，否则不会改变送入模型的输入。</div>',
                unsafe_allow_html=True)
    st.write("")

    up = st.file_uploader("上传 X 光影像（JPG / PNG / BMP / TIFF）",
                          type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    sample_dir = REPO / "assets" / "samples"
    samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
    sample_pick = st.selectbox(
        "或使用内置示例影像（来自 FracAtlas 测试集）",
        ["不使用"] + [p.name for p in samples]) if samples else "不使用"

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        brightness = st.slider("亮度", -60, 60, 0, 5)
    with c2:
        contrast = st.slider("对比度", 0.5, 2.0, 1.0, 0.05)
    with c3:
        use_clahe = st.checkbox("CLAHE 局部增强", value=False)
    sharpen = st.slider("锐化强度", 0.0, 1.5, 0.0, 0.1)
    feed_enhanced = st.checkbox("以增强图像作为模型输入（实验模式）", value=False)

    source_name = None
    raw = None
    if up is not None:
        source_name = up.name
        try:
            raw = load_image(up.read())
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    elif sample_pick != "不使用":
        source_name = sample_pick
        raw = load_image(sample_dir / sample_pick)

    if raw is not None:
        if st.session_state.get("raw_name") != source_name:
            # a new upload invalidates any previous detection / explanation
            st.session_state.pop("result", None)
            st.session_state.pop("explain", None)
        st.session_state["raw_image"] = raw
        st.session_state["raw_name"] = source_name

        view = adjust_brightness_contrast(raw, brightness, contrast)
        if use_clahe:
            view = apply_clahe(view)
        if sharpen > 0:
            view = apply_sharpen(view, sharpen)
        st.session_state["display_image"] = view
        st.session_state["model_input"] = view if feed_enhanced else raw

        meta = image_info(raw)
        m1, m2, m3, m4 = st.columns(4)
        kpi(m1, "影像尺寸", f'{meta["width"]}×{meta["height"]}', f'{meta["megapixels"]} MP')
        kpi(m2, "平均灰度", meta["mean_intensity"], "0–255")
        kpi(m3, "灰度标准差", meta["contrast_std"], "对比度指标")
        kpi(m4, "色彩模式", "灰度" if meta["is_grayscale"] else "彩色",
            source_name.split(".")[-1].upper())

        st.write("")
        a, b = st.columns(2)
        with a:
            st.markdown("**原始影像**")
            show_image(to_pil(raw), width=None)
        with b:
            st.markdown("**显示增强**")
            show_image(to_pil(view), width=None)

        st.caption("提示：增强参数只用于观察，若在“实验模式”下推理，结果不能与标准流程直接比较。")
    else:
        st.info("请先上传影像。若想快速体验，可在下方选择内置示例影像。")

    with st.expander("为什么不做自动裁剪/去噪等“影像预处理”？"):
        st.markdown(
            "本项目的研究对象是**检测模型的表现**，而不是影像增强算法。"
            "为了不引入额外的、未被单独评价的变量，标准流程直接把原始影像送入模型；"
            "图像增强作为独立的可调参数保留在工作台中，并在论文的图像质量实验里被单独量化。")


# ==========================================================================
# 2. detection
# ==========================================================================
with tab_detect:
    st.markdown("### AI 骨折检测")
    if det is None:
        st.error("模型未加载，无法执行真实推理。")
        st.code("python scripts/train.py --epochs 80 --batch 16", language="bash")
    elif "raw_image" not in st.session_state:
        st.info("请先在「影像工作台」选择一张影像。")
    else:
        img_for_model = st.session_state.get("model_input", st.session_state["raw_image"])
        run = st.button("▶ 开始检测", type="primary")

        if run:
            with st.spinner("模型推理中…"):
                # one untimed warm-up call so the displayed latency reflects a
                # steady-state inference rather than CUDA context start-up
                det.predict(img_for_model, conf=conf, iou=iou, imgsz=imgsz, device=device)
                result = det.predict(img_for_model, conf=conf, iou=iou, imgsz=imgsz, device=device)
            st.session_state["result"] = result

        result = st.session_state.get("result")
        if result is not None:
            overlay = draw_detections(img_for_model, result.detections)
            left, right = st.columns([1.4, 1])
            with left:
                show_image(to_pil(overlay), width=None)
            with right:
                if result.n_detections == 0:
                    st.markdown(
                        '<div class="ov-note" style="border-color:#0ca30c">'
                        '<b>未检出疑似骨折区域</b><br>在当前阈值下模型没有给出任何检测框。'
                        '这不等于“正常”，低置信度病灶会被阈值过滤，请结合可解释性与人工阅片判断。</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(f'**检出 {result.n_detections} 个疑似区域**')
                    st.caption("框的位置是模型输出，不是影像学定位金标准。")

                m1, m2 = st.columns(2)
                kpi(m1, "检出数", result.n_detections, "阈值 %.2f" % conf)
                kpi(m2, "最高置信度", f"{result.max_conf:.3f}", "0–1")
                m3, m4 = st.columns(2)
                kpi(m3, "本次推理耗时", f"{result.latency_ms:.1f} ms", f"{imgsz}px · {device}")
                kpi(m4, "输入尺寸", f'{result.image_shape[1]}×{result.image_shape[0]}', "H×W")

                st.download_button("下载标注结果图", png_bytes(overlay),
                                   file_name="orthovision_detection.png", mime="image/png")
                st.caption("单张交互式推理时显卡多处于节能频率，耗时会高于离线评测中"
                           "连续推理的 p50（8.6 ms）——两者衡量的并不是同一件事。")

            if result.n_detections:
                st.markdown("**检测明细**")
                df = pd.DataFrame([d.as_dict() for d in result.detections])
                show_table(df)

                st.markdown("**置信度分布**")
                import plotly.graph_objects as go
                fig = go.Figure(go.Bar(
                    x=[f"#{i+1} {d.label_zh}" for i, d in enumerate(result.detections)],
                    y=[d.conf for d in result.detections],
                    width=0.45,
                    marker_color=C_BLUE, marker_line_width=0,
                    text=[f"{d.conf:.2f}" for d in result.detections],
                    textposition="outside",
                    textfont=dict(color=INK_MUTED, size=11),
                    hovertemplate="%{x}<br>置信度 %{y:.3f}<extra></extra>"))
                fig.add_hline(y=conf, line_dash="dot", line_width=1.5,
                              line_color=S_WARN,
                              annotation_text=f"当前阈值 {conf:.2f}",
                              annotation_position="top right",
                              annotation_font=dict(color=S_WARN, size=11))
                fig.update_yaxes(range=[0, 1.08])
                show_chart(style_fig(fig, height=300, y_title="confidence"))
            st.caption("说明：置信度是模型对“该框属于骨折”的自我评估，未经校准，"
                       "不能直接当作患病的概率解读。")
        else:
            st.info("点击「开始检测」执行真实模型推理。")


# ==========================================================================
# 3. explainability
# ==========================================================================
with tab_explain:
    st.markdown("### 可解释性分析")
    result = st.session_state.get("result")
    if det is None:
        st.error("模型未加载。")
    elif result is None:
        st.info("请先执行一次检测，再查看可解释性结果。")
    else:
        method_name = st.radio("解释方法", list(explain_mod.METHODS.keys()), horizontal=True)
        alpha = st.slider("热力图透明度", 0.15, 0.85, 0.45, 0.05)
        img_for_model = st.session_state.get("model_input", st.session_state["raw_image"])

        with st.spinner("计算特征归因…"):
            ex = explain_mod.METHODS[method_name](det, img_for_model, imgsz=imgsz, device=device)

        if ex.cam is None:
            st.warning(ex.error or "当前环境下无法生成热力图，已隐藏该视图。")
            st.markdown('<div class="ov-note">宁可缺少一张热力图，也不生成伪造的解释图。'
                        '论文中对此的处理方式见“讨论—可解释性的局限”。</div>',
                        unsafe_allow_html=True)
        else:
            heat = overlay_heatmap(img_for_model, ex.cam, alpha=alpha)
            boxes = draw_detections(img_for_model, result.detections)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**原图 + 检测框**")
                show_image(to_pil(boxes), width=None)
            with c2:
                st.markdown(f"**{ex.method} 热力图**")
                show_image(to_pil(heat), width=None)

            ratio = explain_mod.cam_in_boxes(ex.cam, result.detections)
            m1, m2, m3 = st.columns(3)
            kpi(m1, "特征图尺寸", "×".join(map(str, ex.feature_shape or ())), "C×H×W")
            kpi(m2, "框内响应占比", f"{ratio:.2%}" if ratio is not None else "—",
                "CAM 能量落在检测框内的比例")
            kpi(m3, "归因方法", ex.method, "作用特征层")

            st.markdown(
                '<div class="ov-note"><b>如何读这张图：</b>热力图反映的是检测头所依赖的'
                '特征响应强度，<b>不是</b>骨折分割结果，也不能证明模型“看到了骨折线”。'
                '框内响应占比只是一个粗粒度的一致性检查（模型说哪里、特征就在哪里响应），'
                '当图像存在钢板、石膏或体表标记时，响应常落在这些高对比结构上。</div>',
                unsafe_allow_html=True)

            trio = side_by_side([img_for_model, boxes, heat],
                                ["original", "detections", "attribution"])
            st.download_button("导出三联图（原图 / 检测 / 归因）", png_bytes(trio),
                               file_name="orthovision_case.png", mime="image/png")
        st.session_state["explain"] = ex


# ==========================================================================
# 4. evaluation centre
# ==========================================================================
with tab_eval:
    st.markdown("### 实验评价中心")
    st.caption("本页所有数字均来自 results/ 下的真实测试结果文件，不根据单张影像的置信度推算。")

    eval_files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not eval_files:
        st.warning("尚未生成测试结果。请先运行：")
        st.code("python scripts/evaluate.py --split test\n"
                "python scripts/perturbation_experiment.py", language="bash")
    else:
        names = {p.stem.replace("eval_", ""): p for p in eval_files}
        pick = st.selectbox("选择实验记录", list(names.keys()))
        payload = read_json(str(names[pick]))

        cm = payload.get("coco_map", {})
        il = payload.get("image_level", {})
        bl = payload.get("box_level", {})
        lat = payload.get("latency_ms", {})

        r1 = st.columns(4)
        kpi(r1[0], "mAP@0.5", cm.get("mAP50", "—"), "COCO 评测")
        kpi(r1[1], "mAP@0.5:0.95", cm.get("mAP50_95", "—"), "COCO 评测")
        kpi(r1[2], "框级 Precision", bl.get("precision", "—"), f'IoU={payload.get("match_iou")}')
        kpi(r1[3], "框级 Recall", bl.get("recall", "—"), f'conf={payload.get("conf")}')
        r2 = st.columns(4)
        kpi(r2[0], "图级敏感度", il.get("sensitivity", "—"), "有骨折被检出的比例")
        kpi(r2[1], "图级特异度", il.get("specificity", "—"), "无骨折被正确排除的比例")
        kpi(r2[2], "图级准确率", il.get("accuracy", "—"), f'N={payload.get("n_images")}')
        kpi(r2[3], "推理耗时 p50/p95", f'{lat.get("p50","—")} / {lat.get("p95","—")} ms',
            payload.get("gpu", ""))

        st.markdown(f'<div class="ov-note">测试集：{payload.get("n_images")} 张'
                    f'（骨折 {payload.get("n_positive")} 张 / 无骨折 {payload.get("n_negative")} 张）；'
                    f'权重 {Path(payload.get("weights","")).name}；置信度阈值 {payload.get("conf")}。'
                    f'</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ov-summary">{headline(payload)}</div>',
                    unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns(2)
        pr = FIGURES_DIR / f"{pick}_BoxPR_curve.png"
        cmx = FIGURES_DIR / "fig_confusion_image_level.png"
        with c1:
            if pr.exists():
                st.markdown("**Precision-Recall 曲线**")
                show_image(str(pr))
            else:
                st.info("未找到 PR 曲线图")
        with c2:
            if cmx.exists():
                st.markdown("**图级混淆矩阵**")
                show_image(str(cmx))
            else:
                st.info("未找到混淆矩阵")

        per_image = RESULTS_DIR / f"per_image_{pick}.csv"
        if per_image.exists():
            df = read_csv(str(per_image), per_image.stat().st_mtime)
            st.markdown("**逐图判定分布**")
            outcomes = ["TP", "FP", "FN", "TN"]
            zh = {"TP": "TP 正确检出", "FP": "FP 误检", "FN": "FN 漏检", "TN": "TN 正确排除"}
            counts = df["outcome"].value_counts().reindex(outcomes).fillna(0).astype(int)
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=[zh[o] for o in outcomes], y=[counts[o] for o in outcomes],
                width=0.45, marker_line_width=0,
                marker_color=[S_GOOD, S_WARN, S_CRIT, C_BLUE],
                text=[counts[o] for o in outcomes], textposition="outside",
                textfont=dict(color=INK_MUTED, size=12),
                hovertemplate="%{x}<br>%{y} 张<extra></extra>"))
            show_chart(style_fig(fig, height=300, y_title="images"))

            sweep_path = RESULTS_DIR / "threshold_sweep.json"
            if sweep_path.exists():
                sw = read_json(str(sweep_path))
                rows_ = sw.get("sweep", [])
                if rows_:
                    st.markdown("**工作点选择：验证集阈值扫描**")
                    fig2 = go.Figure()
                    for key, name, color in (
                            ("sensitivity", "敏感度", C_BLUE),
                            ("specificity", "特异度", C_ORANGE),
                            ("f1", "图级 F1", C_VIOLET)):
                        fig2.add_trace(go.Scatter(
                            x=[r["threshold"] for r in rows_], y=[r[key] for r in rows_],
                            mode="lines+markers", name=name,
                            line=dict(color=color, width=2),
                            marker=dict(size=8, color=color,
                                        line=dict(width=2, color="#111a2e")),
                            hovertemplate=f"{name} %{{y:.3f}} @ 阈值 %{{x}}<extra></extra>"))
                    sel = sw.get("selected_threshold")
                    if sel is not None:
                        fig2.add_vline(x=sel, line_dash="dot", line_width=1.5,
                                       line_color=INK_MUTED)
                        fig2.add_annotation(x=sel, y=1.04, text=f"选定阈值 {sel}",
                                            showarrow=False,
                                            font=dict(color=INK_MUTED, size=11))
                    fig2.update_yaxes(range=[0, 1.1])
                    fig2.update_xaxes(title_text="confidence threshold")
                    show_chart(style_fig(fig2, height=320, show_legend=True))
                    st.caption("阈值在验证集上按图级 F1 最大原则选定，测试集不参与调参。")

            st.markdown("**错误案例分析**")
            kind = st.radio("查看类型", ["FN 漏检", "FP 误检"], horizontal=True)
            target = "FN" if kind.startswith("FN") else "FP"
            subset = df[df["outcome"] == target]
            if subset.empty:
                st.success(f"该测试集上没有 {target} 案例。")
            else:
                n = st.slider("案例序号", 1, len(subset), 1)
                row = subset.iloc[n - 1]
                img_path = (REPO / "data" / "fracatlas_yolo" / payload.get("split", "test")
                            / "images" / row["image"])
                if img_path.exists():
                    im = load_image(img_path)
                    res = det.predict(im, conf=payload.get("conf", DEFAULT_CONF),
                                      iou=payload.get("nms_iou", DEFAULT_IOU),
                                      imgsz=payload.get("imgsz", IMG_SIZE), device=device)
                    st.image(to_pil(draw_detections(im, res.detections)),
                             caption=f"{target}：{row['image']}｜GT={int(row['n_gt'])} 框，"
                                     f"预测={int(row['n_det'])} 框，最高置信度={row['max_conf']:.2f}")
                    st.caption("本案例重新执行推理用于可视化；判定标签取自离线测试记录。")
                else:
                    st.info("找不到原始测试图像文件。")

            pert_path = RESULTS_DIR / "perturbation.json"
            if pert_path.exists():
                pert = read_json(str(pert_path))
                conds = pert.get("conditions", {})
                if conds:
                    st.markdown("**图像质量扰动实验**")
                    zh_names = {
                        "original": "原始", "blur_mild": "轻度模糊",
                        "blur_moderate": "中度模糊", "brightness_-30": "亮度−30",
                        "brightness_+30": "亮度+30", "contrast_0.7": "对比度×0.7",
                        "noise_sigma12": "噪声σ12",
                    }
                    keys = list(conds)
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        x=[zh_names.get(k, k) for k in keys],
                        y=[conds[k]["box"]["f1"] for k in keys],
                        name="框级 F1", marker_color=C_BLUE, marker_line_width=0, width=0.36,
                        hovertemplate="%{x}<br>F1 %{y:.3f}<extra></extra>"))
                    fig3.add_trace(go.Bar(
                        x=[zh_names.get(k, k) for k in keys],
                        y=[conds[k]["image_level"]["sensitivity"] for k in keys],
                        name="图级敏感度", marker_color=C_ORANGE, marker_line_width=0,
                        width=0.36,
                        hovertemplate="%{x}<br>敏感度 %{y:.3f}<extra></extra>"))
                    fig3.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
                    show_chart(style_fig(fig3, height=330, show_legend=True, y_title="score"))
                    st.caption("扰动只施加在测试图像上，标注框保持不变；结果反映对合成变换的敏感性，"
                               "不等同于真实成像条件差异的影响。")

            note = st.text_input("自己分析的要点（可选，会写入导出报告）", "")
            if st.button("导出本页实验报告"):
                st.session_state["last_report"] = build_report(payload, note)
            if st.session_state.get("last_report"):
                st.download_button("下载 Markdown 报告", st.session_state["last_report"],
                                   file_name=f"orthovision_report_{pick}.md")


# ==========================================================================
# 5. provenance
# ==========================================================================
with tab_about:
    st.markdown("### 数据与资源来源")
    st.markdown("""
<div class="ov-card">

**数据集 · FracAtlas v1**
- 来源：Abedeen I. et al., *FracAtlas: A Dataset for Fracture Classification,
  Localization and Segmentation of Musculoskeletal Radiographs*, Data in Brief, 2023.
- 规模：4,083 张 X 光片（717 张含骨折边界框），单类别 `fractured`。
- 许可：**CC BY 4.0**。
- 下载：figshare 条目 22363012（`Annotations/` 内含 YOLO、COCO、Pascal VOC 三种标注）。
- 本项目的改动：按 70/15/15 分层随机划分 train/val/test（seed=42），
  仅使用其 YOLO 标注，**未修改**任何标签内容。

**模型 · Ultralytics YOLO11-S**
- 代码：https://github.com/ultralytics/ultralytics（**AGPL-3.0**）。
- 初始权重：`yolo11s.pt`（COCO 预训练），在 FracAtlas 上微调得到本项目权重。
- 训练得到的权重新增了 FracAtlas 标签信息，发布时一并遵循 AGPL-3.0。

**依赖**
- PyTorch、OpenCV、Streamlit、Plotly、Pandas、NumPy。

</div>
""", unsafe_allow_html=True)

    st.markdown("### 许可与合规说明")
    st.markdown("""
- 本项目**代码**以 MIT 许可发布（见 `LICENSE`）。
- 由于依赖 Ultralytics（AGPL-3.0），本项目**训练得到的模型权重**按 AGPL-3.0 发布；
  若要在闭源产品中使用，需要替换检测框架或取得商业授权。这一点已在 README 中写明。
- 数据集为 CC BY 4.0，使用时保留原作者署名与数据集名称，并在论文中引用原文。
- 本项目为课程研究原型，**不用于临床诊断**。
""")

    st.markdown("### 系统能力边界")
    st.markdown("""
| 能做什么 | 不能做什么 |
| --- | --- |
| 对单张 X 光给出疑似骨折区域的边界框与置信度 | 不能给出“骨折/正常”的临床诊断结论 |
| 展示特征响应热力图，辅助理解模型关注区域 | 热力图不能作为病灶定位依据 |
| 在固定测试集上报告可复现的指标 | 指标不能外推到真实临床人群 |
| 量化模型对模糊、亮度等扰动的敏感性 | 扰动实验不代表真实成像质量差异的影响 |
""")

st.markdown(
    '<div class="ov-footer"><span>医学人工智能导论 · 结课作业 · OrthoVision v1.0</span>'
    '<span>陈芯芯 · 102404102 · 24级人工智能一班</span></div>',
    unsafe_allow_html=True)
