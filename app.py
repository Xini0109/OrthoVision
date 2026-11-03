"""OrthoVision - Streamlit front-end.

A thin presentation layer over the trained YOLO11 checkpoint and the
JSON/CSV artefacts produced by the evaluation scripts. Nothing on screen is
invented: if a file or a checkpoint is missing, the UI says so.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
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

st.set_page_config(page_title="OrthoVision · 智能骨折辅助检测",
                   page_icon="🦴", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# theme
# --------------------------------------------------------------------------
CSS = """
<style>
:root{
  --bg:#070d1a; --bg2:#0b1424; --panel:#101a2e; --panel2:#152138;
  --line:#213252; --line2:#2b3f63;
  --ink:#eef3fb; --muted:#9aa9c2; --dim:#6f7f9c;
  --blue:#4d8dff; --cyan:#35d0e0; --violet:#8f7bff;
  --good:#21c07a; --warn:#f5b83d; --bad:#f06565;
}
html, body, .stApp{background:var(--bg);}
.stApp{
  background:
    radial-gradient(900px 520px at 8% -8%, #12264a 0%, transparent 55%),
    radial-gradient(700px 420px at 95% 0%, #123a3f 0%, transparent 50%),
    var(--bg);
}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0c1628 0%,#0a1220 100%);
  border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container{padding-top:1.4rem;}
h1,h2,h3,h4,h5{color:var(--ink)!important; letter-spacing:.2px;}
p, span, label, div{color:var(--ink);}
.block-container{padding-top:1.6rem; padding-bottom:3rem; max-width:1560px;}
header[data-testid="stHeader"]{background:transparent;}
div[data-testid="stToolbar"]{right:1rem;}
footer{visibility:hidden;}

/* hero */
.ov-hero{
  display:flex; align-items:center; gap:20px; padding:24px 28px;
  background:linear-gradient(110deg,#10203c 0%,#0e1a30 55%,#0c1729 100%);
  border:1px solid var(--line); border-radius:20px;
  box-shadow:0 18px 50px -28px #000;
}
.ov-logo{
  width:58px;height:58px;border-radius:16px;display:grid;place-items:center;
  font-size:29px;background:linear-gradient(140deg,var(--blue),#6f5cff);
  box-shadow:0 8px 24px -8px #4d8dff;
}
.ov-title{font-size:25px;font-weight:800;line-height:1.18;margin:0;}
.ov-sub{color:var(--muted);font-size:13px;margin-top:5px;}
.ov-hero-right{margin-left:auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:flex-end;}
.ov-chip{
  display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);
  border:1px solid var(--line);background:#0d1729;padding:7px 12px;border-radius:999px;
  white-space:nowrap;
}
.ov-dot{width:7px;height:7px;border-radius:50%;background:var(--good);
  box-shadow:0 0 0 4px rgba(33,192,122,.14);}
.ov-dot.warn{background:var(--warn);box-shadow:0 0 0 4px rgba(245,184,61,.14);}

/* cards */
.ov-card{
  background:linear-gradient(180deg,var(--panel2),var(--panel));
  border:1px solid var(--line);border-radius:18px;padding:18px 20px;
  margin-bottom:14px;
}
.ov-card.tight{padding:14px 16px;}
.ov-card-title{font-size:14px;font-weight:700;color:var(--ink);margin:0 0 12px;}
.ov-kpi{
  background:linear-gradient(165deg,#16233c,#101a2e);
  border:1px solid var(--line);border-radius:16px;padding:15px 17px;
  position:relative;overflow:hidden;
}
.ov-kpi:after{content:"";position:absolute;left:0;top:0;width:100%;height:3px;
  background:linear-gradient(90deg,var(--blue),var(--cyan));opacity:.85;}
.ov-kpi .k{color:var(--muted);font-size:11.5px;letter-spacing:.6px;text-transform:uppercase;}
.ov-kpi .v{font-size:27px;font-weight:800;color:var(--ink);margin-top:3px;line-height:1.1;}
.ov-kpi .d{color:var(--dim);font-size:11.5px;margin-top:3px;}
.ov-kpi.good .v{color:var(--good);} .ov-kpi.warn .v{color:var(--warn);}
.ov-kpi.bad .v{color:var(--bad);}

/* tags and notes */
.ov-tag{display:inline-block;padding:4px 10px;border-radius:999px;font-size:11.5px;
  border:1px solid var(--line);color:var(--muted);background:#0d1729;margin:0 6px 6px 0;}
.ov-tag.blue{color:#bcd3ff;border-color:#2a4d86;background:#10213b;}
.ov-tag.green{color:#a9efc9;border-color:#1f5b44;background:#0d241c;}
.ov-tag.amber{color:#ffe0a3;border-color:#6b5120;background:#241c0c;}
.ov-note{border-left:3px solid var(--blue);background:#0d1930;padding:10px 13px;
  border-radius:10px;color:var(--muted);font-size:12.5px;line-height:1.6;}
.ov-foot{color:var(--dim);font-size:11.5px;margin-top:6px;}
.ov-section{display:flex;align-items:center;gap:10px;margin:8px 0 12px;}
.ov-section .bar{width:4px;height:16px;border-radius:4px;
  background:linear-gradient(180deg,var(--blue),var(--cyan));}
.ov-section .t{font-size:15px;font-weight:750;color:var(--ink);}
.ov-section .s{font-size:12px;color:var(--dim);margin-left:auto;}

/* workflow steps */
.ov-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.ov-step{background:#0e1930;border:1px solid var(--line);border-radius:14px;padding:13px 14px;}
.ov-step .n{font-size:10.5px;color:var(--cyan);letter-spacing:1px;font-weight:700;}
.ov-step .h{font-size:13.5px;font-weight:700;margin-top:3px;}
.ov-step .b{font-size:11.5px;color:var(--muted);margin-top:3px;line-height:1.5;}

/* tabs (Streamlit renders tabs as div[data-testid="stTab"]) */
div[data-testid="stTabs"] div[role="tablist"]{
  gap:8px;padding:4px 0 10px;border-bottom:1px solid var(--line)!important;
}
div[data-testid="stTab"]{
  background:#0e1930;border:1px solid var(--line);border-radius:11px;
  padding:8px 17px;color:var(--muted);font-weight:650;transition:all .15s ease;
}
div[data-testid="stTab"] p{font-size:13px;font-weight:650;margin:0;color:inherit;}
div[data-testid="stTab"]:hover{border-color:#2f4f86;color:#dbe6f7;}
div[data-testid="stTab"][aria-selected="true"]{
  color:#fff!important;border-color:#3f6fd6!important;
  background:linear-gradient(135deg,#17305a,#122340)!important;
  box-shadow:0 6px 18px -12px #4d8dff;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-border"]{display:none!important;}

/* buttons, inputs, tables */
.stButton>button,.stDownloadButton>button{
  background:linear-gradient(135deg,#3f7dff,#6a5cff);color:#fff;border:0;
  border-radius:11px;font-weight:700;padding:.52rem 1.15rem;
  box-shadow:0 10px 24px -14px #4d8dff;
}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.08);}
.stDownloadButton>button{background:linear-gradient(135deg,#173152,#1b2b4a);
  border:1px solid var(--line2);box-shadow:none;}
div[data-testid="stFileUploader"]{
  background:#0d1729;border:1px dashed var(--line2);border-radius:16px;padding:8px;
}
div[data-testid="stFileUploader"] section{background:transparent;}
.stDataFrame{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
div[data-testid="stMetricValue"]{color:var(--ink);}
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div,
div[data-testid="stRadio"] label{color:var(--ink);}
details[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:#0d1729;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# chart / status palette (validated on the dark panel)
C_BLUE = "#4d8dff"
C_CYAN = "#35d0e0"
C_ORANGE = "#f0864a"
C_VIOLET = "#9a8bff"
C_GREEN = "#21c07a"
C_RED = "#f06565"
C_YELLOW = "#f5b83d"
INK_MUTED = "#93a3bd"
GRID = "#22304d"
AXIS = "#2b3f63"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def show_image(img, caption: str = "", width=None):
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


def heatmap_only(img, cam, colormap=cv2.COLORMAP_TURBO):
    cam = np.ascontiguousarray(np.clip(cam, 0, 1), dtype=np.float32)
    if cam.shape[:2] != img.shape[:2]:
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def style_fig(fig, height=290, show_legend=False, y_title=""):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_MUTED, size=12, family="Segoe UI, system-ui, sans-serif"),
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


def kpi(col, label, value, delta="", tone=""):
    col.markdown(
        f'<div class="ov-kpi {tone}"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="d">{delta}</div></div>',
        unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(
        f'<div class="ov-section"><span class="bar"></span>'
        f'<span class="t">{title}</span><span class="s">{sub}</span></div>',
        unsafe_allow_html=True)


def hero(model_ok: bool, device: str):
    dot = "ov-dot" if model_ok else "ov-dot warn"
    state = "模型已加载" if model_ok else "模型未就绪"
    st.markdown(
        f'''<div class="ov-hero">
          <div class="ov-logo">🦴</div>
          <div>
            <div class="ov-title">OrthoVision 智能骨折辅助检测系统</div>
            <div class="ov-sub">基于 YOLO11 的骨骼 X 光影像骨折检测 · 实验评价 · 可解释性分析</div>
          </div>
          <div class="ov-hero-right">
            <span class="ov-chip"><span class="{dot}"></span>{state}</span>
            <span class="ov-chip">推理设备 · {device}</span>
            <span class="ov-chip">v1.1 · 课程研究版</span>
          </div>
        </div>''', unsafe_allow_html=True)


def model_status():
    if "detector" not in st.session_state:
        try:
            st.session_state.detector = get_detector(
                str(DEFAULT_WEIGHTS), st.session_state.get("device", "cuda:0"))
            st.session_state.model_error = None
        except Exception as exc:
            st.session_state.detector = None
            st.session_state.model_error = str(exc)
    return st.session_state.detector


def eval_payload():
    p = RESULTS_DIR / "eval_test.json"
    return read_json(str(p)) if p.exists() else {}


def build_report(payload: dict, note: str = "") -> str:
    cm = payload.get("coco_map", {})
    il = payload.get("image_level", {})
    bl = payload.get("box_level", {})
    lat = payload.get("latency_ms", {})
    lines = [
        "# OrthoVision 实验报告", "",
        f"- 权重：`{Path(str(payload.get('weights', ''))).name}`",
        f"- 测试集：{payload.get('n_images')} 张（骨折 {payload.get('n_positive')} / "
        f"无骨折 {payload.get('n_negative')}）",
        f"- 阈值：conf={payload.get('conf')}, NMS IoU={payload.get('nms_iou')}, "
        f"匹配 IoU={payload.get('match_iou')}",
        f"- 环境：torch {payload.get('torch')} · {payload.get('gpu')}", "",
        "## 指标", "", "| 指标 | 数值 |", "| --- | --- |",
        f"| mAP@0.5 | {cm.get('mAP50')} |",
        f"| mAP@0.5:0.95 | {cm.get('mAP50_95')} |",
        f"| 框级 Precision | {bl.get('precision')} |",
        f"| 框级 Recall | {bl.get('recall')} |",
        f"| 框级 F1 | {bl.get('f1')} |",
        f"| 图级敏感度 | {il.get('sensitivity')} |",
        f"| 图级特异度 | {il.get('specificity')} |",
        f"| 图级准确率 | {il.get('accuracy')} |",
        f"| 推理耗时 p50 / p95 (ms) | {lat.get('p50')} / {lat.get('p95')} |", "",
    ]
    if note.strip():
        lines += ["## 分析备注", "", note.strip(), ""]
    lines += ["---", "", "> " + DISCLAIMER]
    return "\n".join(lines)


def build_case_report(name, result, conf, iou, imgsz, device) -> str:
    lines = [
        "# OrthoVision 单例检测报告", "",
        f"- 影像：`{name}`",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：`{DEFAULT_WEIGHTS.name}`",
        f"- 阈值：conf={conf}, NMS IoU={iou}, 输入 {imgsz}px, 设备 {device}", "",
        f"## 检测结果（{result.n_detections} 个疑似区域）", "",
    ]
    if result.n_detections:
        lines += ["| 序号 | 类别 | 置信度 | x1 | y1 | x2 | y2 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for i, d in enumerate(result.detections, 1):
            x1, y1, x2, y2 = (round(float(v), 1) for v in d.xyxy)
            lines.append(f"| {i} | {d.label_zh} | {d.conf:.4f} | {x1} | {y1} | {x2} | {y2} |")
    else:
        lines.append("当前阈值下未检出疑似骨折区域。**这不等于“正常”**，低置信度病灶可能已被阈值过滤。")
    lines += ["", f"- 本次推理耗时：{result.latency_ms:.1f} ms",
              "", "---", "", "> " + DISCLAIMER]
    return "\n".join(lines)


def headline(payload: dict) -> str:
    il = payload.get("image_level", {})
    cm = payload.get("coco_map", {})
    bl = payload.get("box_level", {})
    try:
        sens = float(il.get("sensitivity", 0.0))
        spec = float(il.get("specificity", 0.0))
    except (TypeError, ValueError):
        return "该记录缺少图级指标。"
    if spec - sens >= 0.15:
        reading = f"排除无骨折影像较可靠（特异度 {spec:.2f}），发现骨折的能力偏弱（敏感度 {sens:.2f}），漏检是主要风险。"
    elif sens - spec >= 0.15:
        reading = f"倾向多报（敏感度 {sens:.2f}），代价是无骨折影像上的误报偏多（特异度 {spec:.2f}）。"
    else:
        reading = f"敏感度 {sens:.2f} 与特异度 {spec:.2f} 接近，误检与漏检相对均衡。"
    return (f"<b>结果解读：</b>{reading} 　mAP@0.5 = {cm.get('mAP50')}，"
            f"框级 F1 = {bl.get('f1')}。指标来自固定测试集，不代表真实临床人群。")


def run_inference(det, img, conf, iou, imgsz, device):
    # Two warm-up calls let the GPU ramp up; the reported latency is the
    # median of three timed calls, so a single clock spike does not dominate.
    for _ in range(2):
        det.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device)
    runs = [det.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device)
            for _ in range(3)]
    result = runs[0]
    result.latency_ms = sorted(r.latency_ms for r in runs)[1]
    return result


def record_history(name, result, conf):
    hist = st.session_state.setdefault("history", [])
    hist.insert(0, {
        "时间": datetime.now().strftime("%H:%M:%S"),
        "影像": name,
        "检出框": result.n_detections,
        "最高置信度": round(result.max_conf, 3),
        "耗时(ms)": round(result.latency_ms, 1),
        "阈值": conf,
    })
    del hist[12:]


def load_eval(name: str):
    p = RESULTS_DIR / f"eval_{name}.json"
    return read_json(str(p)) if p.exists() else {}


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
det = model_status()
device = st.session_state.get("device", "cuda:0")
model_ok = det is not None

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
        '<div style="width:34px;height:34px;border-radius:10px;display:grid;place-items:center;'
        'background:linear-gradient(135deg,#4d8dff,#6f5cff);font-size:18px;">🦴</div>'
        '<div><div style="font-weight:800;font-size:14px;">OrthoVision</div>'
        '<div style="color:#6f7f9c;font-size:11px;">Radiology AI Workstation</div></div></div>',
        unsafe_allow_html=True)

    st.markdown("##### 推理设置")
    device = st.selectbox("推理设备", ["cuda:0", "cpu"],
                          index=0 if device == "cuda:0" else 1)
    st.session_state["device"] = device
    conf = st.slider("置信度阈值", 0.05, 0.95, DEFAULT_CONF, 0.05,
                     help="低于该值的预测框会被过滤")
    iou = st.slider("NMS IoU", 0.10, 0.90, DEFAULT_IOU, 0.05)
    imgsz = st.select_slider("推理分辨率", options=[416, 512, 640, 768, 896], value=IMG_SIZE)

    st.markdown("##### 模型状态")
    if not model_ok:
        st.error("模型未就绪")
        st.code(str(DEFAULT_WEIGHTS), language="text")
        st.caption("运行 scripts/train.py，或把权重放入 weights/")
    else:
        d = det.describe()
        st.markdown(
            f'<div class="ov-card tight">'
            f'<div style="font-size:12px;color:#9aa9c2;">骨折 · 单类检测</div>'
            f'<div style="font-size:22px;font-weight:800;margin-top:2px;">{d["参数量(M)"]}M '
            f'<span style="font-size:11px;color:#6f7f9c;font-weight:500;">参数</span></div>'
            f'<div style="font-size:11px;color:#6f7f9c;margin-top:2px;">'
            f'{d["权重文件"]} · {d["权重大小(MB)"]} MB</div></div>',
            unsafe_allow_html=True)

    stats_path = REPO / "data" / "fracatlas_yolo" / "dataset_stats.json"
    if stats_path.exists():
        s = read_json(str(stats_path))
        st.markdown("##### 数据集")
        st.caption(f"FracAtlas · {s['total_images']} 张 · 阳性 {s['fractured_images']} 张")
        st.markdown(
            f'<span class="ov-tag">训练 {s["splits"]["train"]["images"]}</span>'
            f'<span class="ov-tag">验证 {s["splits"]["val"]["images"]}</span>'
            f'<span class="ov-tag">测试 {s["splits"]["test"]["images"]}</span>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.caption(DISCLAIMER)


hero(model_ok, device)

tab_home, tab_detect, tab_explain, tab_eval, tab_about = st.tabs(
    ["首页", "AI 骨折检测", "可解释性分析", "实验评价中心", "关于系统"])

# ==========================================================================
# 1. home
# ==========================================================================
with tab_home:
    ev = eval_payload()
    cm = ev.get("coco_map", {})
    lat = ev.get("latency_ms", {})
    desc = det.describe() if det is not None else {}

    c = st.columns(4)
    kpi(c[0], "测试集 mAP@0.5", cm.get("mAP50", "—"), "COCO 协议 · 603 张")
    kpi(c[1], "框级 F1", ev.get("box_level", {}).get("f1", "—"),
        f'conf = {ev.get("conf", "—")}')
    kpi(c[2], "图级敏感度", ev.get("image_level", {}).get("sensitivity", "—"),
        "有骨折影像被检出")
    kpi(c[3], "推理 p50", f'{lat.get("p50", "—")} ms', "640 px · RTX 4060")

    left, right = st.columns([1.55, 1])

    with left:
        section("系统工作流", "从影像到结论")
        st.markdown(
            '''<div class="ov-flow">
              <div class="ov-step"><div class="n">STEP 01</div><div class="h">影像载入</div>
                <div class="b">支持 JPG / PNG / BMP / TIFF，自动读取尺寸与灰度信息。</div></div>
              <div class="ov-step"><div class="n">STEP 02</div><div class="h">骨折检测</div>
                <div class="b">YOLO11-S 输出疑似骨折区域、置信度与推理耗时。</div></div>
              <div class="ov-step"><div class="n">STEP 03</div><div class="h">特征归因</div>
                <div class="b">Eigen-CAM / Grad-CAM 展示检测头依赖的特征区域。</div></div>
              <div class="ov-step"><div class="n">STEP 04</div><div class="h">结果导出</div>
                <div class="b">导出标注图、检测明细与结构化报告。</div></div>
            </div>''', unsafe_allow_html=True)

        st.write("")
        section("最近检测", "当前会话")
        hist = st.session_state.get("history", [])
        if hist:
            show_table(pd.DataFrame(hist))
        else:
            st.markdown(
                '<div class="ov-note">尚未执行检测。切换到「AI 骨折检测」上传影像即可开始；'
                '首页会保留本次会话的最近 12 条记录。</div>',
                unsafe_allow_html=True)

        st.write("")
        section("示例影像", "来自 FracAtlas 测试集")
        sample_dir = REPO / "assets" / "samples"
        samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
        if samples:
            cols = st.columns(min(3, len(samples)))
            for i, p in enumerate(samples[:3]):
                with cols[i]:
                    try:
                        st.image(str(p), caption=p.stem, width=210)
                    except TypeError:
                        st.image(str(p), caption=p.stem, use_column_width=True)

    with right:
        section("运行状态")
        gpu_name = "—"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        weight_mtime = (datetime.fromtimestamp(DEFAULT_WEIGHTS.stat().st_mtime).strftime("%m-%d %H:%M")
                        if DEFAULT_WEIGHTS.exists() else "—")
        st.markdown(
            f'<div class="ov-card">'
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;'
            f'border-bottom:1px solid #1b2942;"><span style="color:#9aa9c2;font-size:12px;">检测模型</span>'
            f'<span style="font-size:12px;font-weight:650;">YOLO11-S</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;'
            f'border-bottom:1px solid #1b2942;"><span style="color:#9aa9c2;font-size:12px;">权重版本</span>'
            f'<span style="font-size:12px;font-weight:650;">{weight_mtime}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;'
            f'border-bottom:1px solid #1b2942;"><span style="color:#9aa9c2;font-size:12px;">GPU</span>'
            f'<span style="font-size:12px;font-weight:650;text-align:right;max-width:62%;">{gpu_name}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;'
            f'border-bottom:1px solid #1b2942;"><span style="color:#9aa9c2;font-size:12px;">参数量</span>'
            f'<span style="font-size:12px;font-weight:650;">{desc.get("参数量(M)", "—")} M</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:7px 0;">'
            f'<span style="color:#9aa9c2;font-size:12px;">检测类别</span>'
            f'<span style="font-size:12px;font-weight:650;">骨折（单类）</span></div>'
            f'</div>', unsafe_allow_html=True)

        section("核心指标")
        il = ev.get("image_level", {})
        bl = ev.get("box_level", {})
        st.markdown(
            f'<div class="ov-card tight">'
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
            f'<span style="color:#9aa9c2;font-size:12px;">Precision</span>'
            f'<span style="font-weight:700;">{bl.get("precision","—")}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
            f'<span style="color:#9aa9c2;font-size:12px;">Recall</span>'
            f'<span style="font-weight:700;">{bl.get("recall","—")}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
            f'<span style="color:#9aa9c2;font-size:12px;">特异度</span>'
            f'<span style="font-weight:700;">{il.get("specificity","—")}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;">'
            f'<span style="color:#9aa9c2;font-size:12px;">准确率</span>'
            f'<span style="font-weight:700;">{il.get("accuracy","—")}</span></div>'
            f'<div style="color:#6f7f9c;font-size:11px;margin-top:6px;">'
            f'测试集 {ev.get("n_images","—")} 张，阈值 {ev.get("conf","—")}</div>'
            f'</div>', unsafe_allow_html=True)


# ==========================================================================
# 2. detection
# ==========================================================================
with tab_detect:
    if det is None:
        st.error("模型未加载，无法执行真实推理。")
        st.code("python scripts/train.py --epochs 80 --batch 16", language="bash")
    else:
        section("影像输入", "上传或选择示例")
        up_col, sample_col = st.columns([1.4, 1])
        with up_col:
            up = st.file_uploader("上传 X 光影像（JPG / PNG / BMP / TIFF）",
                                  type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        sample_dir = REPO / "assets" / "samples"
        samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
        with sample_col:
            sample_pick = st.selectbox(
                "使用内置示例影像",
                ["不使用"] + [p.name for p in samples]) if samples else "不使用"

        with st.expander("影像预处理（可选，默认不改变模型输入）"):
            p1, p2, p3 = st.columns(3)
            with p1:
                brightness = st.slider("亮度", -60, 60, 0, 5)
            with p2:
                contrast = st.slider("对比度", 0.5, 2.0, 1.0, 0.05)
            with p3:
                sharpen = st.slider("锐化强度", 0.0, 1.5, 0.0, 0.1)
            q1, q2 = st.columns([1, 1])
            with q1:
                use_clahe = st.checkbox("CLAHE 局部增强", value=False)
            with q2:
                feed_enhanced = st.checkbox("以增强图像推理（实验模式）", value=False)

        source_name, raw = None, None
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
            img_for_model = st.session_state["model_input"]

            meta = image_info(raw)
            c = st.columns(4)
            kpi(c[0], "影像尺寸", f'{meta["width"]}×{meta["height"]}',
                f'{meta["megapixels"]} MP')
            kpi(c[1], "平均灰度", meta["mean_intensity"], "0–255")
            kpi(c[2], "灰度标准差", meta["contrast_std"], "对比度")
            kpi(c[3], "色彩模式", "灰度" if meta["is_grayscale"] else "彩色",
                (source_name or "").split(".")[-1].upper())

            st.write("")
            b1, b2, b3 = st.columns([1, 1, 2.2])
            run = b1.button("开始检测", type="primary", use_container_width=True)
            if b2.button("清空结果", use_container_width=True):
                st.session_state.pop("result", None)
                st.session_state.pop("explain", None)
                st.rerun()

            if run:
                with st.spinner("模型推理中…"):
                    result = run_inference(det, img_for_model, conf, iou, imgsz, device)
                st.session_state["result"] = result
                record_history(source_name, result, conf)
                # Rerun so the home tab (rendered earlier in this pass) picks
                # up the new history entry as well.
                st.rerun()

            result = st.session_state.get("result")
            if result is None:
                st.write("")
                section("影像预览")
                show_image(to_pil(img_for_model), width=None)
            else:
                overlay = draw_detections(img_for_model, result.detections)
                st.write("")
                section("检测结果", f"{result.n_detections} 个疑似区域")
                mode = st.radio("显示方式", ["并排对比", "仅检测结果", "仅原图"],
                                horizontal=True, label_visibility="collapsed")

                if mode == "并排对比":
                    a, b = st.columns(2)
                    with a:
                        st.markdown('<span class="ov-tag blue">原始影像</span>',
                                    unsafe_allow_html=True)
                        show_image(to_pil(img_for_model), width=None)
                    with b:
                        st.markdown('<span class="ov-tag green">检测结果</span>',
                                    unsafe_allow_html=True)
                        show_image(to_pil(overlay), width=None)
                elif mode == "仅检测结果":
                    show_image(to_pil(overlay), width=None)
                else:
                    show_image(to_pil(img_for_model), width=None)

                st.write("")
                c = st.columns(4)
                kpi(c[0], "检出目标数", result.n_detections, f"阈值 {conf:.2f}")
                kpi(c[1], "最高置信度", f"{result.max_conf:.3f}", "0–1")
                kpi(c[2], "推理耗时", f"{result.latency_ms:.1f} ms",
                    f"{imgsz} px · {device}", tone="good")
                kpi(c[3], "输入尺寸",
                    f'{result.image_shape[1]}×{result.image_shape[0]}', "H×W")
                st.caption("单次交互推理受 GPU 节能频率与框架开销影响，耗时会高于离线评测的 "
                           "p50（8.58 ms）；对比实验中的延迟见「实验评价中心 → 对比实验」。")

                if result.n_detections == 0:
                    st.markdown(
                        '<div class="ov-note">当前阈值下未检出可疑区域。这不等于“正常”：'
                        '低置信度病灶可能已被阈值过滤，建议降低阈值并结合可解释性视图复核。</div>',
                        unsafe_allow_html=True)
                else:
                    left, right = st.columns([1.15, 1])
                    with left:
                        section("检测明细")
                        show_table(pd.DataFrame([d.as_dict() for d in result.detections]))
                    with right:
                        section("置信度分布")
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            x=[f"框 {i + 1}" for i in range(len(result.detections))],
                            y=[d.conf for d in result.detections],
                            width=0.5, marker_color=C_BLUE, marker_line_width=0,
                            text=[f"{d.conf:.2f}" for d in result.detections],
                            textposition="outside",
                            textfont=dict(color=INK_MUTED, size=11),
                            hovertemplate="%{x}<br>置信度 %{y:.3f}<extra></extra>"))
                        fig.add_hline(y=conf, line_dash="dot", line_width=1.5,
                                      line_color=C_YELLOW,
                                      annotation_text=f"阈值 {conf:.2f}",
                                      annotation_position="top right",
                                      annotation_font=dict(color=C_YELLOW, size=11))
                        fig.update_yaxes(range=[0, 1.08])
                        fig.update_xaxes(type="category")
                        show_chart(style_fig(fig, height=250, y_title="confidence"))

                st.write("")
                section("结果导出")
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.download_button("标注图 PNG", png_bytes(overlay),
                                       file_name=f"orthovision_{Path(source_name).stem}.png",
                                       mime="image/png", use_container_width=True)
                with e2:
                    if result.n_detections:
                        csv_bytes = pd.DataFrame(
                            [d.as_dict() for d in result.detections]
                        ).to_csv(index=False).encode("utf-8-sig")
                        st.download_button("检测明细 CSV", csv_bytes,
                                           file_name="orthovision_detections.csv",
                                           mime="text/csv", use_container_width=True)
                with e3:
                    report = build_case_report(source_name, result, conf, iou,
                                               imgsz, device)
                    st.download_button("检测报告 MD", report,
                                       file_name=f"orthovision_{Path(source_name).stem}.md",
                                       mime="text/markdown", use_container_width=True)
                st.caption("提示：置信度是模型自评分数，未经概率校准，不能直接当作患病概率。")
        else:
            st.markdown(
                '<div class="ov-note">上传一张 X 光影像，或在上方选择一个内置示例。'
                '检测完成后可直接切换到「可解释性分析」查看特征归因。</div>',
                unsafe_allow_html=True)


# ==========================================================================
# 3. explainability
# ==========================================================================
with tab_explain:
    section("特征归因", "实验性分析")
    result = st.session_state.get("result")
    if det is None:
        st.error("模型未加载。")
    elif result is None:
        st.markdown('<div class="ov-note">请先执行一次检测，再查看可解释性结果。</div>',
                    unsafe_allow_html=True)
    else:
        img_for_model = st.session_state.get("model_input", st.session_state["raw_image"])
        c1, c2, c3 = st.columns([1.5, 1.4, 1])
        with c1:
            method_name = st.radio("解释方法", list(explain_mod.METHODS.keys()),
                                   horizontal=True)
        with c2:
            alpha = st.slider("热力图透明度", 0.15, 0.85, 0.45, 0.05)
        with c3:
            st.markdown(
                '<div style="padding-top:28px;"><span class="ov-tag amber">'
                'CAM ≠ 骨折分割</span></div>', unsafe_allow_html=True)

        with st.spinner("计算特征归因…"):
            ex = explain_mod.METHODS[method_name](det, img_for_model, imgsz=imgsz,
                                                  device=device)

        if ex.cam is None:
            st.warning(ex.error or "当前环境下无法生成热力图，已隐藏该视图。")
            st.markdown('<div class="ov-note">系统不会生成伪造的解释图。</div>',
                        unsafe_allow_html=True)
        else:
            boxes = draw_detections(img_for_model, result.detections)
            heat = overlay_heatmap(img_for_model, ex.cam, alpha=alpha)
            pure = heatmap_only(img_for_model, ex.cam)
            ratio = explain_mod.cam_in_boxes(ex.cam, result.detections)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown('<span class="ov-tag blue">原始影像</span>',
                            unsafe_allow_html=True)
                show_image(to_pil(img_for_model), width=None)
            with r2:
                st.markdown('<span class="ov-tag blue">检测结果</span>',
                            unsafe_allow_html=True)
                show_image(to_pil(boxes), width=None)
            r3, r4 = st.columns(2)
            with r3:
                st.markdown(f'<span class="ov-tag amber">{ex.method} 热力图</span>',
                            unsafe_allow_html=True)
                show_image(to_pil(pure), width=None)
            with r4:
                st.markdown('<span class="ov-tag green">热力图叠加</span>',
                            unsafe_allow_html=True)
                show_image(to_pil(heat), width=None)

            st.write("")
            c = st.columns(3)
            kpi(c[0], "特征图尺寸", "×".join(map(str, ex.feature_shape or ())), "C×H×W")
            kpi(c[1], "框内响应占比",
                f"{ratio:.2%}" if ratio is not None else "—",
                "CAM 能量落在检测框内", tone="good" if (ratio or 0) > 0.5 else "warn")
            kpi(c[2], "归因方法", ex.method, "检测头输入特征层")

            st.markdown(
                '<div class="ov-note"><b>阅读方式：</b>热力图显示模型依赖的特征响应区域，'
                '用于观察“模型在看哪里”，<b>不是</b>骨折分割结果。高响应也可能来自钢板、'
                '石膏或体表标记等高对比结构。</div>', unsafe_allow_html=True)

            trio = side_by_side([img_for_model, boxes, heat],
                                ["original", "detections", "attribution"])
            st.download_button("导出三联图（原图 / 检测 / 归因）", png_bytes(trio),
                               file_name="orthovision_case.png", mime="image/png")
        st.session_state["explain"] = ex


# ==========================================================================
# 4. evaluation centre
# ==========================================================================
with tab_eval:
    section("实验评价中心", "所有数字来自 results/ 下的真实实验文件")

    eval_files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not eval_files:
        st.warning("尚未生成测试结果。请先运行：")
        st.code("python scripts/evaluate.py --split test\n"
                "python scripts/perturbation_experiment.py", language="bash")
    else:
        names = {p.stem.replace("eval_", ""): p for p in eval_files}
        labels = {"test": "YOLO11s · 测试集", "test_yolo11n": "YOLO11n · 测试集"}
        pick = st.selectbox("实验记录", list(names.keys()),
                            format_func=lambda k: labels.get(k, k))
        payload = read_json(str(names[pick]))
        cm = payload.get("coco_map", {})
        il = payload.get("image_level", {})
        bl = payload.get("box_level", {})
        lat = payload.get("latency_ms", {})

        c = st.columns(4)
        kpi(c[0], "mAP@0.5", cm.get("mAP50", "—"), "COCO 评测")
        kpi(c[1], "mAP@0.5:0.95", cm.get("mAP50_95", "—"), "更严格定位")
        kpi(c[2], "Precision", bl.get("precision", "—"),
            f'框级 · IoU={payload.get("match_iou")}')
        kpi(c[3], "Recall", bl.get("recall", "—"), f'阈值 {payload.get("conf")}')
        c = st.columns(4)
        kpi(c[0], "F1", bl.get("f1", "—"), "Precision / Recall 综合")
        kpi(c[1], "图级敏感度", il.get("sensitivity", "—"), "有骨折被检出")
        kpi(c[2], "图级特异度", il.get("specificity", "—"), "无骨折被排除")
        kpi(c[3], "耗时 p50 / p95",
            f'{lat.get("p50", "—")} / {lat.get("p95", "—")} ms', payload.get("gpu", ""))

        st.markdown(f'<div class="ov-note">{headline(payload)}</div>',
                    unsafe_allow_html=True)
        st.caption(f"测试集 {payload.get('n_images')} 张（骨折 {payload.get('n_positive')} / "
                   f"无骨折 {payload.get('n_negative')}）；权重 "
                   f"{Path(str(payload.get('weights', ''))).name}；阈值 {payload.get('conf')}。")

        sec1, sec2, sec3, sec4, sec5 = st.tabs(
            ["性能曲线", "阈值实验", "对比实验", "错误案例", "鲁棒性"])

        with sec1:
            import plotly.graph_objects as go
            sw_path = RESULTS_DIR / f"threshold_experiment_{pick}.json"
            if not sw_path.exists():
                sw_path = RESULTS_DIR / "threshold_experiment_test.json"
            sweep_rows = (read_json(str(sw_path)).get("sweep", [])
                          if sw_path.exists() else [])

            a, b = st.columns(2)
            with a:
                st.markdown("**Precision–Recall 曲线**")
                if sweep_rows:
                    pts = list(reversed(sweep_rows))  # high threshold -> low
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[r["box"]["recall"] for r in pts],
                        y=[r["box"]["precision"] for r in pts],
                        mode="lines+markers", line=dict(color=C_BLUE, width=2.4),
                        marker=dict(size=5, color=C_BLUE),
                        name="PR",
                        customdata=[r["threshold"] for r in pts],
                        hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}"
                                      "<br>阈值 %{customdata}<extra></extra>"))
                    fig.update_xaxes(title_text="Recall", range=[0, 1])
                    fig.update_yaxes(title_text="Precision", range=[0, 1.02])
                    show_chart(style_fig(fig, height=330))
                elif (FIGURES_DIR / f"{pick}_BoxPR_curve.png").exists():
                    show_image(str(FIGURES_DIR / f"{pick}_BoxPR_curve.png"))
                else:
                    st.info("未找到 PR 曲线数据")
            with b:
                st.markdown("**F1–Confidence 曲线**")
                if sweep_rows:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[r["threshold"] for r in sweep_rows],
                        y=[r["box"]["f1"] for r in sweep_rows],
                        mode="lines+markers", name="框级 F1",
                        line=dict(color=C_VIOLET, width=2.4),
                        marker=dict(size=5, color=C_VIOLET),
                        hovertemplate="F1 %{y:.3f} @ %{x}<extra></extra>"))
                    fig.add_trace(go.Scatter(
                        x=[r["threshold"] for r in sweep_rows],
                        y=[r["image"]["f1"] for r in sweep_rows],
                        mode="lines", name="图级 F1",
                        line=dict(color=C_CYAN, width=1.8, dash="dot"),
                        hovertemplate="图级 F1 %{y:.3f} @ %{x}<extra></extra>"))
                    best = max(sweep_rows, key=lambda r: r["box"]["f1"])
                    fig.add_vline(x=best["threshold"], line_dash="dot", line_width=1.2,
                                  line_color=C_YELLOW)
                    fig.add_annotation(x=best["threshold"], y=1.02,
                                       text=f'框级 F1 峰值 {best["threshold"]}',
                                       showarrow=False,
                                       font=dict(color=C_YELLOW, size=10))
                    fig.update_xaxes(title_text="置信度阈值")
                    fig.update_yaxes(title_text="F1", range=[0, 1.08])
                    show_chart(style_fig(fig, height=330, show_legend=True))
                elif (FIGURES_DIR / f"{pick}_BoxF1_curve.png").exists():
                    show_image(str(FIGURES_DIR / f"{pick}_BoxF1_curve.png"))
                else:
                    st.info("未找到 F1 曲线数据")

            a, b = st.columns([1, 1])
            with a:
                cmx = FIGURES_DIR / f"fig_confusion_image_level_{pick}.png"
                if not cmx.exists():
                    cmx = FIGURES_DIR / "fig_confusion_image_level.png"
                if cmx.exists():
                    st.markdown("**图级混淆矩阵**")
                    show_image(str(cmx))

            per_image = RESULTS_DIR / f"per_image_{pick}.csv"
            if per_image.exists():
                df = read_csv(str(per_image), per_image.stat().st_mtime)
                with b:
                    st.markdown("**逐图判定分布**")
                    outcomes = ["TP", "FP", "FN", "TN"]
                    zh = {"TP": "正确检出", "FP": "误检", "FN": "漏检", "TN": "正确排除"}
                    counts = df["outcome"].value_counts().reindex(outcomes).fillna(0).astype(int)
                    fig = go.Figure(go.Bar(
                        x=[zh[o] for o in outcomes], y=[counts[o] for o in outcomes],
                        width=0.5, marker_line_width=0,
                        marker_color=[C_GREEN, C_YELLOW, C_RED, C_BLUE],
                        text=[int(counts[o]) for o in outcomes], textposition="outside",
                        textfont=dict(color=INK_MUTED, size=12),
                        hovertemplate="%{x}<br>%{y} 张<extra></extra>"))
                    show_chart(style_fig(fig, height=330, y_title="images"))
            st.caption("PR / F1 曲线由测试集逐点阈值扫描计算；mAP 由 COCO 协议在全部"
                       "预测上积分，二者口径不同，不应直接比较。")

        with sec2:
            sw_path = RESULTS_DIR / f"threshold_experiment_{pick}.json"
            if not sw_path.exists():
                sw_path = RESULTS_DIR / "threshold_experiment_test.json"
            if sw_path.exists():
                sw = read_json(str(sw_path))
                rows = sw.get("sweep", [])
                import plotly.graph_objects as go
                fig = go.Figure()
                for key, name, color in (
                        ("precision", "框级 Precision", C_BLUE),
                        ("recall", "框级 Recall", C_ORANGE),
                        ("f1", "框级 F1", C_VIOLET)):
                    fig.add_trace(go.Scatter(
                        x=[r["threshold"] for r in rows],
                        y=[r["box"][key] for r in rows],
                        mode="lines+markers", name=name,
                        line=dict(color=color, width=2),
                        marker=dict(size=6, color=color,
                                    line=dict(width=1.5, color="#101a2e")),
                        hovertemplate=f"{name} %{{y:.3f}} @ %{{x}}<extra></extra>"))
                for key, name, color in (
                        ("sensitivity", "图级敏感度", C_CYAN),
                        ("specificity", "图级特异度", C_GREEN)):
                    fig.add_trace(go.Scatter(
                        x=[r["threshold"] for r in rows],
                        y=[r["image"][key] for r in rows],
                        mode="lines", name=name, line=dict(color=color, width=1.6, dash="dot"),
                        hovertemplate=f"{name} %{{y:.3f}} @ %{{x}}<extra></extra>"))
                fig.add_vline(x=0.15, line_dash="dot", line_width=1.2, line_color=INK_MUTED)
                fig.update_xaxes(title_text="置信度阈值")
                fig.update_yaxes(range=[0, 1.05], title_text="得分")
                show_chart(style_fig(fig, height=380, show_legend=True))

                st.markdown("**关键阈值报告点**")
                rp = sw.get("report_points", [])
                tbl = []
                for r in rp:
                    tbl.append({
                        "阈值": r["threshold"],
                        "Precision": r["box"]["precision"],
                        "Recall": r["box"]["recall"],
                        "F1": r["box"]["f1"],
                        "敏感度": r["image"]["sensitivity"],
                        "特异度": r["image"]["specificity"],
                    })
                show_table(pd.DataFrame(tbl))
                st.caption("mAP 由 PR 曲线积分得到，与固定阈值下的 P/R 不可混为一谈；"
                           f"该实验在 {sw.get('split')} 集上完成，n={sw.get('n_images')}。")
            else:
                st.info("未找到阈值实验记录，运行 scripts/threshold_experiment.py")

        with sec3:
            comp_path = RESULTS_DIR / "model_comparison.json"
            if comp_path.exists():
                comp = read_json(str(comp_path))
                rows = comp.get("models", [])
                if rows:
                    st.markdown("**YOLO11n vs YOLO11s**")
                    curated = []
                    for r in rows:
                        curated.append({
                            "模型": r.get("模型"),
                            "参数量(M)": r.get("参数量(M)"),
                            "mAP@0.5": r.get("mAP@0.5"),
                            "mAP@0.5:0.95": r.get("mAP@0.5:0.95"),
                            "Precision": r.get("Precision"),
                            "Recall": r.get("Recall"),
                            "F1": r.get("F1"),
                            "图级敏感度": r.get("图级敏感度"),
                            "图级特异度": r.get("图级特异度"),
                            "p50 (ms)": r.get("p50_ms"),
                        })
                    show_table(pd.DataFrame(curated))
                    import plotly.graph_objects as go
                    a, b = st.columns(2)
                    with a:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[r["模型"] for r in rows],
                            y=[r.get("mAP@0.5", 0) for r in rows],
                            width=0.45, marker_color=C_BLUE, marker_line_width=0,
                            text=[f'{r.get("mAP@0.5", 0):.3f}' for r in rows],
                            textposition="outside", textfont=dict(color=INK_MUTED)))
                        show_chart(style_fig(fig, height=280, y_title="mAP@0.5"))
                    with b:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[r["模型"] for r in rows],
                            y=[r.get("p50_ms", 0) for r in rows],
                            width=0.45, marker_color=C_CYAN, marker_line_width=0,
                            text=[f'{r.get("p50_ms", 0):.1f} ms' for r in rows],
                            textposition="outside", textfont=dict(color=INK_MUTED)))
                        show_chart(style_fig(fig, height=280, y_title="p50 latency (ms)"))
                    st.caption(comp.get("note", "") +
                               " 延迟为同条件基准（200 张、预热、CUDA 同步）。")
            else:
                st.info("模型对比结果尚未生成（results/model_comparison.json）。")

            enh_path = RESULTS_DIR / f"enhancement_{payload.get('split', 'test')}.json"
            if enh_path.exists():
                enh = read_json(str(enh_path))
                conds = enh.get("conditions", {})
                zh = {"original": "原图", "clahe": "CLAHE", "clahe_sharpen": "CLAHE+锐化"}
                keys = list(conds)
                if keys:
                    st.markdown("**图像增强对比（同一模型）**")
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for key, name, color in (("precision", "Precision", C_BLUE),
                                             ("recall", "Recall", C_ORANGE),
                                             ("f1", "F1", C_VIOLET)):
                        fig.add_trace(go.Bar(
                            x=[zh.get(k, k) for k in keys],
                            y=[conds[k]["box"][key] for k in keys],
                            name=name, marker_color=color, marker_line_width=0,
                            hovertemplate=f"{name} %{{y:.3f}}<extra></extra>"))
                    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
                    show_chart(style_fig(fig, height=320, show_legend=True, y_title="score"))
                    st.caption(f"YOLO11s · conf={enh.get('conf')}，n={enh.get('n_images')}。"
                               "增强只在推理前施加，不改变标注。")

        with sec4:
            per_image = RESULTS_DIR / f"per_image_{pick}.csv"
            if not per_image.exists():
                st.info("未找到逐图记录。")
            else:
                df = read_csv(str(per_image), per_image.stat().st_mtime)
                kind = st.radio("类型", ["FN 漏检", "FP 误检"], horizontal=True)
                target = "FN" if kind.startswith("FN") else "FP"
                subset = df[df["outcome"] == target]
                if subset.empty:
                    st.success(f"该测试集上没有 {target} 案例。")
                else:
                    n = st.slider("案例序号", 1, len(subset), 1)
                    row = subset.iloc[n - 1]
                    img_path = (REPO / "data" / "fracatlas_yolo"
                                / payload.get("split", "test") / "images" / row["image"])
                    if img_path.exists():
                        im = load_image(img_path)
                        res = run_inference(det, im, payload.get("conf", DEFAULT_CONF),
                                            payload.get("nms_iou", DEFAULT_IOU),
                                            payload.get("imgsz", IMG_SIZE), device)
                        a, b = st.columns([1.3, 1])
                        with a:
                            show_image(to_pil(draw_detections(im, res.detections)),
                                       caption=row["image"])
                        with b:
                            st.markdown(
                                f'<div class="ov-card">'
                                f'<div style="font-size:12px;color:#9aa9c2;">案例类型</div>'
                                f'<div style="font-size:18px;font-weight:800;">{target}</div>'
                                f'<div style="margin-top:10px;font-size:12px;color:#9aa9c2;">'
                                f'GT 框数</div><div style="font-weight:700;">{int(row["n_gt"])}</div>'
                                f'<div style="margin-top:10px;font-size:12px;color:#9aa9c2;">'
                                f'预测框数</div><div style="font-weight:700;">{int(row["n_det"])}</div>'
                                f'<div style="margin-top:10px;font-size:12px;color:#9aa9c2;">'
                                f'最高置信度</div><div style="font-weight:700;">'
                                f'{row["max_conf"]:.2f}</div></div>',
                                unsafe_allow_html=True)
                            st.caption("判定标签取自离线测试记录；本图重新推理用于可视化。")

            note = st.text_input("分析备注（写入导出报告）", "")
            if st.button("生成实验报告"):
                st.session_state["last_report"] = build_report(payload, note)
            if st.session_state.get("last_report"):
                st.download_button("下载 Markdown 报告",
                                   st.session_state["last_report"],
                                   file_name=f"orthovision_report_{pick}.md")

        with sec5:
            pert_path = RESULTS_DIR / "perturbation.json"
            if pert_path.exists():
                pert = read_json(str(pert_path))
                conds = pert.get("conditions", {})
                if conds:
                    import plotly.graph_objects as go
                    zh_names = {
                        "original": "原始", "blur_mild": "轻度模糊",
                        "blur_moderate": "中度模糊", "brightness_-30": "亮度−30",
                        "brightness_+30": "亮度+30", "contrast_0.7": "对比度×0.7",
                        "noise_sigma12": "噪声σ12",
                    }
                    keys = list(conds)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[zh_names.get(k, k) for k in keys],
                        y=[conds[k]["box"]["f1"] for k in keys],
                        name="框级 F1", marker_color=C_BLUE, marker_line_width=0, width=0.36,
                        hovertemplate="%{x}<br>F1 %{y:.3f}<extra></extra>"))
                    fig.add_trace(go.Bar(
                        x=[zh_names.get(k, k) for k in keys],
                        y=[conds[k]["image_level"]["sensitivity"] for k in keys],
                        name="图级敏感度", marker_color=C_ORANGE, marker_line_width=0,
                        width=0.36,
                        hovertemplate="%{x}<br>敏感度 %{y:.3f}<extra></extra>"))
                    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
                    show_chart(style_fig(fig, height=340, show_legend=True, y_title="score"))
                    st.caption("扰动只施加在测试图像上，标注框保持不变；结果反映对合成变换的"
                               "敏感性，不等同于真实成像条件差异。")
            else:
                st.info("未找到扰动实验记录。")


# ==========================================================================
# 5. about
# ==========================================================================
with tab_about:
    section("数据与资源", "可追溯、可复现")
    a, b = st.columns(2)
    with a:
        st.markdown(
            '<div class="ov-card">'
            '<div class="ov-card-title">数据集 · FracAtlas</div>'
            '<div style="color:#9aa9c2;font-size:12.5px;line-height:1.8;">'
            'Source: Abedeen et al., <i>Data in Brief</i>, 2023<br>'
            '规模：4,083 张 X 光片，717 张含骨折边界框<br>'
            '标注：单类别 <code>fractured</code>，922 个框<br>'
            '许可：CC BY 4.0 · figshare 22363012<br>'
            '划分：70 / 15 / 15 分层随机，seed = 42</div></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ov-card">'
            '<div class="ov-card-title">模型 · YOLO11-S</div>'
            '<div style="color:#9aa9c2;font-size:12.5px;line-height:1.8;">'
            'Ultralytics YOLO11-S（COCO 预训练）<br>'
            '输入 640×640 · 单类别 · NMS IoU 0.45<br>'
            '训练：80 epochs，RTX 4060 Laptop<br>'
            '代码：AGPL-3.0 · 本项目代码：MIT</div></div>',
            unsafe_allow_html=True)
    with b:
        st.markdown(
            '<div class="ov-card">'
            '<div class="ov-card-title">系统能力边界</div>'
            '<div style="color:#9aa9c2;font-size:12.5px;line-height:1.9;">'
            '✓ 输出疑似骨折区域、置信度与推理耗时<br>'
            '✓ 展示检测头特征响应热力图<br>'
            '✓ 在固定测试集上报告可复现指标<br>'
            '✗ 不能给出“骨折 / 正常”的临床诊断<br>'
            '✗ 热力图不是病灶分割结果<br>'
            '✗ 指标不能外推到真实临床人群</div></div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="ov-card">'
            '<div class="ov-card-title">许可与合规</div>'
            '<div style="color:#9aa9c2;font-size:12.5px;line-height:1.8;">'
            '依赖 Ultralytics（AGPL-3.0），训练得到的权重按 AGPL-3.0 发布；'
            '数据集为 CC BY 4.0，使用时保留原作者署名。'
            '本项目为课程研究原型，<b>不用于临床诊断</b>。</div></div>',
            unsafe_allow_html=True)

st.markdown(
    '<div class="ov-foot" style="display:flex;justify-content:space-between;'
    'border-top:1px solid #213252;padding-top:12px;margin-top:26px;">'
    '<span>医学人工智能导论 · 课程结课项目 · OrthoVision v1.1</span>'
    '<span>Xini</span></div>',
    unsafe_allow_html=True)
