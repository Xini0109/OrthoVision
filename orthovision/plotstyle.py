"""Shared visual language for every figure in the paper.

The course paper is set in 仿宋 (FangSong) inside figures, with a light
"fresh gradient" palette: mint, sky, lavender and peach on a near-white
background. Keeping the style in one module means the architecture diagram,
metric bars, curves and qualitative panels all read as one system.

Nothing here changes the data; these are display-only helpers.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# ---- palette --------------------------------------------------------------
MINT = "#63C9B4"
SKY = "#6FA8F5"
LAVENDER = "#A78BFA"
PEACH = "#FFA98A"
CORAL = "#FF8FA3"
INK = "#2B3A55"
MUTED = "#6B7C93"
GRID = "#E4EDF7"
AXIS = "#D3E0EF"

C_MINT = (MINT, "#B7F0DD")
C_SKY = (SKY, "#BBD8FF")
C_LAVENDER = (LAVENDER, "#DCD2FF")
C_PEACH = (PEACH, "#FFD7C4")
C_CORAL = (CORAL, "#FFD1DA")

FONT_CANDIDATES = ["FangSong", "STFangsong", "SimSun", "Microsoft YaHei"]


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_CANDIDATES + ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "none",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 11,
    })


def _cmap(c1: str, c2: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("ov", [c1, c2])


def gradient_background(ax, c1: str = "#F9FDFF", c2: str = "#EDF8F4",
                        alpha: float = 1.0) -> None:
    """A soft vertical wash behind the axes (display only)."""
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, 1, 0, 1], transform=ax.transAxes,
              origin="lower", aspect="auto", cmap=_cmap(c1, c2),
              zorder=-20, alpha=alpha)


def soft_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.9, alpha=0.95, zorder=0)
        ax.set_axisbelow(True)


def gradient_bar(ax, x: float, height: float, width: float, bottom: float = 0.0,
                 colors: tuple[str, str] = C_SKY, label: str | None = None,
                 text: str | None = None, radius: float = 0.04,
                 zorder: int = 3) -> None:
    """A bar filled with a vertical gradient and rounded top corners."""
    if height <= 0:
        return
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[x - width / 2, x + width / 2, bottom, bottom + height],
                   origin="lower", aspect="auto", cmap=_cmap(*colors),
                   zorder=zorder, interpolation="bicubic")
    patch = FancyBboxPatch(
        (x - width / 2, bottom), width, height,
        boxstyle=f"round,pad=0,rounding_size={min(radius, width / 2, height / 2)}",
        transform=ax.transData, facecolor="none", edgecolor="none",
    )
    ax.add_patch(patch)
    im.set_clip_path(patch)
    if text is not None:
        ax.text(x, bottom + height, text, ha="center", va="bottom",
                fontsize=12.5, color=INK, zorder=zorder + 2,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.82))


def gradient_barh(ax, y: float, width: float, height: float, left: float = 0.0,
                  colors: tuple[str, str] = C_SKY, text: str | None = None,
                  radius: float = 0.04, zorder: int = 3) -> None:
    """Horizontal gradient bar for the dataset-composition figure."""
    if width <= 0:
        return
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    im = ax.imshow(grad, extent=[left, left + width, y - height / 2, y + height / 2],
                   origin="lower", aspect="auto", cmap=_cmap(*colors),
                   zorder=zorder, interpolation="bicubic")
    patch = FancyBboxPatch(
        (left, y - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={min(radius, height / 2)}",
        transform=ax.transData, facecolor="none", edgecolor="none",
    )
    ax.add_patch(patch)
    im.set_clip_path(patch)
    if text is not None:
        ax.text(left + width, y, text, ha="left", va="center",
                fontsize=11.5, color=INK, zorder=zorder + 2,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))


def gradient_box(ax, x: float, y: float, w: float, h: float,
                 colors: tuple[str, str] = C_SKY, text: str = "",
                 sub: str = "", edge: str = "#C9DCEF", fontsize: float = 14,
                 subsize: float = 10, radius: float = 0.12) -> None:
    """Rounded card with a diagonal-ish gradient fill (architecture diagram)."""
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[x, x + w, y, y + h], origin="lower",
                   aspect="auto", cmap=_cmap(*colors), zorder=1,
                   interpolation="bicubic")
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        transform=ax.transData, facecolor="none", edgecolor=edge,
        linewidth=1.0, zorder=2,
    )
    ax.add_patch(patch)
    im.set_clip_path(patch)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + 0.10 * h, text, ha="center", va="center",
                fontsize=fontsize, color=INK, zorder=3)
        ax.text(cx, cy - 0.22 * h, sub, ha="center", va="center",
                fontsize=subsize, color=MUTED, zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                color=INK, zorder=3)


def gradient_matrix(ax, mat, c1: str = "#FFFFFF", c2: str = "#7FC8F8"):
    """Soft colormap for confusion matrices."""
    return ax.imshow(mat, cmap=_cmap(c1, c2), vmin=0, vmax=mat.max() * 1.15,
                     aspect="auto")


def add_panel_border(ax, color: str = "#DCE9F5", lw: float = 0.8):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(color)
        s.set_linewidth(lw)
