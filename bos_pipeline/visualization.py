"""Publication-quality BOS visualisation — backend-agnostic.

All functions return ``(fig, axes)`` where *fig* is a
``matplotlib.figure.Figure`` created without touching the pyplot state
machine.  This makes the module safe to import from any backend context
(Agg for the CLI, QtAgg for the GUI app, etc.).

Functions
---------
plot_displacement_magnitude   — turbo colour map, percentile clipping
plot_displacement_components  — signed dx/dy with RdBu_r diverging map
plot_quiver                   — colour-coded arrows overlaid on reference
plot_abel_field               — Abel-inverted density/gradient field
plot_side_by_side             — reference vs measurement comparison
plot_summary                  — 3- or 4-panel overview
save_figure                   — save a Figure to PNG / PDF
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

# Type alias
FigAxes = Tuple[Figure, np.ndarray]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fig(figsize: Tuple[float, float], dpi: int) -> Figure:
    """Create a backend-agnostic Figure (no pyplot involved)."""
    return Figure(figsize=figsize, dpi=dpi)


def _add_colorbar(fig: Figure, ax, im, label: str = "") -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _style_ax(ax, title: str = "", xlabel: str = "x [px]", ylabel: str = "y [px]") -> None:
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _overlay_stats(ax, data: np.ndarray, unit: str = "px") -> None:
    """White-on-black stats box in the top-left corner."""
    txt = f"max  {data.max():.3f} {unit}\nmean {data.mean():.3f} {unit}"
    ax.text(
        0.01, 0.99, txt, transform=ax.transAxes,
        va="top", ha="left", fontsize=6.5, color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55),
    )


def _pclim(data: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Tuple[float, float]:
    """Return (vmin, vmax) from percentiles — robust to outliers."""
    return float(np.percentile(data, lo)), float(np.percentile(data, hi))


def _draw_scale_bar(ax, width_px: int, pixel_size_mm: float) -> None:
    target = width_px * pixel_size_mm * 0.15
    mag = 10 ** math.floor(math.log10(max(target, 1e-9)))
    bar_mm = next((s * mag for s in [1, 2, 5, 10] if s * mag >= target), mag * 10)
    bar_px = bar_mm / pixel_size_mm
    x0 = width_px * 0.80
    y0 = 20
    ax.plot([x0, x0 + bar_px], [y0, y0], "w-", linewidth=3)
    ax.text(x0 + bar_px / 2, y0 - 5, f"{bar_mm:.0f} mm",
            color="white", ha="center", va="bottom", fontsize=8)


# ---------------------------------------------------------------------------
# Displacement magnitude
# ---------------------------------------------------------------------------

def plot_displacement_magnitude(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    title: str = "Displacement Magnitude",
    cmap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    pixel_size_mm: Optional[float] = None,
    dpi: int = 150,
    show: bool = False,
) -> FigAxes:
    """Displacement magnitude ||(dx, dy)|| with turbo colour map and percentile clipping."""
    mag = np.hypot(dx, dy)
    if vmax is None:
        vmax = float(np.percentile(mag, 99)) or 1.0
    if vmin is None:
        vmin = 0.0

    fig = _fig((7, 5), dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(mag, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    _add_colorbar(fig, ax, im, label="Displacement [px]")
    _style_ax(ax, title)
    _overlay_stats(ax, mag)
    if pixel_size_mm is not None:
        _draw_scale_bar(ax, mag.shape[1], pixel_size_mm)
    fig.tight_layout()
    return fig, np.array([ax])


# ---------------------------------------------------------------------------
# Displacement components (signed dx / dy)
# ---------------------------------------------------------------------------

def plot_displacement_components(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    title: str = "Displacement Components",
    cmap: str = "RdBu_r",
    dpi: int = 150,
    show: bool = False,
) -> FigAxes:
    """Side-by-side signed dx and dy maps with a symmetric diverging colourmap."""
    fig = _fig((12, 4.5), dpi)
    axes = fig.subplots(1, 2)

    for ax, data, label in zip(axes, [dx, dy], ["dx [px]", "dy [px]"]):
        lim = float(np.percentile(np.abs(data), 99)) or 0.01
        im = ax.imshow(data, origin="upper", cmap=cmap,
                       vmin=-lim, vmax=lim, interpolation="bilinear")
        _add_colorbar(fig, ax, im, label=label)
        _style_ax(ax, label)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Quiver — colour-coded by magnitude
# ---------------------------------------------------------------------------

def plot_quiver(
    reference: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    downsample: int = 16,
    title: str = "Displacement Field",
    scale: Optional[float] = None,
    dpi: int = 150,
    show: bool = False,
) -> FigAxes:
    """Vector field overlaid on the reference frame.

    Arrows are colour-coded by their magnitude (plasma colourmap).
    ``scale`` is auto-computed so the 95th-percentile arrow spans ~70 % of
    the grid spacing — set it explicitly to override.
    """
    H, W = dx.shape
    ds = max(1, downsample)
    ys = np.arange(0, H, ds)
    xs = np.arange(0, W, ds)
    xx, yy = np.meshgrid(xs, ys)
    u = dx[::ds, ::ds]
    v = -dy[::ds, ::ds]          # flip y for image-space coords
    mag_q = np.hypot(u, v)

    # Adaptive scale: 95th-percentile arrow = 70 % of grid spacing
    d95 = float(np.percentile(mag_q, 95)) or 1.0
    s = scale if scale is not None else (d95 / (0.70 * ds))
    vmax_q = float(np.percentile(mag_q, 99)) or 1.0

    import matplotlib.colors as mcolors

    fig = _fig((8, 6), dpi)
    ax = fig.add_subplot(111)
    ax.imshow(reference, origin="upper", cmap="gray",
              interpolation="bilinear", alpha=0.70)

    q = ax.quiver(
        xx, yy, u, v, mag_q,
        cmap="plasma",
        norm=mcolors.Normalize(vmin=0, vmax=vmax_q),
        angles="xy",
        scale_units="xy",
        scale=s,
        width=0.0018,
        headwidth=4,
        headlength=4,
        minlength=0.3,
        alpha=0.92,
    )
    cb = fig.colorbar(q, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Displacement [px]", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    key_u = max(1.0, round(d95))
    ax.quiverkey(q, X=0.87, Y=1.025, U=key_u,
                 label=f"{key_u:.0f} px", labelpos="E",
                 fontproperties={"size": 7})
    _style_ax(ax, title)
    fig.tight_layout()
    return fig, np.array([ax])


# ---------------------------------------------------------------------------
# Abel-inverted field
# ---------------------------------------------------------------------------

def plot_abel_field(
    inv_field: np.ndarray,
    *,
    title: str = "Abel-Inverted Field",
    cmap: str = "inferno",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    axis_col: Optional[int] = None,
    unit_label: str = "Radial gradient [a.u.]",
    dpi: int = 150,
    show: bool = False,
) -> FigAxes:
    """Abel-inverted radial density gradient field."""
    if vmax is None:
        _, vmax = _pclim(inv_field, 1, 99)

    fig = _fig((7, 5), dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(inv_field, origin="upper", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear")
    _add_colorbar(fig, ax, im, label=unit_label)
    if axis_col is not None:
        ax.axvline(axis_col, color="white", linestyle="--", linewidth=0.8,
                   label=f"Axis (col {axis_col})")
        ax.legend(fontsize=7, loc="upper right")
    _style_ax(ax, title)
    fig.tight_layout()
    return fig, np.array([ax])


# ---------------------------------------------------------------------------
# Side-by-side reference vs measurement
# ---------------------------------------------------------------------------

def plot_side_by_side(
    reference: np.ndarray,
    measurement: np.ndarray,
    *,
    titles: Tuple[str, str] = ("Reference", "Measurement"),
    cmap: str = "gray",
    dpi: int = 150,
    show: bool = False,
) -> FigAxes:
    fig = _fig((11, 4.5), dpi)
    axes = fig.subplots(1, 2)
    for ax, img, ttl in zip(axes, [reference, measurement], titles):
        ax.imshow(img, origin="upper", cmap=cmap,
                  vmin=img.min(), vmax=img.max(), interpolation="bilinear")
        _style_ax(ax, ttl)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------

def plot_summary(
    reference: np.ndarray,
    measurement: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    inv_field: Optional[np.ndarray] = None,
    *,
    dpi: int = 150,
    show: bool = False,
) -> FigAxes:
    n = 4 if inv_field is not None else 3
    fig = _fig((5 * n, 4), dpi)
    axes = fig.subplots(1, n)
    mag = np.hypot(dx, dy)

    axes[0].imshow(reference, cmap="gray", origin="upper", interpolation="bilinear")
    axes[0].set_title("Reference", fontsize=9)

    axes[1].imshow(measurement, cmap="gray", origin="upper", interpolation="bilinear")
    axes[1].set_title("Measurement", fontsize=9)

    vmax_m = float(np.percentile(mag, 99)) or 1.0
    im2 = axes[2].imshow(mag, cmap="turbo", origin="upper",
                          vmin=0, vmax=vmax_m, interpolation="bilinear")
    _add_colorbar(fig, axes[2], im2, label="[px]")
    axes[2].set_title("Magnitude", fontsize=9)

    if inv_field is not None:
        _, vmax_a = _pclim(inv_field, 1, 99)
        im3 = axes[3].imshow(inv_field, cmap="inferno", origin="upper",
                              vmax=vmax_a, interpolation="bilinear")
        _add_colorbar(fig, axes[3], im3, label="[a.u.]")
        axes[3].set_title("Abel Field", fontsize=9)

    for ax in axes:
        ax.set_xlabel("x [px]", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle("BOS Processing Summary", fontsize=12)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------

def save_figure(
    fig: Figure,
    output_dir: Union[str, Path],
    stem: str,
    formats: Sequence[str] = ("png",),
    dpi: int = 200,
) -> List[Path]:
    """Save *fig* to *output_dir/<stem>.<fmt>* for each format in *formats*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for fmt in formats:
        fpath = out / f"{stem}.{fmt}"
        fig.savefig(str(fpath), dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", fpath)
        saved.append(fpath)
    return saved
