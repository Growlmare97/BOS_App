"""Publication-quality BOS visualisation.

All functions return a ``(fig, axes)`` tuple so callers can further customise
or save them.  Pass ``show=True`` to open the interactive Matplotlib window.

Functions
---------
plot_displacement_magnitude   — colour map of sqrt(dx²+dy²)
plot_displacement_components  — side-by-side dx and dy signed maps
plot_quiver                   — quiver arrows overlaid on reference image
plot_abel_field               — Abel-inverted density/gradient field
plot_side_by_side             — reference vs measurement comparison
save_figure                   — wrapper to save fig to PNG and/or PDF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

# Use Agg backend by default so the module is importable headlessly; interactive
# mode is enabled per-call via show= parameter.
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FigAxes = Tuple[plt.Figure, np.ndarray]  # (fig, array-of-axes)


# ---------------------------------------------------------------------------
# Displacement magnitude
# ---------------------------------------------------------------------------


def plot_displacement_magnitude(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    title: str = "Displacement Magnitude",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    pixel_size_mm: Optional[float] = None,
    dpi: int = 300,
    show: bool = False,
) -> FigAxes:
    """Plot displacement magnitude map.

    Parameters
    ----------
    dx, dy:
        Displacement components (H × W).
    pixel_size_mm:
        If provided, draw a scale bar (requires ``matplotlib-scalebar`` or
        axes tick labels in mm).
    """
    mag = np.sqrt(dx ** 2 + dy ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=dpi)
    im = ax.imshow(
        mag,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    _add_colorbar(fig, ax, im, label="Displacement [px]")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    if pixel_size_mm is not None:
        _draw_scale_bar(ax, mag.shape[1], pixel_size_mm)

    fig.tight_layout()
    if show:
        _show_interactive(fig)
    return fig, np.array([ax])


# ---------------------------------------------------------------------------
# Displacement components (signed)
# ---------------------------------------------------------------------------


def plot_displacement_components(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    title: str = "Displacement Components",
    cmap: str = "seismic",
    symmetric: bool = True,
    dpi: int = 300,
    show: bool = False,
) -> FigAxes:
    """Side-by-side dx (horizontal) and dy (vertical) signed displacement maps."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=dpi)

    for ax, data, label in zip(axes, [dx, dy], ["dx [px]", "dy [px]"]):
        lim = np.abs(data).max() if symmetric else None
        vmin_val = -lim if lim else None
        vmax_val = lim if lim else None
        im = ax.imshow(
            data,
            origin="upper",
            cmap=cmap,
            vmin=vmin_val,
            vmax=vmax_val,
            interpolation="nearest",
        )
        _add_colorbar(fig, ax, im, label=label)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if show:
        _show_interactive(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# Quiver plot
# ---------------------------------------------------------------------------


def plot_quiver(
    reference: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    downsample: int = 8,
    title: str = "Displacement Field (Quiver)",
    cmap_bg: str = "gray",
    arrow_color: str = "red",
    scale: Optional[float] = None,
    dpi: int = 300,
    show: bool = False,
) -> FigAxes:
    """Quiver arrows overlaid on the reference image.

    Parameters
    ----------
    downsample:
        Show one arrow every *downsample* pixels in each direction.
    scale:
        Matplotlib quiver scale parameter.  ``None`` = auto.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    ax.imshow(reference, origin="upper", cmap=cmap_bg, aspect="equal")

    H, W = dx.shape
    ys = np.arange(0, H, downsample)
    xs = np.arange(0, W, downsample)
    xx, yy = np.meshgrid(xs, ys)

    u = dx[::downsample, ::downsample]
    v = -dy[::downsample, ::downsample]  # flip y for image coords

    q = ax.quiver(
        xx, yy, u, v,
        color=arrow_color,
        scale=scale,
        scale_units="xy" if scale is not None else None,
        angles="xy",
        width=0.002,
    )
    ax.quiverkey(q, X=0.85, Y=1.02, U=1.0, label="1 px", labelpos="E")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    fig.tight_layout()
    if show:
        _show_interactive(fig)
    return fig, np.array([ax])


# ---------------------------------------------------------------------------
# Abel field
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
    dpi: int = 300,
    show: bool = False,
) -> FigAxes:
    """Plot the Abel-inverted radial density gradient field."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=dpi)
    im = ax.imshow(
        inv_field,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    _add_colorbar(fig, ax, im, label=unit_label)
    if axis_col is not None:
        ax.axvline(axis_col, color="white", linestyle="--", linewidth=0.8,
                   label=f"Axis (col {axis_col})")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    fig.tight_layout()
    if show:
        _show_interactive(fig)
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
    dpi: int = 300,
    show: bool = False,
) -> FigAxes:
    """Reference and measurement frames side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    for ax, img, ttl in zip(axes, [reference, measurement], titles):
        vmin, vmax = img.min(), img.max()
        ax.imshow(img, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
    fig.tight_layout()
    if show:
        _show_interactive(fig)
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
    dpi: int = 300,
    show: bool = False,
) -> FigAxes:
    """4-panel (or 3-panel) summary figure."""
    n_panels = 4 if inv_field is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), dpi=dpi)

    mag = np.sqrt(dx ** 2 + dy ** 2)

    # Panel 0: reference
    axes[0].imshow(reference, cmap="gray", origin="upper")
    axes[0].set_title("Reference")

    # Panel 1: measurement
    axes[1].imshow(measurement, cmap="gray", origin="upper")
    axes[1].set_title("Measurement")

    # Panel 2: displacement magnitude
    im2 = axes[2].imshow(mag, cmap="viridis", origin="upper")
    _add_colorbar(fig, axes[2], im2, label="[px]")
    axes[2].set_title("Displacement Magnitude")

    # Panel 3: Abel field (optional)
    if inv_field is not None:
        im3 = axes[3].imshow(inv_field, cmap="inferno", origin="upper")
        _add_colorbar(fig, axes[3], im3, label="[a.u.]")
        axes[3].set_title("Abel-Inverted Field")

    for ax in axes:
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")

    fig.suptitle("BOS Processing Summary", fontsize=13)
    fig.tight_layout()
    if show:
        _show_interactive(fig)
    return fig, axes


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    output_dir: Union[str, Path],
    stem: str,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> List[Path]:
    """Save *fig* to *output_dir* with *stem* in each requested *format*.

    Returns a list of the saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for fmt in formats:
        fpath = out / f"{stem}.{fmt}"
        fig.savefig(str(fpath), dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure: %s", fpath)
        saved.append(fpath)
    return saved


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_colorbar(fig, ax, im, label: str = "") -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _draw_scale_bar(ax, width_px: int, pixel_size_mm: float) -> None:
    """Draw a simple scale bar in the lower-right corner."""
    bar_mm = _nice_bar_length(width_px * pixel_size_mm * 0.15)
    bar_px = bar_mm / pixel_size_mm
    x0 = width_px * 0.80
    y0 = ax.get_ylim()[0] * 0.95 if ax.get_ylim()[0] > 0 else 20
    ax.plot([x0, x0 + bar_px], [y0, y0], "w-", linewidth=3)
    ax.text(
        x0 + bar_px / 2, y0 - 5, f"{bar_mm:.0f} mm",
        color="white", ha="center", va="bottom", fontsize=8,
    )


def _nice_bar_length(target: float) -> float:
    """Round *target* to a 'nice' number (1, 2, 5, 10, …)."""
    import math
    mag = 10 ** math.floor(math.log10(target))
    for step in [1, 2, 5, 10]:
        if step * mag >= target:
            return step * mag
    return mag * 10


def _show_interactive(fig: plt.Figure) -> None:
    """Temporarily switch to an interactive backend and display *fig*."""
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    plt.figure(fig.number)
    plt.show(block=True)
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
