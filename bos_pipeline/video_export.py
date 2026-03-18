"""Export a sequence of BOS results as an annotated MP4 video.

Uses the Agg matplotlib backend (no display required) to render each frame
into a NumPy buffer, then writes the buffer with OpenCV VideoWriter.

Layout adapts to available data:
  • Displacement only  →  2-panel  (reference | magnitude)
  • With concentration →  3-panel  (reference | magnitude | concentration)

Color scales are fixed globally across all frames so the animation is
directly comparable — the 99th-percentile displacement of the brightest
frame sets the magnitude scale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_results_video(
    results: List[dict],
    output_path: str | Path,
    fps: float = 10.0,
    dpi: int = 120,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Render *results* to an MP4 animation.

    Parameters
    ----------
    results:
        List of result dicts from the pipeline (keys: frame_idx, dx, dy,
        ref, meas, concentration [optional], axis_col [optional]).
    output_path:
        Destination .mp4 file.
    fps:
        Playback frame rate.
    dpi:
        Figure DPI — higher = larger / sharper video.
    progress_cb:
        Called as ``progress_cb(frame_index, total)`` after each frame is
        written.  Useful for progress bars.

    Returns
    -------
    Path
        Absolute path to the written video file.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for video export. Install with: pip install opencv-python"
        ) from exc

    if not results:
        raise ValueError("No results to export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_concentration = any(r.get("concentration") is not None for r in results)
    n_panels = 3 if has_concentration else 2

    # ------------------------------------------------------------------
    # Global color scale — consistent across all frames
    # ------------------------------------------------------------------
    mag_vmax = max(
        float(np.percentile(np.hypot(r["dx"], r["dy"]), 99))
        for r in results
    ) or 1.0

    # Probe figure size from the first rendered frame
    first_img = _render_frame(
        results[0], n_panels, mag_vmax, dpi, has_concentration
    )
    h_px, w_px = first_img.shape[:2]
    logger.info(
        "Video: %d frames  |  %dx%d px  |  %.1f fps  |  %s",
        len(results), w_px, h_px, fps, output_path.name,
    )

    # ------------------------------------------------------------------
    # OpenCV writer — try mp4v codec first, fall back to XVID
    # ------------------------------------------------------------------
    suffix = output_path.suffix.lower()
    if suffix in ("", ".mp4"):
        output_path = output_path.with_suffix(".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_px, h_px))
    if not writer.isOpened():
        # Fallback: write uncompressed AVI
        output_path = output_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_px, h_px))

    try:
        for i, r in enumerate(results):
            rgb = _render_frame(r, n_panels, mag_vmax, dpi, has_concentration)
            bgr = rgb[:, :, ::-1].copy()   # RGB → BGR for OpenCV
            writer.write(bgr)
            if progress_cb:
                progress_cb(i + 1, len(results))
    finally:
        writer.release()

    logger.info("Video saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal frame renderer
# ---------------------------------------------------------------------------

_PANEL_W = 5.0    # inches per panel
_FIG_H   = 4.0    # inches height


def _render_frame(
    result: dict,
    n_panels: int,
    mag_vmax: float,
    dpi: int,
    has_concentration: bool,
) -> np.ndarray:
    """Render a single result dict to an RGB numpy array."""
    dx   = result["dx"]
    dy   = result["dy"]
    ref  = result["ref"]
    fidx = result["frame_idx"]
    mag  = np.hypot(dx, dy)

    fig = Figure(figsize=(_PANEL_W * n_panels, _FIG_H), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1, n_panels)
    if n_panels == 1:
        axes = [axes]

    # ── Panel 0: reference frame ─────────────────────────────────────
    ax0 = axes[0]
    ax0.imshow(ref, cmap="gray", origin="upper", interpolation="bilinear",
               vmin=ref.min(), vmax=ref.max())
    ax0.set_title(f"Reference", fontsize=9, pad=3)
    _style(ax0)

    # ── Panel 1: displacement magnitude ──────────────────────────────
    ax1 = axes[1]
    im1 = ax1.imshow(mag, cmap="turbo", origin="upper",
                     vmin=0, vmax=mag_vmax, interpolation="bilinear")
    _cbar(fig, ax1, im1, "Displacement [px]")
    ax1.set_title(f"Magnitude  — frame {fidx}", fontsize=9, pad=3)

    # Stats overlay
    bg_corners = _bg_noise(mag)
    snr = 20.0 * np.log10(mag.mean() / max(bg_corners, 1e-9))
    ax1.text(
        0.01, 0.99,
        f"max  {mag.max():.2f} px\nmean {mag.mean():.2f} px\nSNR  {snr:.1f} dB",
        transform=ax1.transAxes, va="top", ha="left",
        fontsize=6, color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
    )
    _style(ax1)

    # ── Panel 2: concentration (if present) ──────────────────────────
    if has_concentration and n_panels == 3:
        ax2 = axes[2]
        conc = result.get("concentration")
        if conc is not None:
            im2 = ax2.imshow(conc, cmap="plasma", origin="upper",
                             vmin=0, vmax=1, interpolation="bilinear")
            _cbar(fig, ax2, im2, "c(gas)  [vol. frac.]")
            axis_col = result.get("axis_col")
            if axis_col is not None:
                ax2.axvline(axis_col, color="white", linestyle="--",
                            linewidth=0.7)
            nonzero = conc[conc > 1e-4]
            if nonzero.size:
                ax2.text(
                    0.01, 0.99,
                    f"max  {conc.max():.4f}\nmean {nonzero.mean():.4f}",
                    transform=ax2.transAxes, va="top", ha="left",
                    fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="black", alpha=0.55),
                )
        else:
            ax2.text(0.5, 0.5, "Concentration\nnot computed",
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=9, color="#aaaaaa")
            ax2.set_axis_off()
        ax2.set_title("Concentration", fontsize=9, pad=3)
        _style(ax2)

    fig.tight_layout(pad=0.6)
    canvas.draw()

    # buffer_rgba() returns an RGBA buffer; we only keep RGB
    rgba = np.asarray(canvas.buffer_rgba())   # (H, W, 4) uint8
    return rgba[:, :, :3].copy()


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_xlabel("x [px]", fontsize=7)
    ax.set_ylabel("y [px]", fontsize=7)
    ax.tick_params(labelsize=6)


def _cbar(fig: Figure, ax, im, label: str) -> None:
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.06)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def _bg_noise(mag: np.ndarray, frac: float = 0.08) -> float:
    h, w = mag.shape
    ch = max(1, int(h * frac))
    cw = max(1, int(w * frac))
    corners = np.concatenate([
        mag[:ch, :cw].ravel(), mag[:ch, -cw:].ravel(),
        mag[-ch:, :cw].ravel(), mag[-ch:, -cw:].ravel(),
    ])
    return float(corners.std())
