"""Displacement field computation for BOS.

Three methods are supported:

* ``cross_correlation`` (default) — windowed FFT cross-correlation with
  sub-pixel parabolic or Gaussian peak fitting.
* ``lucas_kanade`` — sparse optical flow via ``cv2.calcOpticalFlowPyrLK``.
* ``farneback`` — dense optical flow via ``cv2.calcOpticalFlowFarneback``.

All methods return (dx, dy) displacement maps in **pixels** as float32
arrays with the same spatial shape as the input images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Method = Literal["cross_correlation", "lucas_kanade", "farneback"]
SubpixelFit = Literal["parabolic", "gaussian"]


@dataclass
class DisplacementResult:
    """Return type for cross-correlation — supports tuple unpacking (dx, dy)
    and also carries the grid coordinate arrays for up-sampling.
    """
    dx: np.ndarray
    dy: np.ndarray
    grid_x: Optional[np.ndarray] = None   # col centres (n_rows × n_cols)
    grid_y: Optional[np.ndarray] = None   # row centres (n_rows × n_cols)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow:  dx, dy = compute_displacement(...)"""
        yield self.dx
        yield self.dy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DisplacementConfig:
    """Parameters for displacement field computation."""

    method: Method = "cross_correlation"
    window_size: int = 64          # interrogation window size [px]
    overlap: float = 0.5           # window overlap fraction  [0–0.9]
    subpixel: SubpixelFit = "parabolic"

    # Lucas-Kanade
    lk_win_size: int = 15
    lk_max_level: int = 3

    # Farneback
    fb_pyr_scale: float = 0.5
    fb_levels: int = 3
    fb_winsize: int = 15
    fb_iterations: int = 3
    fb_poly_n: int = 5
    fb_poly_sigma: float = 1.2

    @classmethod
    def from_dict(cls, d: dict) -> "DisplacementConfig":
        cfg = cls(
            method=d.get("method", "cross_correlation"),
            window_size=d.get("window_size", 64),
            overlap=d.get("overlap", 0.5),
            subpixel=d.get("subpixel", "parabolic"),
        )
        of = d.get("optical_flow", {})
        cfg.lk_win_size = of.get("lk_win_size", 15)
        cfg.lk_max_level = of.get("lk_max_level", 3)
        cfg.fb_pyr_scale = of.get("fb_pyr_scale", 0.5)
        cfg.fb_levels = of.get("fb_levels", 3)
        cfg.fb_winsize = of.get("fb_winsize", 15)
        cfg.fb_iterations = of.get("fb_iterations", 3)
        cfg.fb_poly_n = of.get("fb_poly_n", 5)
        cfg.fb_poly_sigma = of.get("fb_poly_sigma", 1.2)
        return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_displacement(
    reference: np.ndarray,
    measurement: np.ndarray,
    config: Optional[DisplacementConfig] = None,
) -> DisplacementResult:
    """Compute the displacement field between *reference* and *measurement*.

    Parameters
    ----------
    reference:
        2-D background frame (H × W), float32.
    measurement:
        2-D measurement frame (H × W), float32.
    config:
        :class:`DisplacementConfig` instance.

    Returns
    -------
    dx, dy:
        Horizontal and vertical displacement maps in pixels.  Both arrays
        have the same shape as the inputs (H × W, float32) when using
        optical-flow methods, or the *grid* shape for cross-correlation
        (values are then up-sampled / interpolated to H × W).

        .. note::
            Cross-correlation returns the *grid* maps.  Use
            :func:`interpolate_to_full_resolution` to up-sample them.
    """
    cfg = config or DisplacementConfig()

    if reference.shape != measurement.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs "
            f"measurement {measurement.shape}"
        )

    logger.info(
        "Computing displacement: method=%s, shape=%s",
        cfg.method, reference.shape,
    )

    if cfg.method == "cross_correlation":
        return _cross_correlation(reference, measurement, cfg)
    elif cfg.method == "lucas_kanade":
        dx, dy = _lucas_kanade(reference, measurement, cfg)
        return DisplacementResult(dx=dx, dy=dy)
    elif cfg.method == "farneback":
        dx, dy = _farneback(reference, measurement, cfg)
        return DisplacementResult(dx=dx, dy=dy)
    else:
        raise ValueError(f"Unknown displacement method: '{cfg.method}'")


def interpolate_to_full_resolution(
    dx_grid: np.ndarray,
    dy_grid: np.ndarray,
    target_shape: Tuple[int, int],
    grid_x: Optional[np.ndarray] = None,
    grid_y: Optional[np.ndarray] = None,
    result: Optional[DisplacementResult] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear up-sample cross-correlation displacement grids to full resolution.

    Parameters
    ----------
    dx_grid, dy_grid:
        Displacement grids from :func:`compute_displacement`.
    target_shape:
        (H, W) of the full-resolution image.
    grid_x, grid_y:
        Pixel coordinates of each grid node (outputs of
        ``_build_grid``).
    """
    from scipy.interpolate import RegularGridInterpolator

    # Allow passing a DisplacementResult directly
    if result is not None:
        grid_x = result.grid_x
        grid_y = result.grid_y

    if grid_x is None or grid_y is None:
        raise ValueError("grid_x and grid_y must be provided (or pass result=DisplacementResult).")

    H, W = target_shape
    yi = np.arange(H, dtype=np.float32)
    xi = np.arange(W, dtype=np.float32)

    gy = grid_y[:, 0]  # unique row centres
    gx = grid_x[0, :]  # unique col centres

    interp_dx = RegularGridInterpolator(
        (gy, gx), dx_grid, method="linear",
        bounds_error=False, fill_value=0.0,
    )
    interp_dy = RegularGridInterpolator(
        (gy, gx), dy_grid, method="linear",
        bounds_error=False, fill_value=0.0,
    )

    yy, xx = np.meshgrid(yi, xi, indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], axis=1)

    dx_full = interp_dx(pts).reshape(H, W).astype(np.float32)
    dy_full = interp_dy(pts).reshape(H, W).astype(np.float32)
    return dx_full, dy_full


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------


def _cross_correlation(
    ref: np.ndarray,
    meas: np.ndarray,
    cfg: DisplacementConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Windowed FFT cross-correlation with sub-pixel peak fitting."""
    H, W = ref.shape
    ws = cfg.window_size
    step = max(1, int(ws * (1.0 - cfg.overlap)))

    grid_rows, grid_cols = _build_grid(H, W, ws, step)
    n_rows, n_cols = len(grid_rows), len(grid_cols)

    dx_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    dy_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    half = ws // 2
    win_func = np.outer(np.hanning(ws), np.hanning(ws))  # 2-D Hanning window

    for ri, row in enumerate(grid_rows):
        for ci, col in enumerate(grid_cols):
            r0, r1 = max(0, row - half), min(H, row + half)
            c0, c1 = max(0, col - half), min(W, col + half)

            patch_ref = ref[r0:r1, c0:c1]
            patch_meas = meas[r0:r1, c0:c1]

            if patch_ref.shape != (ws, ws):
                # Edge: skip (leave zeros) or zero-pad
                ph, pw = patch_ref.shape
                pad_r = ((0, ws - ph), (0, ws - pw))
                patch_ref = np.pad(patch_ref, pad_r)
                patch_meas = np.pad(patch_meas, pad_r)

            # Normalise and window
            pr = _normalise_patch(patch_ref) * win_func
            pm = _normalise_patch(patch_meas) * win_func

            # FFT cross-correlation (normalised)
            # correlating (meas, ref) gives peak at (+dx, +dy) = displacement
            corr = _fft_xcorr(pm, pr)

            # Find peak
            peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)

            # Sub-pixel refinement
            dy_sub, dx_sub = _subpixel(corr, peak_y, peak_x, cfg.subpixel)

            # Convert correlation map indices to displacement (centre = 0)
            cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
            dy_grid[ri, ci] = (peak_y - cy) + dy_sub
            dx_grid[ri, ci] = (peak_x - cx) + dx_sub

    # Build full-resolution coordinate meshes (for optional up-sampling)
    gx_arr, gy_arr = np.meshgrid(grid_cols, grid_rows)

    logger.debug(
        "Cross-correlation complete: grid shape (%d × %d), "
        "max |dx|=%.2f px, max |dy|=%.2f px.",
        n_rows, n_cols, np.abs(dx_grid).max(), np.abs(dy_grid).max(),
    )
    return DisplacementResult(dx=dx_grid, dy=dy_grid, grid_x=gx_arr, grid_y=gy_arr)


def _build_grid(H: int, W: int, ws: int, step: int):
    """Return lists of window centre coordinates."""
    half = ws // 2
    rows = list(range(half, H - half + 1, step))
    cols = list(range(half, W - half + 1, step))
    if not rows:
        rows = [H // 2]
    if not cols:
        cols = [W // 2]
    return rows, cols


def _normalise_patch(patch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation."""
    p = patch - patch.mean()
    std = patch.std()
    if std < 1e-12:
        return p
    return p / std


def _fft_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised FFT cross-correlation; output is shifted so dc is at centre."""
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    corr = np.fft.ifft2(fa * np.conj(fb)).real
    return np.fft.fftshift(corr)


def _subpixel(
    corr: np.ndarray,
    py: int,
    px: int,
    method: SubpixelFit,
) -> Tuple[float, float]:
    """Return sub-pixel offset (dy, dx) from a 3-point fit around the peak."""
    h, w = corr.shape

    def _fit(c_minus, c_zero, c_plus) -> float:
        if method == "gaussian":
            # Gaussian: ln values
            try:
                lm = np.log(max(c_minus, 1e-12))
                l0 = np.log(max(c_zero, 1e-12))
                lp = np.log(max(c_plus, 1e-12))
                denom = 2 * (lm - 2 * l0 + lp)
                if abs(denom) < 1e-12:
                    return 0.0
                return (lm - lp) / denom
            except Exception:
                return 0.0
        else:  # parabolic
            denom = 2 * (c_minus - 2 * c_zero + c_plus)
            if abs(denom) < 1e-12:
                return 0.0
            return (c_minus - c_plus) / denom

    # Y direction
    if 0 < py < h - 1:
        dy_sub = _fit(corr[py - 1, px], corr[py, px], corr[py + 1, px])
    else:
        dy_sub = 0.0

    # X direction
    if 0 < px < w - 1:
        dx_sub = _fit(corr[py, px - 1], corr[py, px], corr[py, px + 1])
    else:
        dx_sub = 0.0

    return float(dy_sub), float(dx_sub)


# ---------------------------------------------------------------------------
# Optical flow — Lucas-Kanade (sparse → dense via interpolation)
# ---------------------------------------------------------------------------


def _lucas_kanade(
    ref: np.ndarray,
    meas: np.ndarray,
    cfg: DisplacementConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense field via sparse LK flow on a regular grid."""
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for Lucas-Kanade optical flow. "
            "Install it with: pip install opencv-python"
        ) from exc

    ref8 = _to_uint8(ref)
    meas8 = _to_uint8(meas)

    H, W = ref.shape
    step = max(1, int(cfg.window_size * (1.0 - cfg.overlap)))
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    p0 = p0.reshape(-1, 1, 2)

    lk_params = dict(
        winSize=(cfg.lk_win_size, cfg.lk_win_size),
        maxLevel=cfg.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30, 0.01,
        ),
    )

    p1, status, _ = cv2.calcOpticalFlowPyrLK(ref8, meas8, p0, None, **lk_params)

    good = status.ravel() == 1
    src = p0.reshape(-1, 2)[good]
    dst = p1.reshape(-1, 2)[good]
    disp = dst - src  # (N, 2) — dx in col 0, dy in col 1

    # Interpolate sparse field to full resolution
    from scipy.interpolate import griddata
    yi = np.arange(H, dtype=np.float32)
    xi = np.arange(W, dtype=np.float32)
    xx, yy = np.meshgrid(xi, yi)

    dx = griddata(src, disp[:, 0], (xx, yy), method="linear", fill_value=0.0)
    dy = griddata(src, disp[:, 1], (xx, yy), method="linear", fill_value=0.0)

    return dx.astype(np.float32), dy.astype(np.float32)


# ---------------------------------------------------------------------------
# Optical flow — Farneback (dense)
# ---------------------------------------------------------------------------


def _farneback(
    ref: np.ndarray,
    meas: np.ndarray,
    cfg: DisplacementConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense Gunnar Farneback optical flow."""
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for Farneback optical flow. "
            "Install it with: pip install opencv-python"
        ) from exc

    ref8 = _to_uint8(ref)
    meas8 = _to_uint8(meas)

    flow = cv2.calcOpticalFlowFarneback(
        ref8, meas8,
        None,
        cfg.fb_pyr_scale,
        cfg.fb_levels,
        cfg.fb_winsize,
        cfg.fb_iterations,
        cfg.fb_poly_n,
        cfg.fb_poly_sigma,
        0,
    )

    dx = flow[..., 0].astype(np.float32)
    dy = flow[..., 1].astype(np.float32)
    return dx, dy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise float array to uint8 [0, 255] for OpenCV."""
    a_min, a_max = arr.min(), arr.max()
    if a_max - a_min < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - a_min) / (a_max - a_min) * 255
    return scaled.astype(np.uint8)


def displacement_magnitude(
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    """Return pixel-displacement magnitude sqrt(dx² + dy²)."""
    return np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
