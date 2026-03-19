"""BOS velocimetry — extracting 2-D velocity fields from BOS image sequences.

Two measurement methods are provided:

Frame-to-frame cross-correlation (Method A)
-------------------------------------------
BOS frames (raw or pre-computed density-gradient images) are treated as a PIV
image pair.  FFT cross-correlation is applied in interrogation windows; the
resulting pixel displacements are converted to physical velocities via:

    u = dx_px * (pixel_scale_mm * 1e-3) / dt   [m/s]

Outliers are rejected using the Normalised Median Test (NMT) of Westerweel &
Scarano (2005).  Vorticity and divergence are computed from the cleaned
velocity fields using second-order central differences.

References:

    Zhou et al., "Background-oriented-schlieren-based optical velocimetry of
    low-convective-Mach-number turbulent shear layers", Experiments in Fluids,
    2025.  DOI: 10.1007/s00348-024-03926-8

    Settles & Liberzon (2022), kymography-based BOS velocimetry.

    Westerweel & Scarano, "Universal outlier detection for PIV data",
    Exp. Fluids 39 (2005) 1096-1100.

Kymography (Method B)
---------------------
A single line (row or column) is extracted from each frame to build a
space-time (kymograph) image.  Dominant streak slopes are detected using the
Radon transform (via ``skimage.transform.radon``) or as a fallback via the
power-spectral-slope method.  The convective velocity is:

    U_c = pixel_scale_m / (slope_px_per_frame * dt)   [m/s]

References:

    Settles & Liberzon (2022), Exp. Fluids 63, 89.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import generic_filter, median_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class VelocityConfig:
    """Parameters for BOS velocimetry.

    Attributes
    ----------
    method:
        ``"frame_to_frame"`` for windowed cross-correlation between consecutive
        frames, or ``"kymography"`` for space-time streak analysis.
    window_size:
        Interrogation window size [px] for cross-correlation.
    overlap:
        Fractional overlap between adjacent interrogation windows [0 – 0.9].
    dt:
        Time interval between frames [s].  For a 1000 fps camera: ``1e-3``.
    pixel_scale_mm:
        Physical size of one pixel at the measurement plane [mm/px].
    nmt_threshold:
        Normalised Median Test threshold for outlier rejection.
        Westerweel & Scarano (2005) recommend values between 2 and 3.
    kymo_axis:
        Axis along which the kymograph line is extracted.
        ``"horizontal"`` → a row is extracted (x-axis is spatial, y-axis is time).
        ``"vertical"``   → a column is extracted.
    kymo_line_pos:
        Pixel position of the kymograph extraction line (row index for
        ``"horizontal"``, column index for ``"vertical"``).
    """

    method: Literal["frame_to_frame", "kymography"] = "frame_to_frame"

    # Cross-correlation
    window_size: int = 32
    overlap: float = 0.5

    # Physical calibration
    dt: float = 1.0 / 1000.0          # time step between frames [s]
    pixel_scale_mm: float = 0.1       # mm per pixel at measurement plane

    # NMT outlier rejection
    nmt_threshold: float = 2.0

    # Kymography-specific
    kymo_axis: Literal["horizontal", "vertical"] = "horizontal"
    kymo_line_pos: int = 128


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VelocityResult:
    """Output of :func:`compute_velocity_frame_to_frame`.

    Attributes
    ----------
    u:
        x-velocity component [m/s].  Shape is (n_rows_grid, n_cols_grid) for
        cross-correlation, or (H, W) for full-resolution methods.
    v:
        y-velocity component [m/s].  Positive y is downward (image convention).
    magnitude:
        Velocity magnitude :math:`|\\mathbf{V}| = \\sqrt{u^2 + v^2}` [m/s].
    vorticity:
        Out-of-plane vorticity :math:`\\omega_z = \\partial v/\\partial x -
        \\partial u/\\partial y` [1/s].
    divergence:
        2-D divergence :math:`\\nabla\\cdot\\mathbf{V} = \\partial u/\\partial x +
        \\partial v/\\partial y` [1/s].
    grid_x:
        x (column) coordinates of the velocity grid nodes [px].
        Shape (n_rows_grid, n_cols_grid).  ``None`` for full-resolution results.
    grid_y:
        y (row) coordinates of the velocity grid nodes [px].
        Shape (n_rows_grid, n_cols_grid).  ``None`` for full-resolution results.
    method:
        Name of the method that produced this result.
    """

    u: np.ndarray
    v: np.ndarray
    magnitude: np.ndarray
    vorticity: np.ndarray
    divergence: np.ndarray
    grid_x: Optional[np.ndarray]
    grid_y: Optional[np.ndarray]
    method: str


# ---------------------------------------------------------------------------
# Method A — Frame-to-frame cross-correlation
# ---------------------------------------------------------------------------


def compute_velocity_frame_to_frame(
    frame1: np.ndarray,
    frame2: np.ndarray,
    config: Optional[VelocityConfig] = None,
) -> VelocityResult:
    """Compute the 2-D velocity field between two consecutive BOS frames.

    The frames may be raw camera images or pre-computed density-gradient
    fields (e.g., the output of :func:`~bos_pipeline.processing.displacement`
    mapped back to intensity).  The algorithm mirrors PIV processing:

    1. Divide images into overlapping interrogation windows.
    2. FFT cross-correlate each window pair.
    3. Locate correlation peak with sub-pixel parabolic refinement.
    4. Convert pixel displacement to velocity (Eq. 1):

    .. math::

        u = \\frac{\\Delta x_{\\text{px}} \\cdot s}{\\Delta t}, \\quad
        v = \\frac{\\Delta y_{\\text{px}} \\cdot s}{\\Delta t}

    where :math:`s = \\text{pixel\\_scale\\_mm} \\times 10^{-3}` [m/px] and
    :math:`\\Delta t` = ``config.dt`` [s].

    5. Apply the Normalised Median Test (NMT) to flag and replace outliers.
    6. Compute vorticity and divergence via central differences.

    Parameters
    ----------
    frame1:
        First frame (H x W), float32 or float64.
    frame2:
        Second frame (H x W), same shape as *frame1*.
    config:
        :class:`VelocityConfig` instance.  Defaults are used if ``None``.

    Returns
    -------
    VelocityResult
        Velocity field, derived scalar fields, and grid coordinates.

    Raises
    ------
    ValueError
        If ``frame1`` and ``frame2`` have different shapes.

    References
    ----------
    Zhou et al. (2025), Exp. Fluids.
    Westerweel & Scarano (2005), Exp. Fluids 39, 1096.
    """
    cfg = config or VelocityConfig()

    if frame1.shape != frame2.shape:
        raise ValueError(
            f"Frame shape mismatch: {frame1.shape} vs {frame2.shape}."
        )

    H, W = frame1.shape
    ws = cfg.window_size
    step = max(1, int(ws * (1.0 - cfg.overlap)))
    pixel_scale_m = cfg.pixel_scale_mm * 1e-3

    # Build grid of window centres
    grid_rows, grid_cols = _build_grid(H, W, ws, step)
    n_rows, n_cols = len(grid_rows), len(grid_cols)

    dx_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
    dy_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    half = ws // 2
    win2d = np.outer(np.hanning(ws), np.hanning(ws))

    f1 = frame1.astype(np.float64)
    f2 = frame2.astype(np.float64)

    for ri, row in enumerate(grid_rows):
        for ci, col in enumerate(grid_cols):
            r0, r1 = max(0, row - half), min(H, row + half)
            c0, c1 = max(0, col - half), min(W, col + half)

            p1 = f1[r0:r1, c0:c1]
            p2 = f2[r0:r1, c0:c1]

            if p1.shape != (ws, ws):
                ph, pw = p1.shape
                pad = ((0, ws - ph), (0, ws - pw))
                p1 = np.pad(p1, pad)
                p2 = np.pad(p2, pad)

            pr = _normalise_patch(p1) * win2d
            pm = _normalise_patch(p2) * win2d

            corr = _fft_xcorr(pm, pr)

            peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
            dy_sub, dx_sub = _subpixel_parabolic(corr, peak_y, peak_x)

            cy_c, cx_c = corr.shape[0] // 2, corr.shape[1] // 2
            dy_grid[ri, ci] = float((peak_y - cy_c) + dy_sub)
            dx_grid[ri, ci] = float((peak_x - cx_c) + dx_sub)

    # Convert pixel displacement → velocity  [m/s]  (Eq. 1)
    u_raw = dx_grid * pixel_scale_m / cfg.dt
    v_raw = dy_grid * pixel_scale_m / cfg.dt

    # NMT outlier rejection
    u_clean, v_clean, _mask = _nmt_filter(u_raw, v_raw, threshold=cfg.nmt_threshold)

    n_outliers = int(np.sum(_mask))
    if n_outliers:
        logger.debug(
            "NMT rejected %d / %d vectors (%.1f%%).",
            n_outliers, n_rows * n_cols,
            100.0 * n_outliers / (n_rows * n_cols),
        )

    # Physical grid spacing [m]
    dx_m = pixel_scale_m * step
    dy_m = pixel_scale_m * step

    vorticity, divergence = compute_derived_fields(u_clean, v_clean, dx_m, dy_m)

    magnitude = np.sqrt(u_clean ** 2 + v_clean ** 2).astype(np.float32)

    gx_arr, gy_arr = np.meshgrid(grid_cols, grid_rows)

    logger.info(
        "Frame-to-frame velocity: grid (%d x %d), "
        "max |u|=%.3f m/s, max |v|=%.3f m/s.",
        n_rows, n_cols, np.abs(u_clean).max(), np.abs(v_clean).max(),
    )

    return VelocityResult(
        u=u_clean.astype(np.float32),
        v=v_clean.astype(np.float32),
        magnitude=magnitude,
        vorticity=vorticity.astype(np.float32),
        divergence=divergence.astype(np.float32),
        grid_x=gx_arr.astype(np.float32),
        grid_y=gy_arr.astype(np.float32),
        method="frame_to_frame",
    )


# ---------------------------------------------------------------------------
# NMT outlier rejection
# ---------------------------------------------------------------------------


def _nmt_filter(
    u: np.ndarray,
    v: np.ndarray,
    threshold: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised Median Test (NMT) outlier rejection for 2-D velocity fields.

    For each vector :math:`(u_i, v_i)`, the residual normalised by the
    local median absolute deviation (MAD) of its 3x3 neighbourhood is
    computed.  If the normalised residual exceeds ``threshold``, the vector
    is flagged as an outlier and replaced by the local neighbourhood median.

    The normalised residual for component :math:`q \\in \\{u, v\\}` is:

    .. math::

        r_q = \\frac{|q_i - q_{\\text{med}}|}{\\text{MAD}_q + \\epsilon}

    where :math:`q_{\\text{med}}` is the median of the 3x3 neighbourhood
    (excluding the centre) and :math:`\\epsilon = 0.1` is a noise floor
    that prevents false rejections in uniform flow regions.  A vector is an
    outlier if :math:`r_u > t` **or** :math:`r_v > t`.

    Parameters
    ----------
    u:
        x-velocity grid (n_rows x n_cols), float.
    v:
        y-velocity grid (n_rows x n_cols), float.
    threshold:
        Normalised residual threshold ``t``.  Westerweel & Scarano (2005)
        recommend ``t = 2``.

    Returns
    -------
    u_clean:
        Cleaned x-velocity (outliers replaced by neighbourhood median).
    v_clean:
        Cleaned y-velocity.
    outlier_mask:
        Boolean array (True = outlier).

    References
    ----------
    Westerweel & Scarano, "Universal outlier detection for PIV data",
    Exp. Fluids 39 (2005) 1096-1100.  doi:10.1007/s00348-005-0016-6
    """
    eps = 0.1   # noise floor, Westerweel & Scarano (2005) Eq. 5

    def _med3x3(arr: np.ndarray) -> np.ndarray:
        """3x3 neighbourhood median (edge-padded with reflect mode)."""
        return median_filter(arr, size=3, mode="reflect")

    def _mad3x3(arr: np.ndarray, med: np.ndarray) -> np.ndarray:
        """Median absolute deviation of 3x3 neighbourhood."""
        return median_filter(np.abs(arr - med), size=3, mode="reflect")

    u_med = _med3x3(u)
    v_med = _med3x3(v)

    u_mad = _mad3x3(u, u_med)
    v_mad = _mad3x3(v, v_med)

    # Normalised residuals
    r_u = np.abs(u - u_med) / (u_mad + eps)
    r_v = np.abs(v - v_med) / (v_mad + eps)

    outlier_mask = (r_u > threshold) | (r_v > threshold)

    u_clean = np.where(outlier_mask, u_med, u).astype(u.dtype)
    v_clean = np.where(outlier_mask, v_med, v).astype(v.dtype)

    return u_clean, v_clean, outlier_mask


# ---------------------------------------------------------------------------
# Method B — Kymography
# ---------------------------------------------------------------------------


def compute_velocity_kymography(
    frames: Sequence[np.ndarray],
    config: Optional[VelocityConfig] = None,
) -> Dict:
    """Estimate convective velocity from a kymograph of a BOS image sequence.

    A 1-D spatial profile along ``config.kymo_line_pos`` is extracted from
    every frame and assembled into a 2-D space-time array (the kymograph).
    Coherent structures in the flow appear as inclined streaks whose slope is
    inversely proportional to the convective velocity.

    The dominant slope is estimated by three approaches (in order of
    preference):

    1. **Radon transform** (primary): ``skimage.transform.radon`` projects the
       kymograph onto a set of angles; the angle giving the maximum projection
       variance corresponds to the dominant streak direction.  The convective
       velocity is (Eq. 2):

    .. math::

        U_c = \\frac{s}{m \\cdot \\Delta t}

    where :math:`s` = ``pixel_scale_mm * 1e-3`` [m/px], :math:`m` is the
    streak slope in [px/frame], and :math:`\\Delta t` = ``dt`` [s].

    2. **Power-spectrum fallback**: the 2-D FFT of the kymograph is used to
       find the dominant wavenumber-frequency ridge, giving an independent
       estimate of :math:`U_c = \\omega / k`.

    Parameters
    ----------
    frames:
        Sequence of 2-D numpy arrays (each H x W), ordered in time.
    config:
        :class:`VelocityConfig` instance.

    Returns
    -------
    dict with keys:
        ``"kymograph"``
            2-D float32 array of shape (n_spatial, n_frames), i.e., space
            along rows and time along columns.
        ``"velocity_profile"``
            1-D float32 array of length *n_spatial* with per-position
            velocity estimates [m/s].  For kymography this is a constant
            value equal to the dominant convective velocity, but is returned
            as an array for API consistency.
        ``"convective_velocity"``
            Dominant convective velocity scalar [m/s].
        ``"kymograph_lines"``
            List of dicts, each with keys ``"slope_px_per_frame"`` and
            ``"convective_velocity_m_s"`` describing detected streak lines.

    Raises
    ------
    ValueError
        If ``frames`` is empty or contains fewer than 2 frames.

    References
    ----------
    Settles & Liberzon (2022), Exp. Fluids 63, 89.
    Zhou et al. (2025), Exp. Fluids.
    """
    cfg = config or VelocityConfig()

    if len(frames) < 2:
        raise ValueError("At least 2 frames are required for kymography.")

    pixel_scale_m = cfg.pixel_scale_mm * 1e-3

    # ------------------------------------------------------------------
    # 1. Build the kymograph  (space x time)
    # ------------------------------------------------------------------
    profiles: List[np.ndarray] = []
    for frame in frames:
        arr = frame.astype(np.float32)
        if cfg.kymo_axis == "horizontal":
            # Extract a row: spatial axis is x (columns)
            row = int(np.clip(cfg.kymo_line_pos, 0, arr.shape[0] - 1))
            profiles.append(arr[row, :])
        else:
            # Extract a column: spatial axis is y (rows)
            col = int(np.clip(cfg.kymo_line_pos, 0, arr.shape[1] - 1))
            profiles.append(arr[:, col])

    # kymograph shape: (n_spatial, n_frames)
    kymo = np.stack(profiles, axis=1).astype(np.float32)
    n_spatial, n_frames = kymo.shape

    logger.info(
        "Kymograph shape: %d spatial x %d frames (axis=%s, line=%d).",
        n_spatial, n_frames, cfg.kymo_axis, cfg.kymo_line_pos,
    )

    # Normalise kymograph for better streak detection
    kymo_norm = kymo - kymo.mean()
    std_k = kymo_norm.std()
    if std_k > 1e-12:
        kymo_norm /= std_k

    # ------------------------------------------------------------------
    # 2. Radon-transform streak detection (primary method)
    # ------------------------------------------------------------------
    convective_velocity: float
    kymograph_lines: List[Dict] = []

    try:
        from skimage.transform import radon  # type: ignore[import]

        # Test angles: slopes from -80 to +80 degrees.
        # Angle 0° = projection onto rows (horizontal streaks → infinite velocity).
        # Angle ~±45° → slope ≈ 1 px/frame → moderate velocity.
        theta_deg = np.linspace(-80.0, 80.0, 321)
        sinogram = radon(kymo_norm, theta=theta_deg, circle=False)

        # Variance along each projection angle
        proj_var = sinogram.var(axis=0)
        best_idx = int(np.argmax(proj_var))
        best_angle_deg = float(theta_deg[best_idx])

        # Convert Radon angle to slope in px/frame:
        # Radon angle theta is measured from the horizontal axis of the image.
        # A streak running at angle alpha from the vertical (time axis) has
        # Radon angle = 90 - alpha.
        # slope_px_per_frame = tan(alpha) = tan(90 - theta) = cot(theta)
        angle_rad = math.radians(best_angle_deg)
        if abs(math.sin(angle_rad)) < 1e-6:
            # Near-horizontal streak → very high or infinite velocity
            slope_px_per_frame = np.inf
        else:
            slope_px_per_frame = math.cos(angle_rad) / math.sin(angle_rad)

        if abs(slope_px_per_frame) < 1e-12 or not math.isfinite(slope_px_per_frame):
            convective_velocity = 0.0
        else:
            # Eq. 2:  U_c = s / (m * dt)
            convective_velocity = float(pixel_scale_m / (slope_px_per_frame * cfg.dt))

        kymograph_lines.append({
            "slope_px_per_frame": slope_px_per_frame,
            "convective_velocity_m_s": convective_velocity,
            "method": "radon",
            "radon_angle_deg": best_angle_deg,
        })

        logger.info(
            "Radon kymography: best angle=%.1f deg, slope=%.3f px/frame, "
            "U_c=%.3f m/s.",
            best_angle_deg, slope_px_per_frame, convective_velocity,
        )

    except ImportError:
        logger.warning(
            "scikit-image not available; falling back to power-spectrum method."
        )
        convective_velocity, kymograph_lines = _kymo_power_spectrum(
            kymo_norm, pixel_scale_m, cfg.dt
        )

    # ------------------------------------------------------------------
    # 3. Power-spectrum fallback / cross-check
    # ------------------------------------------------------------------
    uc_ps, lines_ps = _kymo_power_spectrum(kymo_norm, pixel_scale_m, cfg.dt)
    kymograph_lines.extend(lines_ps)

    logger.info("Power-spectrum cross-check: U_c=%.3f m/s.", uc_ps)

    # ------------------------------------------------------------------
    # 4. Build velocity_profile (constant for single-line kymography)
    # ------------------------------------------------------------------
    velocity_profile = np.full(n_spatial, convective_velocity, dtype=np.float32)

    return {
        "kymograph": kymo,
        "velocity_profile": velocity_profile,
        "convective_velocity": convective_velocity,
        "kymograph_lines": kymograph_lines,
    }


def _kymo_power_spectrum(
    kymo_norm: np.ndarray,
    pixel_scale_m: float,
    dt: float,
) -> Tuple[float, List[Dict]]:
    """Estimate convective velocity from the 2-D power spectrum of the kymograph.

    The 2-D FFT of the kymograph produces a wavenumber-frequency spectrum.
    For convected structures (Taylor's frozen turbulence hypothesis), energy
    concentrates along the line :math:`\\omega = k \\cdot U_c` in wavenumber-
    frequency space.  The slope of this ridge gives :math:`U_c`.

    The slope is found by fitting the peak of each 1-D frequency slice
    (for each non-zero wavenumber bin) and taking the median slope.

    Parameters
    ----------
    kymo_norm:
        Normalised kymograph array (n_spatial x n_frames), float.
    pixel_scale_m:
        Physical pixel size [m/px].
    dt:
        Time step between frames [s].

    Returns
    -------
    convective_velocity:
        Dominant convective velocity [m/s].
    lines:
        List with a single entry describing the detection.
    """
    n_space, n_time = kymo_norm.shape

    win = np.outer(np.hanning(n_space), np.hanning(n_time))
    psd2 = np.abs(np.fft.fftshift(np.fft.fft2(kymo_norm * win))) ** 2

    # Frequency axes
    k_axis = np.fft.fftshift(np.fft.fftfreq(n_space))   # cycles/px
    f_axis = np.fft.fftshift(np.fft.fftfreq(n_time))     # cycles/frame

    # Only positive spatial frequencies
    pos_k = k_axis > 0
    psd_pos = psd2[pos_k, :]

    # For each k, find the dominant f
    dominant_f = np.array([
        float(f_axis[np.argmax(row)]) for row in psd_pos
    ])
    k_vals = k_axis[pos_k]

    # Convert:  f [cy/frame] / k [cy/px] = px/frame  → * pixel_scale_m / dt = m/s
    # Use median to robustify
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(np.abs(k_vals) > 1e-9, dominant_f / k_vals, np.nan)

    slopes_valid = slopes[np.isfinite(slopes)]
    if slopes_valid.size == 0:
        slope_median = 0.0
    else:
        slope_median = float(np.median(slopes_valid))

    if abs(slope_median) < 1e-12:
        uc = 0.0
    else:
        # slope [px/frame] * pixel_scale_m [m/px] / dt [s/frame] = [m/s]
        uc = float(slope_median * pixel_scale_m / dt)

    lines: List[Dict] = [{
        "slope_px_per_frame": slope_median,
        "convective_velocity_m_s": uc,
        "method": "power_spectrum",
    }]

    return uc, lines


# ---------------------------------------------------------------------------
# Derived fields: vorticity and divergence
# ---------------------------------------------------------------------------


def compute_derived_fields(
    u: np.ndarray,
    v: np.ndarray,
    dx_m: float,
    dy_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute vorticity and divergence from 2-D velocity fields.

    Uses second-order central differences for interior points and
    first-order forward/backward differences at the domain boundaries.

    .. math::

        \\omega_z = \\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y}
        \\quad [\\text{1/s}]

    .. math::

        \\nabla\\cdot\\mathbf{V} =
        \\frac{\\partial u}{\\partial x} + \\frac{\\partial v}{\\partial y}
        \\quad [\\text{1/s}]

    Parameters
    ----------
    u:
        x-velocity field (n_rows x n_cols) [m/s].
    v:
        y-velocity field (n_rows x n_cols) [m/s].
    dx_m:
        Physical grid spacing in the x-direction [m].
    dy_m:
        Physical grid spacing in the y-direction [m].

    Returns
    -------
    vorticity:
        Out-of-plane vorticity field [1/s], same shape as *u*.
    divergence:
        2-D divergence field [1/s], same shape as *u*.
    """
    u = u.astype(np.float64)
    v = v.astype(np.float64)

    nr, nc = u.shape
    dvdx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dudx = np.zeros_like(u)
    dvdy = np.zeros_like(u)

    # Interior: central differences
    if nc >= 3:
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * dx_m)
        dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx_m)
    # Boundaries: forward/backward
    if nc >= 2:
        dvdx[:, 0]  = (v[:, 1] - v[:, 0])    / dx_m
        dvdx[:, -1] = (v[:, -1] - v[:, -2])   / dx_m
        dudx[:, 0]  = (u[:, 1] - u[:, 0])     / dx_m
        dudx[:, -1] = (u[:, -1] - u[:, -2])   / dx_m

    if nr >= 3:
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dy_m)
        dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy_m)
    if nr >= 2:
        dudy[0, :]  = (u[1, :] - u[0, :])    / dy_m
        dudy[-1, :] = (u[-1, :] - u[-2, :])   / dy_m
        dvdy[0, :]  = (v[1, :] - v[0, :])     / dy_m
        dvdy[-1, :] = (v[-1, :] - v[-2, :])   / dy_m

    vorticity = dvdx - dudy
    divergence = dudx + dvdy

    return vorticity, divergence


# ---------------------------------------------------------------------------
# Private helpers (shared with Method A)
# ---------------------------------------------------------------------------


def _build_grid(
    H: int,
    W: int,
    ws: int,
    step: int,
) -> Tuple[List[int], List[int]]:
    """Return lists of window centre row and column coordinates."""
    half = ws // 2
    rows = list(range(half, H - half + 1, step))
    cols = list(range(half, W - half + 1, step))
    if not rows:
        rows = [H // 2]
    if not cols:
        cols = [W // 2]
    return rows, cols


def _normalise_patch(patch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation of an interrogation window."""
    p = patch - patch.mean()
    std = p.std()
    return p / std if std > 1e-12 else p


def _fft_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised FFT cross-correlation; output is shifted so DC is at centre."""
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    corr = np.fft.ifft2(fa * np.conj(fb)).real
    return np.fft.fftshift(corr)


def _subpixel_parabolic(
    corr: np.ndarray,
    py: int,
    px: int,
) -> Tuple[float, float]:
    """Sub-pixel peak refinement using a 3-point parabolic fit.

    Returns (dy_sub, dx_sub) — the fractional offsets from the integer peak.

    The parabolic estimator is (Willert & Gharib 1991):

    .. math::

        \\delta = \\frac{C_{-1} - C_{+1}}{2(C_{-1} - 2C_0 + C_{+1})}

    Parameters
    ----------
    corr:
        2-D cross-correlation map (DC at centre after ``fftshift``).
    py, px:
        Integer row and column of the correlation peak.

    Returns
    -------
    dy_sub, dx_sub:
        Sub-pixel offsets (both in [-0.5, 0.5]).

    References
    ----------
    Willert & Gharib (1991), Exp. Fluids 10, 181-193.
    """
    h, w = corr.shape

    def _parabolic(cm: float, c0: float, cp: float) -> float:
        denom = 2.0 * (cm - 2.0 * c0 + cp)
        if abs(denom) < 1e-12:
            return 0.0
        return float((cm - cp) / denom)

    dy_sub = (
        _parabolic(corr[py - 1, px], corr[py, px], corr[py + 1, px])
        if 0 < py < h - 1 else 0.0
    )
    dx_sub = (
        _parabolic(corr[py, px - 1], corr[py, px], corr[py, px + 1])
        if 0 < px < w - 1 else 0.0
    )

    return dy_sub, dx_sub
