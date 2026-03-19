"""Background pattern generation for BOS (Background-Oriented Schlieren) experiments.

This module implements three background pattern types relevant to BOS and, in particular,
the Adaptive Stream Stripe BOS (ASS-BOS) technique described in:

    Najibi et al., "Measurement of the concentration distribution of hydrogen jets
    using adaptive stream stripe-background oriented schlieren (ASS-BOS)",
    International Journal of Hydrogen Energy, Vol. 77, 2024, pp. 281-295.
    DOI: 10.1016/j.ijhydene.2024.06.298

Pattern types
-------------
* ``random_dots``   — Random dot pattern (classical BOS background).
* ``checkerboard``  — Binary checkerboard (useful for calibration / comparison).
* ``ass_stripe``    — Adaptive stream stripe pattern whose local stripe orientation
                      is aligned with expected flow streamlines.  For an axisymmetric
                      vertical jet, streamlines are radial from the jet origin and
                      the ASS-BOS approach orients stripes perpendicular to them,
                      maximising sensitivity along the dominant density-gradient
                      direction.

Quality metrics
---------------
:class:`PatternQualityMetric` and :func:`compute_pattern_quality` estimate how well
a pattern will perform in cross-correlation displacement measurements by computing:

* Cross-correlation SNR (synthetic 1 px shift test).
* Contrast (coefficient of variation: std / mean).
* Dominant spatial frequency [cycles/px].

All patterns are returned as ``uint8`` numpy arrays (values 0 or 255 for binary
patterns, 0-255 for smooth variants).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import label as ndimage_label
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class PatternConfig:
    """Parameters controlling background pattern generation.

    Attributes
    ----------
    pattern_type:
        One of ``"random_dots"``, ``"checkerboard"``, ``"ass_stripe"``.
    image_height:
        Output image height in pixels.
    image_width:
        Output image width in pixels.
    dot_density:
        Fraction of pixels covered by dots (used by ``random_dots``).
        Typical range: 0.02 – 0.20.
    dot_radius_px:
        Radius of each dot in pixels (used by ``random_dots``).
    seed:
        Random number generator seed for reproducibility.
    checker_size_px:
        Side length of each checkerboard square in pixels.
    stripe_width_px:
        Half-period of the stripe pattern in pixels (one light + one dark band
        equals ``2 * stripe_width_px`` pixels).
    stripe_orientation_deg:
        Global stripe orientation for uniform stripes [degrees].
        0° → horizontal stripes (light/dark bands run left-right, varying top-bottom).
        90° → vertical stripes.
        Ignored when ``orientation_field`` is provided.
    orientation_field:
        2-D array of shape (H, W) containing per-pixel stripe orientation angles
        [radians].  When supplied, overrides ``stripe_orientation_deg`` and
        produces spatially varying ASS-BOS stripes.  ``None`` → uniform stripes.
    """

    pattern_type: Literal["random_dots", "checkerboard", "ass_stripe"] = "random_dots"
    image_height: int = 512
    image_width: int = 512

    # Random dot parameters
    dot_density: float = 0.05
    dot_radius_px: float = 2.0
    seed: int = 42

    # Checkerboard parameters
    checker_size_px: int = 16

    # ASS-stripe parameters
    stripe_width_px: int = 8
    stripe_orientation_deg: float = 0.0           # global orientation [deg]
    orientation_field: Optional[np.ndarray] = None  # per-pixel angles [rad]


# ---------------------------------------------------------------------------
# Public pattern generators
# ---------------------------------------------------------------------------


class RandomDotPattern:
    """Generates a random-dot BOS background pattern.

    The pattern consists of filled circular dots scattered uniformly at random
    across the image.  Dot density is controlled as a target area fraction; the
    actual number of dots is chosen so that non-overlapping dots would cover
    approximately ``dot_density`` of the image area.

    Parameters
    ----------
    config:
        :class:`PatternConfig` instance (``pattern_type`` need not be set).

    Notes
    -----
    Dots are drawn by rasterising circles of radius ``dot_radius_px`` centred
    at Poisson-distributed random locations with mean separation derived from
    the target density.
    """

    def __init__(self, config: PatternConfig) -> None:
        self._cfg = config

    def generate(self) -> np.ndarray:
        """Generate the random dot pattern.

        Returns
        -------
        np.ndarray
            Uint8 image of shape (H, W); values are 0 (background) or 255 (dot).
        """
        cfg = self._cfg
        H, W = cfg.image_height, cfg.image_width
        rng = np.random.default_rng(cfg.seed)

        canvas = np.zeros((H, W), dtype=np.uint8)

        r = cfg.dot_radius_px
        dot_area = math.pi * r ** 2
        total_area = H * W
        n_dots = max(1, int(round(cfg.dot_density * total_area / dot_area)))

        logger.debug(
            "RandomDotPattern: %d dots, radius=%.1f px, density=%.3f",
            n_dots, r, cfg.dot_density,
        )

        # Draw dots
        cx_all = rng.uniform(0, W, size=n_dots)
        cy_all = rng.uniform(0, H, size=n_dots)

        # Pre-build circular mask template
        ri = int(math.ceil(r))
        ys_t, xs_t = np.ogrid[-ri: ri + 1, -ri: ri + 1]
        circle_mask = (xs_t ** 2 + ys_t ** 2) <= r ** 2

        for cx, cy in zip(cx_all, cy_all):
            # Bounding box in image coordinates
            row_c, col_c = int(round(cy)), int(round(cx))
            r0 = row_c - ri
            r1 = row_c + ri + 1
            c0 = col_c - ri
            c1 = col_c + ri + 1

            # Clip to image bounds and corresponding mask indices
            mr0 = max(0, -r0)
            mc0 = max(0, -c0)
            mr1 = mr0 + (min(r1, H) - max(r0, 0))
            mc1 = mc0 + (min(c1, W) - max(c0, 0))

            ir0, ir1 = max(r0, 0), min(r1, H)
            ic0, ic1 = max(c0, 0), min(c1, W)

            if ir0 >= ir1 or ic0 >= ic1:
                continue

            canvas[ir0:ir1, ic0:ic1] = np.where(
                circle_mask[mr0:mr1, mc0:mc1],
                255,
                canvas[ir0:ir1, ic0:ic1],
            )

        return canvas


class CheckerboardPattern:
    """Generates a binary checkerboard BOS background pattern.

    Parameters
    ----------
    config:
        :class:`PatternConfig` instance.
    """

    def __init__(self, config: PatternConfig) -> None:
        self._cfg = config

    def generate(self) -> np.ndarray:
        """Generate the checkerboard pattern.

        Returns
        -------
        np.ndarray
            Uint8 image of shape (H, W); values are 0 or 255.
        """
        cfg = self._cfg
        H, W = cfg.image_height, cfg.image_width
        sq = max(1, cfg.checker_size_px)

        row_idx = np.arange(H) // sq
        col_idx = np.arange(W) // sq
        rr, cc = np.meshgrid(row_idx, col_idx, indexing="ij")
        pattern = ((rr + cc) % 2).astype(np.uint8) * 255

        logger.debug(
            "CheckerboardPattern: H=%d W=%d square_size=%d px", H, W, sq,
        )
        return pattern


class ASSStripePattern:
    """Generates an Adaptive Stream Stripe (ASS-BOS) background pattern.

    In ASS-BOS (Najibi et al. 2024) the stripe orientation is locally aligned
    with the expected flow streamlines so that the cross-correlation
    displacement signal is maximised in the direction of the dominant
    density gradient.

    The stripe intensity at a pixel (y, x) is computed as:

    .. math::

        I(y, x) = \\frac{1}{2}\\left[
            1 + \\operatorname{sgn}\\!\\left(
                \\sin\\!\\left(
                    \\frac{\\pi}{w}\\,\\bigl(x\\cos\\theta + y\\sin\\theta\\bigr)
                \\right)
            \\right)
        \\right] \\times 255

    where :math:`\\theta` is the local stripe orientation (from the
    ``orientation_field`` array or the global ``stripe_orientation_deg``
    parameter) and :math:`w` is ``stripe_width_px``.

    When *uniform* stripes are requested (``orientation_field=None``), the
    phase argument reduces to a simple scan direction projected at angle
    :math:`\\theta`, giving perfectly parallel straight stripes.

    For spatially varying stripes, each pixel uses its own :math:`\\theta`
    value, which produces curved stripe patterns that follow streamlines.

    Parameters
    ----------
    config:
        :class:`PatternConfig` instance.  Set ``orientation_field`` for
        spatially adaptive stripes.

    See Also
    --------
    ASSStripePattern.generate_from_streamlines :
        Classmethod to build a streamline-aligned orientation field for a
        vertical axisymmetric jet.
    """

    def __init__(self, config: PatternConfig) -> None:
        self._cfg = config

    def generate(self) -> np.ndarray:
        """Generate the ASS-BOS stripe pattern.

        Returns
        -------
        np.ndarray
            Uint8 image of shape (H, W); values are 0 or 255.
        """
        cfg = self._cfg
        H, W = cfg.image_height, cfg.image_width
        w = max(1, cfg.stripe_width_px)

        yy, xx = np.indices((H, W), dtype=np.float64)

        if cfg.orientation_field is not None:
            theta = cfg.orientation_field.astype(np.float64)
            if theta.shape != (H, W):
                raise ValueError(
                    f"orientation_field shape {theta.shape} does not match "
                    f"image shape ({H}, {W})."
                )
        else:
            theta = np.full((H, W), math.radians(cfg.stripe_orientation_deg))

        # Phase: projection of pixel position onto the direction perpendicular
        # to the stripe (i.e., the direction theta gives the stripe normal).
        # Eq. (per Najibi et al. 2024, adapted):
        #   phase(y, x) = (pi / w) * (x * cos(theta) + y * sin(theta))
        phase = (math.pi / w) * (xx * np.cos(theta) + yy * np.sin(theta))
        pattern = ((np.sin(phase) >= 0)).astype(np.uint8) * 255

        logger.debug(
            "ASSStripePattern: H=%d W=%d stripe_width=%d px, "
            "orientation=%s",
            H, W, w,
            "spatially varying" if cfg.orientation_field is not None
            else f"{cfg.stripe_orientation_deg:.1f} deg",
        )
        return pattern

    @classmethod
    def generate_from_streamlines(
        cls,
        jet_axis_col: int,
        jet_origin_row: int,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Compute a streamline-aligned orientation field for a vertical axisymmetric jet.

        For an axisymmetric jet issuing vertically (along the image column
        direction), the streamlines are radial lines emanating from the nozzle
        exit point ``(jet_origin_row, jet_axis_col)``.  The ASS-BOS stripe
        should run *perpendicular* to the radial direction (i.e., the stripes
        are concentric arcs centred on the nozzle).

        The orientation angle at pixel (y, x) is the angle of the radial
        vector from the jet origin to that pixel, measured from the positive
        x-axis:

        .. math::

            \\theta(y, x) = \\operatorname{atan2}(y - y_0,\\; x - x_0) + \\frac{\\pi}{2}

        where :math:`(x_0, y_0)` is the jet origin.  The :math:`+\\pi/2`
        rotates the stripe normal by 90° so that the stripes themselves run
        along the radial direction (sensitivity in the cross-radial, i.e.,
        azimuthal, direction).

        Parameters
        ----------
        jet_axis_col:
            Column index of the jet symmetry axis.
        jet_origin_row:
            Row index of the jet nozzle exit (origin of streamlines).
        image_shape:
            (H, W) of the target image.

        Returns
        -------
        np.ndarray
            Float64 array of shape (H, W) containing per-pixel orientation
            angles in radians.  Pass this as ``orientation_field`` in
            :class:`PatternConfig`.

        Notes
        -----
        Pixels exactly at the jet origin (where ``atan2`` is undefined) are
        assigned the global orientation :math:`\\theta = 0` (horizontal
        stripes).
        """
        H, W = image_shape
        yy, xx = np.indices((H, W), dtype=np.float64)

        dy = yy - jet_origin_row
        dx = xx - jet_axis_col

        # Radial angle of each pixel from the jet origin
        radial_angle = np.arctan2(dy, dx)

        # Stripe perpendicular to radial → rotate by 90°
        orientation = radial_angle + math.pi / 2.0

        # Fix the singularity at the jet origin
        origin_mask = (dy == 0) & (dx == 0)
        orientation[origin_mask] = 0.0

        return orientation


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def generate_pattern(config: PatternConfig) -> np.ndarray:
    """Generate a BOS background pattern array from a :class:`PatternConfig`.

    Parameters
    ----------
    config:
        Configuration dataclass specifying pattern type and parameters.

    Returns
    -------
    np.ndarray
        Uint8 array of shape (H, W) with values 0 or 255.

    Raises
    ------
    ValueError
        If ``config.pattern_type`` is not recognised.

    Examples
    --------
    >>> cfg = PatternConfig(pattern_type="random_dots", image_height=256,
    ...                     image_width=256, dot_density=0.05, seed=0)
    >>> pattern = generate_pattern(cfg)
    >>> pattern.dtype
    dtype('uint8')
    """
    if config.pattern_type == "random_dots":
        return RandomDotPattern(config).generate()
    elif config.pattern_type == "checkerboard":
        return CheckerboardPattern(config).generate()
    elif config.pattern_type == "ass_stripe":
        return ASSStripePattern(config).generate()
    else:
        raise ValueError(
            f"Unknown pattern_type: '{config.pattern_type}'. "
            "Choose from 'random_dots', 'checkerboard', 'ass_stripe'."
        )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


@dataclass
class PatternQualityMetric:
    """Quality metrics for a BOS background pattern.

    Attributes
    ----------
    peak_snr:
        Signal-to-noise ratio of the cross-correlation displacement peak
        when a synthetic 1 px horizontal shift is applied.
        Computed as:

        .. math::

            \\text{SNR} = 20 \\log_{10}\\!\\left(
                \\frac{p_{\\max}}{\\sigma_{\\text{bg}}}
            \\right) \\quad [\\text{dB}]

        where :math:`p_{\\max}` is the peak value and
        :math:`\\sigma_{\\text{bg}}` is the RMS of the correlation map
        excluding a small region around the peak.
    contrast:
        Coefficient of variation of the pattern intensity:

        .. math::

            C = \\frac{\\sigma_I}{\\mu_I}

        where :math:`\\sigma_I` and :math:`\\mu_I` are the standard deviation
        and mean of the pixel values.  Higher contrast generally yields better
        cross-correlation performance.
    spatial_frequency_peak:
        Dominant spatial frequency of the pattern in cycles per pixel,
        estimated from the radially averaged power spectral density.
        Optimal BOS patterns have a spatial frequency in the range
        0.1 – 0.4 cycles/px (Hargather & Settles 2012).
    """

    peak_snr: float
    contrast: float
    spatial_frequency_peak: float


def compute_pattern_quality(
    pattern: np.ndarray,
    window_size: int = 32,
) -> PatternQualityMetric:
    """Compute quality metrics for a BOS background pattern.

    Applies a synthetic horizontal shift of 1 pixel to a central window of
    the pattern and measures the cross-correlation peak SNR.  Also computes
    pattern contrast and the dominant spatial frequency from the 2-D power
    spectral density.

    Parameters
    ----------
    pattern:
        2-D uint8 array (H x W) — the background pattern to evaluate.
    window_size:
        Side length of the interrogation window used for the SNR test.

    Returns
    -------
    PatternQualityMetric
        Dataclass containing ``peak_snr``, ``contrast``, and
        ``spatial_frequency_peak``.

    Notes
    -----
    The SNR formula follows:

    .. math::

        \\text{SNR [dB]} = 20 \\log_{10}\\!\\left(
            \\frac{C_{\\max}}{C_{\\text{rms,bg}}}
        \\right)

    where :math:`C_{\\text{rms,bg}}` is the root-mean-square of the
    correlation map pixels *outside* a 3-pixel exclusion zone around the
    peak — i.e., the background noise level of the cross-correlation.

    The dominant spatial frequency is found by computing the 2-D FFT of the
    full pattern, forming the radially averaged power spectrum, and returning
    the frequency bin with the highest power (excluding the DC term).

    References
    ----------
    Westerweel & Scarano (2005), Exp. Fluids 39, 1096-1100 (SNR definition).
    Hargather & Settles (2012), Appl. Opt. 51, 4443-4450 (spatial frequency
    recommendations for BOS).
    """
    arr = pattern.astype(np.float64)
    H, W = arr.shape
    ws = window_size

    # ------------------------------------------------------------------
    # 1. Contrast  C = sigma / mu
    # ------------------------------------------------------------------
    mu = arr.mean()
    sigma = arr.std()
    contrast = float(sigma / mu) if mu > 1e-12 else 0.0

    # ------------------------------------------------------------------
    # 2. Dominant spatial frequency  (radially averaged PSD)
    # ------------------------------------------------------------------
    # Apply 2-D Hanning window to reduce spectral leakage
    win2d = np.outer(np.hanning(H), np.hanning(W))
    psd = np.abs(np.fft.fftshift(np.fft.fft2(arr * win2d))) ** 2

    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    freq_axis = np.fft.fftfreq(H)[: cy]  # positive-frequency half

    # Build radial bin indices
    yy, xx = np.indices((H, W), dtype=np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radial_bins = np.round(rr).astype(int)

    # Radially average PSD
    psd_radial = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.int64)
    for r_bin in range(max_r):
        mask = radial_bins == r_bin
        psd_radial[r_bin] = psd[mask].sum()
        counts[r_bin] = mask.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        psd_radial = np.where(counts > 0, psd_radial / counts, 0.0)

    # Exclude DC (bin 0)
    if max_r > 1:
        peak_bin = int(np.argmax(psd_radial[1:])) + 1
    else:
        peak_bin = 1

    # Convert bin number to cycles/pixel
    # Bin r corresponds to frequency r / max(H, W) cycles/px
    spatial_frequency_peak = float(peak_bin / max(H, W))

    # ------------------------------------------------------------------
    # 3. Cross-correlation peak SNR  (synthetic 1 px shift)
    # ------------------------------------------------------------------
    # Extract a central window
    r0 = max(0, cy - ws // 2)
    r1 = min(H, r0 + ws)
    c0 = max(0, cx - ws // 2)
    c1 = min(W, c0 + ws)

    if r1 - r0 < ws:
        r0 = max(0, H - ws)
        r1 = H
    if c1 - c0 < ws:
        c0 = max(0, W - ws)
        c1 = W

    patch_ref = arr[r0:r1, c0:c1].copy()

    # Synthetic 1 px horizontal shift via array roll
    patch_shifted = np.roll(patch_ref, shift=1, axis=1)

    # Zero-mean, unit-variance normalise
    def _norm(p: np.ndarray) -> np.ndarray:
        p = p - p.mean()
        s = p.std()
        return p / s if s > 1e-12 else p

    win1d = np.hanning(patch_ref.shape[0])
    win2d_w = np.outer(win1d, np.hanning(patch_ref.shape[1]))

    pr = _norm(patch_ref) * win2d_w
    ps = _norm(patch_shifted) * win2d_w

    # FFT cross-correlation
    corr = np.fft.fftshift(
        np.fft.ifft2(np.fft.fft2(ps) * np.conj(np.fft.fft2(pr))).real
    )

    peak_val = float(corr.max())

    # Background RMS: exclude 3-pixel radius around peak
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    yy_c, xx_c = np.indices(corr.shape)
    dist = np.sqrt((yy_c - peak_y) ** 2 + (xx_c - peak_x) ** 2)
    bg_mask = dist > 3.0
    bg_vals = corr[bg_mask]
    bg_rms = float(np.sqrt(np.mean(bg_vals ** 2))) if bg_vals.size > 0 else 1e-12

    if bg_rms < 1e-12:
        bg_rms = 1e-12

    peak_snr = float(20.0 * math.log10(abs(peak_val) / bg_rms + 1e-12))

    logger.debug(
        "Pattern quality: SNR=%.2f dB, contrast=%.4f, f_peak=%.4f cy/px",
        peak_snr, contrast, spatial_frequency_peak,
    )

    return PatternQualityMetric(
        peak_snr=peak_snr,
        contrast=contrast,
        spatial_frequency_peak=spatial_frequency_peak,
    )
