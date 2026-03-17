"""Preprocessing utilities for BOS frames.

Pipeline (all steps optional):
  1. Dark-frame subtraction
  2. Flat-field correction
  3. Background subtraction  (reference – measurement difference image)
  4. Gaussian pre-filter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Parameters controlling the preprocessing pipeline."""

    # Temporal averaging
    reference_avg_frames: int = 1      # number of reference frames to average
    measurement_avg_frames: int = 1    # number of measurement frames to average

    # Calibration frames (optional paths or pre-loaded arrays)
    flat_field: Optional[Union[str, Path, np.ndarray]] = None
    dark_frame: Optional[Union[str, Path, np.ndarray]] = None

    # Gaussian pre-filter sigma (pixels).  0 → disabled.
    gaussian_sigma: float = 1.5

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(
            reference_avg_frames=d.get("reference_avg_frames", 1),
            measurement_avg_frames=d.get("measurement_avg_frames", 1),
            flat_field=d.get("flat_field_path"),
            dark_frame=d.get("dark_frame_path"),
            gaussian_sigma=d.get("gaussian_sigma", 1.5),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess(
    reference: np.ndarray,
    measurement: np.ndarray,
    config: Optional[PreprocessConfig] = None,
    flat_field: Optional[np.ndarray] = None,
    dark_frame: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full preprocessing pipeline to a reference/measurement pair.

    Parameters
    ----------
    reference:
        2-D reference (background) frame, dtype float32 or float64.
    measurement:
        2-D measurement frame with the flow feature.
    config:
        :class:`PreprocessConfig` instance.  Defaults are used if ``None``.
    flat_field:
        Optional flat-field correction image (same shape as *reference*).
        Overrides ``config.flat_field`` if provided.
    dark_frame:
        Optional dark frame (same shape as *reference*).
        Overrides ``config.dark_frame`` if provided.

    Returns
    -------
    ref_proc, meas_proc:
        Preprocessed reference and measurement arrays (float32).
    """
    cfg = config or PreprocessConfig()

    ref = reference.astype(np.float64)
    meas = measurement.astype(np.float64)

    # ------------------------------------------------------------------
    # Resolve calibration frames
    # ------------------------------------------------------------------
    ff = _resolve_array(flat_field or cfg.flat_field, "flat-field")
    dk = _resolve_array(dark_frame or cfg.dark_frame, "dark frame")

    # ------------------------------------------------------------------
    # 1. Dark-frame subtraction
    # ------------------------------------------------------------------
    if dk is not None:
        logger.debug("Applying dark-frame subtraction.")
        ref = np.maximum(ref - dk, 0.0)
        meas = np.maximum(meas - dk, 0.0)

    # ------------------------------------------------------------------
    # 2. Flat-field correction:  img_corrected = img / flat_normalised
    # ------------------------------------------------------------------
    if ff is not None:
        logger.debug("Applying flat-field correction.")
        ff_norm = _safe_normalise(ff)
        ref = ref / ff_norm
        meas = meas / ff_norm

    # ------------------------------------------------------------------
    # 3. Gaussian pre-filter
    # ------------------------------------------------------------------
    sigma = cfg.gaussian_sigma
    if sigma > 0:
        logger.debug("Gaussian pre-filter sigma=%.2f px.", sigma)
        ref = gaussian_filter(ref, sigma=sigma)
        meas = gaussian_filter(meas, sigma=sigma)

    return ref.astype(np.float32), meas.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal averaging helpers
# ---------------------------------------------------------------------------

def average_frames(
    reader,
    indices,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Stream-average frames from any :class:`~bos_pipeline.io.base.CameraReader`.

    Parameters
    ----------
    reader:
        An open CameraReader instance.
    indices:
        Sequence of integer frame indices to average.
    dtype:
        Accumulator dtype (float64 recommended to avoid overflow).
    """
    return reader.get_average(indices, dtype=dtype)


def build_reference(
    reader,
    ref_index: int,
    n_avg: int = 1,
) -> np.ndarray:
    """Build a temporal-average reference frame.

    Parameters
    ----------
    reader:
        Open CameraReader.
    ref_index:
        Central frame index (or first frame of the averaging window).
    n_avg:
        Number of consecutive frames to average, starting from *ref_index*.
    """
    indices = range(ref_index, ref_index + n_avg)
    meta = reader.metadata
    total = meta.total_frames or 0
    # Clamp to valid range
    indices = [i for i in indices if 0 <= i < total]
    if not indices:
        raise ValueError(
            f"No valid frames in range [{ref_index}, {ref_index + n_avg})"
        )
    logger.debug("Building reference from frames %s.", list(indices))
    return reader.get_average(indices)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_array(
    source: Optional[Union[str, Path, np.ndarray]],
    label: str,
) -> Optional[np.ndarray]:
    """Load *source* as a float64 array, or return ``None``."""
    if source is None:
        return None
    if isinstance(source, np.ndarray):
        return source.astype(np.float64)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    # Support .npy and TIFF via tifffile
    if path.suffix.lower() == ".npy":
        return np.load(str(path)).astype(np.float64)
    try:
        import tifffile  # type: ignore[import]
        return tifffile.imread(str(path)).astype(np.float64)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {label} from {path}: {exc}"
        ) from exc


def _safe_normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise *arr* to its mean, replacing zeros with 1 to avoid division."""
    mean_val = arr.mean()
    if mean_val == 0.0:
        logger.warning("Flat-field mean is zero — skipping normalisation.")
        return np.ones_like(arr)
    normed = arr / mean_val
    # Replace near-zero pixels to avoid inf/nan
    normed = np.where(normed < 1e-6, 1.0, normed)
    return normed
