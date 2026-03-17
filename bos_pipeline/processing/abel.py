"""Abel inversion for axisymmetric BOS flows (e.g. vertical hydrogen jets).

Wraps the PyAbel library and provides:
* Automatic or manual symmetry-axis detection.
* Inversion of displacement magnitude or individual dx/dy components.
* Output: radial density gradient and optionally reconstructed density field
  (requires Gladstone-Dale constant and reference density).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

AbelMethod = Literal["three_point", "basex", "hansenlaw"]
AxisMode = Literal["auto", "manual"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AbelConfig:
    """Parameters for Abel inversion."""

    enabled: bool = False
    method: AbelMethod = "three_point"
    component: Literal["magnitude", "dx", "dy"] = "magnitude"
    axis_mode: AxisMode = "auto"
    axis_pos: Optional[int] = None     # pixel column; required for manual mode
    dr: float = 1.0                    # radial step [px]

    @classmethod
    def from_dict(cls, d: dict) -> "AbelConfig":
        return cls(
            enabled=d.get("enabled", False),
            method=d.get("method", "three_point"),
            component=d.get("component", "magnitude"),
            axis_mode=d.get("axis_mode", "auto"),
            axis_pos=d.get("axis_pos"),
            dr=d.get("dr", 1.0),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_symmetry_axis(field: np.ndarray) -> int:
    """Auto-detect the symmetry axis column by intensity centroid per row.

    Computes the intensity-weighted column centroid for each row and returns
    the median centroid as the axis position.  Works well for jet-like flows
    that are brightest along the centreline.

    Parameters
    ----------
    field:
        2-D array (H × W).

    Returns
    -------
    axis_col:
        Estimated symmetry axis column index.
    """
    H, W = field.shape
    cols = np.arange(W, dtype=np.float64)
    centroids = []
    for row in field:
        total = row.sum()
        if total > 0:
            centroids.append((row * cols).sum() / total)
    if not centroids:
        return W // 2
    axis_col = int(round(np.median(centroids)))
    logger.debug("Auto-detected symmetry axis at column %d.", axis_col)
    return axis_col


def abel_invert(
    dx: np.ndarray,
    dy: np.ndarray,
    config: Optional[AbelConfig] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply Abel inversion to a displacement field.

    Parameters
    ----------
    dx, dy:
        Horizontal and vertical displacement maps (H × W, float32).
    config:
        :class:`AbelConfig` instance.

    Returns
    -------
    inv_field:
        Abel-inverted field (radial density gradient), same shape as input.
    axis_col:
        The symmetry axis column that was used (wrapped in a 1-element array
        for easy inspection).

    Raises
    ------
    ImportError
        If PyAbel is not installed.
    """
    try:
        import abel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyAbel is required for Abel inversion. "
            "Install it with: pip install PyAbel"
        ) from exc

    cfg = config or AbelConfig()

    # ------------------------------------------------------------------
    # Select input component
    # ------------------------------------------------------------------
    if cfg.component == "dx":
        data = dx.copy()
    elif cfg.component == "dy":
        data = dy.copy()
    else:  # magnitude
        data = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)

    # ------------------------------------------------------------------
    # Axis detection
    # ------------------------------------------------------------------
    if cfg.axis_mode == "manual":
        if cfg.axis_pos is None:
            raise ValueError(
                "axis_pos must be set when axis_mode='manual'."
            )
        axis_col = int(cfg.axis_pos)
    else:
        axis_col = find_symmetry_axis(data)

    logger.info(
        "Abel inversion: method=%s, component=%s, axis_col=%d",
        cfg.method, cfg.component, axis_col,
    )

    # ------------------------------------------------------------------
    # Inversion row-by-row
    # ------------------------------------------------------------------
    H, W = data.shape
    inv_field = np.zeros_like(data, dtype=np.float64)

    method_map = {
        "three_point": "three_point",
        "basex": "basex",
        "hansenlaw": "hansenlaw",
    }
    abel_method = method_map.get(cfg.method, "three_point")

    # Build the right half of the projection (Abel expects a half-profile)
    # We fold the image at the symmetry axis and invert each row.
    for row_idx in range(H):
        row = data[row_idx, :]
        half_row = _extract_half_profile(row, axis_col)
        if half_row.size < 3:
            continue
        try:
            inv_row = abel.Transform(
                half_row.reshape(1, -1),
                method=abel_method,
                direction="inverse",
                verbose=False,
            ).transform.ravel()
        except Exception as exc:
            logger.debug("Abel row %d failed: %s", row_idx, exc)
            continue

        # Place back into the full-width array (mirrored)
        _place_half_profile(inv_field[row_idx, :], inv_row, axis_col)

    return inv_field.astype(np.float32), np.array([axis_col], dtype=np.int32)


def reconstruct_density(
    inv_field: np.ndarray,
    gladstone_dale: float,
    reference_density: float,
    dr_m: float = 1e-4,
) -> np.ndarray:
    """Integrate the radial density gradient to obtain the density field.

    Uses cumulative trapezoidal integration from the symmetry axis outwards.

    Parameters
    ----------
    inv_field:
        Abel-inverted radial density gradient field (H × W).
    gladstone_dale:
        Gladstone-Dale constant ``K`` [m³/kg].
    reference_density:
        Ambient density ``ρ₀`` [kg/m³].
    dr_m:
        Radial step size in metres.

    Returns
    -------
    density:
        Estimated density field [kg/m³] (H × W, float32).
    """
    from scipy.integrate import cumulative_trapezoid

    H, W = inv_field.shape
    density = np.full_like(inv_field, fill_value=reference_density, dtype=np.float64)

    for row_idx in range(H):
        grad = inv_field[row_idx, :]
        # Integrate from centre outwards — cumulative trapz
        integrated = cumulative_trapezoid(grad, dx=dr_m, initial=0.0)
        # Δn = K * Δρ  →  Δρ = Δn / K  (refractive index change → density)
        delta_rho = integrated / gladstone_dale
        density[row_idx, :] = reference_density + delta_rho

    return density.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_half_profile(row: np.ndarray, axis: int) -> np.ndarray:
    """Return the right half of *row* starting from *axis*.

    Pixels at *axis* and to the right are used.  If *axis* is to the right
    of centre, use the left half (flipped) instead.
    """
    W = len(row)
    left_len = axis
    right_len = W - axis

    if right_len >= left_len:
        return row[axis:].copy()
    else:
        # Flip left half
        return row[:axis + 1][::-1].copy()


def _place_half_profile(
    full_row: np.ndarray,
    half: np.ndarray,
    axis: int,
) -> None:
    """Write *half* into *full_row* symmetrically around *axis* (in-place)."""
    W = len(full_row)
    n = len(half)
    # Right side
    right_end = min(axis + n, W)
    full_row[axis:right_end] = half[:right_end - axis]
    # Left side (mirror)
    for i, val in enumerate(half):
        left_idx = axis - i
        if left_idx >= 0:
            full_row[left_idx] = val
