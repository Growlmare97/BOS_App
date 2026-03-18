"""H₂ concentration measurement from BOS images (axisymmetric jets).

Algorithm (per-row Abel inversion):
------------------------------------
1.  Convert pixel displacement to deflection angle:
        ε_x(ρ) = dx(ρ) · (mm_per_px / Z_f_mm)          [rad]
2.  Integrate ε_x from the far side inward to build the line-of-sight
    refractive-index projection G (Abel transform of Δn):
        G(ρ) = -∫_ρ^∞ ε_x(ρ') dρ'   ≈  -cumsum from far edge
3.  Abel-invert G (PyAbel, row by row on a full symmetric image) to get
    the radial refractive-index change Δn(r).
4.  Convert to molar concentration fraction:
        c_H₂(r) = Δn(r) / (n_H₂ − n_air)    clipped to [0, 1]

Reference:
    Surber et al., "Evaluation of hydrogen concentration from BOS images",
    axisymmetric hydrogen jet study.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConcentrationConfig:
    """Parameters for H₂ concentration measurement."""

    enabled: bool = False

    # Spatial calibration
    mm_per_px: float = 0.1      # [mm/px]  pixel size at the background screen
    Z_f_mm: float = 1000.0      # [mm]     camera-to-background distance

    # Refractive indices at measurement conditions (default: H₂ in air, STP)
    n_gas: float = 1.000132     # H₂ refractive index
    n_ambient: float = 1.000293 # air refractive index

    # Which displacement component carries the radial signal
    # For a *vertical* jet (axis along Y): use 'dx' (horizontal deflection)
    # For a *horizontal* jet (axis along X): use 'dy'
    component: Literal["dx", "dy"] = "dx"

    # PyAbel inversion method
    abel_method: str = "three_point"

    # Symmetry axis
    axis_mode: Literal["auto", "manual"] = "auto"
    axis_pos: Optional[int] = None   # column index when axis_mode == "manual"

    @classmethod
    def from_dict(cls, d: dict) -> "ConcentrationConfig":
        return cls(
            enabled=d.get("enabled", False),
            mm_per_px=d.get("mm_per_px", 0.1),
            Z_f_mm=d.get("Z_f_mm", 1000.0),
            n_gas=d.get("n_gas", 1.000132),
            n_ambient=d.get("n_ambient", 1.000293),
            component=d.get("component", "dx"),
            abel_method=d.get("abel_method", "three_point"),
            axis_mode=d.get("axis_mode", "auto"),
            axis_pos=d.get("axis_pos"),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_h2_concentration(
    dx: np.ndarray,
    dy: np.ndarray,
    config: Optional[ConcentrationConfig] = None,
) -> Tuple[np.ndarray, int]:
    """Compute volume-fraction H₂ concentration map from a BOS displacement field.

    Parameters
    ----------
    dx, dy:
        Full-resolution displacement fields (H × W, float32) in pixels.
    config:
        :class:`ConcentrationConfig`.

    Returns
    -------
    concentration:
        H₂ concentration field (H × W, float32) in the range [0, 1].
    axis_col:
        Symmetry axis column that was used.

    Raises
    ------
    ImportError
        If PyAbel is not installed.
    """
    try:
        import abel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyAbel is required for concentration measurement. "
            "Install with: pip install PyAbel"
        ) from exc

    cfg = config or ConcentrationConfig()
    dn_diff = cfg.n_gas - cfg.n_ambient   # typically negative for H₂ in air

    if abs(dn_diff) < 1e-12:
        raise ValueError(
            "n_gas == n_ambient: cannot compute concentration (division by zero)."
        )

    # ------------------------------------------------------------------
    # 1. Select displacement component and convert to deflection angle
    # ------------------------------------------------------------------
    raw = dx if cfg.component == "dx" else dy
    sensitivity = cfg.mm_per_px / cfg.Z_f_mm   # [rad/px]
    eps = raw.astype(np.float64) * sensitivity   # deflection angle [rad]

    logger.info(
        "Concentration: component=%s, mm_per_px=%.4f, Z_f=%.1f mm, "
        "sensitivity=%.2e rad/px",
        cfg.component, cfg.mm_per_px, cfg.Z_f_mm, sensitivity,
    )

    # ------------------------------------------------------------------
    # 2. Detect symmetry axis
    # ------------------------------------------------------------------
    from bos_pipeline.processing.abel import find_symmetry_axis

    if cfg.axis_mode == "manual":
        if cfg.axis_pos is None:
            raise ValueError("axis_pos must be set when axis_mode='manual'.")
        axis_col = int(cfg.axis_pos)
    else:
        axis_col = find_symmetry_axis(np.abs(eps))

    logger.info("Symmetry axis at column %d.", axis_col)

    H, W = eps.shape

    # ------------------------------------------------------------------
    # 3. Build G(ρ) = −∫_ρ^∞ ε_x dρ' (Abel projection of Δn)
    #    by cumulative-summing ε_x from the far (right) edge inward.
    #    Then mirror to a full symmetric image for PyAbel.
    # ------------------------------------------------------------------
    r_half = W - axis_col          # pixels from axis to right edge

    # Right half: columns [axis_col, axis_col + r_half)
    eps_right = eps[:, axis_col: axis_col + r_half]   # shape (H, r_half)

    # G(ρ) = −cumsum from far edge inward (flip → cumsum → flip back)
    G_right = -np.cumsum(eps_right[:, ::-1], axis=1)[:, ::-1]   # (H, r_half)

    # Mirror: build a full (H × 2*r_half − 1) symmetric image
    # (PyAbel expects symmetry about col 0 for the three_point method with
    #  the 'forward'/'inverse' transform on a half-image, OR full symmetric)
    G_sym = np.hstack([G_right[:, ::-1][:, :-1], G_right])  # (H, 2*r_half-1)

    # ------------------------------------------------------------------
    # 4. Abel inversion of G_sym → Δn radial profile
    # ------------------------------------------------------------------
    dn_sym = _abel_invert_image(G_sym, cfg.abel_method)   # (H, 2*r_half-1)

    # ------------------------------------------------------------------
    # 5. Concentration: c = Δn / (n_gas − n_ambient)
    # ------------------------------------------------------------------
    conc_sym = np.clip(dn_sym / dn_diff, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 6. Place result back into a full-resolution (H × W) canvas
    # ------------------------------------------------------------------
    concentration = np.zeros((H, W), dtype=np.float32)
    sym_w = conc_sym.shape[1]

    # Centre of G_sym maps to axis_col in the original image
    sym_centre = r_half - 1  # index in G_sym that corresponds to axis_col
    x_start = axis_col - sym_centre
    x_end   = x_start + sym_w
    # Clip to valid range
    src_l = max(0, -x_start)
    src_r = sym_w - max(0, x_end - W)
    dst_l = max(0, x_start)
    dst_r = min(W, x_end)
    concentration[:, dst_l:dst_r] = conc_sym[:, src_l:src_r]

    logger.info(
        "Concentration: max=%.4f, mean=%.4f",
        concentration.max(), concentration.mean(),
    )
    return concentration, axis_col


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abel_invert_image(image: np.ndarray, method: str = "three_point") -> np.ndarray:
    """Invert a full symmetric Abel-projected image row by row using PyAbel.

    Parameters
    ----------
    image:
        2-D array (H × W) assumed symmetric about the centre column.
    method:
        PyAbel method string.

    Returns
    -------
    inv:
        Inverted image, same shape.
    """
    import abel  # type: ignore[import]

    H, W = image.shape
    inv = np.zeros_like(image, dtype=np.float64)

    # PyAbel can handle 2-D arrays directly (all rows at once) for methods
    # that support it; three_point and hansenlaw do.  Use the bulk transform.
    try:
        result = abel.Transform(
            image,
            method=method,
            direction="inverse",
            verbose=False,
        ).transform
        inv[:] = result
    except Exception as exc:
        logger.warning(
            "Bulk Abel inversion failed (%s); falling back to row-by-row.", exc
        )
        for row_idx in range(H):
            row = image[row_idx, :]
            try:
                row_inv = abel.Transform(
                    row.reshape(1, -1),
                    method=method,
                    direction="inverse",
                    verbose=False,
                ).transform.ravel()
                inv[row_idx, :] = row_inv
            except Exception as row_exc:
                logger.debug("Row %d Abel failed: %s", row_idx, row_exc)

    return inv
