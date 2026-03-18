"""Gas concentration measurement from BOS images (axisymmetric jets).

Algorithm (per-row Abel inversion):
------------------------------------
1.  Convert pixel displacement to deflection angle:
        ε_x(ρ) = dx(ρ) · (mm_per_px / Z_f_mm)          [rad]
2.  Integrate ε_x from the far side inward to build the line-of-sight
    refractive-index projection G (Abel transform of Δn):
        G(ρ) = -∫_ρ^∞ ε_x(ρ') dρ'   ≈  -cumsum from far edge
3.  Abel-invert G (PyAbel, row by row on a full symmetric image) to get
    the radial refractive-index change Δn(r).
4.  Convert to volume-fraction concentration:
        c(r) = Δn(r) / (n_gas − n_ambient)    clipped to [0, 1]

Temperature / pressure dependence (Gladstone-Dale):
----------------------------------------------------
The refractive index of any ideal gas depends on T and P via:
    n(T, P) = 1 + C_GD · ρ(T, P) = 1 + C_GD · P · M / (R · T)

where C_GD [m³/kg] is the Gladstone-Dale constant (gas property),
M [kg/mol] is the molar mass, R = 8.314 J/(mol·K).

When a gas type is selected from the built-in database, n_gas and
n_ambient are computed automatically from T and P — no need to look up
refractive-index tables.

Reference:
    Venkatakrishnan & Meier (2004), "Density measurements using the
    Background Oriented Schlieren technique", Exp. Fluids 37, 237-247.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Universal gas constant [J/(mol·K)]
_R = 8.314462618

# ---------------------------------------------------------------------------
# Gas Gladstone-Dale database
# ---------------------------------------------------------------------------
# Values: C_GD [m³/kg], M [kg/mol]
# C_GD verified from published n values at STP (0 °C, 101 325 Pa):
#   C_GD = (n_STP - 1) / ρ_STP   where ρ_STP = P_STP · M / (R · T_STP)
#
#  Gas   n_STP        ρ_STP [kg/m³]  C_GD [m³/kg]
#  H₂    1.000 132    0.0899         1.468e-3
#  N₂    1.000 297    1.2506         2.375e-4
#  He    1.000 035    0.1786         1.960e-5  (approx; wavelength-dependent)
#  CO₂   1.000 449    1.9769         2.271e-4
#  Air   1.000 293    1.2929         2.267e-4

GAS_DB: Dict[str, Dict] = {
    "H2":  {"C_GD": 1.468e-3,  "M": 2.01588e-3,  "label": "H₂  (Hydrogen)"},
    "N2":  {"C_GD": 2.375e-4,  "M": 28.0134e-3,  "label": "N₂  (Nitrogen)"},
    "He":  {"C_GD": 1.960e-5,  "M": 4.00260e-3,  "label": "He  (Helium)"},
    "CO2": {"C_GD": 2.271e-4,  "M": 44.0095e-3,  "label": "CO₂ (Carbon dioxide)"},
    "air": {"C_GD": 2.267e-4,  "M": 28.9647e-3,  "label": "Air"},
}

# Gases that make sense as the jet gas (exclude 'air' from jet choices
# since ambient default is air)
JET_GASES = ["H2", "N2", "He", "CO2", "custom"]
AMBIENT_GASES = ["air", "N2", "custom"]


# ---------------------------------------------------------------------------
# Gladstone-Dale helper
# ---------------------------------------------------------------------------


def n_from_gladstone_dale(
    C_GD: float,
    M_kg_per_mol: float,
    T_K: float,
    P_Pa: float,
) -> float:
    """Return refractive index n from the Gladstone-Dale relation.

    Parameters
    ----------
    C_GD:
        Gladstone-Dale constant [m³/kg].
    M_kg_per_mol:
        Molar mass [kg/mol].
    T_K:
        Temperature [K].
    P_Pa:
        Pressure [Pa].
    """
    rho = P_Pa * M_kg_per_mol / (_R * T_K)   # ideal-gas density [kg/m³]
    return 1.0 + C_GD * rho


def compute_n_pair(
    gas_type: str,
    ambient_gas: str,
    temperature_K: float,
    pressure_Pa: float,
    n_gas_custom: float,
    n_ambient_custom: float,
) -> Tuple[float, float]:
    """Return (n_gas, n_ambient) using the Gladstone-Dale database.

    Falls back to the user-supplied custom values when gas_type or
    ambient_gas == 'custom'.
    """
    if gas_type == "custom":
        n_gas = n_gas_custom
    else:
        g = GAS_DB[gas_type]
        n_gas = n_from_gladstone_dale(g["C_GD"], g["M"], temperature_K, pressure_Pa)

    if ambient_gas == "custom":
        n_ambient = n_ambient_custom
    else:
        a = GAS_DB[ambient_gas]
        n_ambient = n_from_gladstone_dale(a["C_GD"], a["M"], temperature_K, pressure_Pa)

    return n_gas, n_ambient


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConcentrationConfig:
    """Parameters for gas concentration measurement via Abel inversion."""

    enabled: bool = False

    # Spatial calibration
    mm_per_px: float = 0.1      # [mm/px]  pixel size at the background screen
    Z_f_mm: float = 1000.0      # [mm]     camera-to-background distance

    # Gas selection for implicit n computation (Gladstone-Dale)
    gas_type: str = "H2"         # key in GAS_DB, or "custom"
    ambient_gas: str = "air"     # key in GAS_DB, or "custom"

    # Ambient conditions used for Gladstone-Dale calculation
    temperature_K: float = 293.15   # 20 °C
    pressure_Pa: float = 101325.0   # 1 atm

    # Manual override (only used when gas_type / ambient_gas == "custom")
    n_gas_custom: float = 1.000132
    n_ambient_custom: float = 1.000293

    # Which displacement component carries the radial signal
    # Vertical jet (axis along Y)  → 'dx' (horizontal deflection)
    # Horizontal jet (axis along X) → 'dy'
    component: Literal["dx", "dy"] = "dx"

    # PyAbel inversion method
    abel_method: str = "three_point"

    # Symmetry axis
    axis_mode: Literal["auto", "manual"] = "auto"
    axis_pos: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ConcentrationConfig":
        return cls(
            enabled=d.get("enabled", False),
            mm_per_px=d.get("mm_per_px", 0.1),
            Z_f_mm=d.get("Z_f_mm", 1000.0),
            gas_type=d.get("gas_type", "H2"),
            ambient_gas=d.get("ambient_gas", "air"),
            temperature_K=d.get("temperature_K", 293.15),
            pressure_Pa=d.get("pressure_Pa", 101325.0),
            n_gas_custom=d.get("n_gas_custom", 1.000132),
            n_ambient_custom=d.get("n_ambient_custom", 1.000293),
            component=d.get("component", "dx"),
            abel_method=d.get("abel_method", "three_point"),
            axis_mode=d.get("axis_mode", "auto"),
            axis_pos=d.get("axis_pos"),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_concentration(
    dx: np.ndarray,
    dy: np.ndarray,
    config: Optional[ConcentrationConfig] = None,
) -> Tuple[np.ndarray, int, float, float]:
    """Compute gas volume-fraction concentration from a BOS displacement field.

    Parameters
    ----------
    dx, dy:
        Full-resolution displacement fields (H × W, float32) in pixels.
    config:
        :class:`ConcentrationConfig`.

    Returns
    -------
    concentration:
        Gas concentration field (H × W, float32) in [0, 1].
    axis_col:
        Symmetry axis column used.
    n_gas:
        Refractive index of the pure jet gas (computed or custom).
    n_ambient:
        Refractive index of the ambient (computed or custom).

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

    # ------------------------------------------------------------------
    # 1. Compute refractive indices (Gladstone-Dale or custom)
    # ------------------------------------------------------------------
    n_gas, n_ambient = compute_n_pair(
        cfg.gas_type,
        cfg.ambient_gas,
        cfg.temperature_K,
        cfg.pressure_Pa,
        cfg.n_gas_custom,
        cfg.n_ambient_custom,
    )
    dn_diff = n_gas - n_ambient

    logger.info(
        "Concentration: gas=%s n_gas=%.6f, ambient=%s n_ambient=%.6f, "
        "Δn=%.2e, T=%.1f K, P=%.0f Pa",
        cfg.gas_type, n_gas, cfg.ambient_gas, n_ambient,
        dn_diff, cfg.temperature_K, cfg.pressure_Pa,
    )

    if abs(dn_diff) < 1e-12:
        raise ValueError(
            f"n_gas ({n_gas:.6f}) ≈ n_ambient ({n_ambient:.6f}): "
            "cannot compute concentration (division by zero)."
        )

    # ------------------------------------------------------------------
    # 2. Select displacement component → deflection angle
    # ------------------------------------------------------------------
    raw = dx if cfg.component == "dx" else dy
    sensitivity = cfg.mm_per_px / cfg.Z_f_mm   # [rad/px]
    eps = raw.astype(np.float64) * sensitivity   # deflection angle [rad]

    logger.info(
        "Sensitivity: %.4f mm/px / %.1f mm = %.2e rad/px",
        cfg.mm_per_px, cfg.Z_f_mm, sensitivity,
    )

    # ------------------------------------------------------------------
    # 3. Detect symmetry axis
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
    r_half = W - axis_col

    # ------------------------------------------------------------------
    # 4. Build G(ρ) = −∫_ρ^∞ ε_x dρ'  (Abel projection of Δn)
    # ------------------------------------------------------------------
    eps_right = eps[:, axis_col: axis_col + r_half]
    G_right = -np.cumsum(eps_right[:, ::-1], axis=1)[:, ::-1]

    # Full symmetric image for PyAbel (mirrors right half)
    G_sym = np.hstack([G_right[:, ::-1][:, :-1], G_right])

    # ------------------------------------------------------------------
    # 5. Abel inversion → Δn(r)
    # ------------------------------------------------------------------
    dn_sym = _abel_invert_image(G_sym, cfg.abel_method)

    # ------------------------------------------------------------------
    # 6. c = Δn / (n_gas − n_ambient),  clipped to [0, 1]
    # ------------------------------------------------------------------
    conc_sym = np.clip(dn_sym / dn_diff, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # 7. Place back into full-resolution (H × W) canvas
    # ------------------------------------------------------------------
    concentration = np.zeros((H, W), dtype=np.float32)
    sym_w = conc_sym.shape[1]
    sym_centre = r_half - 1          # G_sym column that maps to axis_col
    x_start = axis_col - sym_centre
    x_end   = x_start + sym_w
    src_l = max(0, -x_start)
    src_r = sym_w - max(0, x_end - W)
    dst_l = max(0, x_start)
    dst_r = min(W, x_end)
    concentration[:, dst_l:dst_r] = conc_sym[:, src_l:src_r]

    logger.info(
        "Concentration: max=%.4f, mean(>0)=%.4f",
        concentration.max(),
        float(concentration[concentration > 1e-4].mean())
        if (concentration > 1e-4).any() else 0.0,
    )
    return concentration, axis_col, n_gas, n_ambient


# Keep old name as alias for backwards compat
compute_h2_concentration = compute_concentration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abel_invert_image(image: np.ndarray, method: str = "three_point") -> np.ndarray:
    """Invert a full symmetric Abel-projected image using PyAbel.

    Attempts bulk 2-D transform first; falls back to row-by-row on failure.
    """
    import abel  # type: ignore[import]

    H, W = image.shape
    inv = np.zeros_like(image, dtype=np.float64)

    try:
        inv[:] = abel.Transform(
            image,
            method=method,
            direction="inverse",
            verbose=False,
        ).transform
    except Exception as exc:
        logger.warning("Bulk Abel inversion failed (%s); row-by-row fallback.", exc)
        for row_idx in range(H):
            try:
                inv[row_idx] = abel.Transform(
                    image[row_idx].reshape(1, -1),
                    method=method,
                    direction="inverse",
                    verbose=False,
                ).transform.ravel()
            except Exception as row_exc:
                logger.debug("Row %d Abel failed: %s", row_idx, row_exc)

    return inv
