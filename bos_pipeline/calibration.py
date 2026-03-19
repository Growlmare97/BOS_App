"""Physical calibration for BOS measurements.

Converts pixel-space displacements to physical angular deflections,
density gradients, and refractive-index fields.

This module is concerned with *physical unit conversion* only and is
intentionally separate from :mod:`bos_pipeline.processing.concentration`,
which handles Gladstone-Dale-based gas concentration via Abel inversion.
Here the focus is on the camera/optical geometry (sensitivity, working
distances, magnification) and the mapping from pixel displacement to
physical quantities.

References
----------
Settles, G.S. (2001). Schlieren and Shadowgraph Techniques. Springer.
    Chapter 8 covers BOS sensitivity and the geometry of background-oriented
    schlieren.

Meier, G.E.A. (2002). Computerized background-oriented schlieren.
    Exp. Fluids 33, 181–187.
    DOI: 10.1007/s00348-002-0450-7
    Equations (3) and (4) are the basis for the pixel-to-angle and
    angle-to-density-gradient conversions implemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BOSCalibrationConfig:
    """Physical parameters describing the BOS optical set-up.

    All length quantities are in **millimetres** unless stated otherwise.

    Parameters
    ----------
    pixel_pitch_mm:
        Physical size of one camera sensor pixel [mm].  Also called the
        pixel size or pixel pitch.  Typical values range from 0.003 mm
        (3 µm) to 0.020 mm (20 µm).
    magnification:
        Lens magnification M = image_size / object_size (dimensionless).
        For most long-working-distance BOS set-ups M < 1.
    focal_length_mm:
        Effective focal length of the imaging lens [mm].
    ZD_mm:
        Distance from the camera to the background (dot pattern) [mm].
        This is Z_D in Meier (2002).
    ZA_mm:
        Distance from the camera to the object (region of interest) [mm].
        This is Z_A in Meier (2002).  Must satisfy ZA_mm < ZD_mm.
    gladstone_dale_K:
        Gladstone-Dale constant K = (n − 1) / ρ [m³/kg] for the gas of
        interest.  Default is for molecular hydrogen H₂.
        Common values:
          * H₂  : 1.56e-4 m³/kg  (or 1.468e-3 at standard conditions,
                   value depends on wavelength and source; 1.56e-4 is the
                   value cited by Settles (2001) for visible light)
          * Air : 2.27e-4 m³/kg
          * He  : 1.96e-5 m³/kg
    ambient_density_kg_m3:
        Density of the ambient (undisturbed) fluid [kg/m³].
        Default is air at 20 °C, 1 atm.
    reference_n:
        Refractive index of the undisturbed ambient medium.
        Default is air at standard temperature and pressure.
    """

    pixel_pitch_mm: float = 0.005       # sensor pixel size [mm]
    magnification: float = 1.0          # lens magnification (image / object)
    focal_length_mm: float = 50.0       # lens focal length [mm]
    ZD_mm: float = 1000.0               # camera-to-background distance [mm]  (Z_D, Meier 2002)
    ZA_mm: float = 500.0                # camera-to-object distance [mm]       (Z_A, Meier 2002)
    gladstone_dale_K: float = 1.56e-4   # Gladstone-Dale constant for H₂ [m³/kg]
    ambient_density_kg_m3: float = 1.204  # air density at 20 °C, 1 atm [kg/m³]
    reference_n: float = 1.000293       # ambient refractive index (air at STP)

    @classmethod
    def from_dict(cls, d: dict) -> "BOSCalibrationConfig":
        """Construct from a plain dictionary (e.g. loaded from YAML)."""
        return cls(
            pixel_pitch_mm=d.get("pixel_pitch_mm", 0.005),
            magnification=d.get("magnification", 1.0),
            focal_length_mm=d.get("focal_length_mm", 50.0),
            ZD_mm=d.get("ZD_mm", 1000.0),
            ZA_mm=d.get("ZA_mm", 500.0),
            gladstone_dale_K=d.get("gladstone_dale_K", 1.56e-4),
            ambient_density_kg_m3=d.get("ambient_density_kg_m3", 1.204),
            reference_n=d.get("reference_n", 1.000293),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pixel_to_deflection_angle(
    dx_px: np.ndarray,
    config: BOSCalibrationConfig,
) -> np.ndarray:
    """Convert pixel displacement to angular deflection [rad].

    Uses the BOS sensitivity formula from Settles (2001, Ch. 8) and
    Meier (2002, Eq. 3):

        ε_x = dx_px × pixel_pitch_mm / (focal_length_mm × magnification)

    where ε_x is the ray deflection angle in radians (mm/mm is
    dimensionless, giving radians directly).

    Parameters
    ----------
    dx_px:
        Pixel displacement array (any shape, float32/64).  Typically the
        horizontal component ``dx`` from
        :func:`bos_pipeline.processing.displacement.compute_displacement`.
    config:
        :class:`BOSCalibrationConfig`.

    Returns
    -------
    epsilon:
        Angular deflection field in **radians**, same shape as *dx_px*.

    Notes
    -----
    The formula is derived from the thin-lens image-shift relationship.
    A ray deflected by angle ε at the object plane produces a shift of
    Δx = f × M × ε on the sensor.  Inverting: ε = Δx_sensor / (f × M)
    where Δx_sensor = dx_px × pixel_pitch_mm.
    Ref: Meier (2002) Eq. 3; Settles (2001) Ch. 8.
    """
    # ε = dx_px × p / (f × M)   [rad]   — Meier (2002) Eq. 3
    epsilon = (
        dx_px.astype(np.float64)
        * config.pixel_pitch_mm
        / (config.focal_length_mm * config.magnification)
    )
    logger.debug(
        "pixel_to_deflection_angle: pixel_pitch=%.4f mm, f=%.1f mm, M=%.3f "
        "→ scale factor=%.4e rad/px",
        config.pixel_pitch_mm,
        config.focal_length_mm,
        config.magnification,
        config.pixel_pitch_mm / (config.focal_length_mm * config.magnification),
    )
    return epsilon.astype(np.float32)


def deflection_to_density_gradient(
    epsilon_x: np.ndarray,
    config: BOSCalibrationConfig,
) -> np.ndarray:
    """Convert angular deflection [rad] to density gradient [kg/m⁴].

    Two-step conversion following Meier (2002, Eq. 4):

    Step 1 — refractive-index gradient integrated along the line of sight
    (approximated as the lateral gradient of the path-integrated field):

        dn/dx ≈ ε_x / Z_D        [1/mm]

    where Z_D is the camera-to-background distance.  Convert to SI:
        dn/dx  [1/m] = (dn/dx [1/mm]) × 1000

    Step 2 — density gradient via Gladstone-Dale relation
    (n − 1 = K × ρ  →  dn/dx = K × dρ/dx):

        dρ/dx = (dn/dx) / K      [kg/m⁴]

    Parameters
    ----------
    epsilon_x:
        Angular deflection field [rad] from :func:`pixel_to_deflection_angle`.
    config:
        :class:`BOSCalibrationConfig`.

    Returns
    -------
    drho_dx:
        Density gradient field [kg/m⁴], same shape as *epsilon_x*.

    Notes
    -----
    This gives the *path-integrated* density gradient divided by the
    integration path length (an average gradient), not the local value.
    For local values an Abel inversion is required (axisymmetric flows)
    or tomographic reconstruction (3-D flows).
    Ref: Meier (2002) Eq. 4.
    """
    # Step 1: dn/dx [1/mm] = ε_x / Z_D_mm   — Meier (2002) Eq. 4
    dndx_per_mm = epsilon_x.astype(np.float64) / config.ZD_mm
    # Convert [1/mm] → [1/m]
    dndx_per_m = dndx_per_mm * 1.0e3

    # Step 2: dρ/dx [kg/m⁴] = (dn/dx [1/m]) / K [m³/kg]
    drho_dx = dndx_per_m / config.gladstone_dale_K

    logger.debug(
        "deflection_to_density_gradient: ZD=%.1f mm, K=%.3e m³/kg "
        "→ max |dρ/dx|=%.3e kg/m⁴",
        config.ZD_mm, config.gladstone_dale_K, np.abs(drho_dx).max(),
    )
    return drho_dx.astype(np.float32)


def density_gradient_to_concentration(
    dndx: np.ndarray,
    config: BOSCalibrationConfig,
) -> np.ndarray:
    """Estimate H₂ concentration from the refractive-index gradient field.

    For a binary mixture of H₂ in air, the refractive-index perturbation
    relative to the ambient is related to the H₂ mass-density perturbation
    through the Gladstone-Dale relation:

        Δn = K_H2 × Δρ_H2

    This function integrates *dndx* along the first axis (rows) via a
    cumulative sum to estimate Δn(x, y), then converts to a dimensionless
    volume-fraction concentration:

        c(x, y) = Δn(x, y) / (n_H2_pure − n_ambient)

    where ``n_H2_pure = 1 + K × ρ_H2_pure`` and
    ``ρ_H2_pure = ambient_density × (M_H2 / M_air)`` (simplified).

    .. note::
        This is a *simplified* 1-D integration and is only valid as a first
        approximation for planar (non-axisymmetric) flows or as a qualitative
        indicator.  For quantitative axisymmetric jets use
        :func:`bos_pipeline.processing.concentration.compute_concentration`
        with full Abel inversion.

    Parameters
    ----------
    dndx:
        Refractive-index gradient field [1/m], e.g. from
        :func:`deflection_to_density_gradient` × K.  Shape: (H, W).
    config:
        :class:`BOSCalibrationConfig`.

    Returns
    -------
    concentration:
        Estimated H₂ volume-fraction field [0, 1], shape (H, W), float32.
    """
    # Approximate pixel pitch in metres for integration step
    dx_m = config.pixel_pitch_mm * 1.0e-3  # [m]

    # Integrate along columns (X direction) to get Δn
    delta_n = np.cumsum(dndx.astype(np.float64), axis=1) * dx_m  # [dimensionless]

    # Pure H₂ refractive index (approximate at ambient conditions)
    # Using Gladstone-Dale: n_H2_pure = 1 + K × ρ_H2_pure
    # ρ_H2_pure estimated from ambient density rescaled by molar mass ratio
    M_H2 = 2.01588e-3   # [kg/mol]
    M_air = 28.9647e-3  # [kg/mol]
    rho_H2_pure = config.ambient_density_kg_m3 * (M_H2 / M_air)  # [kg/m³]
    n_H2_pure = 1.0 + config.gladstone_dale_K * rho_H2_pure
    dn_diff = n_H2_pure - config.reference_n

    logger.debug(
        "density_gradient_to_concentration: n_H2_pure=%.6f, n_ambient=%.6f, "
        "Δn=%.2e",
        n_H2_pure, config.reference_n, dn_diff,
    )

    if abs(dn_diff) < 1e-12:
        raise ValueError(
            "n_H2_pure ≈ n_ambient: cannot compute concentration (Δn ≈ 0). "
            "Check gladstone_dale_K and reference_n values."
        )

    concentration = np.clip(delta_n / dn_diff, 0.0, 1.0).astype(np.float32)
    return concentration


def compute_sensitivity(config: BOSCalibrationConfig) -> float:
    """Return the BOS optical sensitivity in pixels per radian [px/rad].

    Definition (Settles 2001, Ch. 8):

        S = f × M / pixel_pitch

    A higher sensitivity means a larger pixel displacement per unit
    deflection angle, i.e. the system is more sensitive to density
    gradients.

    Parameters
    ----------
    config:
        :class:`BOSCalibrationConfig`.

    Returns
    -------
    sensitivity:
        Sensitivity in [px/rad].  This is the inverse of the scale factor
        used in :func:`pixel_to_deflection_angle`.
    """
    # S = f × M / p   [px/rad]   — Settles (2001) Ch. 8
    sensitivity = (config.focal_length_mm * config.magnification) / config.pixel_pitch_mm
    logger.debug(
        "BOS sensitivity: f=%.1f mm, M=%.3f, p=%.4f mm → S=%.1f px/rad",
        config.focal_length_mm, config.magnification,
        config.pixel_pitch_mm, sensitivity,
    )
    return float(sensitivity)


def describe_calibration(config: BOSCalibrationConfig) -> Dict[str, object]:
    """Return a dict of all derived calibration parameters for logging.

    Useful for writing to a JSON processing log via
    :func:`bos_pipeline.export.write_log`.

    Parameters
    ----------
    config:
        :class:`BOSCalibrationConfig`.

    Returns
    -------
    info:
        Dictionary containing:

        * ``sensitivity_px_per_rad`` — optical sensitivity [px/rad]
        * ``scale_rad_per_px`` — deflection per pixel [rad/px]
        * ``working_distance_mm`` — camera-to-object distance Z_A [mm]
        * ``background_distance_mm`` — camera-to-background distance Z_D [mm]
        * ``ZA_over_ZD`` — geometric ratio Z_A / Z_D (BOS fringe visibility)
        * ``magnification`` — lens magnification (dimensionless)
        * ``focal_length_mm`` — lens focal length [mm]
        * ``pixel_pitch_mm`` — sensor pixel pitch [mm]
        * ``gladstone_dale_K_m3_per_kg`` — K constant [m³/kg]
        * ``ambient_density_kg_m3`` — ambient density [kg/m³]
        * ``reference_n`` — ambient refractive index
        * ``min_detectable_dndx_per_m`` — minimum detectable refractive-index
          gradient [1/m] assuming 0.1 px displacement resolution
    """
    sensitivity = compute_sensitivity(config)
    scale_rad_per_px = 1.0 / sensitivity if sensitivity != 0.0 else float("inf")

    # Minimum detectable angular deflection (0.1 px sub-pixel resolution)
    min_eps_rad = 0.1 * scale_rad_per_px

    # Minimum detectable dn/dx [1/m]
    min_dndx_per_m = (min_eps_rad / config.ZD_mm) * 1.0e3  # [1/m]

    info: Dict[str, object] = {
        "sensitivity_px_per_rad": round(sensitivity, 4),
        "scale_rad_per_px": float(f"{scale_rad_per_px:.6e}"),
        "working_distance_mm": config.ZA_mm,
        "background_distance_mm": config.ZD_mm,
        "ZA_over_ZD": round(config.ZA_mm / config.ZD_mm, 4) if config.ZD_mm != 0 else None,
        "magnification": config.magnification,
        "focal_length_mm": config.focal_length_mm,
        "pixel_pitch_mm": config.pixel_pitch_mm,
        "gladstone_dale_K_m3_per_kg": config.gladstone_dale_K,
        "ambient_density_kg_m3": config.ambient_density_kg_m3,
        "reference_n": config.reference_n,
        "min_detectable_dndx_per_m": float(f"{min_dndx_per_m:.4e}"),
    }

    logger.info(
        "BOSCalibrationConfig summary: sensitivity=%.1f px/rad, "
        "ZD=%.0f mm, ZA=%.0f mm, f=%.1f mm, M=%.3f",
        sensitivity, config.ZD_mm, config.ZA_mm,
        config.focal_length_mm, config.magnification,
    )
    return info
