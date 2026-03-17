"""Synthetic BOS test-data generator.

Creates realistic reference/measurement image pairs by computing the
refractive deflection field of a Gaussian refractive-index blob and
applying sub-pixel shifts via bilinear interpolation — matching what a
real BOS camera would record.

Two geometries are provided:

* **Cartesian** — 2-D Gaussian blob in free space (asymmetric, for
  testing the full dx/dy displacement pipeline).
* **Axisymmetric** — axially symmetric Gaussian jet (for testing Abel
  inversion).

Usage (standalone)
------------------
::

    python tests/generate_synthetic_data.py --output ./tests/synthetic_data
    python tests/generate_synthetic_data.py --geometry axisymmetric --show
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    """Parameters for the synthetic BOS dataset."""

    # Image size
    height: int = 512
    width: int = 512

    # Gaussian blob (refractive index perturbation)
    blob_amplitude: float = 0.05    # peak Δn (refractive index change)
    blob_sigma_x: float = 40.0      # [px]
    blob_sigma_y: float = 30.0      # [px]
    blob_center_x: float = 0.5      # fraction of width
    blob_center_y: float = 0.5      # fraction of height

    # Background dot pattern
    dot_density: float = 0.005      # fraction of pixels that are dots
    dot_radius: int = 2             # dot radius [px]
    dot_intensity: float = 1.0      # normalised dot brightness
    background_level: float = 0.3   # normalised background grey level

    # Physical parameters
    sensitivity: float = 50.0      # pixel displacement per unit Δn gradient
    noise_std: float = 0.002        # additive Gaussian noise (normalised)

    # Random seed for reproducibility
    seed: int = 42

    geometry: str = "cartesian"     # "cartesian" | "axisymmetric"


# ---------------------------------------------------------------------------
# Background pattern
# ---------------------------------------------------------------------------


def make_background_pattern(cfg: SyntheticConfig) -> np.ndarray:
    """Generate a random dot pattern on a grey background (float32, 0–1)."""
    rng = np.random.default_rng(cfg.seed)
    H, W = cfg.height, cfg.width

    img = np.full((H, W), cfg.background_level, dtype=np.float32)

    # Random dot centres
    n_dots = int(H * W * cfg.dot_density)
    cx = rng.integers(0, W, size=n_dots)
    cy = rng.integers(0, H, size=n_dots)

    r = cfg.dot_radius
    for x, y in zip(cx, cy):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        img[y0:y1, x0:x1] = cfg.dot_intensity

    return img


# ---------------------------------------------------------------------------
# Refractive-index fields
# ---------------------------------------------------------------------------


def make_cartesian_blob(cfg: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D asymmetric Gaussian Δn field and its x/y gradients.

    Returns
    -------
    dn:
        Δn field (H × W).
    grad_x, grad_y:
        Spatial gradients of Δn (H × W) — proportional to BOS deflection.
    """
    H, W = cfg.height, cfg.width
    cy = cfg.blob_center_y * H
    cx = cfg.blob_center_x * W
    sx, sy = cfg.blob_sigma_x, cfg.blob_sigma_y

    y_idx, x_idx = np.mgrid[0:H, 0:W].astype(np.float64)
    dn = cfg.blob_amplitude * np.exp(
        -0.5 * (((x_idx - cx) / sx) ** 2 + ((y_idx - cy) / sy) ** 2)
    )

    # Analytical gradients
    grad_x = dn * (-(x_idx - cx) / sx ** 2)
    grad_y = dn * (-(y_idx - cy) / sy ** 2)

    return dn.astype(np.float32), grad_x.astype(np.float32), grad_y.astype(np.float32)


def make_axisymmetric_blob(cfg: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axially symmetric Gaussian Δn field (vertical jet geometry).

    The symmetry axis is the vertical centreline of the image.

    Returns
    -------
    dn, grad_x, grad_y:
        Same as :func:`make_cartesian_blob`.
    """
    H, W = cfg.height, cfg.width
    cx = W // 2
    cy = cfg.blob_center_y * H
    sr = (cfg.blob_sigma_x + cfg.blob_sigma_y) / 2  # isotropic
    sz = cfg.blob_sigma_y

    y_idx, x_idx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.abs(x_idx - cx)
    z = y_idx - cy

    dn = cfg.blob_amplitude * np.exp(
        -0.5 * ((r / sr) ** 2 + (z / sz) ** 2)
    )

    # Radial gradient: dΔn/dx (x is the integration direction for Abel)
    grad_x = dn * (-(x_idx - cx) / sr ** 2)
    grad_y = dn * (-z / sz ** 2)

    return dn.astype(np.float32), grad_x.astype(np.float32), grad_y.astype(np.float32)


# ---------------------------------------------------------------------------
# Image synthesis
# ---------------------------------------------------------------------------


def apply_displacement(
    background: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> np.ndarray:
    """Shift background pixels according to (dx, dy) via bilinear interpolation.

    This simulates what the BOS camera records: the background pattern
    appears shifted by the deflection field.

    Parameters
    ----------
    background:
        Reference background image (H × W, float32, 0–1).
    dx, dy:
        Per-pixel horizontal and vertical displacement [px].

    Returns
    -------
    shifted:
        Displaced background image (H × W, float32).
    """
    from scipy.ndimage import map_coordinates

    H, W = background.shape
    y_idx, x_idx = np.mgrid[0:H, 0:W].astype(np.float64)

    # New sampling coordinates: source pixel at (x + dx, y + dy)
    coords = np.array([
        (y_idx + dy).ravel(),
        (x_idx + dx).ravel(),
    ])

    shifted = map_coordinates(
        background, coords, order=1, mode="nearest"
    ).reshape(H, W)
    return shifted.astype(np.float32)


def add_noise(image: np.ndarray, std: float, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noisy = image + rng.normal(0.0, std, size=image.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level generator
# ---------------------------------------------------------------------------


def generate(
    cfg: Optional[SyntheticConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a complete synthetic BOS dataset.

    Parameters
    ----------
    cfg:
        Generator configuration.  Uses defaults if ``None``.

    Returns
    -------
    reference:
        Clean background dot-pattern image (H × W, float32).
    measurement:
        Background image with BOS displacement applied + noise (H × W, float32).
    dx_true:
        Ground-truth horizontal displacement field [px].
    dy_true:
        Ground-truth vertical displacement field [px].
    dn:
        True refractive-index perturbation field Δn.
    """
    cfg = cfg or SyntheticConfig()

    background = make_background_pattern(cfg)

    if cfg.geometry == "axisymmetric":
        dn, grad_x, grad_y = make_axisymmetric_blob(cfg)
    else:
        dn, grad_x, grad_y = make_cartesian_blob(cfg)

    # Displacement = sensitivity × gradient
    dx_true = (cfg.sensitivity * grad_x).astype(np.float32)
    dy_true = (cfg.sensitivity * grad_y).astype(np.float32)

    shifted = apply_displacement(background, dx_true, dy_true)
    measurement = add_noise(shifted, cfg.noise_std, seed=cfg.seed + 1)
    reference = add_noise(background, cfg.noise_std, seed=cfg.seed)

    return reference, measurement, dx_true, dy_true, dn


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_dataset(
    output_dir: str | Path,
    cfg: Optional[SyntheticConfig] = None,
) -> dict:
    """Generate and save a synthetic dataset to *output_dir*.

    Saves reference.npy, measurement.npy, dx_true.npy, dy_true.npy, dn.npy
    and also exports as 16-bit TIFFs for testing the TIFF reader.

    Returns a dict with all saved paths.
    """
    try:
        import tifffile  # type: ignore[import]
        _has_tiff = True
    except ImportError:
        _has_tiff = False

    cfg = cfg or SyntheticConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    reference, measurement, dx_true, dy_true, dn = generate(cfg)

    paths = {}
    for name, arr in [
        ("reference", reference),
        ("measurement", measurement),
        ("dx_true", dx_true),
        ("dy_true", dy_true),
        ("dn", dn),
    ]:
        p_npy = out / f"{name}.npy"
        np.save(str(p_npy), arr)
        paths[name] = p_npy

        if _has_tiff and name in ("reference", "measurement"):
            # Save as 16-bit TIFF (scale [0,1] → [0, 65535])
            arr_16 = (arr * 65535).astype(np.uint16)
            p_tif = out / f"{name}.tiff"
            tifffile.imwrite(str(p_tif), arr_16)
            paths[f"{name}_tiff"] = p_tif

    # Also save a TIFF sequence (frames 0..4 = reference copies; frame 5 = measurement)
    seq_dir = out / "tiff_sequence"
    seq_dir.mkdir(exist_ok=True)
    if _has_tiff:
        for i in range(5):
            tifffile.imwrite(str(seq_dir / f"frame_{i:05d}.tiff"),
                             (reference * 65535).astype(np.uint16))
        tifffile.imwrite(str(seq_dir / "frame_00005.tiff"),
                         (measurement * 65535).astype(np.uint16))
        paths["tiff_sequence_dir"] = seq_dir

    print(f"Synthetic dataset saved to: {out}")
    for name, p in paths.items():
        print(f"  {name}: {p}")

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic BOS test data (Gaussian blob)."
    )
    p.add_argument("--output", "-o", default="./tests/synthetic_data",
                   help="Output directory (default: ./tests/synthetic_data)")
    p.add_argument("--geometry", choices=["cartesian", "axisymmetric"],
                   default="cartesian", help="Blob geometry (default: cartesian)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--sigma-x", type=float, default=40.0,
                   dest="sigma_x", help="Blob sigma x [px]")
    p.add_argument("--sigma-y", type=float, default=30.0,
                   dest="sigma_y", help="Blob sigma y [px]")
    p.add_argument("--amplitude", type=float, default=0.05,
                   help="Peak refractive index change Δn")
    p.add_argument("--sensitivity", type=float, default=50.0,
                   help="px displacement per unit Δn gradient")
    p.add_argument("--noise", type=float, default=0.002,
                   help="Additive Gaussian noise std (normalised)")
    p.add_argument("--show", action="store_true",
                   help="Display generated images with matplotlib")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = SyntheticConfig(
        height=args.height,
        width=args.width,
        blob_sigma_x=args.sigma_x,
        blob_sigma_y=args.sigma_y,
        blob_amplitude=args.amplitude,
        sensitivity=args.sensitivity,
        noise_std=args.noise,
        geometry=args.geometry,
    )
    paths = save_dataset(args.output, cfg)

    if args.show:
        import matplotlib.pyplot as plt

        reference = np.load(str(paths["reference"]))
        measurement = np.load(str(paths["measurement"]))
        dx_true = np.load(str(paths["dx_true"]))
        dy_true = np.load(str(paths["dy_true"]))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(reference, cmap="gray")
        axes[0].set_title("Reference")
        axes[1].imshow(measurement, cmap="gray")
        axes[1].set_title("Measurement")
        axes[2].imshow(dx_true, cmap="seismic")
        axes[2].set_title("dx_true [px]")
        axes[3].imshow(dy_true, cmap="seismic")
        axes[3].set_title("dy_true [px]")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
