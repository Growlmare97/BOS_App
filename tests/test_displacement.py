"""Tests for the displacement computation module.

Run with:  pytest tests/test_displacement.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.generate_synthetic_data import SyntheticConfig, generate
from bos_pipeline.processing.displacement import (
    DisplacementConfig,
    compute_displacement,
    displacement_magnitude,
    interpolate_to_full_resolution,
)
from bos_pipeline.processing.preprocess import PreprocessConfig, preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_cartesian():
    """Pre-generated cartesian synthetic dataset (module-scoped for speed)."""
    cfg = SyntheticConfig(
        height=256,
        width=256,
        blob_sigma_x=30.0,
        blob_sigma_y=25.0,
        blob_amplitude=0.05,
        sensitivity=30.0,
        noise_std=0.001,
        geometry="cartesian",
        seed=0,
    )
    ref, meas, dx_true, dy_true, dn = generate(cfg)
    return ref, meas, dx_true, dy_true, dn, cfg


@pytest.fixture(scope="module")
def synthetic_axisymmetric():
    cfg = SyntheticConfig(
        height=256,
        width=256,
        blob_sigma_x=25.0,
        blob_sigma_y=25.0,
        blob_amplitude=0.05,
        sensitivity=30.0,
        noise_std=0.001,
        geometry="axisymmetric",
        seed=1,
    )
    ref, meas, dx_true, dy_true, dn = generate(cfg)
    return ref, meas, dx_true, dy_true, dn, cfg


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocess:
    def test_output_shape_preserved(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        pre_cfg = PreprocessConfig(gaussian_sigma=1.0)
        ref_p, meas_p = preprocess(ref, meas, config=pre_cfg)
        assert ref_p.shape == ref.shape
        assert meas_p.shape == meas.shape

    def test_output_dtype_float32(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        ref_p, meas_p = preprocess(ref, meas)
        assert ref_p.dtype == np.float32
        assert meas_p.dtype == np.float32

    def test_gaussian_filter_reduces_std(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        pre_cfg = PreprocessConfig(gaussian_sigma=5.0)
        ref_p, _ = preprocess(ref, ref, config=pre_cfg)  # same img pair
        # Filtered image should have lower std (smoother)
        assert ref_p.std() <= ref.std() + 1e-6

    def test_no_filter(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        pre_cfg = PreprocessConfig(gaussian_sigma=0.0)
        ref_p, meas_p = preprocess(ref, meas, config=pre_cfg)
        np.testing.assert_allclose(ref_p, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------


class TestCrossCorrelation:
    def test_returns_correct_grid_shape(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        cfg = DisplacementConfig(
            method="cross_correlation", window_size=64, overlap=0.5
        )
        dx, dy = compute_displacement(ref, meas, config=cfg)
        H, W = ref.shape
        ws, step = 64, 32  # overlap=0.5 → step=32
        expected_rows = len(range(ws // 2, H - ws // 2 + 1, step))
        expected_cols = len(range(ws // 2, W - ws // 2 + 1, step))
        assert dx.shape == (expected_rows, expected_cols)
        assert dy.shape == (expected_rows, expected_cols)

    def test_displacement_float32(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        cfg = DisplacementConfig(method="cross_correlation", window_size=64)
        dx, dy = compute_displacement(ref, meas, config=cfg)
        assert dx.dtype == np.float32
        assert dy.dtype == np.float32

    def test_zero_displacement_for_identical_images(self):
        """Identical images → near-zero displacement."""
        rng = np.random.default_rng(42)
        img = rng.random((128, 128)).astype(np.float32)
        cfg = DisplacementConfig(method="cross_correlation", window_size=32)
        dx, dy = compute_displacement(img, img, config=cfg)
        assert np.abs(dx).max() < 1.5, f"max |dx|={np.abs(dx).max():.3f} px"
        assert np.abs(dy).max() < 1.5, f"max |dy|={np.abs(dy).max():.3f} px"

    def test_known_shift_recovery(self):
        """Apply a uniform shift of (sx, sy) px and check recovery."""
        from scipy.ndimage import shift as nd_shift

        rng = np.random.default_rng(99)
        bg = rng.random((256, 256)).astype(np.float32)
        sx, sy = 3.0, 2.0
        shifted = nd_shift(bg, (sy, sx), mode="nearest").astype(np.float32)

        cfg = DisplacementConfig(
            method="cross_correlation", window_size=64, overlap=0.5
        )
        dx, dy = compute_displacement(bg, shifted, config=cfg)

        # Central grid region (avoid edges)
        cx, cy = dx.shape[1] // 2, dx.shape[0] // 2
        pad = 2
        dx_center = dx[cy - pad:cy + pad + 1, cx - pad:cx + pad + 1]
        dy_center = dy[cy - pad:cy + pad + 1, cx - pad:cx + pad + 1]

        assert abs(dx_center.mean() - sx) < 1.5, (
            f"Expected dx≈{sx}, got {dx_center.mean():.2f}"
        )
        assert abs(dy_center.mean() - sy) < 1.5, (
            f"Expected dy≈{sy}, got {dy_center.mean():.2f}"
        )

    def test_interpolation_to_full_resolution(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        cfg = DisplacementConfig(method="cross_correlation", window_size=64)
        dx_grid, dy_grid = compute_displacement(ref, meas, config=cfg)

        gx = dx_grid._grid_x  # type: ignore[attr-defined]
        gy = dx_grid._grid_y  # type: ignore[attr-defined]
        dx_full, dy_full = interpolate_to_full_resolution(
            dx_grid, dy_grid, ref.shape, gx, gy
        )
        assert dx_full.shape == ref.shape
        assert dy_full.shape == ref.shape
        assert dx_full.dtype == np.float32


# ---------------------------------------------------------------------------
# Farneback optical flow
# ---------------------------------------------------------------------------


class TestFarneback:
    def test_output_shape_full_resolution(self, synthetic_cartesian):
        pytest.importorskip("cv2")
        ref, meas, *_ = synthetic_cartesian
        cfg = DisplacementConfig(method="farneback")
        dx, dy = compute_displacement(ref, meas, config=cfg)
        assert dx.shape == ref.shape
        assert dy.shape == ref.shape

    def test_zero_displacement_for_identical_images(self):
        pytest.importorskip("cv2")
        rng = np.random.default_rng(42)
        img = rng.random((128, 128)).astype(np.float32)
        cfg = DisplacementConfig(method="farneback")
        dx, dy = compute_displacement(img, img, config=cfg)
        assert np.abs(dx).max() < 0.5
        assert np.abs(dy).max() < 0.5


# ---------------------------------------------------------------------------
# Lucas-Kanade optical flow
# ---------------------------------------------------------------------------


class TestLucasKanade:
    def test_output_shape_full_resolution(self, synthetic_cartesian):
        pytest.importorskip("cv2")
        ref, meas, *_ = synthetic_cartesian
        cfg = DisplacementConfig(method="lucas_kanade", window_size=32)
        dx, dy = compute_displacement(ref, meas, config=cfg)
        assert dx.shape == ref.shape
        assert dy.shape == ref.shape


# ---------------------------------------------------------------------------
# Displacement magnitude
# ---------------------------------------------------------------------------


class TestDisplacementMagnitude:
    def test_magnitude_nonneg(self, synthetic_cartesian):
        ref, meas, *_ = synthetic_cartesian
        cfg = DisplacementConfig(method="cross_correlation", window_size=64)
        dx, dy = compute_displacement(ref, meas, config=cfg)
        mag = displacement_magnitude(dx, dy)
        assert (mag >= 0).all()

    def test_magnitude_zeros(self):
        dx = np.zeros((8, 8), dtype=np.float32)
        dy = np.zeros((8, 8), dtype=np.float32)
        assert displacement_magnitude(dx, dy).max() == 0.0

    def test_magnitude_pythagorean(self):
        dx = np.full((4, 4), 3.0, dtype=np.float32)
        dy = np.full((4, 4), 4.0, dtype=np.float32)
        mag = displacement_magnitude(dx, dy)
        np.testing.assert_allclose(mag, 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_shape_mismatch_raises(self):
        a = np.zeros((64, 64), dtype=np.float32)
        b = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_displacement(a, b)

    def test_unknown_method_raises(self):
        img = np.zeros((32, 32), dtype=np.float32)
        cfg = DisplacementConfig(method="nonexistent")  # type: ignore
        with pytest.raises(ValueError, match="Unknown displacement method"):
            compute_displacement(img, img, config=cfg)
