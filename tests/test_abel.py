"""Tests for Abel inversion module.

Run with:  pytest tests/test_abel.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.generate_synthetic_data import SyntheticConfig, generate
from bos_pipeline.processing.abel import (
    AbelConfig,
    abel_invert,
    find_symmetry_axis,
    reconstruct_density,
)
from bos_pipeline.processing.displacement import (
    DisplacementConfig,
    compute_displacement,
)
from bos_pipeline.processing.preprocess import PreprocessConfig, preprocess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def axisym_dataset():
    cfg = SyntheticConfig(
        height=256,
        width=256,
        blob_sigma_x=25.0,
        blob_sigma_y=25.0,
        blob_amplitude=0.05,
        sensitivity=30.0,
        noise_std=0.001,
        geometry="axisymmetric",
        seed=7,
    )
    ref, meas, dx_true, dy_true, dn = generate(cfg)
    return ref, meas, dx_true, dy_true, dn, cfg


@pytest.fixture(scope="module")
def displacement_pair(axisym_dataset):
    ref, meas, dx_true, dy_true, dn, cfg = axisym_dataset
    pre_cfg = PreprocessConfig(gaussian_sigma=1.5)
    ref_p, meas_p = preprocess(ref, meas, config=pre_cfg)
    disp_cfg = DisplacementConfig(
        method="cross_correlation", window_size=32, overlap=0.5
    )
    dx, dy = compute_displacement(ref_p, meas_p, config=disp_cfg)
    return dx, dy, cfg


# ---------------------------------------------------------------------------
# Axis detection
# ---------------------------------------------------------------------------


class TestFindSymmetryAxis:
    def test_detects_centre_for_symmetric_field(self):
        H, W = 64, 64
        x = np.arange(W, dtype=np.float64)
        cx = W // 2
        row = np.exp(-0.5 * ((x - cx) / 10.0) ** 2)
        field = np.tile(row, (H, 1))
        detected = find_symmetry_axis(field)
        assert abs(detected - cx) <= 2, f"Detected axis {detected}, expected ~{cx}"

    def test_returns_integer(self):
        field = np.ones((16, 32), dtype=np.float32)
        axis = find_symmetry_axis(field)
        assert isinstance(axis, int)

    def test_uniform_field_returns_half_width(self):
        field = np.ones((32, 64), dtype=np.float32)
        axis = find_symmetry_axis(field)
        assert axis == 32  # centroid of uniform row = W/2 = 32


# ---------------------------------------------------------------------------
# Abel inversion
# ---------------------------------------------------------------------------


class TestAbelInvert:
    def test_requires_pyabel(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, method="three_point", axis_mode="auto")
        inv_field, axis_arr = abel_invert(dx, dy, config=cfg)
        assert inv_field is not None

    def test_output_shape_matches_input(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        assert inv_field.shape == dx.shape

    def test_output_dtype_float32(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        assert inv_field.dtype == np.float32

    def test_axis_returned(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="auto")
        _, axis_arr = abel_invert(dx, dy, config=cfg)
        assert axis_arr is not None
        assert axis_arr.shape == (1,)

    def test_manual_axis(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="manual", axis_pos=dx.shape[1] // 2)
        inv_field, axis_arr = abel_invert(dx, dy, config=cfg)
        assert int(axis_arr[0]) == dx.shape[1] // 2

    def test_manual_axis_requires_axis_pos(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="manual", axis_pos=None)
        with pytest.raises(ValueError, match="axis_pos"):
            abel_invert(dx, dy, config=cfg)

    @pytest.mark.parametrize("method", ["three_point", "hansenlaw"])
    def test_methods_run_without_error(self, displacement_pair, method):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, method=method, axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        assert inv_field is not None

    def test_component_dx(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, component="dx", axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        assert inv_field.shape == dx.shape

    def test_component_magnitude(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, component="magnitude", axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        assert inv_field.shape == dx.shape

    def test_abel_inversion_recovers_gaussian_radial_profile(self):
        """Abel inversion of a Gaussian Abel-transform should recover a
        profile peaked at r=0.

        We construct the Abel transform (projection) of a Gaussian radial
        function g(r)=exp(-r²/σ²) analytically:
            f(y) = σ√π · exp(-y²/σ²)
        and verify that the inverse Abel transform of f recovers a profile
        with its maximum near r=0.
        """
        pytest.importorskip("abel")
        import abel as abel_lib

        N = 128
        sigma = 20.0
        r = np.arange(N, dtype=np.float64)

        # Analytical Abel transform of exp(-r²/σ²) is σ√π·exp(-r²/σ²)
        projection = sigma * np.sqrt(np.pi) * np.exp(-(r / sigma) ** 2)
        # PyAbel requires at least 3 rows — tile the same row
        projection_2d = np.tile(projection.astype(np.float32), (5, 1))

        result = abel_lib.Transform(
            projection_2d,
            method="three_point",
            direction="inverse",
            verbose=False,
        ).transform[2, :]  # use the central row

        # The recovered radial function should have its maximum near r=0
        peak_idx = int(np.argmax(result))
        assert peak_idx <= 5, (
            f"Expected peak near r=0, found at r={peak_idx}"
        )


# ---------------------------------------------------------------------------
# Density reconstruction
# ---------------------------------------------------------------------------


class TestReconstructDensity:
    def test_output_shape(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        density = reconstruct_density(
            inv_field,
            gladstone_dale=2.259e-4,
            reference_density=1.2,
            dr_m=1e-4,
        )
        assert density.shape == inv_field.shape

    def test_reference_density_at_boundary(self, displacement_pair):
        """At the first column (zero displacement), density ≈ reference."""
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        rho_ref = 1.2
        density = reconstruct_density(
            inv_field,
            gladstone_dale=2.259e-4,
            reference_density=rho_ref,
            dr_m=1e-4,
        )
        # First column should be very close to reference density
        np.testing.assert_allclose(density[:, 0], rho_ref, atol=0.5)

    def test_output_dtype_float32(self, displacement_pair):
        pytest.importorskip("abel")
        dx, dy, _ = displacement_pair
        cfg = AbelConfig(enabled=True, axis_mode="auto")
        inv_field, _ = abel_invert(dx, dy, config=cfg)
        density = reconstruct_density(inv_field, 2.259e-4, 1.2)
        assert density.dtype == np.float32


# ---------------------------------------------------------------------------
# AbelConfig parsing
# ---------------------------------------------------------------------------


class TestAbelConfigParsing:
    def test_from_dict_defaults(self):
        cfg = AbelConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.method == "three_point"
        assert cfg.component == "magnitude"

    def test_from_dict_custom(self):
        d = {
            "enabled": True,
            "method": "basex",
            "component": "dx",
            "axis_mode": "manual",
            "axis_pos": 128,
        }
        cfg = AbelConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.method == "basex"
        assert cfg.axis_pos == 128

    def test_missing_pyabel_raises_import_error(self):
        import unittest.mock as mock
        dx = np.zeros((16, 16), dtype=np.float32)
        dy = np.zeros((16, 16), dtype=np.float32)
        cfg = AbelConfig(enabled=True)
        with mock.patch.dict("sys.modules", {"abel": None}):
            with pytest.raises(ImportError, match="PyAbel"):
                abel_invert(dx, dy, config=cfg)
