"""Pydantic models for the BOS Analysis REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ProcessingRequest(BaseModel):
    """Full processing request submitted to POST /api/process."""

    # --- Camera / input --------------------------------------------------
    camera_type: Literal["photron", "photron_avi", "dalsa", "tiff_sequence"]
    input_path: str
    metadata_path: Optional[str] = None

    # --- Frame selection -------------------------------------------------
    reference_frame: int = 0
    measurement_frames: Union[str, List[int]] = "all"  # "all" or explicit list

    # --- Displacement ----------------------------------------------------
    method: Literal["cross_correlation", "lucas_kanade", "farneback"] = (
        "cross_correlation"
    )
    window_size: int = Field(32, ge=8, le=256)
    overlap: float = Field(0.75, ge=0.0, le=0.95)
    sigma: float = Field(1.0, ge=0.0, le=10.0)
    ensemble_averaging: bool = False
    multi_pass: bool = False

    # --- Abel inversion --------------------------------------------------
    abel_enabled: bool = False
    abel_method: Literal["three_point", "hansenlaw", "basex"] = "three_point"

    # --- Concentration ---------------------------------------------------
    concentration_enabled: bool = False
    gas_type: str = "H2"
    ambient_gas: str = "air"
    temperature_K: float = 293.15
    pressure_Pa: float = 101325.0

    # --- Velocity --------------------------------------------------------
    velocity_enabled: bool = False
    velocity_method: Literal["frame_to_frame", "kymography"] = "frame_to_frame"

    # --- Physical calibration -------------------------------------------
    pixel_scale_mm: float = 0.1   # [mm/px]
    ZD_mm: float = 1000.0         # camera-to-background distance [mm]
    ZA_mm: float = 500.0          # camera-to-object distance [mm]
    focal_length_mm: float = 50.0

    # --- Output ----------------------------------------------------------
    output_formats: List[Literal["npy", "hdf5", "csv", "vtk"]] = ["npy"]


class KymographRequest(BaseModel):
    """Request body for POST /api/kymograph."""

    job_id: str
    frame_indices: List[int]
    axis: Literal["horizontal", "vertical"]
    line_position: int
    pixel_scale_mm: float = 0.1
    dt: float = 0.001  # time step between frames [s]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FrameMetadataResponse(BaseModel):
    """Metadata returned by POST /api/probe."""

    camera_type: str
    frame_count: int
    frame_rate: float
    width: int
    height: int
    bit_depth: int
    trigger_frame: Optional[int] = None


class ResultSummary(BaseModel):
    """Per-frame statistics included in ProcessingResponse."""

    frame_idx: int
    max_displacement: float
    mean_displacement: float
    bg_noise: float
    snr_db: float
    has_concentration: bool
    has_velocity: bool


class ProcessingResponse(BaseModel):
    """Returned by GET /api/jobs/{job_id}."""

    job_id: str
    status: Literal["queued", "running", "done", "error"]
    results: Optional[List[ResultSummary]] = None
    output_paths: Optional[Dict[str, str]] = None
    error: Optional[str] = None
