"""REST API routes for the BOS Analysis backend.

All endpoints live under the /api prefix (applied when the router is
included in main.py).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from bos_pipeline.api.models import (
    FrameMetadataResponse,
    KymographRequest,
    ProcessingRequest,
    ProcessingResponse,
    ResultSummary,
)
from bos_pipeline.api.websocket import broadcast_progress

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process job store
# ---------------------------------------------------------------------------

# Each entry: {
#   "status": "queued"|"running"|"done"|"error",
#   "request": ProcessingRequest,
#   "results": list[dict],          # per-frame result dicts
#   "output_paths": dict[str, str],
#   "error": str | None,
# }
_jobs: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helper: detect camera type from path
# ---------------------------------------------------------------------------


def _detect_camera_type(path: str) -> str:
    """Return a camera_type string inferred from the file extension / path.

    Falls back to ``"dalsa"`` (TIFF sequence folder) for unrecognised inputs.
    """
    p = Path(path)
    if p.is_dir():
        return "dalsa"
    suffix = p.suffix.lower()
    if suffix == ".mraw":
        return "photron"
    if suffix == ".avi":
        return "photron_avi"
    # Single TIFF → treat as a one-file sequence handled by DalsaReader
    return "dalsa"


# ---------------------------------------------------------------------------
# POST /api/probe
# ---------------------------------------------------------------------------


class ProbeBody(BaseModel):
    path: str
    camera_type: str = "auto"


@router.post("/probe", response_model=FrameMetadataResponse)
async def probe(body: ProbeBody) -> FrameMetadataResponse:
    """Read file metadata without running the processing pipeline."""
    camera_type = body.camera_type
    if camera_type == "auto":
        camera_type = _detect_camera_type(body.path)

    try:
        from bos_pipeline.io import get_reader

        reader = get_reader(camera_type, path=body.path)
        with reader:
            meta = reader.metadata
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FrameMetadataResponse(
        camera_type=camera_type,
        frame_count=int(meta.total_frames or 0),
        frame_rate=float(meta.frame_rate or 0.0),
        width=int(meta.width or 0),
        height=int(meta.height or 0),
        bit_depth=int(meta.bit_depth or 16),
        trigger_frame=meta.trigger_frame,
    )


# ---------------------------------------------------------------------------
# POST /api/process
# ---------------------------------------------------------------------------


@router.post("/process")
async def process(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Queue a full processing job and return immediately."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "request": request,
        "results": [],
        "output_paths": {},
        "error": None,
    }
    background_tasks.add_task(_run_pipeline_task, job_id, request)
    logger.info("Job queued: %s", job_id)
    return {"job_id": job_id, "status": "queued"}


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=ProcessingResponse)
async def get_job(job_id: str) -> ProcessingResponse:
    """Return current status and results for *job_id*."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    summaries: Optional[List[ResultSummary]] = None
    raw_results: List[dict] = job.get("results") or []
    if raw_results:
        summaries = [
            ResultSummary(
                frame_idx=r["frame_idx"],
                max_displacement=r.get("max_displacement", 0.0),
                mean_displacement=r.get("mean_displacement", 0.0),
                bg_noise=r.get("bg_noise", 0.0),
                snr_db=r.get("snr_db", 0.0),
                has_concentration="concentration" in r,
                has_velocity="velocity_u" in r,
            )
            for r in raw_results
        ]

    return ProcessingResponse(
        job_id=job_id,
        status=job["status"],
        results=summaries,
        output_paths=job.get("output_paths") or None,
        error=job.get("error"),
    )


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}/frame/{frame_idx}/image/{field}
# ---------------------------------------------------------------------------

_FIELD_UNITS = {
    "magnitude": "px",
    "dx": "px",
    "dy": "px",
    "concentration": "vol. frac.",
    "u": "m/s",
    "v": "m/s",
    "vorticity": "1/s",
}

_FIELD_CMAPS = {
    "magnitude": "turbo",
    "dx": "RdBu_r",
    "dy": "RdBu_r",
    "concentration": "plasma",
    "u": "RdBu_r",
    "v": "RdBu_r",
    "vorticity": "seismic",
}


def _get_field_array(job: dict, frame_idx: int, field: str) -> np.ndarray:
    """Extract the requested *field* array from stored job results.

    Raises ``HTTPException(404)`` if the frame or field is unavailable.
    """
    results: List[dict] = job.get("results") or []
    frame_result = next(
        (r for r in results if r["frame_idx"] == frame_idx), None
    )
    if frame_result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Frame {frame_idx} not found in job results",
        )

    # Map field name to stored key
    _key_map = {
        "magnitude": "magnitude",
        "dx": "dx",
        "dy": "dy",
        "concentration": "concentration",
        "u": "velocity_u",
        "v": "velocity_v",
        "vorticity": "vorticity",
    }
    key = _key_map.get(field)
    if key is None or key not in frame_result:
        raise HTTPException(
            status_code=404,
            detail=f"Field '{field}' not available for frame {frame_idx}",
        )
    return np.asarray(frame_result[key], dtype=np.float32)


@router.get("/results/{job_id}/frame/{frame_idx}/image/{field}")
async def get_frame_image(
    job_id: str,
    frame_idx: int,
    field: str,
) -> StreamingResponse:
    """Return a PNG image of *field* for *frame_idx* rendered with Matplotlib."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    arr = _get_field_array(job, frame_idx, field)
    cmap = _FIELD_CMAPS.get(field, "viridis")
    unit = _FIELD_UNITS.get(field, "")

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    vmin = float(np.percentile(arr, 1))
    vmax = float(np.percentile(arr, 99)) or float(arr.max()) or 1.0
    # For signed fields centre the colour map at zero
    if field in ("dx", "dy", "u", "v", "vorticity"):
        lim = max(abs(vmin), abs(vmax)) or 1.0
        vmin, vmax = -lim, lim

    fig = Figure(figsize=(7, 5), dpi=120)
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(unit, fontsize=8)
    ax.set_title(f"{field}  — frame {frame_idx}", fontsize=10)
    ax.set_xlabel("x [px]", fontsize=8)
    ax.set_ylabel("y [px]", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------------------------------------------------------------------
# GET /api/results/{job_id}/frame/{frame_idx}/data/{field}
# ---------------------------------------------------------------------------


@router.get("/results/{job_id}/frame/{frame_idx}/data/{field}")
async def get_frame_data(
    job_id: str,
    frame_idx: int,
    field: str,
) -> dict:
    """Return raw field data as a JSON-serialisable dict for Plotly rendering."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    arr = _get_field_array(job, frame_idx, field)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    unit = _FIELD_UNITS.get(field, "")

    return {
        "data": arr.tolist(),
        "shape": list(arr.shape),
        "vmin": vmin,
        "vmax": vmax,
        "unit": unit,
    }


# ---------------------------------------------------------------------------
# POST /api/kymograph
# ---------------------------------------------------------------------------


@router.post("/kymograph")
async def kymograph(body: KymographRequest) -> dict:
    """Build a kymograph and estimate convective velocity."""
    job = _jobs.get(body.job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"Job '{body.job_id}' not found"
        )

    results: List[dict] = job.get("results") or []
    frames_lookup = {r["frame_idx"]: r for r in results}

    # Collect line profiles from each requested frame
    profiles: List[np.ndarray] = []
    for fidx in body.frame_indices:
        frame_result = frames_lookup.get(fidx)
        if frame_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {fidx} not in job results",
            )
        mag = np.asarray(frame_result.get("magnitude", [[]]), dtype=np.float32)
        if body.axis == "horizontal":
            # Horizontal kymograph: extract one row
            row = min(body.line_position, mag.shape[0] - 1)
            profiles.append(mag[row, :])
        else:
            # Vertical kymograph: extract one column
            col = min(body.line_position, mag.shape[1] - 1)
            profiles.append(mag[:, col])

    if not profiles:
        raise HTTPException(status_code=422, detail="No valid frame profiles found")

    kymo = np.stack(profiles, axis=0)  # (n_frames, n_pixels)

    # Estimate convective velocity via cross-correlation between successive frames
    velocity_profile: List[float] = []
    for i in range(len(profiles) - 1):
        p0 = profiles[i].astype(np.float64)
        p1 = profiles[i + 1].astype(np.float64)
        p0 -= p0.mean()
        p1 -= p1.mean()
        corr = np.correlate(p0, p1, mode="full")
        lag = int(np.argmax(corr)) - (len(p0) - 1)
        vel = (lag * body.pixel_scale_mm * 1e-3) / body.dt  # [m/s]
        velocity_profile.append(float(vel))

    convective_velocity = (
        float(np.median(velocity_profile)) if velocity_profile else 0.0
    )

    return {
        "kymograph": kymo.tolist(),
        "velocity_profile": velocity_profile,
        "convective_velocity": convective_velocity,
    }


# ---------------------------------------------------------------------------
# POST /api/export/{job_id}
# ---------------------------------------------------------------------------


class ExportBody(BaseModel):
    formats: List[str] = ["npy"]
    output_dir: str = ""


@router.post("/export/{job_id}")
async def export_job(job_id: str, body: ExportBody) -> dict:
    """Export all displacement results for *job_id* to *output_dir*."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] not in ("done",):
        raise HTTPException(
            status_code=409, detail="Job must be in 'done' state before export"
        )

    output_dir = body.output_dir or str(
        Path(tempfile.gettempdir()) / "bos_exports" / job_id
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from bos_pipeline import export as bos_export

    results: List[dict] = job.get("results") or []
    all_paths: Dict[str, str] = {}

    for r in results:
        fidx = r["frame_idx"]
        dx = np.asarray(r.get("dx", [[]]), dtype=np.float32)
        dy = np.asarray(r.get("dy", [[]]), dtype=np.float32)

        for fmt in body.formats:
            if fmt in ("npy", "hdf5", "csv"):
                try:
                    paths = bos_export.export_displacement(
                        dx, dy,
                        output_dir=output_dir,
                        stem=f"frame_{fidx:04d}",
                        fmt=fmt,
                    )
                    for k, v in paths.items():
                        all_paths[f"frame_{fidx:04d}_{k}"] = str(v)
                except Exception as exc:
                    logger.warning(
                        "Export failed for frame %d fmt=%s: %s", fidx, fmt, exc
                    )
            elif fmt == "vtk":
                # Simple VTK rectilinear grid export
                try:
                    vtk_path = Path(output_dir) / f"frame_{fidx:04d}.vtk"
                    _export_vtk(dx, dy, vtk_path)
                    all_paths[f"frame_{fidx:04d}_vtk"] = str(vtk_path)
                except Exception as exc:
                    logger.warning("VTK export failed for frame %d: %s", fidx, exc)

        # Export concentration if present
        if "concentration" in r:
            try:
                conc = np.asarray(r["concentration"], dtype=np.float32)
                for fmt in body.formats:
                    if fmt == "npy":
                        p = Path(output_dir) / f"frame_{fidx:04d}_conc.npy"
                        np.save(str(p), conc)
                        all_paths[f"frame_{fidx:04d}_conc"] = str(p)
            except Exception as exc:
                logger.warning("Concentration export failed: %s", exc)

    job["output_paths"] = all_paths
    return {"paths": all_paths}


def _export_vtk(dx: np.ndarray, dy: np.ndarray, out_path: Path) -> None:
    """Write a minimal ASCII VTK structured points file."""
    H, W = dx.shape
    mag = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
    with open(str(out_path), "w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("BOS displacement field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {W} {H} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n")
        f.write(f"POINT_DATA {H * W}\n")
        f.write("SCALARS magnitude float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in mag.ravel():
            f.write(f"{val:.6f}\n")
        f.write("VECTORS displacement float\n")
        for row in range(H):
            for col in range(W):
                f.write(f"{dx[row, col]:.6f} {dy[row, col]:.6f} 0.0\n")


# ---------------------------------------------------------------------------
# GET /api/download/{job_id}/{filename}
# ---------------------------------------------------------------------------


@router.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str) -> FileResponse:
    """Download an exported file by name from the job's output directory."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    output_paths: Dict[str, str] = job.get("output_paths") or {}
    # Search by the filename component of any stored path
    match = next(
        (p for p in output_paths.values() if Path(p).name == filename), None
    )
    if match is None or not Path(match).exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found for job '{job_id}'",
        )
    return FileResponse(
        path=match,
        filename=filename,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Background pipeline task
# ---------------------------------------------------------------------------


async def _run_pipeline_task(
    job_id: str,
    request: ProcessingRequest,
) -> None:
    """Run the full BOS pipeline asynchronously, updating _jobs in-place."""

    async def _progress(stage: str, pct: int, msg: str) -> None:
        _jobs[job_id]["status"] = "running"
        await broadcast_progress(job_id, stage, pct, msg)
        logger.info("[%s] %s %d%% — %s", job_id, stage, pct, msg)

    try:
        # ----------------------------------------------------------------
        # Stage: loading
        # ----------------------------------------------------------------
        await _progress("loading", 0, "Opening camera reader…")

        from bos_pipeline.io import get_reader
        from bos_pipeline.processing.preprocess import (
            PreprocessConfig,
            preprocess,
        )
        from bos_pipeline.processing.displacement import (
            DisplacementConfig,
            DisplacementResult,
            compute_displacement,
            displacement_magnitude,
            interpolate_to_full_resolution,
        )

        camera_type = request.camera_type
        reader = get_reader(camera_type, path=request.input_path)

        with reader:
            meta = reader.metadata
            total_frames = meta.total_frames or 0

            await _progress("loading", 10, f"Loaded {total_frames} frames")

            # Determine which measurement frames to process
            if request.measurement_frames == "all":
                meas_indices = [
                    i for i in range(total_frames)
                    if i != request.reference_frame
                ]
            else:
                meas_indices = list(request.measurement_frames)  # type: ignore[arg-type]

            # ----------------------------------------------------------------
            # Stage: preprocessing — build reference frame
            # ----------------------------------------------------------------
            await _progress(
                "preprocessing", 15,
                f"Building reference frame {request.reference_frame}…",
            )

            pp_cfg = PreprocessConfig(gaussian_sigma=request.sigma)
            ref_raw = reader.get_frame(request.reference_frame).astype(np.float32)

            disp_cfg = DisplacementConfig(
                method=request.method,
                window_size=request.window_size,
                overlap=request.overlap,
                ensemble_averaging=request.ensemble_averaging,
                multi_pass=request.multi_pass,
            )

            n_frames = len(meas_indices)
            frame_results: List[dict] = []

            # ----------------------------------------------------------------
            # Stage: displacement — iterate over measurement frames
            # ----------------------------------------------------------------
            for fi, frame_idx in enumerate(meas_indices):
                pct_base = 20 + int(60 * fi / max(n_frames, 1))
                await _progress(
                    "displacement",
                    pct_base,
                    f"Frame {fi + 1}/{n_frames} (idx {frame_idx})…",
                )

                meas_raw = reader.get_frame(frame_idx).astype(np.float32)

                # Preprocess
                ref_proc, meas_proc = await asyncio.get_event_loop().run_in_executor(
                    None, lambda r=ref_raw, m=meas_raw: preprocess(r, m, pp_cfg)
                )

                # Displacement
                disp_result: DisplacementResult = (
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda r=ref_proc, m=meas_proc: compute_displacement(
                            r, m, disp_cfg
                        ),
                    )
                )

                # Up-sample cross-correlation grids to full resolution
                if disp_result.grid_x is not None:
                    dx_full, dy_full = interpolate_to_full_resolution(
                        disp_result.dx,
                        disp_result.dy,
                        target_shape=ref_raw.shape,
                        result=disp_result,
                    )
                else:
                    dx_full = disp_result.dx
                    dy_full = disp_result.dy

                mag = displacement_magnitude(dx_full, dy_full)

                # Basic statistics
                flat_mag = mag[np.isfinite(mag)]
                max_disp = float(flat_mag.max()) if flat_mag.size else 0.0
                mean_disp = float(flat_mag.mean()) if flat_mag.size else 0.0
                # Noise estimate: 5th-percentile displacement value
                bg_noise = float(np.percentile(flat_mag, 5)) if flat_mag.size else 0.0
                snr_db = (
                    float(10 * np.log10(mean_disp / bg_noise))
                    if bg_noise > 1e-9
                    else 0.0
                )

                frame_data: dict = {
                    "frame_idx": frame_idx,
                    "dx": dx_full,
                    "dy": dy_full,
                    "magnitude": mag,
                    "max_displacement": max_disp,
                    "mean_displacement": mean_disp,
                    "bg_noise": bg_noise,
                    "snr_db": snr_db,
                }

                # --------------------------------------------------------
                # Stage: abel (optional)
                # --------------------------------------------------------
                if request.abel_enabled:
                    await _progress(
                        "abel",
                        pct_base + 2,
                        f"Abel inversion frame {fi + 1}/{n_frames}…",
                    )
                    from bos_pipeline.processing.abel import (
                        AbelConfig,
                        abel_invert,
                    )

                    abel_cfg = AbelConfig(
                        enabled=True,
                        method=request.abel_method,
                    )
                    try:
                        inv_field, _ = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda dx=dx_full, dy=dy_full, c=abel_cfg: abel_invert(
                                dx, dy, c
                            ),
                        )
                        frame_data["abel"] = inv_field
                    except ImportError as exc:
                        logger.warning("PyAbel not available: %s", exc)
                    except Exception as exc:
                        logger.warning("Abel inversion failed frame %d: %s", frame_idx, exc)

                # --------------------------------------------------------
                # Stage: concentration (optional)
                # --------------------------------------------------------
                if request.concentration_enabled:
                    await _progress(
                        "concentration",
                        pct_base + 3,
                        f"Concentration frame {fi + 1}/{n_frames}…",
                    )
                    from bos_pipeline.processing.concentration import (
                        ConcentrationConfig,
                        compute_concentration,
                    )

                    conc_cfg = ConcentrationConfig(
                        enabled=True,
                        mm_per_px=request.pixel_scale_mm,
                        Z_f_mm=request.ZD_mm,
                        gas_type=request.gas_type,
                        ambient_gas=request.ambient_gas,
                        temperature_K=request.temperature_K,
                        pressure_Pa=request.pressure_Pa,
                    )
                    try:
                        conc, _axis, _n_gas, _n_amb = (
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda dx=dx_full, dy=dy_full, c=conc_cfg: compute_concentration(
                                    dx, dy, c
                                ),
                            )
                        )
                        frame_data["concentration"] = conc
                    except ImportError as exc:
                        logger.warning("PyAbel not available for concentration: %s", exc)
                    except Exception as exc:
                        logger.warning(
                            "Concentration failed frame %d: %s", frame_idx, exc
                        )

                # --------------------------------------------------------
                # Stage: velocity (optional — frame-to-frame)
                # --------------------------------------------------------
                if request.velocity_enabled and fi > 0:
                    await _progress(
                        "velocity",
                        pct_base + 4,
                        f"Velocity frame {fi + 1}/{n_frames}…",
                    )
                    prev = frame_results[-1]
                    dx_prev = np.asarray(prev["dx"], dtype=np.float32)
                    dy_prev = np.asarray(prev["dy"], dtype=np.float32)
                    # Simple finite-difference velocity [m/s]
                    # dt estimated from frame rate; falls back to 1/25 s
                    dt = 1.0 / float(meta.frame_rate or 25.0)
                    px_m = request.pixel_scale_mm * 1e-3  # [m/px]

                    u = (dx_full - dx_prev) / dt * px_m
                    v = (dy_full - dy_prev) / dt * px_m
                    # Vorticity: ∂v/∂x − ∂u/∂y  (finite differences)
                    dvdx = np.gradient(v, axis=1)
                    dudy = np.gradient(u, axis=0)
                    vorticity = dvdx - dudy

                    frame_data["velocity_u"] = u
                    frame_data["velocity_v"] = v
                    frame_data["vorticity"] = vorticity

                frame_results.append(frame_data)
                _jobs[job_id]["results"] = frame_results

        # ----------------------------------------------------------------
        # Stage: exporting
        # ----------------------------------------------------------------
        await _progress("exporting", 85, "Exporting results…")

        output_dir = str(
            Path(tempfile.gettempdir()) / "bos_exports" / job_id
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        from bos_pipeline import export as bos_export

        all_paths: Dict[str, str] = {}
        for r in frame_results:
            fidx = r["frame_idx"]
            dx_arr = np.asarray(r["dx"], dtype=np.float32)
            dy_arr = np.asarray(r["dy"], dtype=np.float32)
            for fmt in request.output_formats:
                if fmt in ("npy", "hdf5", "csv"):
                    try:
                        paths = bos_export.export_displacement(
                            dx_arr,
                            dy_arr,
                            output_dir=output_dir,
                            stem=f"frame_{fidx:04d}",
                            fmt=fmt,
                        )
                        for k, v in paths.items():
                            all_paths[f"frame_{fidx:04d}_{k}"] = str(v)
                    except Exception as exc:
                        logger.warning(
                            "Export fmt=%s frame %d: %s", fmt, fidx, exc
                        )
                elif fmt == "vtk":
                    try:
                        vtk_path = Path(output_dir) / f"frame_{fidx:04d}.vtk"
                        _export_vtk(dx_arr, dy_arr, vtk_path)
                        all_paths[f"frame_{fidx:04d}_vtk"] = str(vtk_path)
                    except Exception as exc:
                        logger.warning("VTK export frame %d: %s", fidx, exc)

        _jobs[job_id]["output_paths"] = all_paths

        # ----------------------------------------------------------------
        # Stage: done
        # ----------------------------------------------------------------
        _jobs[job_id]["status"] = "done"
        await broadcast_progress(job_id, "done", 100, "Processing complete")
        logger.info("Job done: %s  frames=%d", job_id, len(frame_results))

    except Exception as exc:
        import traceback

        err_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Pipeline error job=%s: %s", job_id, err_msg)
        logger.debug(traceback.format_exc())
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = err_msg
        await broadcast_progress(job_id, "error", 0, err_msg)
