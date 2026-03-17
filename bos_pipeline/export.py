"""Data export utilities for BOS pipeline results.

Supported displacement field formats:
* NumPy ``.npy`` (lossless, fast)
* HDF5  ``.h5``  (structured, metadata-rich)
* CSV         (human-readable, single component per file)

Additionally writes a JSON processing log summarising all parameters,
file paths, and timestamps.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Displacement field export
# ---------------------------------------------------------------------------


def export_displacement(
    dx: np.ndarray,
    dy: np.ndarray,
    output_dir: Union[str, Path],
    stem: str = "displacement",
    fmt: str = "hdf5",
) -> Dict[str, Path]:
    """Save displacement arrays to *output_dir*.

    Parameters
    ----------
    dx, dy:
        Displacement component arrays (H × W, float32).
    output_dir:
        Destination directory (created if it does not exist).
    stem:
        Base filename without extension.
    fmt:
        One of ``"npy"``, ``"hdf5"``, ``"csv"``.

    Returns
    -------
    paths:
        Dict mapping component names to saved file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()
    if fmt == "npy":
        return _export_npy(dx, dy, out, stem)
    elif fmt in ("hdf5", "h5"):
        return _export_hdf5(dx, dy, out, stem)
    elif fmt == "csv":
        return _export_csv(dx, dy, out, stem)
    else:
        raise ValueError(
            f"Unknown export format '{fmt}'. Choose from: npy, hdf5, csv"
        )


def _export_npy(
    dx: np.ndarray,
    dy: np.ndarray,
    out: Path,
    stem: str,
) -> Dict[str, Path]:
    paths = {}
    for name, arr in [("dx", dx), ("dy", dy)]:
        p = out / f"{stem}_{name}.npy"
        np.save(str(p), arr)
        logger.info("Saved %s → %s", name, p)
        paths[name] = p
    return paths


def _export_hdf5(
    dx: np.ndarray,
    dy: np.ndarray,
    out: Path,
    stem: str,
) -> Dict[str, Path]:
    try:
        import h5py  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 export. "
            "Install it with: pip install h5py"
        ) from exc

    p = out / f"{stem}.h5"
    with h5py.File(str(p), "w") as f:
        f.create_dataset("dx", data=dx, compression="gzip", compression_opts=4)
        f.create_dataset("dy", data=dy, compression="gzip", compression_opts=4)
        mag = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
        f.create_dataset("magnitude", data=mag, compression="gzip", compression_opts=4)
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.attrs["shape_hw"] = list(dx.shape)
        f.attrs["dtype"] = str(dx.dtype)

    logger.info("Saved HDF5 displacement → %s", p)
    return {"dx": p, "dy": p, "hdf5": p}


def _export_csv(
    dx: np.ndarray,
    dy: np.ndarray,
    out: Path,
    stem: str,
) -> Dict[str, Path]:
    paths = {}
    for name, arr in [("dx", dx), ("dy", dy)]:
        p = out / f"{stem}_{name}.csv"
        np.savetxt(str(p), arr, delimiter=",", fmt="%.6f")
        logger.info("Saved %s → %s", name, p)
        paths[name] = p
    return paths


# ---------------------------------------------------------------------------
# Abel field export
# ---------------------------------------------------------------------------


def export_abel(
    inv_field: np.ndarray,
    output_dir: Union[str, Path],
    stem: str = "abel",
    fmt: str = "hdf5",
) -> Path:
    """Save Abel-inverted field.  Same format options as displacement."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()
    if fmt == "npy":
        p = out / f"{stem}_inv.npy"
        np.save(str(p), inv_field)
    elif fmt in ("hdf5", "h5"):
        try:
            import h5py  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("h5py required for HDF5 export.") from exc
        p = out / f"{stem}_inv.h5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset(
                "inv_field", data=inv_field,
                compression="gzip", compression_opts=4,
            )
    elif fmt == "csv":
        p = out / f"{stem}_inv.csv"
        np.savetxt(str(p), inv_field, delimiter=",", fmt="%.6f")
    else:
        raise ValueError(f"Unknown format: {fmt}")

    logger.info("Saved Abel field → %s", p)
    return p


# ---------------------------------------------------------------------------
# JSON processing log
# ---------------------------------------------------------------------------


def write_log(
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    processing_stats: Optional[Dict[str, Any]] = None,
    stem: str = "processing_log",
) -> Path:
    """Write a JSON summary of the processing run.

    Parameters
    ----------
    output_dir:
        Where to save the log file.
    config:
        Full resolved config dict (YAML params).
    input_paths:
        Mapping of role → path string (e.g. ``{"mraw": "/data/shot.mraw"}``).
    output_paths:
        Mapping of output role → path string.
    processing_stats:
        Optional dict with computed statistics (max displacement, etc.).
    stem:
        Log filename without ``.json`` extension.

    Returns
    -------
    log_path:
        Path to the written JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bos_pipeline_version": _get_version(),
        "config": _serialise(config),
        "inputs": input_paths,
        "outputs": {k: str(v) for k, v in output_paths.items()},
    }
    if processing_stats:
        log["stats"] = _serialise(processing_stats)

    log_path = out / f"{stem}.json"
    with open(str(log_path), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, default=str)

    logger.info("Processing log → %s", log_path)
    return log_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_version() -> str:
    try:
        from bos_pipeline import __version__
        return __version__
    except Exception:
        return "unknown"


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy types and Path objects to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
