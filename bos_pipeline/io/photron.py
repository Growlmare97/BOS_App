"""Photron FASTCAM reader — wraps pyMRAW with memory-mapped access.

Supports .mraw files paired with .cihx (XML) or legacy .cih (text) metadata.
Frames are accessed via ``np.memmap`` for 8-bit and 16-bit files; 12-bit files
require a full load and will emit a warning.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from bos_pipeline.io.base import CameraReader, FrameMetadata

logger = logging.getLogger(__name__)


class PhotronReader(CameraReader):
    """Read Photron .mraw files paired with .cihx / .cih metadata.

    Parameters
    ----------
    path:
        Path to the ``.mraw`` file.
    metadata_file:
        Path to the accompanying ``.cihx`` or ``.cih`` file.
        If ``None``, the reader will look for a file with the same stem.
    """

    def __init__(
        self,
        path: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(path)
        self._meta_path: Optional[Path] = (
            Path(metadata_file) if metadata_file else None
        )
        self._mraw = None          # pyMRAW object
        self._memmap: Optional[np.memmap] = None
        self._metadata: Optional[FrameMetadata] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return
        try:
            import pymraw  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pyMRAW is required for Photron files. "
                "Install it with: pip install pyMRAW"
            ) from exc

        mraw_path = self._path
        if not mraw_path.exists():
            raise FileNotFoundError(f"MRAW file not found: {mraw_path}")

        # Locate metadata file
        meta_path = self._resolve_metadata_path()

        logger.debug("Opening Photron file: %s (meta: %s)", mraw_path, meta_path)
        self._mraw = pymraw.get_mraw(str(mraw_path), str(meta_path))
        self._metadata = self._parse_metadata()

        bit_depth = self._metadata.bit_depth or 16
        if bit_depth == 12:
            warnings.warn(
                "12-bit MRAW files require loading the full dataset into RAM. "
                "This may exhaust memory for large recordings.",
                UserWarning,
                stacklevel=2,
            )
            # pyMRAW handles 12-bit loading internally; no custom memmap
        else:
            self._memmap = self._build_memmap(bit_depth)

        self._is_open = True

    def close(self) -> None:
        if self._memmap is not None:
            del self._memmap
            self._memmap = None
        self._mraw = None
        self._is_open = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> FrameMetadata:
        if self._metadata is None:
            raise RuntimeError("Reader is not open.")
        return self._metadata

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, index: int) -> np.ndarray:
        """Return frame *index* as a 2-D float32 array (H × W)."""
        if not self._is_open:
            raise RuntimeError("Reader is not open.")

        meta = self._metadata
        total = meta.total_frames or 0
        if index < 0 or index >= total:
            raise IndexError(
                f"Frame index {index} out of range [0, {total - 1}]"
            )

        bit_depth = meta.bit_depth or 16

        if bit_depth == 12 or self._memmap is None:
            # Fall back to pyMRAW's own frame accessor
            frame = self._mraw[index]
        else:
            frame = self._memmap[index]

        return frame.astype(np.float32)

    def get_average(
        self,
        indices: Sequence[int],
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """Temporal mean — uses memmap slice for 8/16-bit, streaming for 12-bit."""
        if not self._is_open:
            raise RuntimeError("Reader is not open.")
        if len(indices) == 0:
            raise ValueError("indices must be non-empty")

        bit_depth = (self._metadata.bit_depth or 16) if self._metadata else 16

        if bit_depth != 12 and self._memmap is not None:
            idx_arr = np.asarray(indices)
            return self._memmap[idx_arr].mean(axis=0).astype(dtype)

        # Streaming fallback (12-bit)
        return super().get_average(indices, dtype=dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_metadata_path(self) -> Path:
        if self._meta_path is not None:
            if not self._meta_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {self._meta_path}"
                )
            return self._meta_path

        stem = self._path.stem
        parent = self._path.parent
        for ext in (".cihx", ".cih"):
            candidate = parent / (stem + ext)
            if candidate.exists():
                logger.debug("Auto-detected metadata file: %s", candidate)
                return candidate

        raise FileNotFoundError(
            f"Could not find .cihx or .cih metadata for {self._path}. "
            "Provide it explicitly via the metadata_file argument."
        )

    def _parse_metadata(self) -> FrameMetadata:
        """Extract metadata from the pyMRAW object."""
        mraw = self._mraw
        # pyMRAW exposes metadata via the .cih attribute (a dict-like object)
        cih = getattr(mraw, "cih", {}) or {}

        def _get(key: str, cast=None):
            val = cih.get(key)
            if val is not None and cast is not None:
                try:
                    val = cast(val)
                except (ValueError, TypeError):
                    val = None
            return val

        width = _get("Image Width", int)
        height = _get("Image Height", int)
        bit_depth = _get("Color Bit", int)
        frame_rate = _get("Record Rate(fps)", float)
        total_frames = _get("Total Frame", int)
        trigger_frame = _get("Trigger Frame", int)
        exposure_us = _get("Shutter Speed(s)", float)
        if exposure_us is not None:
            exposure_us = exposure_us * 1e6  # convert s → µs

        # Fallback: interrogate array shape
        try:
            arr_shape = mraw.shape  # (n_frames, H, W) for recent pyMRAW
            if total_frames is None and len(arr_shape) == 3:
                total_frames = arr_shape[0]
            if height is None and len(arr_shape) >= 2:
                height = arr_shape[-2]
            if width is None and len(arr_shape) >= 1:
                width = arr_shape[-1]
        except AttributeError:
            pass

        return FrameMetadata(
            width=width,
            height=height,
            bit_depth=bit_depth,
            frame_rate=frame_rate,
            total_frames=total_frames,
            trigger_frame=trigger_frame,
            exposure_us=exposure_us,
            raw_tags=dict(cih),
        )

    def _build_memmap(self, bit_depth: int) -> Optional[np.memmap]:
        """Build a read-only np.memmap over the raw binary .mraw data.

        The MRAW binary layout is a flat row-major array of frames immediately
        following a small header.  pyMRAW exposes the header offset via
        ``self._mraw.data_start`` (or similar); we fall back to 0 when not
        available so that real offset detection is delegated to pyMRAW itself.
        """
        meta = self._metadata
        if meta is None or meta.total_frames is None:
            return None
        if meta.height is None or meta.width is None:
            return None

        dtype_map = {8: np.uint8, 16: np.uint16}
        np_dtype = dtype_map.get(bit_depth)
        if np_dtype is None:
            return None  # unsupported, will use pyMRAW streaming

        offset = getattr(self._mraw, "data_start", 0) or 0

        try:
            mmap = np.memmap(
                self._path,
                dtype=np_dtype,
                mode="r",
                offset=offset,
                shape=(meta.total_frames, meta.height, meta.width),
            )
            logger.debug(
                "memmap shape=%s dtype=%s offset=%d",
                mmap.shape, mmap.dtype, offset,
            )
            return mmap
        except Exception as exc:
            logger.warning(
                "Could not create memmap for %s (%s). "
                "Falling back to pyMRAW streaming.",
                self._path, exc,
            )
            return None
