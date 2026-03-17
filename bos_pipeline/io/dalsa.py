"""Teledyne DALSA camera reader.

Supports:
* TIFF sequences exported from Sapera LT (single-page 16-bit mono TIFFs,
  numbered sequentially).
* Multi-page TIFF stacks.
* Live GigE Vision / GenICam acquisition via the ``harvesters`` library
  (camera-agnostic fallback if the Sapera SDK is unavailable).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np

from bos_pipeline.io.base import CameraReader, FrameMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TIFF Sequence Reader
# ---------------------------------------------------------------------------


class DalsaReader(CameraReader):
    """Read a TIFF image sequence (or multi-page TIFF) exported from Sapera LT.

    Parameters
    ----------
    path:
        Path to either:

        * A **folder** containing numbered single-page TIFFs, or
        * A single **multi-page TIFF** file.
    pattern:
        ``str.format``-style pattern for numbered files, e.g.
        ``"frame_{:05d}.tiff"``.  Used only when *path* is a directory.
        If ``None``, all ``*.tif`` / ``*.tiff`` files are sorted
        lexicographically.
    start_index:
        First frame index for pattern-based enumeration.
    multipage:
        Force multi-page TIFF mode even when *path* is a directory.
    """

    def __init__(
        self,
        path: Union[str, Path],
        pattern: Optional[str] = None,
        start_index: int = 0,
        multipage: bool = False,
    ) -> None:
        super().__init__(path)
        self._pattern = pattern
        self._start_index = start_index
        self._multipage = multipage

        self._file_list: List[Path] = []
        self._tifffile = None          # tifffile.TiffFile for multipage
        self._metadata: Optional[FrameMetadata] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return
        try:
            import tifffile  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "tifffile is required for TIFF sequences. "
                "Install it with: pip install tifffile"
            ) from exc

        self._tifflib = tifffile

        if self._path.is_file() or self._multipage:
            self._open_multipage()
        else:
            self._open_sequence()

        self._metadata = self._parse_metadata()
        self._is_open = True

    def close(self) -> None:
        if self._tifffile is not None:
            self._tifffile.close()
            self._tifffile = None
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

        total = self._metadata.total_frames if self._metadata else 0
        if index < 0 or (total and index >= total):
            raise IndexError(
                f"Frame index {index} out of range [0, {(total or 1) - 1}]"
            )

        if self._tifffile is not None:
            # Multi-page TIFF
            page = self._tifffile.pages[index]
            frame = page.asarray()
        else:
            # Sequence
            tiff_path = self._file_list[index]
            frame = self._tifflib.imread(str(tiff_path))

        return frame.astype(np.float32)

    def iter_frames(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> Iterator[np.ndarray]:
        """Yield frames one at a time (lazy, no full-stack load)."""
        if not self._is_open:
            raise RuntimeError("Reader is not open.")
        meta = self.metadata
        if indices is None:
            indices = range(meta.total_frames or 0)
        for idx in indices:
            yield self.get_frame(idx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_multipage(self) -> None:
        """Open a multi-page TIFF file."""
        if not self._path.exists():
            raise FileNotFoundError(f"TIFF file not found: {self._path}")
        logger.debug("Opening multi-page TIFF: %s", self._path)
        self._tifffile = self._tifflib.TiffFile(str(self._path))

    def _open_sequence(self) -> None:
        """Discover and sort single-page TIFF files in a directory."""
        folder = self._path
        if not folder.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {folder}. "
                "For a single TIFF file use multipage=True."
            )

        if self._pattern:
            # Pattern-based enumeration — find how many files exist
            files: List[Path] = []
            idx = self._start_index
            while True:
                candidate = folder / self._pattern.format(idx)
                if not candidate.exists():
                    break
                files.append(candidate)
                idx += 1
        else:
            # Lexicographic sort of all TIFF files
            files = sorted(
                list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
            )

        if not files:
            raise FileNotFoundError(
                f"No TIFF files found in {folder} "
                f"(pattern={self._pattern!r})"
            )

        logger.debug(
            "Found %d TIFF frames in %s (first: %s)",
            len(files), folder, files[0].name,
        )
        self._file_list = files

    def _parse_metadata(self) -> FrameMetadata:
        """Read image dimensions and optional embedded tags from the first frame."""
        width = height = bit_depth = None
        frame_rate = exposure_us = None
        raw_tags: dict = {}

        # Read from the first frame
        if self._tifffile is not None:
            first_page = self._tifffile.pages[0]
            arr = first_page.asarray()
            height, width = arr.shape[-2], arr.shape[-1]
            bit_depth = arr.dtype.itemsize * 8
            total_frames = len(self._tifffile.pages)
            raw_tags = _extract_tiff_tags(first_page)
        elif self._file_list:
            arr = self._tifflib.imread(str(self._file_list[0]))
            height, width = arr.shape[-2], arr.shape[-1]
            bit_depth = arr.dtype.itemsize * 8
            total_frames = len(self._file_list)
            try:
                with self._tifflib.TiffFile(str(self._file_list[0])) as tf:
                    raw_tags = _extract_tiff_tags(tf.pages[0])
            except Exception:
                pass
        else:
            total_frames = 0

        # Try to extract frame rate / exposure from known TIFF tags
        frame_rate = _tag_to_float(raw_tags, "FrameRate")
        exposure_us = _tag_to_float(raw_tags, "ExposureTime")

        return FrameMetadata(
            width=width,
            height=height,
            bit_depth=bit_depth,
            frame_rate=frame_rate,
            total_frames=total_frames,
            exposure_us=exposure_us,
            raw_tags=raw_tags,
        )


def _extract_tiff_tags(page) -> dict:
    """Return a flat dict of TIFF tag name → value for a tifffile page."""
    tags = {}
    try:
        for tag in page.tags.values():
            tags[tag.name] = tag.value
    except Exception:
        pass
    return tags


def _tag_to_float(tags: dict, key: str) -> Optional[float]:
    val = tags.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Live acquisition via Harvesters (GenICam / GigE Vision)
# ---------------------------------------------------------------------------


class HarvestersLiveReader(CameraReader):
    """Live camera acquisition using the ``harvesters`` GenICam library.

    This is a camera-agnostic fallback for GigE Vision / USB3 Vision devices
    when the proprietary Sapera SDK is unavailable.

    Parameters
    ----------
    cti_file:
        Path to the GenTL producer ``.cti`` shared library.
    exposure_us:
        Sensor exposure time in microseconds.
    gain:
        Analogue gain.
    buffer_count:
        Number of image buffers to pre-allocate in the acquisition queue.
    """

    def __init__(
        self,
        cti_file: Union[str, Path],
        exposure_us: float = 1000.0,
        gain: float = 1.0,
        buffer_count: int = 16,
    ) -> None:
        super().__init__(Path(cti_file))
        self._exposure_us = exposure_us
        self._gain = gain
        self._buffer_count = buffer_count
        self._harvester = None
        self._ia = None            # ImageAcquirer
        self._metadata: Optional[FrameMetadata] = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return
        try:
            from harvesters.core import Harvester  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "harvesters is required for live acquisition. "
                "Install it with: pip install harvesters genicam"
            ) from exc

        cti = str(self._path)
        logger.info("Initialising Harvesters with CTI: %s", cti)

        h = Harvester()
        h.add_file(cti)
        h.update()

        if not h.device_info_list:
            h.reset()
            raise RuntimeError(
                f"No GenICam devices found via CTI: {cti}"
            )

        ia = h.create()
        node_map = ia.remote_device.node_map

        # Configure camera
        try:
            node_map.ExposureTime.value = self._exposure_us
        except Exception:
            logger.warning("Could not set ExposureTime — node may not exist.")
        try:
            node_map.Gain.value = self._gain
        except Exception:
            logger.warning("Could not set Gain — node may not exist.")

        ia.num_buffers = self._buffer_count
        ia.start()

        self._harvester = h
        self._ia = ia
        self._metadata = self._build_metadata(node_map)
        self._is_open = True
        logger.info("Live acquisition started.")

    def close(self) -> None:
        if self._ia is not None:
            try:
                self._ia.stop()
                self._ia.destroy()
            except Exception as exc:
                logger.warning("Error stopping image acquirer: %s", exc)
            self._ia = None
        if self._harvester is not None:
            self._harvester.reset()
            self._harvester = None
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
    # Frame access — live grab
    # ------------------------------------------------------------------

    def get_frame(self, index: int = 0) -> np.ndarray:  # type: ignore[override]
        """Grab the *next* frame from the live stream.

        The *index* parameter is ignored for live acquisition (each call
        returns the next available frame).
        """
        if not self._is_open or self._ia is None:
            raise RuntimeError("Reader is not open.")

        with self._ia.fetch() as buffer:
            component = buffer.payload.components[0]
            data = component.data.copy()
            width = component.width
            height = component.height

        frame = data.reshape(height, width).astype(np.float32)
        self._frame_count += 1
        return frame

    def iter_frames(  # type: ignore[override]
        self,
        n_frames: Optional[int] = None,
        timeout_s: float = 10.0,
    ) -> Iterator[np.ndarray]:
        """Yield frames from the live stream.

        Parameters
        ----------
        n_frames:
            Stop after *n_frames* frames.  ``None`` = stream indefinitely.
        timeout_s:
            Per-frame grab timeout in seconds.
        """
        if not self._is_open:
            raise RuntimeError("Reader is not open.")
        count = 0
        deadline = time.monotonic() + timeout_s
        while n_frames is None or count < n_frames:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Live stream timed out after {timeout_s} s"
                )
            yield self.get_frame()
            count += 1
            deadline = time.monotonic() + timeout_s  # reset per-frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(node_map) -> FrameMetadata:
        def _safe(attr: str, cast=float):
            try:
                return cast(getattr(node_map, attr).value)
            except Exception:
                return None

        return FrameMetadata(
            width=_safe("Width", int),
            height=_safe("Height", int),
            bit_depth=_safe("PixelSize", int),
            frame_rate=_safe("AcquisitionFrameRate"),
            exposure_us=_safe("ExposureTime"),
        )
