"""Photron AVI reader — reads .avi video paired with .cihx/.cih metadata.

Uses OpenCV (cv2.VideoCapture) for frame access and parses the Photron
.cihx (XML) or .cih (text key=value) metadata file for acquisition parameters.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from bos_pipeline.io.base import CameraReader, FrameMetadata

logger = logging.getLogger(__name__)


class PhotronAviReader(CameraReader):
    """Read a Photron .avi file paired with a .cihx or .cih metadata file.

    Parameters
    ----------
    path:
        Path to the ``.avi`` file.
    metadata_file:
        Path to ``.cihx`` / ``.cih``.  Auto-detected from the same folder
        and stem if not provided.
    grayscale:
        If ``True`` (default) convert frames to single-channel float32.
        Set to ``False`` to keep colour frames (H × W × 3).
    """

    def __init__(
        self,
        path: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        grayscale: bool = True,
    ) -> None:
        super().__init__(path)
        self._meta_path: Optional[Path] = (
            Path(metadata_file) if metadata_file else None
        )
        self._grayscale = grayscale
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[FrameMetadata] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return
        if not self._path.exists():
            raise FileNotFoundError(f"AVI file not found: {self._path}")

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {self._path}")

        self._cap = cap
        meta_path = self._resolve_metadata_path()
        self._metadata = self._parse_metadata(meta_path)
        self._is_open = True
        logger.info(
            "Opened Photron AVI: %s  |  %dx%d px, %d frames, %.1f fps",
            self._path.name,
            self._metadata.width or 0,
            self._metadata.height or 0,
            self._metadata.total_frames or 0,
            self._metadata.frame_rate or 0,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
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
        if not self._is_open or self._cap is None:
            raise RuntimeError("Reader is not open.")

        total = self._metadata.total_frames if self._metadata else 0
        if total and (index < 0 or index >= total):
            raise IndexError(
                f"Frame index {index} out of range [0, {total - 1}]"
            )

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(
                f"Could not read frame {index} from {self._path}"
            )

        if self._grayscale:
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame.astype(np.float32)
        return frame.astype(np.float32)

    def get_average(
        self,
        indices: Sequence[int],
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """Stream-average frames (no full-video load)."""
        if not self._is_open:
            raise RuntimeError("Reader is not open.")
        if len(indices) == 0:
            raise ValueError("indices must be non-empty")
        acc: Optional[np.ndarray] = None
        for idx in indices:
            frame = self.get_frame(idx).astype(dtype)
            acc = frame if acc is None else acc + frame
        assert acc is not None
        return acc / len(indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_metadata_path(self) -> Optional[Path]:
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
                logger.debug("Auto-detected metadata: %s", candidate)
                return candidate

        logger.warning(
            "No .cihx/.cih found for %s — using OpenCV header only.",
            self._path.name,
        )
        return None

    def _parse_metadata(self, meta_path: Optional[Path]) -> FrameMetadata:
        """Merge OpenCV video properties with .cihx/.cih metadata."""
        cap = self._cap

        # --- OpenCV fallback values ---
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_tags: dict = {}

        # --- Parse .cihx (XML) or .cih (key=value text) ---
        if meta_path is not None:
            try:
                if meta_path.suffix.lower() == ".cihx":
                    raw_tags = _parse_cihx(meta_path)
                else:
                    raw_tags = _parse_cih(meta_path)
            except Exception as exc:
                logger.warning("Could not parse metadata %s: %s", meta_path, exc)

        def _get(key: str, cast=None):
            val = raw_tags.get(key)
            if val is not None and cast is not None:
                try:
                    return cast(val)
                except (ValueError, TypeError):
                    pass
            return val

        # Prefer metadata file values over OpenCV header
        width_meta  = _get("Image Width", int) or _get("imageWidth", int)
        height_meta = _get("Image Height", int) or _get("imageHeight", int)
        fps_meta    = (_get("Record Rate(fps)", float)
                       or _get("recordRate", float))
        total_meta  = (_get("Total Frame", int)
                       or _get("totalFrame", int))
        trigger     = (_get("Trigger Frame", int)
                       or _get("triggerFrame", int))
        exposure_s  = (_get("Shutter Speed(s)", float)
                       or _get("shutterSpeed", float))
        bit_depth   = (_get("Color Bit", int) or _get("colorBit", int))

        return FrameMetadata(
            width=width_meta or width,
            height=height_meta or height,
            bit_depth=bit_depth or 8,
            frame_rate=fps_meta or (fps if fps > 0 else None),
            total_frames=total_meta or (total if total > 0 else None),
            trigger_frame=trigger,
            exposure_us=(exposure_s * 1e6) if exposure_s else None,
            raw_tags=raw_tags,
        )


# ---------------------------------------------------------------------------
# .cihx XML parser
# ---------------------------------------------------------------------------

def _parse_cihx(path: Path) -> dict:
    """Parse a Photron .cihx XML file into a flat key→value dict.

    Photron .cihx files have a binary header before the XML payload.
    We scan for the ``<?xml`` marker and parse from there.
    """
    with open(str(path), "rb") as fh:
        raw = fh.read()

    xml_start = raw.find(b"<?xml")
    if xml_start == -1:
        raise ValueError("No XML content found in .cihx file.")

    xml_bytes = raw[xml_start:]
    root = ET.fromstring(xml_bytes.decode("utf-8", errors="replace"))
    tags: dict = {}

    # Strip namespace if present
    def _strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _walk(node):
        for child in node:
            key = _strip_ns(child.tag)
            if len(child) == 0:          # leaf node
                tags[key] = child.text
            else:
                _walk(child)

    _walk(root)
    return tags


# ---------------------------------------------------------------------------
# .cih text parser  (key : value  or  key = value)
# ---------------------------------------------------------------------------

def _parse_cih(path: Path) -> dict:
    """Parse a Photron legacy .cih text file into a flat dict."""
    tags: dict = {}
    with open(str(path), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            for sep in (" : ", " = ", "=", ":"):
                if sep in line:
                    key, _, val = line.partition(sep)
                    tags[key.strip()] = val.strip()
                    break
    return tags
