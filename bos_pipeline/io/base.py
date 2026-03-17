"""Abstract base class for all camera readers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np


@dataclass
class FrameMetadata:
    """Metadata associated with a camera acquisition.

    All fields are *optional* — readers populate only those available from
    the camera format.  Consumer code should check for ``None`` before use.
    """

    # Spatial
    width: Optional[int] = None
    height: Optional[int] = None
    bit_depth: Optional[int] = None

    # Temporal
    frame_rate: Optional[float] = None       # frames per second
    total_frames: Optional[int] = None
    trigger_frame: Optional[int] = None      # index of trigger event
    exposure_us: Optional[float] = None      # exposure time in microseconds

    # Timestamps — one entry per frame (seconds from start), if available
    timestamps: Optional[np.ndarray] = None

    # Raw tag dict (format-specific key/value pairs)
    raw_tags: dict = field(default_factory=dict)


class CameraReader(abc.ABC):
    """Abstract interface that every camera reader must implement.

    Usage pattern::

        reader = SomeCameraReader(path)
        with reader:
            meta = reader.metadata
            ref  = reader.get_frame(0)                  # single frame
            avg  = reader.get_average(range(0, 10))     # temporal average
            for frame in reader.iter_frames([0, 5, 10]):
                process(frame)
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._is_open: bool = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraReader":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def open(self) -> None:
        """Open file handles / connections.  Must be idempotent."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release all resources."""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def metadata(self) -> FrameMetadata:
        """Return acquisition metadata."""

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_frame(self, index: int) -> np.ndarray:
        """Return a single frame as a 2-D ``float32`` array (H × W).

        Implementations must *not* load the entire file into RAM.
        """

    def get_average(
        self,
        indices: Sequence[int],
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """Return the temporal mean of the frames at *indices*.

        The default implementation streams frames one by one to keep
        memory usage bounded.  Readers may override this for efficiency.
        """
        if len(indices) == 0:
            raise ValueError("indices must be non-empty")
        acc: Optional[np.ndarray] = None
        for idx in indices:
            frame = self.get_frame(idx).astype(dtype)
            if acc is None:
                acc = frame
            else:
                acc += frame
        assert acc is not None
        return acc / len(indices)

    def iter_frames(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> Iterator[np.ndarray]:
        """Yield frames one at a time.

        Parameters
        ----------
        indices:
            Frame indices to yield.  If ``None``, yields all frames in order.
        """
        if not self._is_open:
            raise RuntimeError(
                f"{self.__class__.__name__} is not open. "
                "Use it as a context manager or call open() first."
            )
        meta = self.metadata
        if indices is None:
            if meta.total_frames is None:
                raise RuntimeError(
                    "Cannot iterate all frames: total_frames not available. "
                    "Provide explicit indices."
                )
            indices = range(meta.total_frames)
        for idx in indices:
            yield self.get_frame(idx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"path={self._path!r}, open={self._is_open})"
        )
