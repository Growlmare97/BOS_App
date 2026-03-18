"""Camera I/O sub-package."""

from bos_pipeline.io.base import CameraReader, FrameMetadata  # noqa: F401
from bos_pipeline.io.photron import PhotronReader  # noqa: F401
from bos_pipeline.io.dalsa import DalsaReader  # noqa: F401
from bos_pipeline.io.avi import PhotronAviReader  # noqa: F401


def get_reader(camera_type: str, **kwargs) -> CameraReader:
    """Factory: return the appropriate CameraReader for *camera_type*.

    Parameters
    ----------
    camera_type:
        One of ``"photron"``, ``"photron_avi"``, ``"dalsa"``, ``"tiff_sequence"``.
    **kwargs:
        Forwarded to the reader constructor.
    """
    registry = {
        "photron": PhotronReader,
        "photron_avi": PhotronAviReader,
        "dalsa": DalsaReader,
        "tiff_sequence": DalsaReader,
    }
    key = camera_type.lower()
    if key not in registry:
        raise ValueError(
            f"Unknown camera type '{camera_type}'. "
            f"Choose from: {list(registry)}"
        )
    return registry[key](**kwargs)
