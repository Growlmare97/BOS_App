"""
bos_pipeline — Background-Oriented Schlieren image processing pipeline.

Supports Photron FASTCAM (.mraw/.cihx) and Teledyne DALSA (TIFF sequences,
GigE Vision live) camera inputs.  Provides cross-correlation and optical-flow
displacement computation, optional Abel inversion for axisymmetric flows,
publication-quality visualisation, and flexible data export.
"""

__version__ = "0.1.0"
__author__ = "CoStudy GmbH"

from bos_pipeline import io, processing, visualization, export, cli  # noqa: F401
