"""Command-line interface for the BOS processing pipeline.

Usage examples
--------------
::

    bos_process --config config.yaml

    bos_process --input ./data --camera photron \
                --reference 0 --output ./output

    bos_process --live --camera dalsa \
                --cti /opt/genicam/DALSA.cti --output ./output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger("bos_pipeline")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bos_process",
        description=(
            "Background-Oriented Schlieren (BOS) image processing pipeline.\n"
            "Supports Photron FASTCAM and Teledyne DALSA cameras."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file (overrides all CLI args if provided)
    p.add_argument(
        "--config", "-c",
        metavar="CONFIG.yaml",
        help="Path to YAML config file.  All parameters can be set here; "
             "CLI flags override individual keys.",
    )

    # Input
    input_grp = p.add_argument_group("Input")
    input_grp.add_argument(
        "--input", "-i",
        metavar="PATH",
        help=".mraw file (Photron) or folder of TIFFs (DALSA/tiff_sequence).",
    )
    input_grp.add_argument(
        "--camera",
        choices=["photron", "photron_avi", "dalsa", "tiff_sequence"],
        default="photron",
        help="Camera type / input format (default: photron).",
    )
    input_grp.add_argument(
        "--metadata",
        metavar="FILE",
        help="Path to .cihx / .cih metadata (Photron only; auto-detected if omitted).",
    )
    input_grp.add_argument(
        "--reference",
        metavar="FRAME",
        default="0",
        help="Reference frame index or range 'start:stop' (default: 0).",
    )
    input_grp.add_argument(
        "--measurement",
        metavar="FRAMES",
        default="all",
        help="Measurement frame indices: single int, comma-separated list, "
             "range 'start:stop', or 'all' (default: all).",
    )

    # Live acquisition
    live_grp = p.add_argument_group("Live acquisition")
    live_grp.add_argument(
        "--live",
        action="store_true",
        help="Enable live GigE Vision acquisition via Harvesters.",
    )
    live_grp.add_argument(
        "--cti",
        metavar="FILE",
        help="GenTL producer .cti file path (required for --live).",
    )
    live_grp.add_argument(
        "--n-frames",
        type=int,
        default=100,
        metavar="N",
        help="Number of frames to acquire in live mode (default: 100).",
    )

    # Processing
    proc_grp = p.add_argument_group("Processing")
    proc_grp.add_argument(
        "--method",
        choices=["cross_correlation", "lucas_kanade", "farneback"],
        default=None,
        help="Displacement computation method.",
    )
    proc_grp.add_argument(
        "--window",
        type=int,
        default=None,
        metavar="PX",
        help="Interrogation window size in pixels.",
    )
    proc_grp.add_argument(
        "--overlap",
        type=float,
        default=None,
        metavar="FRAC",
        help="Window overlap fraction [0–0.9].",
    )
    proc_grp.add_argument(
        "--no-abel",
        action="store_true",
        help="Disable Abel inversion even if enabled in config.",
    )
    proc_grp.add_argument(
        "--abel-method",
        choices=["three_point", "basex", "hansenlaw"],
        default=None,
        help="Abel inversion method.",
    )
    proc_grp.add_argument(
        "--sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Gaussian pre-filter sigma [px] (0 = disabled).",
    )

    # Output
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument(
        "--output", "-o",
        metavar="DIR",
        default="./output",
        help="Output directory (default: ./output).",
    )
    out_grp.add_argument(
        "--format",
        choices=["npy", "hdf5", "csv"],
        default=None,
        help="Displacement field export format.",
    )
    out_grp.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation.",
    )
    out_grp.add_argument(
        "--show",
        action="store_true",
        help="Open interactive Matplotlib windows.",
    )

    # Misc
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=_get_version(),
    )

    return p


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    # ------------------------------------------------------------------
    # Load + merge config
    # ------------------------------------------------------------------
    config = _load_config(args.config) if args.config else {}
    config = _merge_cli_args(config, args)

    logger.debug("Resolved config:\n%s", yaml.dump(config, default_flow_style=False))

    # ------------------------------------------------------------------
    # Dispatch: live vs. file-based
    # ------------------------------------------------------------------
    if args.live or config.get("live", {}).get("enabled", False):
        return _run_live(config)
    else:
        return _run_file(config)


# ---------------------------------------------------------------------------
# File-based processing
# ---------------------------------------------------------------------------


def _run_file(config: Dict[str, Any]) -> int:
    from bos_pipeline.io import get_reader
    from bos_pipeline.processing.preprocess import PreprocessConfig, preprocess
    from bos_pipeline.processing.displacement import (
        DisplacementConfig, compute_displacement, displacement_magnitude,
        interpolate_to_full_resolution,
    )
    from bos_pipeline.processing.abel import AbelConfig, abel_invert
    from bos_pipeline import visualization as viz
    from bos_pipeline import export

    cam_cfg = config.get("camera", {})
    camera_type = cam_cfg.get("type", "photron")
    input_path = cam_cfg.get("input_path") or config.get("input")

    if not input_path:
        logger.error("No input path specified. Use --input or set camera.input_path in config.")
        return 1

    reader_kwargs: Dict[str, Any] = {"path": input_path}
    if camera_type in ("photron", "photron_avi"):
        if meta := cam_cfg.get("metadata_file"):
            reader_kwargs["metadata_file"] = meta
    elif camera_type in ("dalsa", "tiff_sequence"):
        tiff_cfg = config.get("tiff", {})
        if pattern := tiff_cfg.get("pattern"):
            reader_kwargs["pattern"] = pattern
        reader_kwargs["start_index"] = tiff_cfg.get("start_index", 0)
        reader_kwargs["multipage"] = tiff_cfg.get("multipage", False)

    # Parse frame indices
    ref_index = _parse_frame_index(cam_cfg.get("reference_frame", 0))
    meas_indices = _parse_frame_indices(
        cam_cfg.get("measurement_frames", "all")
    )

    output_dir = Path(config.get("export", {}).get("output_dir", "./output"))
    disp_fmt = config.get("export", {}).get("displacement_format", "hdf5")
    save_figs = config.get("export", {}).get("save_figures", True)
    fig_formats = config.get("export", {}).get("figure_format", ["png"])
    show = config.get("visualization", {}).get("show_interactive", False)
    dpi = config.get("visualization", {}).get("dpi", 300)

    pre_cfg = PreprocessConfig.from_dict(config.get("preprocessing", {}))
    disp_cfg = DisplacementConfig.from_dict(config.get("displacement", {}))
    abel_cfg = AbelConfig.from_dict(config.get("abel", {}))

    output_paths: Dict[str, str] = {}

    with get_reader(camera_type, **reader_kwargs) as reader:
        meta = reader.metadata
        logger.info(
            "Opened %s: %dx%d px, %d frames, %.0f fps",
            camera_type,
            meta.width or 0, meta.height or 0,
            meta.total_frames or 0, meta.frame_rate or 0,
        )

        total = meta.total_frames or 0
        if meas_indices is None:
            meas_indices = list(range(total))

        # Build reference
        n_ref = pre_cfg.reference_avg_frames
        ref_idxs = list(range(ref_index, min(ref_index + n_ref, total)))
        logger.info("Building reference from %d frame(s): %s", len(ref_idxs), ref_idxs)
        reference_raw = reader.get_average(ref_idxs)

        all_dx, all_dy = [], []
        saved_disp_paths: Dict[str, Any] = {}

        for meas_idx in meas_indices:
            n_meas = pre_cfg.measurement_avg_frames
            m_idxs = list(range(meas_idx, min(meas_idx + n_meas, total)))
            meas_raw = reader.get_average(m_idxs)

            ref_proc, meas_proc = preprocess(reference_raw, meas_raw, config=pre_cfg)

            result = compute_displacement(ref_proc, meas_proc, config=disp_cfg)
            dx, dy = result.dx, result.dy

            # Up-sample cross-correlation grid to full image resolution
            if disp_cfg.method == "cross_correlation" and result.grid_x is not None:
                dx, dy = interpolate_to_full_resolution(
                    dx, dy, ref_proc.shape, result=result
                )
                logger.info("Up-sampled displacement to full resolution %s.", ref_proc.shape)

            all_dx.append(dx)
            all_dy.append(dy)

            stem = f"frame_{meas_idx:05d}"
            paths = export.export_displacement(dx, dy, output_dir, stem=stem, fmt=disp_fmt)
            saved_disp_paths[stem] = {k: str(v) for k, v in paths.items()}

            if save_figs and not config.get("no_figures", False):
                cmap_mag = config.get("visualization", {}).get("colormap_magnitude", "viridis")
                ds = config.get("visualization", {}).get("quiver_downsample", 16)

                fig_mag, _ = viz.plot_displacement_magnitude(
                    dx, dy, cmap=cmap_mag, dpi=dpi, show=show
                )
                viz.save_figure(fig_mag, output_dir / "figures", f"{stem}_magnitude", fig_formats, dpi)

                fig_comp, _ = viz.plot_displacement_components(
                    dx, dy, dpi=dpi, show=show
                )
                viz.save_figure(fig_comp, output_dir / "figures", f"{stem}_components", fig_formats, dpi)

                fig_q, _ = viz.plot_quiver(
                    ref_proc, dx, dy, downsample=ds, dpi=dpi, show=show
                )
                viz.save_figure(fig_q, output_dir / "figures", f"{stem}_quiver", fig_formats, dpi)

                import matplotlib.pyplot as plt
                plt.close("all")

        # Abel inversion on the last (or mean) frame
        if abel_cfg.enabled and all_dx:
            dx_last, dy_last = all_dx[-1], all_dy[-1]
            inv_field, axis_arr = abel_invert(dx_last, dy_last, config=abel_cfg)
            axis_col = int(axis_arr[0]) if axis_arr is not None else None

            abel_path = export.export_abel(inv_field, output_dir, stem="abel", fmt=disp_fmt)
            output_paths["abel"] = str(abel_path)

            if save_figs:
                fig_ab, _ = viz.plot_abel_field(
                    inv_field, axis_col=axis_col, dpi=dpi, show=show
                )
                viz.save_figure(fig_ab, output_dir / "figures", "abel_field", fig_formats, dpi)

        output_paths["displacement"] = str(output_dir)

        # JSON log
        stats = {
            "n_measurement_frames": len(all_dx),
            "max_dx_px": float(np.max([np.abs(d).max() for d in all_dx])) if all_dx else 0,
            "max_dy_px": float(np.max([np.abs(d).max() for d in all_dy])) if all_dy else 0,
        }
        log_path = export.write_log(
            output_dir,
            config=config,
            input_paths={"input": str(input_path)},
            output_paths=output_paths,
            processing_stats=stats,
        )
        logger.info("Done. Log: %s", log_path)

    return 0


# ---------------------------------------------------------------------------
# Live acquisition processing
# ---------------------------------------------------------------------------


def _run_live(config: Dict[str, Any]) -> int:
    from bos_pipeline.io.dalsa import HarvestersLiveReader
    from bos_pipeline.processing.preprocess import PreprocessConfig, preprocess
    from bos_pipeline.processing.displacement import DisplacementConfig, compute_displacement
    from bos_pipeline import visualization as viz
    from bos_pipeline import export
    import matplotlib.pyplot as plt

    live_cfg = config.get("live", {})
    cti = live_cfg.get("cti_file") or config.get("cti")
    if not cti:
        logger.error("No CTI file specified for live acquisition (--cti or live.cti_file).")
        return 1

    n_frames = live_cfg.get("n_frames", config.get("n_frames", 100))
    output_dir = Path(config.get("export", {}).get("output_dir", "./output"))
    disp_fmt = config.get("export", {}).get("displacement_format", "hdf5")

    pre_cfg = PreprocessConfig.from_dict(config.get("preprocessing", {}))
    disp_cfg = DisplacementConfig.from_dict(config.get("displacement", {}))

    reader = HarvestersLiveReader(
        cti_file=cti,
        exposure_us=live_cfg.get("exposure_us", 1000.0),
        gain=live_cfg.get("gain", 1.0),
    )

    with reader:
        logger.info("Capturing reference frame...")
        reference = reader.get_frame()

        logger.info("Acquiring %d measurement frames...", n_frames)
        for i, meas_raw in enumerate(reader.iter_frames(n_frames=n_frames)):
            ref_proc, meas_proc = preprocess(
                reference, meas_raw, config=pre_cfg
            )
            dx, dy = compute_displacement(ref_proc, meas_proc, config=disp_cfg)
            export.export_displacement(
                dx, dy, output_dir, stem=f"live_{i:05d}", fmt=disp_fmt
            )

    logger.info("Live acquisition complete.")
    return 0


# ---------------------------------------------------------------------------
# Config / argument helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(str(p), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.debug("Loaded config from %s", p)
    return cfg


def _merge_cli_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Overlay explicit CLI args onto *config* dict."""
    cam = config.setdefault("camera", {})
    if args.input:
        cam["input_path"] = args.input
    if args.camera:
        cam["type"] = args.camera
    if args.metadata:
        cam["metadata_file"] = args.metadata
    if args.reference != "0":
        cam["reference_frame"] = args.reference
    if args.measurement != "all":
        cam["measurement_frames"] = args.measurement

    disp = config.setdefault("displacement", {})
    if args.method:
        disp["method"] = args.method
    if args.window:
        disp["window_size"] = args.window
    if args.overlap is not None:
        disp["overlap"] = args.overlap

    pre = config.setdefault("preprocessing", {})
    if args.sigma is not None:
        pre["gaussian_sigma"] = args.sigma

    abel = config.setdefault("abel", {})
    if args.no_abel:
        abel["enabled"] = False
    if args.abel_method:
        abel["method"] = args.abel_method

    exp = config.setdefault("export", {})
    if args.output:
        exp["output_dir"] = args.output
    if args.format:
        exp["displacement_format"] = args.format
    if args.no_figures:
        exp["save_figures"] = False

    vis = config.setdefault("visualization", {})
    if args.show:
        vis["show_interactive"] = True

    if args.live:
        config.setdefault("live", {})["enabled"] = True
    if hasattr(args, "cti") and args.cti:
        config.setdefault("live", {})["cti_file"] = args.cti
    if hasattr(args, "n_frames"):
        config.setdefault("live", {})["n_frames"] = args.n_frames

    return config


def _parse_frame_index(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _parse_frame_indices(val) -> Optional[List[int]]:
    if val is None or val == "all":
        return None
    if isinstance(val, (list, range)):
        return list(val)
    s = str(val)
    if ":" in s:
        parts = s.split(":")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else None
        if stop is None:
            return None
        return list(range(start, stop))
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    try:
        return [int(s)]
    except ValueError:
        return None


def _get_version() -> str:
    try:
        from bos_pipeline import __version__
        return f"bos_process {__version__}"
    except Exception:
        return "bos_process (unknown version)"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    sys.exit(main())
