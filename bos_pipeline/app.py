"""BOS Analysis — PySide6 Desktop Application.

Launch with:
    bos_app                      # after pip install -e .
    python -m bos_pipeline.app   # without installing

Features
--------
* Load a Photron .avi + .cihx pair via file dialogs (auto-detected)
* Live frame preview — slider scrubs through the video in real time
* Configurable processing parameters (method, window, overlap, sigma)
* Background pipeline thread — UI stays responsive during computation
* Six result tabs: Frame View · Magnitude · Components · Quiver ·
  Concentration · Velocity
* Velocity estimation from consecutive displacement fields
* Stats bar: max/mean displacement, background noise, SNR
* Result navigator — switch between computed measurement frames
* Auto-save all figures and .npy displacement fields on completion
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# ── Matplotlib backend must be set BEFORE any bos_pipeline import ────────────
import matplotlib
matplotlib.use("QtAgg")

# ── PySide6 ──────────────────────────────────────────────────────────────────
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QStatusBar, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

# ── Matplotlib / NumPy ───────────────────────────────────────────────────────
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# ── BOS pipeline (after matplotlib.use()) ────────────────────────────────────
from bos_pipeline.io.avi import PhotronAviReader
from bos_pipeline.processing.preprocess import PreprocessConfig, preprocess
from bos_pipeline.processing.displacement import (
    DisplacementConfig,
    compute_displacement,
    interpolate_to_full_resolution,
)
from bos_pipeline.processing.concentration import (
    ConcentrationConfig,
    compute_concentration,
    compute_n_pair,
    GAS_DB, JET_GASES, AMBIENT_GASES,
)
from bos_pipeline.processing.velocity import (
    VelocityConfig,
    compute_velocity_frame_to_frame,
)
from bos_pipeline import export as _export
import bos_pipeline.visualization as viz
from bos_pipeline.video_export import export_results_video


# =============================================================================
# Qt stylesheet — Windows-11 inspired Fluent-ish light theme
# =============================================================================

_STYLE = """
* {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 9pt;
}
QLabel {
    color: #202020;
    background: transparent;
}
QMainWindow, QWidget#central {
    background-color: #f3f3f3;
}
QGroupBox {
    font-weight: 600;
    color: #202020;
    border: 1px solid #d8d8d8;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 8px 8px 8px;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 5px;
    color: #0067c0;
}
/* ── Inputs ── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    color: #202020;
    border: 1px solid #d0d0d0;
    border-radius: 5px;
    padding: 4px 7px;
    background: #fafafa;
    selection-background-color: #0067c0;
    selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #0067c0;
}
QLineEdit::placeholder { color: #aaaaaa; }
QComboBox QAbstractItemView { color: #202020; background: #ffffff; }
QComboBox::drop-down { border: none; }
/* ── Buttons ── */
QPushButton {
    border: 1px solid #c8c8c8;
    border-radius: 6px;
    padding: 5px 14px;
    background-color: #f9f9f9;
    color: #202020;
}
QPushButton:hover  { background-color: #e8f0fb; border-color: #0067c0; }
QPushButton:pressed{ background-color: #cde0f8; }
QPushButton:disabled { color: #b0b0b0; background: #f0f0f0; border-color: #e0e0e0; }
QPushButton#run_btn {
    background-color: #0067c0;
    color: #ffffff;
    font-weight: 700;
    border: none;
    padding: 9px;
    border-radius: 7px;
}
QPushButton#run_btn:hover    { background-color: #0055a3; }
QPushButton#run_btn:disabled { background-color: #7fb3e0; }
QPushButton#cancel_btn {
    background-color: #c42b1c;
    color: #ffffff;
    font-weight: 700;
    border: none;
    padding: 9px;
    border-radius: 7px;
}
QPushButton#cancel_btn:hover    { background-color: #a32316; }
QPushButton#cancel_btn:disabled { background-color: #e09a94; }
QPushButton#save_btn {
    background-color: #107c10;
    color: #ffffff;
    font-weight: 700;
    border: none;
    padding: 9px;
    border-radius: 7px;
}
QPushButton#save_btn:hover    { background-color: #0b5e0b; }
QPushButton#save_btn:disabled { background-color: #88c888; }
/* ── Slider ── */
QSlider::groove:horizontal {
    height: 5px;
    background: #d8d8d8;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #0067c0;
    width: 16px; height: 16px;
    border-radius: 8px;
    margin: -6px 0;
}
QSlider::sub-page:horizontal { background: #0067c0; border-radius: 3px; }
/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    background: #ffffff;
}
QTabBar::tab {
    padding: 8px 20px;
    border: 1px solid #d0d0d0;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    background: #f0f0f0;
    color: #555;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #0067c0;
    font-weight: 700;
}
QTabBar::tab:hover:!selected { background: #e8f0fb; }
/* ── Log box ── */
QTextEdit#log_box {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 8pt;
    border-radius: 6px;
    border: 1px solid #333;
}
/* ── Stats bar ── */
QFrame#stats_bar {
    background-color: #fafafa;
    border-top: 1px solid #e0e0e0;
}
QLabel#stat_label {
    color: #444;
    padding: 2px 6px;
    font-size: 8.5pt;
}
QLabel#stat_sep {
    color: #bbb;
}
/* ── Progress ── */
QProgressBar {
    border: none;
    background: #e8e8e8;
    border-radius: 3px;
    max-height: 5px;
    text-align: center;
}
QProgressBar::chunk { background: #0067c0; border-radius: 3px; }
/* ── Status bar ── */
QStatusBar {
    background: #0067c0;
    color: #ffffff;
    font-size: 8.5pt;
    padding: 2px 8px;
}
/* ── Scroll area ── */
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical {
    background: #eeeeee; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #c0c0c0; border-radius: 4px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #a0a0a0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
/* ── Checkable group box ── */
QGroupBox::indicator {
    width: 14px; height: 14px;
}
QGroupBox::indicator:checked { image: none; background: #0067c0; border: 1px solid #0067c0; border-radius: 3px; }
QGroupBox::indicator:unchecked { image: none; background: #fafafa; border: 1px solid #c0c0c0; border-radius: 3px; }
"""


# =============================================================================
# Background pipeline worker
# =============================================================================

class PipelineWorker(QObject):
    """Runs the full BOS pipeline in a background QThread.

    Signals
    -------
    log_msg(message, level)  — "info" | "ok" | "warn" | "err"
    status_msg(message)      — short string for the status bar
    result_ready(dict)       — dict with frame_idx, dx, dy, ref, meas
    finished()               — all frames done (or cancelled)
    error(traceback_str)     — unhandled exception
    """

    log_msg    = Signal(str, str)
    status_msg = Signal(str)
    result_ready = Signal(dict)
    finished   = Signal()
    error      = Signal(str)

    def __init__(self, params: dict) -> None:
        super().__init__()
        self._params = params
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            self._execute()
        except Exception:
            self.error.emit(traceback.format_exc())

    # ------------------------------------------------------------------
    def _execute(self) -> None:
        p = self._params
        pre_cfg  = PreprocessConfig(gaussian_sigma=p["sigma"])
        disp_cfg = DisplacementConfig(
            method=p["method"],
            window_size=p["window"],
            overlap=p["overlap"],
        )

        vel_enabled = p.get("vel_enabled", False)
        vel_cfg: Optional[VelocityConfig] = None
        if vel_enabled:
            vel_cfg = VelocityConfig(
                method="frame_to_frame",
                window_size=p.get("vel_window", 32),
                overlap=p.get("vel_overlap", 0.5),
                dt=p.get("vel_dt", 1e-3),
                pixel_scale_mm=p.get("vel_mm_per_px", 0.1),
                nmt_threshold=p.get("vel_nmt_threshold", 2.0),
            )

        self.log_msg.emit("Opening AVI…", "info")
        kwargs: dict = {"path": p["avi_path"]}
        if p.get("cihx_path"):
            kwargs["metadata_file"] = p["cihx_path"]

        with PhotronAviReader(**kwargs) as reader:
            meta  = reader.metadata
            total = meta.total_frames or 0
            fps   = meta.frame_rate or 0
            w, h  = meta.width or 0, meta.height or 0
            self.log_msg.emit(
                f"Opened: {w}×{h} px  |  {total} frames  |  {fps:.0f} fps",
                "info",
            )

            # Auto-set dt from camera fps if user left it at default
            if vel_cfg and fps > 0:
                vel_cfg.dt = 1.0 / fps

            self.status_msg.emit(f"Building reference (frame {p['ref_idx']})…")
            ref_raw = reader.get_frame(p["ref_idx"]).astype(np.float32)

            meas_indices: List[int] = p["meas_indices"]
            n = len(meas_indices)
            prev_meas_raw: Optional[np.ndarray] = None

            for i, meas_idx in enumerate(meas_indices):
                if self._cancelled:
                    self.log_msg.emit("Cancelled by user.", "warn")
                    break

                self.status_msg.emit(
                    f"Computing frame {meas_idx}  ({i + 1} / {n})…"
                )
                self.log_msg.emit(f"  Processing frame {meas_idx}…", "info")

                meas_raw = reader.get_frame(meas_idx).astype(np.float32)
                ref_p, meas_p = preprocess(ref_raw, meas_raw, config=pre_cfg)
                result = compute_displacement(ref_p, meas_p, config=disp_cfg)
                dx, dy = result.dx, result.dy

                if disp_cfg.method == "cross_correlation" and result.grid_x is not None:
                    dx, dy = interpolate_to_full_resolution(
                        dx, dy, ref_p.shape, result=result
                    )

                mag = np.hypot(dx, dy)
                self.log_msg.emit(
                    f"    max|d|={mag.max():.2f} px   mean|d|={mag.mean():.2f} px",
                    "ok",
                )

                # Concentration (optional)
                concentration = None
                axis_col = None
                if p.get("conc_enabled"):
                    try:
                        conc_cfg = ConcentrationConfig(
                            enabled=True,
                            mm_per_px=p["conc_mm_per_px"],
                            Z_f_mm=p["conc_Z_f_mm"],
                            gas_type=p["conc_gas_type"],
                            ambient_gas=p["conc_ambient_gas"],
                            temperature_K=p["conc_temperature_K"],
                            pressure_Pa=p["conc_pressure_Pa"],
                            n_gas_custom=p["conc_n_gas_custom"],
                            n_ambient_custom=p["conc_n_ambient_custom"],
                            component=p["conc_component"],
                            abel_method=p["conc_abel_method"],
                            axis_mode=p["conc_axis_mode"],
                            axis_pos=p.get("conc_axis_pos"),
                        )
                        concentration, axis_col, n_gas, n_amb = compute_concentration(
                            dx, dy, conc_cfg
                        )
                        self.log_msg.emit(
                            f"    Conc: gas={p['conc_gas_type']} "
                            f"n_gas={n_gas:.6f}  n_amb={n_amb:.6f}  "
                            f"max={concentration.max():.4f}  axis={axis_col}",
                            "ok",
                        )
                    except Exception as exc:
                        self.log_msg.emit(
                            f"    Concentration failed: {exc}", "warn"
                        )

                # Velocity (optional — needs two consecutive raw frames)
                velocity = None
                if vel_enabled and vel_cfg and prev_meas_raw is not None:
                    try:
                        vel_result = compute_velocity_frame_to_frame(
                            prev_meas_raw, meas_raw, config=vel_cfg,
                        )
                        velocity = {
                            "u": vel_result.u,
                            "v": vel_result.v,
                            "magnitude": vel_result.magnitude,
                            "vorticity": vel_result.vorticity,
                            "divergence": vel_result.divergence,
                        }
                        vmax = float(vel_result.magnitude.max())
                        vmean = float(vel_result.magnitude.mean())
                        self.log_msg.emit(
                            f"    Vel: max={vmax:.3f} m/s  mean={vmean:.3f} m/s",
                            "ok",
                        )
                    except Exception as exc:
                        self.log_msg.emit(
                            f"    Velocity failed: {exc}", "warn"
                        )

                prev_meas_raw = meas_raw

                self.result_ready.emit({
                    "frame_idx": meas_idx,
                    "dx": dx,
                    "dy": dy,
                    "ref": ref_p,
                    "meas": meas_p,
                    "concentration": concentration,
                    "axis_col": axis_col,
                    "velocity": velocity,
                })

        self.finished.emit()


# =============================================================================
# Video export worker
# =============================================================================

class VideoExportWorker(QObject):
    """Renders all results to an MP4 in a background thread."""

    progress = Signal(int, int)    # (current, total)
    finished = Signal(str)         # output path
    error    = Signal(str)

    def __init__(self, results: list, output_path: str, fps: float) -> None:
        super().__init__()
        self._results = results
        self._output_path = output_path
        self._fps = fps

    @Slot()
    def run(self) -> None:
        try:
            path = export_results_video(
                self._results,
                self._output_path,
                fps=self._fps,
                progress_cb=lambda i, n: self.progress.emit(i, n),
            )
            self.finished.emit(str(path))
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# =============================================================================
# Embedded matplotlib canvas
# =============================================================================

class MplCanvas(QWidget):
    """A matplotlib Figure embedded in a QWidget with a navigation toolbar."""

    def __init__(self, parent: Optional[QWidget] = None,
                 figsize: tuple = (8, 5.5), dpi: int = 100) -> None:
        super().__init__(parent)
        self.fig    = Figure(figsize=figsize, dpi=dpi, facecolor="#ffffff")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def draw(self) -> None:
        self.canvas.draw_idle()

    def show_placeholder(self, name: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, f"{name}\n(run pipeline to see results)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#aaaaaa")
        ax.set_axis_off()
        self.draw()


# =============================================================================
# Main window
# =============================================================================

class BOSWindow(QMainWindow):

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BOS Analysis  —  Background-Oriented Schlieren")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 650)

        # State
        self._avi_path:  Optional[Path] = None
        self._cihx_path: Optional[Path] = None
        self._n_frames:  int = 0
        self._results:   List[dict] = []
        self._result_idx: int = 0

        # Worker / thread (pipeline)
        self._thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None
        # Worker / thread (video export)
        self._vid_thread: Optional[QThread] = None
        self._vid_worker: Optional[VideoExportWorker] = None

        # Debounce timer for slider preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._do_preview_frame)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = self._make_left_panel()
        left.setFixedWidth(310)

        right = self._make_right_panel()

        root.addWidget(left)
        root.addWidget(right, stretch=1)

        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("Ready — load an AVI file to begin.")

    # ── Left panel (scrollable) ─────────────────────────────────────────────

    def _make_left_panel(self) -> QWidget:
        outer = QWidget()
        outer_vl = QVBoxLayout(outer)
        outer_vl.setContentsMargins(0, 0, 0, 0)
        outer_vl.setSpacing(0)

        # Scrollable area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        inner = QWidget()
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(0, 0, 2, 0)
        vl.setSpacing(6)
        vl.addWidget(self._make_files_group())
        vl.addWidget(self._make_params_group())
        vl.addWidget(self._make_viewer_group())
        vl.addWidget(self._make_velocity_group())
        vl.addWidget(self._make_concentration_group())
        vl.addStretch()

        scroll.setWidget(inner)
        outer_vl.addWidget(scroll, stretch=1)
        outer_vl.addWidget(self._make_buttons_section())
        outer_vl.addWidget(self._make_log_group())
        return outer

    def _make_files_group(self) -> QGroupBox:
        grp = QGroupBox("Files")
        fl = QFormLayout(grp)
        fl.setSpacing(7)

        self._avi_edit  = QLineEdit(); self._avi_edit.setPlaceholderText("Select .avi file…")
        self._cihx_edit = QLineEdit(); self._cihx_edit.setPlaceholderText("Auto-detected or select…")
        self._out_edit  = QLineEdit(); self._out_edit.setPlaceholderText("Output folder…")

        fl.addRow("AVI file:",    self._browse_row(self._avi_edit,  self._browse_avi))
        fl.addRow("CIHX file:",   self._browse_row(self._cihx_edit, self._browse_cihx))
        fl.addRow("Output folder:", self._browse_row(self._out_edit, self._browse_output, folder=True))
        return grp

    def _browse_row(self, edit: QLineEdit, callback, folder: bool = False) -> QWidget:
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(callback)
        hl.addWidget(edit)
        hl.addWidget(btn)
        return row

    def _make_params_group(self) -> QGroupBox:
        grp = QGroupBox("Displacement")
        fl = QFormLayout(grp)
        fl.setSpacing(6)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["cross_correlation", "farneback", "lucas_kanade"])
        self._method_combo.setToolTip(
            "cross_correlation — windowed FFT (PIV-style, robust)\n"
            "farneback — dense optical flow (Gunnar Farneback)\n"
            "lucas_kanade — sparse optical flow (Lucas-Kanade pyramid)"
        )
        fl.addRow("Method:", self._method_combo)

        self._ref_spin = QSpinBox()
        self._ref_spin.setRange(0, 9999)
        self._ref_spin.setToolTip("Index of the reference frame (no-flow background)")
        fl.addRow("Ref frame:", self._ref_spin)

        self._meas_edit = QLineEdit("5,6,7,8,9,10")
        self._meas_edit.setPlaceholderText("e.g. 5,6,7  or  5:20  or  all")
        self._meas_edit.setToolTip(
            "Frames to process.\n"
            "Formats:  5,6,7  |  5:25  (range)  |  all"
        )
        fl.addRow("Meas frames:", self._meas_edit)

        self._window_spin = QSpinBox()
        self._window_spin.setRange(8, 512)
        self._window_spin.setSingleStep(8)
        self._window_spin.setValue(32)
        self._window_spin.setToolTip("Interrogation window size [px] (power of 2 recommended)")
        fl.addRow("Window [px]:", self._window_spin)

        self._overlap_spin = QDoubleSpinBox()
        self._overlap_spin.setRange(0.0, 0.95)
        self._overlap_spin.setSingleStep(0.05)
        self._overlap_spin.setValue(0.75)
        self._overlap_spin.setDecimals(2)
        self._overlap_spin.setToolTip("Overlap fraction between adjacent windows (0.5 – 0.75 typical)")
        fl.addRow("Overlap:", self._overlap_spin)

        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 10.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setValue(1.0)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setToolTip("Gaussian pre-filter σ [px] — smooths noise before correlation (0 = off)")
        fl.addRow("Pre-filter σ:", self._sigma_spin)

        return grp

    def _make_viewer_group(self) -> QGroupBox:
        grp = QGroupBox("Frame Preview")
        vl = QVBoxLayout(grp)
        vl.setSpacing(6)

        self._frame_label = QLabel("No video loaded")
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setStyleSheet("font-size: 8pt; color: #666;")
        vl.addWidget(self._frame_label)

        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        vl.addWidget(self._frame_slider)

        nav = QWidget()
        nl = QHBoxLayout(nav)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(4)
        btn_prev = QPushButton("◀")
        btn_prev.setFixedWidth(34)
        btn_prev.clicked.connect(lambda: self._step_frame(-1))
        btn_next = QPushButton("▶")
        btn_next.setFixedWidth(34)
        btn_next.clicked.connect(lambda: self._step_frame(+1))
        btn_show = QPushButton("Preview Frame")
        btn_show.clicked.connect(self._do_preview_frame)
        nl.addWidget(btn_prev)
        nl.addWidget(btn_show, stretch=1)
        nl.addWidget(btn_next)
        vl.addWidget(nav)
        return grp

    def _make_velocity_group(self) -> QGroupBox:
        grp = QGroupBox("Velocity")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.setToolTip(
            "Estimate 2-D velocity field from consecutive BOS frames.\n"
            "Uses FFT cross-correlation between frame pairs.\n"
            "Requires at least two measurement frames.\n"
            "Output: u, v [m/s], vorticity [1/s], divergence [1/s]."
        )
        self._vel_group = grp

        fl = QFormLayout(grp)
        fl.setSpacing(6)

        self._vel_mm_per_px = QDoubleSpinBox()
        self._vel_mm_per_px.setRange(0.001, 100.0)
        self._vel_mm_per_px.setDecimals(4)
        self._vel_mm_per_px.setSingleStep(0.01)
        self._vel_mm_per_px.setValue(0.1)
        self._vel_mm_per_px.setToolTip(
            "Physical pixel size at the measurement plane [mm/px].\n"
            "Used to convert pixel displacements to m/s."
        )
        fl.addRow("mm / px:", self._vel_mm_per_px)

        self._vel_dt_spin = QDoubleSpinBox()
        self._vel_dt_spin.setRange(1e-6, 10.0)
        self._vel_dt_spin.setDecimals(6)
        self._vel_dt_spin.setSingleStep(0.0001)
        self._vel_dt_spin.setValue(0.001)
        self._vel_dt_spin.setToolTip(
            "Time step between consecutive frames [s].\n"
            "Auto-filled from camera frame rate when video is loaded.\n"
            "For 1000 fps → dt = 0.001 s"
        )
        fl.addRow("dt [s]:", self._vel_dt_spin)

        self._vel_auto_dt = QCheckBox("Auto from fps")
        self._vel_auto_dt.setChecked(True)
        self._vel_auto_dt.setToolTip("Automatically compute dt = 1 / fps from the camera metadata")
        self._vel_auto_dt.toggled.connect(
            lambda on: self._vel_dt_spin.setEnabled(not on)
        )
        self._vel_dt_spin.setEnabled(False)
        fl.addRow("", self._vel_auto_dt)

        self._vel_window_spin = QSpinBox()
        self._vel_window_spin.setRange(8, 512)
        self._vel_window_spin.setSingleStep(8)
        self._vel_window_spin.setValue(32)
        self._vel_window_spin.setToolTip("Velocity interrogation window [px]")
        fl.addRow("Window [px]:", self._vel_window_spin)

        self._vel_overlap_spin = QDoubleSpinBox()
        self._vel_overlap_spin.setRange(0.0, 0.9)
        self._vel_overlap_spin.setSingleStep(0.05)
        self._vel_overlap_spin.setValue(0.5)
        self._vel_overlap_spin.setDecimals(2)
        self._vel_overlap_spin.setToolTip("Window overlap fraction for velocity cross-correlation")
        fl.addRow("Overlap:", self._vel_overlap_spin)

        self._vel_nmt_spin = QDoubleSpinBox()
        self._vel_nmt_spin.setRange(0.5, 10.0)
        self._vel_nmt_spin.setSingleStep(0.5)
        self._vel_nmt_spin.setValue(2.0)
        self._vel_nmt_spin.setDecimals(1)
        self._vel_nmt_spin.setToolTip(
            "Normalised Median Test threshold for outlier rejection.\n"
            "Westerweel & Scarano (2005) recommend 2.0 – 3.0.\n"
            "Lower = more aggressive filtering."
        )
        fl.addRow("NMT thresh:", self._vel_nmt_spin)

        return grp

    def _make_concentration_group(self) -> QGroupBox:
        grp = QGroupBox("Gas Concentration")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.setToolTip(
            "Enable Abel-inversion-based gas concentration measurement.\n"
            "Requires axisymmetric jet and calibrated pixel size + Z distance.\n"
            "Refractive indices are computed implicitly from Gladstone-Dale\n"
            "theory at the specified temperature and pressure."
        )
        self._conc_group = grp

        fl = QFormLayout(grp)
        fl.setSpacing(6)

        # ── Gas and ambient selection ──────────────────────────────────
        self._conc_gas_type = QComboBox()
        for key in JET_GASES:
            label = GAS_DB[key]["label"] if key != "custom" else "Custom"
            self._conc_gas_type.addItem(label, key)
        self._conc_gas_type.setToolTip("Jet gas species")
        self._conc_gas_type.currentIndexChanged.connect(self._on_conc_gas_changed)
        fl.addRow("Jet gas:", self._conc_gas_type)

        self._conc_ambient_gas = QComboBox()
        for key in AMBIENT_GASES:
            label = GAS_DB[key]["label"] if key != "custom" else "Custom"
            self._conc_ambient_gas.addItem(label, key)
        self._conc_ambient_gas.setToolTip("Ambient (background) gas species")
        self._conc_ambient_gas.currentIndexChanged.connect(self._on_conc_gas_changed)
        fl.addRow("Ambient:", self._conc_ambient_gas)

        # ── Ambient conditions ─────────────────────────────────────────
        self._conc_temp = QDoubleSpinBox()
        self._conc_temp.setRange(100.0, 3000.0)
        self._conc_temp.setDecimals(1)
        self._conc_temp.setSingleStep(1.0)
        self._conc_temp.setValue(293.15)
        self._conc_temp.setToolTip("Ambient temperature [K]  (20 °C = 293.15 K)")
        self._conc_temp.valueChanged.connect(self._on_conc_gas_changed)
        fl.addRow("T [K]:", self._conc_temp)

        self._conc_pressure = QDoubleSpinBox()
        self._conc_pressure.setRange(1000.0, 1_000_000.0)
        self._conc_pressure.setDecimals(0)
        self._conc_pressure.setSingleStep(100.0)
        self._conc_pressure.setValue(101325.0)
        self._conc_pressure.setToolTip("Ambient pressure [Pa]  (1 atm = 101 325 Pa)")
        self._conc_pressure.valueChanged.connect(self._on_conc_gas_changed)
        fl.addRow("P [Pa]:", self._conc_pressure)

        # ── Computed / custom n display ────────────────────────────────
        self._conc_n_gas_lbl = QLabel("n_gas = 1.000 123")
        self._conc_n_gas_lbl.setStyleSheet("color: #555; font-size: 8pt;")
        self._conc_n_gas_spin = QDoubleSpinBox()
        self._conc_n_gas_spin.setRange(1.0, 2.0)
        self._conc_n_gas_spin.setDecimals(6)
        self._conc_n_gas_spin.setSingleStep(0.000001)
        self._conc_n_gas_spin.setValue(1.000132)
        self._conc_n_gas_spin.setVisible(False)
        self._conc_n_gas_spin.setToolTip("Custom refractive index of pure jet gas")
        self._n_gas_stack = QWidget()
        _hl = QHBoxLayout(self._n_gas_stack)
        _hl.setContentsMargins(0, 0, 0, 0)
        _hl.addWidget(self._conc_n_gas_lbl)
        _hl.addWidget(self._conc_n_gas_spin)
        fl.addRow("n_gas:", self._n_gas_stack)

        self._conc_n_amb_lbl = QLabel("n_amb = 1.000 273")
        self._conc_n_amb_lbl.setStyleSheet("color: #555; font-size: 8pt;")
        self._conc_n_amb_spin = QDoubleSpinBox()
        self._conc_n_amb_spin.setRange(1.0, 2.0)
        self._conc_n_amb_spin.setDecimals(6)
        self._conc_n_amb_spin.setSingleStep(0.000001)
        self._conc_n_amb_spin.setValue(1.000293)
        self._conc_n_amb_spin.setVisible(False)
        self._conc_n_amb_spin.setToolTip("Custom refractive index of ambient gas")
        self._n_amb_stack = QWidget()
        _hl2 = QHBoxLayout(self._n_amb_stack)
        _hl2.setContentsMargins(0, 0, 0, 0)
        _hl2.addWidget(self._conc_n_amb_lbl)
        _hl2.addWidget(self._conc_n_amb_spin)
        fl.addRow("n_ambient:", self._n_amb_stack)

        # ── Separator line ─────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #e0e0e0;")
        fl.addRow(sep)

        # ── Spatial calibration ────────────────────────────────────────
        self._conc_mm_per_px = QDoubleSpinBox()
        self._conc_mm_per_px.setRange(0.001, 100.0)
        self._conc_mm_per_px.setDecimals(4)
        self._conc_mm_per_px.setSingleStep(0.01)
        self._conc_mm_per_px.setValue(0.1)
        self._conc_mm_per_px.setToolTip("Physical pixel size at the background screen [mm/px]")
        fl.addRow("mm / px:", self._conc_mm_per_px)

        self._conc_Zf = QDoubleSpinBox()
        self._conc_Zf.setRange(1.0, 100000.0)
        self._conc_Zf.setDecimals(1)
        self._conc_Zf.setSingleStep(10.0)
        self._conc_Zf.setValue(1000.0)
        self._conc_Zf.setToolTip("Distance from object to background screen [mm]")
        fl.addRow("Z_f [mm]:", self._conc_Zf)

        # ── Signal component ───────────────────────────────────────────
        self._conc_component = QComboBox()
        self._conc_component.addItems(["dx", "dy"])
        self._conc_component.setToolTip(
            "Displacement component carrying the radial signal.\n"
            "Vertical jet (axis ∥ Y) → dx  |  Horizontal jet (axis ∥ X) → dy"
        )
        fl.addRow("Component:", self._conc_component)

        self._conc_abel_method = QComboBox()
        self._conc_abel_method.addItems(["three_point", "hansenlaw", "basex"])
        self._conc_abel_method.setToolTip(
            "PyAbel inversion method.\n"
            "three_point: fast & robust\n"
            "hansenlaw:   better for noisy data\n"
            "basex:       basis-set expansion"
        )
        fl.addRow("Abel method:", self._conc_abel_method)

        self._conc_axis_mode = QComboBox()
        self._conc_axis_mode.addItems(["auto", "manual"])
        self._conc_axis_mode.currentTextChanged.connect(self._on_conc_axis_mode_changed)
        fl.addRow("Axis mode:", self._conc_axis_mode)

        self._conc_axis_pos = QSpinBox()
        self._conc_axis_pos.setRange(0, 9999)
        self._conc_axis_pos.setValue(512)
        self._conc_axis_pos.setEnabled(False)
        self._conc_axis_pos.setToolTip("Manual symmetry axis column [px]")
        fl.addRow("Axis col [px]:", self._conc_axis_pos)

        # Initial label update
        self._on_conc_gas_changed()
        return grp

    def _on_conc_gas_changed(self, *_) -> None:
        """Recompute and display n_gas / n_ambient from Gladstone-Dale."""
        gas_key = self._conc_gas_type.currentData()
        amb_key = self._conc_ambient_gas.currentData()
        T = self._conc_temp.value()
        P = self._conc_pressure.value()

        custom_gas = (gas_key == "custom")
        custom_amb = (amb_key == "custom")

        self._conc_n_gas_lbl.setVisible(not custom_gas)
        self._conc_n_gas_spin.setVisible(custom_gas)
        self._conc_n_amb_lbl.setVisible(not custom_amb)
        self._conc_n_amb_spin.setVisible(custom_amb)

        if not custom_gas:
            n_gas, n_amb = compute_n_pair(
                gas_key, amb_key if not custom_amb else "air",
                T, P,
                self._conc_n_gas_spin.value(),
                self._conc_n_amb_spin.value(),
            )
            self._conc_n_gas_lbl.setText(f"n_gas = {n_gas:.6f}")
        if not custom_amb:
            _, n_amb = compute_n_pair(
                gas_key if not custom_gas else "air",
                amb_key, T, P,
                self._conc_n_gas_spin.value(),
                self._conc_n_amb_spin.value(),
            )
            self._conc_n_amb_lbl.setText(f"n_amb = {n_amb:.6f}")

    def _on_conc_axis_mode_changed(self, mode: str) -> None:
        self._conc_axis_pos.setEnabled(mode == "manual")

    def _make_buttons_section(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(5)

        self._run_btn = QPushButton("▶  Run Pipeline")
        self._run_btn.setObjectName("run_btn")
        self._run_btn.clicked.connect(self._run_pipeline)

        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setObjectName("cancel_btn")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_pipeline)

        self._save_btn = QPushButton("💾  Save All Results")
        self._save_btn.setObjectName("save_btn")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_results)

        # Video export row: button + fps spinner
        vid_row = QWidget()
        vr_hl = QHBoxLayout(vid_row)
        vr_hl.setContentsMargins(0, 0, 0, 0)
        vr_hl.setSpacing(4)

        self._video_btn = QPushButton("🎬  Export Video")
        self._video_btn.setObjectName("save_btn")
        self._video_btn.setEnabled(False)
        self._video_btn.setToolTip(
            "Render all processed frames into an MP4 animation.\n"
            "Shows displacement magnitude (and concentration if enabled)."
        )
        self._video_btn.clicked.connect(self._export_video)

        fps_lbl = QLabel("fps:")
        fps_lbl.setFixedWidth(26)
        self._video_fps = QDoubleSpinBox()
        self._video_fps.setRange(1.0, 60.0)
        self._video_fps.setDecimals(1)
        self._video_fps.setSingleStep(1.0)
        self._video_fps.setValue(10.0)
        self._video_fps.setFixedWidth(58)
        self._video_fps.setToolTip("Video playback frame rate")

        vr_hl.addWidget(self._video_btn, stretch=1)
        vr_hl.addWidget(fps_lbl)
        vr_hl.addWidget(self._video_fps)

        self._progress = QProgressBar()
        self._progress.setMaximum(0)      # indeterminate by default
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)

        vl.addWidget(self._run_btn)
        vl.addWidget(self._cancel_btn)
        vl.addWidget(self._save_btn)
        vl.addWidget(vid_row)
        vl.addWidget(self._progress)
        return w

    def _make_log_group(self) -> QGroupBox:
        grp = QGroupBox("Log")
        vl = QVBoxLayout(grp)
        vl.setContentsMargins(4, 4, 4, 4)
        self._log_box = QTextEdit()
        self._log_box.setObjectName("log_box")
        self._log_box.setReadOnly(True)
        self._log_box.setFixedHeight(120)
        vl.addWidget(self._log_box)
        return grp

    # ── Right panel ───────────────────────────────────────────────────────────

    def _make_right_panel(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)

        # Tab widget with 4 embedded canvases
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        vl.addWidget(self._tabs, stretch=1)

        self._canvases: Dict[str, MplCanvas] = {}
        for key, label in [
            ("frame_view",    "  Frame View    "),
            ("magnitude",     "  Magnitude     "),
            ("components",    "  Components    "),
            ("quiver",        "    Quiver      "),
            ("velocity",      "   Velocity     "),
            ("concentration", " Concentration  "),
        ]:
            canvas = MplCanvas(dpi=100)
            canvas.show_placeholder(label.strip())
            self._tabs.addTab(canvas, label)
            self._canvases[key] = canvas

        # Stats bar
        vl.addWidget(self._make_stats_bar())
        return w

    def _make_stats_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("stats_bar")
        bar.setFixedHeight(38)

        hl = QHBoxLayout(bar)
        hl.setContentsMargins(12, 0, 12, 0)
        hl.setSpacing(0)

        self._stat_lbls: Dict[str, QLabel] = {}
        items = [
            ("frame",    "Frame: —"),
            ("sep1",     " │ "),
            ("max_d",    "Max |d|: —"),
            ("sep2",     " │ "),
            ("mean_d",   "Mean |d|: —"),
            ("sep3",     " │ "),
            ("bg_noise", "BG noise: —"),
            ("sep4",     " │ "),
            ("snr",      "SNR: —"),
        ]
        for key, text in items:
            lbl = QLabel(text)
            lbl.setObjectName("stat_sep" if key.startswith("sep") else "stat_label")
            if key.startswith("sep"):
                lbl.setStyleSheet("color: #cccccc;")
            self._stat_lbls[key] = lbl
            hl.addWidget(lbl)

        hl.addStretch(1)

        hl.addWidget(QLabel("Result: "))
        self._result_combo = QComboBox()
        self._result_combo.setFixedWidth(120)
        self._result_combo.currentIndexChanged.connect(self._on_result_changed)
        hl.addWidget(self._result_combo)
        return bar

    # ── File browsers ─────────────────────────────────────────────────────────

    def _browse_avi(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select AVI file", "", "AVI video (*.avi);;All files (*.*)"
        )
        if not path:
            return
        self._avi_path = Path(path)
        self._avi_edit.setText(path)
        self._load_video_info()

        # Auto-detect CIHX in same folder
        for ext in (".cihx", ".cih"):
            cand = self._avi_path.with_suffix(ext)
            if cand.exists() and not self._cihx_edit.text():
                self._cihx_edit.setText(str(cand))
                self._cihx_path = cand
                self._log(f"Auto-detected metadata: {cand.name}")
                break

        # Auto-suggest output folder
        if not self._out_edit.text():
            out = self._avi_path.parent / "bos_output"
            self._out_edit.setText(str(out))

    def _browse_cihx(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CIHX metadata", "",
            "Photron metadata (*.cihx *.cih);;All files (*.*)"
        )
        if path:
            self._cihx_path = Path(path)
            self._cihx_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._out_edit.setText(path)

    # ── Video loading ─────────────────────────────────────────────────────────

    def _load_video_info(self) -> None:
        if not self._avi_path:
            return
        try:
            import cv2
            cap = cv2.VideoCapture(str(self._avi_path))
            if not cap.isOpened():
                self._log(f"Could not open: {self._avi_path.name}", "err")
                return
            self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            self._frame_slider.setRange(0, max(0, self._n_frames - 1))
            self._frame_slider.setValue(0)
            self._frame_label.setText(f"Frame 0 / {self._n_frames - 1}")
            self._ref_spin.setMaximum(self._n_frames - 1)

            # Auto-fill velocity dt from fps
            if fps > 0 and self._vel_auto_dt.isChecked():
                self._vel_dt_spin.setValue(1.0 / fps)

            msg = (f"Loaded: {self._avi_path.name}  "
                   f"|  {w}×{h} px  |  {self._n_frames} frames  |  {fps:.0f} fps")
            self._log(msg)
            self.statusBar().showMessage(msg)

            # Show first frame immediately
            QTimer.singleShot(120, self._do_preview_frame)
        except Exception as exc:
            self._log(f"Error reading video: {exc}", "err")

    # ── Frame preview ─────────────────────────────────────────────────────────

    def _on_slider_changed(self, val: int) -> None:
        self._frame_label.setText(f"Frame {val} / {self._n_frames - 1}")
        self._preview_timer.start(250)   # debounce 250 ms

    def _step_frame(self, delta: int) -> None:
        new = max(0, min(self._n_frames - 1, self._frame_slider.value() + delta))
        self._frame_slider.setValue(new)

    def _do_preview_frame(self) -> None:
        if not self._avi_path:
            return
        idx = self._frame_slider.value()
        try:
            import cv2
            cap = cv2.VideoCapture(str(self._avi_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32)

            canvas = self._canvases["frame_view"]
            canvas.fig.clear()
            ax = canvas.fig.add_subplot(111)
            ax.imshow(frame, cmap="gray", origin="upper", interpolation="bilinear")
            ax.set_title(f"Frame {idx}  —  single frame preview", fontsize=10, pad=5)
            ax.set_xlabel("x [px]", fontsize=8)
            ax.set_ylabel("y [px]", fontsize=8)
            ax.tick_params(labelsize=7)
            canvas.fig.tight_layout()
            canvas.draw()
            self._tabs.setCurrentIndex(0)
        except Exception as exc:
            self._log(f"Preview error: {exc}", "err")

    # ── Run pipeline ──────────────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        if not self._avi_edit.text():
            QMessageBox.warning(self, "No file", "Please select an AVI file first.")
            return
        if self._thread and self._thread.isRunning():
            return

        self._avi_path  = Path(self._avi_edit.text())
        self._cihx_path = Path(self._cihx_edit.text()) if self._cihx_edit.text() else None

        meas_str = self._meas_edit.text().strip()
        total    = self._n_frames or 200
        try:
            meas_indices = _parse_indices(meas_str, total)
        except Exception as exc:
            QMessageBox.warning(self, "Parse error",
                                f"Could not parse measurement frames:\n{exc}")
            return

        conc_enabled = self._conc_group.isChecked()
        vel_enabled  = self._vel_group.isChecked()
        axis_mode = self._conc_axis_mode.currentText()
        params = {
            "avi_path":    str(self._avi_path),
            "cihx_path":   str(self._cihx_path) if self._cihx_path else None,
            "ref_idx":     self._ref_spin.value(),
            "meas_indices": meas_indices,
            "method":      self._method_combo.currentText(),
            "window":      self._window_spin.value(),
            "overlap":     self._overlap_spin.value(),
            "sigma":       self._sigma_spin.value(),
            # Velocity
            "vel_enabled":        vel_enabled,
            "vel_mm_per_px":      self._vel_mm_per_px.value(),
            "vel_dt":             self._vel_dt_spin.value(),
            "vel_window":         self._vel_window_spin.value(),
            "vel_overlap":        self._vel_overlap_spin.value(),
            "vel_nmt_threshold":  self._vel_nmt_spin.value(),
            # Concentration
            "conc_enabled":        conc_enabled,
            "conc_gas_type":       self._conc_gas_type.currentData(),
            "conc_ambient_gas":    self._conc_ambient_gas.currentData(),
            "conc_temperature_K":  self._conc_temp.value(),
            "conc_pressure_Pa":    self._conc_pressure.value(),
            "conc_n_gas_custom":   self._conc_n_gas_spin.value(),
            "conc_n_ambient_custom": self._conc_n_amb_spin.value(),
            "conc_mm_per_px":      self._conc_mm_per_px.value(),
            "conc_Z_f_mm":         self._conc_Zf.value(),
            "conc_component":      self._conc_component.currentText(),
            "conc_abel_method":    self._conc_abel_method.currentText(),
            "conc_axis_mode":      axis_mode,
            "conc_axis_pos":       (self._conc_axis_pos.value()
                                    if axis_mode == "manual" else None),
        }

        self._results.clear()
        self._result_combo.clear()

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._video_btn.setEnabled(False)
        self._progress.setVisible(True)

        self._log("─" * 44)
        self._log(f"Starting pipeline  —  {len(meas_indices)} frame(s)…")

        # Move worker to a QThread
        self._worker = PipelineWorker(params)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log_msg.connect(self._on_worker_log)
        self._worker.status_msg.connect(self._on_worker_status)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _cancel_pipeline(self) -> None:
        if self._worker:
            self._worker.cancel()
        self.statusBar().showMessage("Cancelling…")

    # ── Worker signal slots ───────────────────────────────────────────────────

    @Slot(str, str)
    def _on_worker_log(self, msg: str, level: str) -> None:
        self._log(msg, level)

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    @Slot(dict)
    def _on_result_ready(self, result: dict) -> None:
        self._results.append(result)
        idx = len(self._results) - 1
        self._result_combo.addItem(f"Frame {result['frame_idx']}")
        self._result_combo.setCurrentIndex(idx)
        self._result_idx = idx
        self._render_all(result)

    @Slot()
    def _on_pipeline_finished(self) -> None:
        n = len(self._results)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self.statusBar().showMessage(
            f"Done — {n} frame{'s' if n != 1 else ''} processed."
        )
        self._log(f"Pipeline complete.  {n} result(s) available.", "ok")
        if n:
            self._save_btn.setEnabled(True)
            self._video_btn.setEnabled(True)
            self._auto_save()

    @Slot(str)
    def _on_pipeline_error(self, tb: str) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._log(f"ERROR:\n{tb}", "err")
        self.statusBar().showMessage("Error — see log.")

    @Slot(int)
    def _on_result_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._results):
            self._result_idx = idx
            self._render_all(self._results[idx])

    def _on_tab_changed(self, _idx: int) -> None:
        pass  # all tabs are rendered together when a result arrives

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_all(self, result: dict) -> None:
        dx, dy = result["dx"], result["dy"]
        ref    = result["ref"]
        meas   = result["meas"]
        fidx   = result["frame_idx"]
        mag    = np.hypot(dx, dy)

        # Update stats bar
        bg   = _bg_noise(mag)
        snr  = 20.0 * np.log10(mag.mean() / max(bg, 1e-9))
        self._stat_lbls["frame"].setText(f"Frame: {fidx}")
        self._stat_lbls["max_d"].setText(f"Max |d|: {mag.max():.2f} px")
        self._stat_lbls["mean_d"].setText(f"Mean |d|: {mag.mean():.2f} px")
        self._stat_lbls["bg_noise"].setText(f"BG noise: {bg:.3f} px")
        self._stat_lbls["snr"].setText(f"SNR: {snr:.1f} dB")

        self._render_frame_view(ref, meas, fidx)
        self._render_magnitude(dx, dy, fidx)
        self._render_components(dx, dy, fidx)
        self._render_quiver(ref, dx, dy, fidx)
        self._render_velocity(result)
        self._render_concentration(result)

    # -- Frame view tab --

    def _render_frame_view(self, ref: np.ndarray, meas: np.ndarray, fidx: int) -> None:
        canvas = self._canvases["frame_view"]
        canvas.fig.clear()
        axes = canvas.fig.subplots(1, 2)
        vmin = min(ref.min(), meas.min())
        vmax = max(ref.max(), meas.max())
        for ax, img, ttl in zip(axes, [ref, meas],
                                 ["Reference", f"Measurement  (frame {fidx})"]):
            ax.imshow(img, cmap="gray", origin="upper",
                      vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_title(ttl, fontsize=9, pad=4)
            ax.set_xlabel("x [px]", fontsize=7)
            ax.set_ylabel("y [px]", fontsize=7)
            ax.tick_params(labelsize=6)
        canvas.fig.suptitle("Reference vs Measurement", fontsize=10, y=1.01)
        canvas.fig.tight_layout()
        canvas.draw()

    # -- Magnitude tab --

    def _render_magnitude(self, dx: np.ndarray, dy: np.ndarray, fidx: int) -> None:
        canvas = self._canvases["magnitude"]
        canvas.fig.clear()
        ax  = canvas.fig.add_subplot(111)
        mag = np.hypot(dx, dy)
        vmax = float(np.percentile(mag, 99)) or 1.0

        im = ax.imshow(mag, origin="upper", cmap="turbo",
                       vmin=0, vmax=vmax, interpolation="bilinear")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.08)
        cb  = canvas.fig.colorbar(im, cax=cax)
        cb.set_label("Displacement [px]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        ax.set_title(f"Displacement Magnitude  —  Frame {fidx}", fontsize=10, pad=5)
        ax.set_xlabel("x [px]", fontsize=8)
        ax.set_ylabel("y [px]", fontsize=8)
        ax.tick_params(labelsize=7)

        # Stats overlay
        bg  = _bg_noise(mag)
        snr = 20.0 * np.log10(mag.mean() / max(bg, 1e-9))
        txt = (f"max  {mag.max():.3f} px\n"
               f"mean {mag.mean():.3f} px\n"
               f"BG σ {bg:.4f} px\n"
               f"SNR  {snr:.1f} dB")
        ax.text(0.01, 0.99, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=6.5, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.55))

        canvas.fig.tight_layout()
        canvas.draw()

    # -- Components tab --

    def _render_components(self, dx: np.ndarray, dy: np.ndarray, fidx: int) -> None:
        canvas = self._canvases["components"]
        canvas.fig.clear()
        axes = canvas.fig.subplots(1, 2)

        for ax, data, label in zip(axes, [dx, dy], ["dx [px]", "dy [px]"]):
            lim = float(np.percentile(np.abs(data), 99)) or 0.01
            im  = ax.imshow(data, origin="upper", cmap="RdBu_r",
                            vmin=-lim, vmax=lim, interpolation="bilinear")
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="4%", pad=0.08)
            cb  = canvas.fig.colorbar(im, cax=cax)
            cb.set_label(label, fontsize=8)
            cb.ax.tick_params(labelsize=7)
            ax.set_title(label, fontsize=9, pad=4)
            ax.set_xlabel("x [px]", fontsize=7)
            ax.set_ylabel("y [px]", fontsize=7)
            ax.tick_params(labelsize=6)

        canvas.fig.suptitle(
            f"Signed Displacement Components  —  Frame {fidx}", fontsize=10, y=1.01
        )
        canvas.fig.tight_layout()
        canvas.draw()

    # -- Quiver tab --

    def _render_quiver(self, ref: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                       fidx: int) -> None:
        canvas = self._canvases["quiver"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        ax.imshow(ref, origin="upper", cmap="gray",
                  interpolation="bilinear", alpha=0.72)

        H, W = dx.shape
        ds   = max(4, H // 52)           # ~52 arrows per row
        ys   = np.arange(0, H, ds)
        xs   = np.arange(0, W, ds)
        xx, yy = np.meshgrid(xs, ys)
        u    = dx[::ds, ::ds]
        v    = -dy[::ds, ::ds]           # flip y for image coords
        mag_q = np.hypot(u, v)

        d95   = float(np.percentile(mag_q, 95)) or 1.0
        scale = d95 / (0.70 * ds)
        vmax_q = float(np.percentile(mag_q, 99)) or 1.0

        q = ax.quiver(
            xx, yy, u, v, mag_q,
            cmap="plasma",
            norm=mcolors.Normalize(vmin=0, vmax=vmax_q),
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.0018,
            headwidth=4,
            headlength=4,
            minlength=0.3,
            alpha=0.92,
        )
        cb = canvas.fig.colorbar(q, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Displacement [px]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        key_u = max(1.0, round(d95))
        ax.quiverkey(q, X=0.86, Y=1.025, U=key_u,
                     label=f"{key_u:.0f} px", labelpos="E",
                     fontproperties={"size": 7})

        ax.set_title(f"Displacement Field  —  Frame {fidx}", fontsize=10, pad=5)
        ax.set_xlabel("x [px]", fontsize=8)
        ax.set_ylabel("y [px]", fontsize=8)
        ax.tick_params(labelsize=7)
        canvas.fig.tight_layout()
        canvas.draw()

    # -- Velocity tab --

    def _render_velocity(self, result: dict) -> None:
        canvas = self._canvases["velocity"]
        canvas.fig.clear()
        velocity = result.get("velocity")
        fidx = result["frame_idx"]

        if velocity is None:
            ax = canvas.fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "Velocity not computed.\n"
                "Enable 'Velocity' in the left panel and process\n"
                "at least two consecutive measurement frames.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888",
            )
            ax.set_axis_off()
            canvas.draw()
            return

        u = velocity["u"]
        v = velocity["v"]
        vel_mag = velocity["magnitude"]
        vort = velocity["vorticity"]

        axes = canvas.fig.subplots(1, 2)

        # Left: velocity magnitude
        ax = axes[0]
        vmax_v = float(np.percentile(vel_mag, 99)) or 1.0
        im = ax.imshow(vel_mag, origin="upper", cmap="turbo",
                       vmin=0, vmax=vmax_v, interpolation="bilinear")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.08)
        cb = canvas.fig.colorbar(im, cax=cax)
        cb.set_label("|V| [m/s]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title("Velocity Magnitude", fontsize=9, pad=4)
        ax.set_xlabel("x [px]", fontsize=7)
        ax.set_ylabel("y [px]", fontsize=7)
        ax.tick_params(labelsize=6)

        txt = (f"max  {vel_mag.max():.3f} m/s\n"
               f"mean {vel_mag.mean():.3f} m/s")
        ax.text(0.01, 0.99, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=6.5, color="white",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.55))

        # Right: vorticity
        ax = axes[1]
        vlim = float(np.percentile(np.abs(vort), 99)) or 1.0
        im = ax.imshow(vort, origin="upper", cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim, interpolation="bilinear")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.08)
        cb = canvas.fig.colorbar(im, cax=cax)
        cb.set_label("ω_z [1/s]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title("Vorticity", fontsize=9, pad=4)
        ax.set_xlabel("x [px]", fontsize=7)
        ax.set_ylabel("y [px]", fontsize=7)
        ax.tick_params(labelsize=6)

        canvas.fig.suptitle(
            f"Velocity Field  —  Frame {fidx}", fontsize=10, y=1.01
        )
        canvas.fig.tight_layout()
        canvas.draw()

    # -- Concentration tab --

    def _render_concentration(self, result: dict) -> None:
        canvas = self._canvases["concentration"]
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        concentration = result.get("concentration")
        fidx = result["frame_idx"]

        if concentration is None:
            ax.text(0.5, 0.5,
                    "Concentration not computed.\nEnable 'H₂ Concentration' in the left panel\nand re-run the pipeline.",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#888888")
            ax.set_axis_off()
            canvas.draw()
            return

        axis_col = result.get("axis_col")
        im = ax.imshow(concentration, origin="upper", cmap="plasma",
                       vmin=0.0, vmax=1.0, interpolation="bilinear")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.08)
        cb  = canvas.fig.colorbar(im, cax=cax)
        cb.set_label("c(H₂)  [vol. frac.]", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        if axis_col is not None:
            ax.axvline(axis_col, color="white", linestyle="--",
                       linewidth=0.8, label=f"Axis col {axis_col}")
            ax.legend(fontsize=7, loc="upper right",
                      facecolor="#00000099", labelcolor="white")

        nonzero = concentration[concentration > 1e-4]
        if nonzero.size:
            txt = (f"max   {concentration.max():.4f}\n"
                   f"mean  {nonzero.mean():.4f}\n"
                   f"area  {nonzero.size / concentration.size:.3f}")
            ax.text(0.01, 0.99, txt, transform=ax.transAxes,
                    va="top", ha="left", fontsize=6.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="black", alpha=0.55))

        ax.set_title(f"H₂ Concentration  —  Frame {fidx}", fontsize=10, pad=5)
        ax.set_xlabel("x [px]", fontsize=8)
        ax.set_ylabel("y [px]", fontsize=8)
        ax.tick_params(labelsize=7)
        canvas.fig.tight_layout()
        canvas.draw()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _export_video(self) -> None:
        if not self._results:
            QMessageBox.information(self, "Nothing to export", "Run the pipeline first.")
            return
        if self._vid_thread and self._vid_thread.isRunning():
            return

        out_dir = Path(self._out_edit.text() or "./bos_output")
        out_path = str(out_dir / "bos_animation.mp4")
        fps = self._video_fps.value()

        self._video_btn.setEnabled(False)
        self._progress.setMaximum(len(self._results))
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setVisible(True)
        self._log(f"Exporting video ({len(self._results)} frames @ {fps:.0f} fps)…")
        self.statusBar().showMessage("Exporting video…")

        self._vid_worker = VideoExportWorker(self._results, out_path, fps)
        self._vid_thread = QThread(self)
        self._vid_worker.moveToThread(self._vid_thread)
        self._vid_thread.started.connect(self._vid_worker.run)
        self._vid_worker.progress.connect(self._on_vid_progress)
        self._vid_worker.finished.connect(self._on_vid_finished)
        self._vid_worker.error.connect(self._on_vid_error)
        self._vid_worker.finished.connect(self._vid_thread.quit)
        self._vid_worker.error.connect(self._vid_thread.quit)
        self._vid_thread.start()

    @Slot(int, int)
    def _on_vid_progress(self, current: int, total: int) -> None:
        self._progress.setValue(current)
        self.statusBar().showMessage(f"Rendering frame {current} / {total}…")

    @Slot(str)
    def _on_vid_finished(self, path: str) -> None:
        self._video_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        self._progress.setMaximum(0)
        self._log(f"Video saved → {path}", "ok")
        self.statusBar().showMessage(f"Video saved: {Path(path).name}")

    @Slot(str)
    def _on_vid_error(self, tb: str) -> None:
        self._video_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        self._progress.setMaximum(0)
        self._log(f"Video export error:\n{tb}", "err")
        self.statusBar().showMessage("Video export failed — see log.")

    def _save_results(self) -> None:
        if not self._results:
            QMessageBox.information(self, "Nothing to save",
                                    "Run the pipeline first.")
            return
        out = self._out_edit.text() or "./bos_output"
        self._save_to_disk(Path(out))

    def _auto_save(self) -> None:
        out = self._out_edit.text()
        if out:
            self._save_to_disk(Path(out))

    def _save_to_disk(self, out_dir: Path) -> None:
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"Saving results to {out_dir}…")
        try:
            for r in self._results:
                stem = f"frame_{r['frame_idx']:05d}"
                _export.export_displacement(
                    r["dx"], r["dy"], out_dir, stem=stem, fmt="npy"
                )
                # Generate high-DPI figures via visualization.py
                dx, dy, ref, fidx = r["dx"], r["dy"], r["ref"], r["frame_idx"]
                figs = {
                    "_magnitude":  viz.plot_displacement_magnitude(
                        dx, dy, title=f"Magnitude — Frame {fidx}", dpi=200)[0],
                    "_components": viz.plot_displacement_components(
                        dx, dy, title=f"Components — Frame {fidx}", dpi=200)[0],
                    "_quiver":     viz.plot_quiver(
                        ref, dx, dy, title=f"Field — Frame {fidx}", dpi=200)[0],
                }
                if r.get("velocity") is not None:
                    vel = r["velocity"]
                    np.save(str(out_dir / f"{stem}_vel_u.npy"), vel["u"])
                    np.save(str(out_dir / f"{stem}_vel_v.npy"), vel["v"])
                    np.save(str(out_dir / f"{stem}_vel_mag.npy"), vel["magnitude"])
                    np.save(str(out_dir / f"{stem}_vorticity.npy"), vel["vorticity"])

                if r.get("concentration") is not None:
                    axis_col = r.get("axis_col")
                    figs["_concentration"] = viz.plot_concentration(
                        r["concentration"],
                        title=f"H₂ Concentration — Frame {fidx}",
                        axis_col=axis_col,
                        dpi=200,
                    )[0]
                    # Save concentration .npy alongside dx/dy
                    np.save(
                        str(out_dir / f"{stem}_concentration.npy"),
                        r["concentration"],
                    )

                for suffix, fig in figs.items():
                    fpath = fig_dir / f"{stem}{suffix}.png"
                    fig.savefig(str(fpath), dpi=200, bbox_inches="tight")
                    del fig

            self._log(
                f"Saved {len(self._results)} result(s) → {out_dir}", "ok"
            )
            self.statusBar().showMessage(f"Saved to {out_dir}")
        except Exception as exc:
            self._log(f"Save error: {exc}", "err")
            self._log(traceback.format_exc(), "err")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "info") -> None:
        colours = {
            "info": "#d4d4d4",
            "ok":   "#6ec96e",
            "warn": "#ffb347",
            "err":  "#ff6b6b",
        }
        c    = colours.get(level, "#d4d4d4")
        html = f'<span style="color:{c};">{_esc(msg)}</span>'
        self._log_box.append(html)
        sb = self._log_box.verticalScrollBar()
        sb.setValue(sb.maximum())


# =============================================================================
# Module-level helpers
# =============================================================================

def _parse_indices(s: str, total: int) -> List[int]:
    s = s.strip().lower()
    if s in ("all", ""):
        return list(range(1, total))
    indices: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, _, b = part.partition(":")
            indices.extend(range(int(a), int(b)))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def _bg_noise(mag: np.ndarray, frac: float = 0.08) -> float:
    """Std of magnitude in the four image corners (background estimate)."""
    h, w = mag.shape
    ch   = max(1, int(h * frac))
    cw   = max(1, int(w * frac))
    corners = np.concatenate([
        mag[:ch,  :cw ].ravel(),
        mag[:ch,  -cw:].ravel(),
        mag[-ch:, :cw ].ravel(),
        mag[-ch:, -cw:].ravel(),
    ])
    return float(corners.std())


def _esc(s: str) -> str:
    """Minimal HTML escaping for log messages."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace("\n", "<br>"))


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("BOS Analysis")
    app.setApplicationVersion("0.1.0")
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLE)

    window = BOSWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
