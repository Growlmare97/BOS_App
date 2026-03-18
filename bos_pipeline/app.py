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
* Four result tabs: Frame View · Magnitude · Components · Quiver
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
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSpinBox, QStatusBar, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
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
from bos_pipeline import export as _export
import bos_pipeline.visualization as viz


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

            self.status_msg.emit(f"Building reference (frame {p['ref_idx']})…")
            ref_raw = reader.get_frame(p["ref_idx"]).astype(np.float32)

            meas_indices: List[int] = p["meas_indices"]
            n = len(meas_indices)

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
                self.result_ready.emit({
                    "frame_idx": meas_idx,
                    "dx": dx,
                    "dy": dy,
                    "ref": ref_p,
                    "meas": meas_p,
                })

        self.finished.emit()


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

        # Worker / thread
        self._thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None

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

    # ── Left panel ────────────────────────────────────────────────────────────

    def _make_left_panel(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)
        vl.addWidget(self._make_files_group())
        vl.addWidget(self._make_params_group())
        vl.addWidget(self._make_viewer_group())
        vl.addWidget(self._make_buttons_section())
        vl.addWidget(self._make_log_group(), stretch=1)
        return w

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
        grp = QGroupBox("Processing Parameters")
        fl = QFormLayout(grp)
        fl.setSpacing(7)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["cross_correlation", "farneback", "lucas_kanade"])
        fl.addRow("Method:", self._method_combo)

        self._ref_spin = QSpinBox()
        self._ref_spin.setRange(0, 9999)
        fl.addRow("Ref frame:", self._ref_spin)

        self._meas_edit = QLineEdit("5,6,7,8,9,10")
        self._meas_edit.setPlaceholderText("e.g. 5,6,7 or 5:20")
        fl.addRow("Meas frames:", self._meas_edit)

        self._window_spin = QSpinBox()
        self._window_spin.setRange(8, 512)
        self._window_spin.setSingleStep(8)
        self._window_spin.setValue(32)
        fl.addRow("Window [px]:", self._window_spin)

        self._overlap_spin = QDoubleSpinBox()
        self._overlap_spin.setRange(0.0, 0.95)
        self._overlap_spin.setSingleStep(0.05)
        self._overlap_spin.setValue(0.75)
        self._overlap_spin.setDecimals(2)
        fl.addRow("Overlap:", self._overlap_spin)

        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 10.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setValue(1.0)
        self._sigma_spin.setDecimals(1)
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

        self._progress = QProgressBar()
        self._progress.setMaximum(0)      # indeterminate
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)

        vl.addWidget(self._run_btn)
        vl.addWidget(self._cancel_btn)
        vl.addWidget(self._save_btn)
        vl.addWidget(self._progress)
        return w

    def _make_log_group(self) -> QGroupBox:
        grp = QGroupBox("Log")
        vl = QVBoxLayout(grp)
        vl.setContentsMargins(4, 4, 4, 4)
        self._log_box = QTextEdit()
        self._log_box.setObjectName("log_box")
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(80)
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
            ("frame_view",  "  Frame View  "),
            ("magnitude",   "  Magnitude   "),
            ("components",  "  Components  "),
            ("quiver",      "    Quiver    "),
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

        params = {
            "avi_path":    str(self._avi_path),
            "cihx_path":   str(self._cihx_path) if self._cihx_path else None,
            "ref_idx":     self._ref_spin.value(),
            "meas_indices": meas_indices,
            "method":      self._method_combo.currentText(),
            "window":      self._window_spin.value(),
            "overlap":     self._overlap_spin.value(),
            "sigma":       self._sigma_spin.value(),
        }

        self._results.clear()
        self._result_combo.clear()

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
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

    # ── Save ──────────────────────────────────────────────────────────────────

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
