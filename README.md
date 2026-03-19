# BOS Pipeline

Background-Oriented Schlieren (BOS) image processing pipeline for optical
analysis of gas flows (hydrogen jets, combustion processes) in industrial
safety research.


---

## Features

| Capability | Details |
|---|---|
| **Camera support** | Photron FASTCAM (`.mraw` + `.cihx/.cih`) · Teledyne DALSA (16-bit mono TIFF sequences, multi-page TIFFs, GigE Vision live) |
| **Memory-efficient I/O** | `np.memmap` for 8/16-bit Photron files; lazy TIFF reading — never loads full datasets into RAM |
| **Displacement methods** | Windowed FFT cross-correlation (sub-pixel parabolic/Gaussian peak fitting) · Lucas-Kanade sparse optical flow · Farneback dense optical flow |
| **Abel inversion** | Three-point, BASEX, Hansen-Law via PyAbel; auto or manual symmetry-axis detection |
| **Physical calibration** | Pixel → angular deflection → density (Gladstone-Dale) conversion |
| **Visualisation** | Magnitude maps, signed dx/dy, quiver overlay, Abel field, side-by-side, summary panel — 300 DPI publication quality |
| **Export** | NumPy `.npy` · HDF5 (compressed) · CSV · JSON processing log |
| **CLI** | `bos_process --config config.yaml` or direct flag overrides |

---

## Installation

```bash
git clone https://github.com/Growlmare97/BOS_App.git
cd BOS_App
pip install -e ".[dev]"
```

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations, memmap |
| scipy | ≥1.11 | FFT correlation, interpolation, integration |
| scikit-image | ≥0.21 | Image utilities |
| opencv-python | ≥4.8 | Optical flow (Lucas-Kanade, Farneback) |
| matplotlib | ≥3.7 | Visualisation |
| tifffile | ≥2023.1 | TIFF I/O (Teledyne DALSA) |
| pyMRAW | ≥1.1 | Photron `.mraw` reader |
| PyAbel | ≥0.9 | Abel inversion |
| harvesters | ≥1.4 | GenICam/GigE Vision live acquisition |
| h5py | ≥3.9 | HDF5 export |
| pyyaml | ≥6.0 | YAML config |

Tested on: **Python 3.10, 3.11, 3.12** · Windows 11 / Ubuntu 22.04.

---

## Quickstart

### File-based processing

```bash
# Using a YAML config file (recommended)
bos_process --config config_example.yaml

# Photron — override individual parameters
bos_process --input ./data/shot_001.mraw \
            --camera photron \
            --reference 0 \
            --measurement 10,20,30 \
            --method cross_correlation \
            --window 64 \
            --output ./results

# Teledyne DALSA TIFF sequence
bos_process --input ./data/frames/ \
            --camera dalsa \
            --reference 0 \
            --output ./results

# Live GigE Vision acquisition
bos_process --live \
            --camera dalsa \
            --cti /opt/genicam/producers/DALSA.cti \
            --n-frames 200 \
            --output ./results
```

### Python API

```python
from bos_pipeline.io import get_reader
from bos_pipeline.processing.preprocess import PreprocessConfig, preprocess
from bos_pipeline.processing.displacement import DisplacementConfig, compute_displacement
from bos_pipeline.processing.abel import AbelConfig, abel_invert
from bos_pipeline import visualization as viz, export

# --- Open camera file ---
with get_reader("photron", path="shot_001.mraw") as reader:
    meta = reader.metadata
    print(f"{meta.total_frames} frames @ {meta.frame_rate} fps")

    reference = reader.get_average(range(0, 10))   # avg first 10 frames
    measurement = reader.get_frame(50)

# --- Preprocess ---
ref_p, meas_p = preprocess(
    reference, measurement,
    config=PreprocessConfig(gaussian_sigma=1.5),
)

# --- Displacement ---
dx, dy = compute_displacement(
    ref_p, meas_p,
    config=DisplacementConfig(method="cross_correlation", window_size=64),
)

# --- Abel inversion (axisymmetric flow) ---
inv_field, axis = abel_invert(
    dx, dy,
    config=AbelConfig(enabled=True, method="three_point"),
)

# --- Visualise ---
fig, _ = viz.plot_summary(ref_p, meas_p, dx, dy, inv_field=inv_field, show=True)
viz.save_figure(fig, "./output", "bos_result", ["png", "pdf"])

# --- Export ---
export.export_displacement(dx, dy, "./output", fmt="hdf5")
export.write_log("./output", config={}, input_paths={}, output_paths={})
```

---

## YAML Configuration

Copy `config_example.yaml` and adjust:

```yaml
camera:
  type: photron                        # photron | dalsa | tiff_sequence
  input_path: ./data/shot_001.mraw
  metadata_file: ./data/shot_001.cihx
  reference_frame: 0
  measurement_frames: [10, 20, 30]

preprocessing:
  reference_avg_frames: 10
  gaussian_sigma: 1.5

displacement:
  method: cross_correlation            # cross_correlation | lucas_kanade | farneback
  window_size: 64
  overlap: 0.5

abel:
  enabled: true
  method: three_point                  # three_point | basex | hansenlaw
  component: dx
  axis_mode: auto

export:
  output_dir: ./output
  displacement_format: hdf5            # npy | hdf5 | csv
  save_figures: true
  figure_format: [png, pdf]
```

Full reference: see [config_example.yaml](config_example.yaml).

---

## Project Structure

```
bos_pipeline/
├── pyproject.toml
├── requirements.txt
├── config_example.yaml
├── README.md
├── bos_pipeline/
│   ├── __init__.py
│   ├── io/
│   │   ├── base.py          # Abstract CameraReader + FrameMetadata
│   │   ├── photron.py       # pyMRAW wrapper (memmap for 8/16-bit)
│   │   └── dalsa.py         # TIFF sequence + Harvesters live reader
│   ├── processing/
│   │   ├── preprocess.py    # Dark/flat-field correction, Gaussian filter
│   │   ├── displacement.py  # Cross-correlation, Lucas-Kanade, Farneback
│   │   └── abel.py          # PyAbel integration + density reconstruction
│   ├── visualization.py     # All plot types, scale bars, colorbar helpers
│   ├── export.py            # npy/HDF5/CSV/JSON export
│   └── cli.py               # argparse CLI (bos_process)
├── notebooks/
│   └── bos_demo.ipynb       # End-to-end demo on synthetic data
└── tests/
    ├── __init__.py
    ├── generate_synthetic_data.py   # Gaussian blob generator (Cartesian + axisymmetric)
    ├── test_displacement.py
    └── test_abel.py
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=bos_pipeline --cov-report=term-missing

# Generate synthetic test data first (optional — tests generate inline)
python tests/generate_synthetic_data.py --output tests/synthetic_data --show
```

Tests cover:
- Preprocessing pipeline (dtype, shape, filter effects)
- Cross-correlation: grid shape, sub-pixel shift recovery, known-shift test
- Optical flow: Farneback and Lucas-Kanade output shapes
- Displacement magnitude (Pythagorean identity, edge cases)
- Abel inversion: output shape/dtype, axis detection, method variants
- Density reconstruction
- Error handling (shape mismatch, missing dependencies)

---

## Synthetic Data Generator

```bash
# Cartesian Gaussian blob
python tests/generate_synthetic_data.py --geometry cartesian --show

# Axisymmetric hydrogen jet
python tests/generate_synthetic_data.py \
    --geometry axisymmetric \
    --amplitude 0.06 \
    --sensitivity 60 \
    --output ./tests/synthetic_data \
    --show
```

Outputs: `reference.npy`, `measurement.npy`, `dx_true.npy`, `dy_true.npy`,
`dn.npy`, `reference.tiff`, `measurement.tiff`, `tiff_sequence/` folder.

---

## Physical Unit Conversion

Set `calibration.enabled: true` in your config and provide:

| Parameter | Description |
|---|---|
| `focal_length_mm` | Camera focal length [mm] |
| `distance_to_background_m` | Distance from lens to background pattern [m] |
| `pixel_size_um` | Sensor pixel pitch [µm] |
| `gladstone_dale` | Gladstone-Dale constant K [m³/kg] — H₂: 1.55×10⁻⁴, air: 2.26×10⁻⁴ |
| `reference_density_kg_m3` | Ambient density ρ₀ [kg/m³] |

The pipeline then converts pixel displacements → angular deflections →
refractive-index gradients → density field via Abel inversion.

---

## Notes on Large Files (Photron)

Photron high-speed recordings can be **tens of GB**.  The pipeline handles
this via:

- **`np.memmap`** for 8-bit and 16-bit MRAW files — only the requested
  frames are read from disk.
- **12-bit MRAW** requires a full RAM load (pyMRAW limitation) — the
  reader emits a `UserWarning` when this is detected.
- `get_average()` on the Photron reader uses direct memmap indexing
  (no full-stack load).
- Displacement computation processes one frame pair at a time.

---

## License

MIT © CoStudy GmbH
