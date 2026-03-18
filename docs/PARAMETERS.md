# BOS Analysis — Parameter Reference

Complete guide to every parameter in the **BOS Analysis** desktop app and CLI.

---

## Files

| Field | What to put here |
|---|---|
| **AVI file** | The `.avi` video exported from Photron PFV. The CIHX field auto-fills if the metadata file is in the same folder. |
| **CIHX file** | The `.cihx` (XML) or `.cih` (text) metadata file produced by the Photron camera. Contains frame rate, resolution, trigger frame, exposure, etc. Auto-detected — only browse manually if it lives in a different folder. |
| **Output folder** | Where displacement fields (`.npy`) and figures (`.png`) are saved. Auto-set to a `bos_output/` folder next to your AVI. You can change it to any folder you like. |

---

## Processing Parameters

### Method

How the displacement between the reference and each measurement frame is computed.

| Option | Description | Best for |
|---|---|---|
| `cross_correlation` | **(Default)** Divides the image into small windows and finds the peak of the FFT cross-correlation between each window pair. Very robust and accurate. | Most BOS experiments |
| `farneback` | Dense optical flow (Gunnar Farnebäck algorithm via OpenCV). Runs faster than cross-correlation and can capture large displacements. Less accurate at sub-pixel level. | Rapid preview, large displacements |
| `lucas_kanade` | Sparse Lucas-Kanade flow on a regular grid, then interpolated to full resolution. Good for tracking small, sharp features. | Fine-structure flows |

---

### Ref frame

**Default: `0`**

Index of the frame used as the *reference* (background, no flow).
This is the undisturbed background image — ideally recorded just before the event starts.

- Frame indices start at **0**.
- For a 1000 fps Photron recording where the jet starts at frame 5, use frame `0`, `1`, or `2` as the reference (pure background before anything happens).
- You can scrub the **Frame Preview** slider to find the cleanest background frame.

---

### Meas frames

**Default: `5,6,7,8,9,10`**

Which frames to analyse as *measurement* frames (with flow/density change).

Accepted formats:

| Format | Example | Meaning |
|---|---|---|
| Comma-separated list | `5,6,7,10,20` | Process exactly those frames |
| Range (start:stop) | `5:25` | Process frames 5, 6, 7 … 24 |
| Mixed | `5:10,15,20` | Frames 5–9 plus 15 and 20 |
| `all` | `all` | Every frame except the reference |

**Tip:** Start with a few key frames (e.g. `5,10,20,50`) to check the result quickly before running all frames.

---

### Window [px]

**Default: `32` px**

Side length of the interrogation window used by cross-correlation (ignored for optical-flow methods).

The image is divided into overlapping square tiles of this size. The cross-correlation peak inside each tile gives the local displacement.

| Value | Effect |
|---|---|
| Small (16–32 px) | Higher spatial resolution — detects fine structures. Needs a good signal-to-noise ratio. |
| Large (64–128 px) | Smoother result — more robust to noise and low-contrast backgrounds. Lower spatial resolution. |

**Rule of thumb:** the window should be at least 4–6× larger than the maximum expected displacement in pixels.

---

### Overlap

**Default: `0.75`**  (range 0.0 – 0.95)

Fraction by which adjacent interrogation windows overlap each other.

| Value | Effect |
|---|---|
| `0.0` | Windows are adjacent, no overlap. Coarsest grid — fastest. |
| `0.50` | Windows overlap by 50%. Standard setting. |
| `0.75` | **(Default)** 75% overlap — 4× denser grid than 0.0, good balance of resolution and speed. |
| `0.90` | Very dense grid — highest visual resolution but slowest. |

Overlap does **not** add new information (the underlying pixel data is the same), but it interpolates the vector field more densely, making the displacement maps look smoother and sharper.

---

### Pre-filter σ (sigma)

**Default: `1.0` px**  (range 0.0 – 10.0)

Standard deviation of a Gaussian blur applied to both the reference and measurement frames *before* computing displacement.

| Value | Effect |
|---|---|
| `0.0` | No smoothing. Use if images are already clean. |
| `0.5–1.5` | **(Recommended)** Mild smoothing — removes sensor noise without blurring real structures. |
| `2.0–4.0` | Stronger smoothing. Helps with very noisy images or bright hot pixels. Reduces spatial resolution. |
| `> 5.0` | Heavy blur — only for extremely noisy data. |

**Tip:** If you see a speckled, salt-and-pepper pattern in the displacement map, increase sigma slightly.

---

## H₂ Concentration Measurement

Enable the **H₂ Concentration** group box in the left panel (tick the checkbox) to compute the volumetric hydrogen concentration field from the BOS displacement map.

> **Prerequisite:** The jet must be **axisymmetric** and oriented vertically (axis along Y). The method uses Abel inversion to reconstruct the radial refractive-index profile.

### Physics

1. **Deflection angle** — converts pixel displacement to angular deflection:
   `ε_x(ρ) = dx(ρ) × (mm_per_px / Z_f)`   \[rad\]

2. **Line-of-sight projection** — cumulative integral of `ε_x` from the far edge inward:
   `G(ρ) = −∫_ρ^∞ ε_x dρ'`   (Abel transform of Δn)

3. **Abel inversion** — recovers the radial refractive-index change `Δn(r)` from `G(ρ)` using PyAbel.

4. **Concentration** — divides by the refractive-index contrast between pure H₂ and ambient air:
   `c_H₂(r) = Δn(r) / (n_H₂ − n_air)`   clipped to \[0, 1\]

### Parameters

| Parameter | Default | Description |
|---|---|---|
| **mm / px** | `0.1` | Physical size of one pixel at the background screen \[mm/px\]. Measure with a calibration target placed at the background. |
| **Z_f \[mm\]** | `1000` | Distance from the object (jet) to the background screen \[mm\]. |
| **n_gas** | `1.000132` | Refractive index of the pure gas. H₂ at STP: `1.000132`. Change for other gases. |
| **n_ambient** | `1.000293` | Refractive index of the surrounding ambient. Air at STP: `1.000293`. |
| **Component** | `dx` | Which displacement component carries the radial deflection signal. For a **vertical** jet (axis along Y) use `dx`. For a **horizontal** jet (axis along X) use `dy`. |
| **Abel method** | `three_point` | PyAbel inversion algorithm. `three_point` is fast and robust; `hansenlaw` can be more accurate for noisy data; `basex` uses basis-set expansion. |
| **Axis mode** | `auto` | `auto` detects the symmetry axis from the deflection field intensity centroid. `manual` lets you specify the exact column. |
| **Axis col \[px\]** | — | Only active in `manual` mode. Column index of the jet centreline in the image. |

### Output

When concentration is enabled:
- An extra **Concentration** tab appears in the result panel showing a colour map (plasma) of `c_H₂ ∈ [0, 1]`.
- On save: `frame_NNNNN_concentration.npy` (float32) and `frame_NNNNN_concentration.png` are written to the output folder.

### Tips

- **Calibrate carefully.** The result is linearly proportional to `mm_per_px / Z_f`. A 10% error in either parameter gives a 10% error in concentration.
- **Start with `auto` axis.** Only switch to `manual` if the auto-detected axis is clearly wrong (visible from the dashed white line on the plot).
- If the map shows negative concentrations or large-amplitude artifacts, check that the correct `component` is selected (dx vs dy) and that the jet is centred in the field of view.
- The `three_point` Abel method requires at least ~20 pixels of radial extent. Use a larger window size or zoom into the jet region if the jet is narrow.

---

## Frame Preview

| Control | What it does |
|---|---|
| **Slider** | Scrubs through the video. The Frame View tab updates ~250 ms after you release the slider. |
| **◀ / ▶** | Step one frame at a time. |
| **Preview Frame** | Force-refresh the Frame View tab with the current slider position. |

Use this to find the right reference frame (flat, no flow) and to check which frames show the event of interest.

---

## Result Tabs

| Tab | What is shown |
|---|---|
| **Frame View** | Reference (left) and the selected measurement frame (right), same brightness scale. Good for spotting the density disturbance visually. |
| **Magnitude** | Colour map of `sqrt(dx² + dy²)` — total displacement in pixels. Uses the *turbo* colormap; the colour range is clipped at the 99th percentile so outliers don't collapse the scale. |
| **Components** | Signed `dx` (horizontal) and `dy` (vertical) maps side by side. Red/blue diverging colourmap (*RdBu_r*) — blue = negative displacement, red = positive. Symmetric scale. |
| **Quiver** | Vector arrows overlaid on the reference image. Arrows are colour-coded by magnitude (*plasma* colourmap). Arrow length is scaled so the 95th-percentile vector fills ~70% of the grid spacing. |

---

## Stats Bar

Shown below the result tabs after the pipeline runs.

| Stat | Meaning |
|---|---|
| **Frame** | Which measurement frame is currently displayed. |
| **Max \|d\|** | Largest displacement anywhere in the field [px]. |
| **Mean \|d\|** | Average displacement over the whole frame [px]. |
| **BG noise** | Standard deviation of displacement in the four image corners, where there should be no flow. This is the measurement noise floor [px]. A good BOS setup has BG noise < 0.1 px. |
| **SNR** | Signal-to-Noise Ratio = `20 · log₁₀(mean|d| / BG_noise)` in dB. Higher is better. Typical values: 10–30 dB. |

---

## Result Navigator

The **Result** dropdown in the stats bar lets you switch between all the measurement frames you processed in one run without recomputing. Each frame's displacement map, components, and quiver are cached in memory.

---

## Output Files

After the pipeline finishes (or when you click **Save All Results**), the following files are written to the output folder:

```
bos_output/
├── frame_00005_dx.npy       ← horizontal displacement field [px], float32
├── frame_00005_dy.npy       ← vertical displacement field [px], float32
├── frame_00006_dx.npy
├── frame_00006_dy.npy
│   ...
├── figures/
│   ├── frame_00005_magnitude.png
│   ├── frame_00005_components.png
│   ├── frame_00005_quiver.png
│   ├── frame_00006_magnitude.png
│   │   ...
└── processing_log.json      ← (CLI only) full run parameters and stats
```

The `.npy` files can be loaded in Python with `np.load("frame_00005_dx.npy")`.

---

## CLI Usage

The same pipeline can be run from the command line without the GUI:

```bash
bos_process \
  --input  "path/to/video.avi" \
  --camera photron_avi \
  --metadata "path/to/video.cihx" \
  --reference 0 \
  --measurement "5,6,7,8,9,10,15,20" \
  --window 32 \
  --overlap 0.75 \
  --sigma 1.0 \
  --output ./bos_output
```

All parameters map directly to the GUI fields above.

---

## Quick-start Checklist

1. Record a clean **reference frame** before the event (no flow, same background).
2. Open the app with `bos_app`.
3. Load your `.avi` — CIHX auto-detects.
4. Scrub the slider to confirm the reference frame is flat and measurement frames show the event.
5. Set **Ref frame** to the index of your clean background.
6. Set **Meas frames** to the frames of interest (e.g. `5:30` for the first 25 frames of the event).
7. Start with **Window = 32**, **Overlap = 0.75**, **Sigma = 1.0**.
8. Click **▶ Run Pipeline**.
9. Check **BG noise** in the stats bar — if it's above ~0.2 px, try increasing Sigma slightly.
10. Results and figures are auto-saved to the output folder.
