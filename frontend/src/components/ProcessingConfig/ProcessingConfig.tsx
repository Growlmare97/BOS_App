import React, { useState, useCallback } from 'react';
import { useAnalysisStore } from '../../store/analysisStore';
import { useProcessing } from '../../hooks/useProcessing';
import styles from './ProcessingConfig.module.css';

// ── SVG Pattern Icons ────────────────────────────────────────────────────────

function RandomDotsSvg() {
  return (
    <svg className={styles.patternSvg} viewBox="0 0 40 40" fill="none">
      <rect width="40" height="40" rx="3" fill="#0d1424" />
      {[
        [6, 8], [14, 4], [22, 12], [30, 6], [36, 14],
        [4, 18], [18, 22], [28, 20], [10, 30], [24, 34],
        [34, 28], [8, 36], [20, 36], [32, 36],
      ].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="1.8" fill="#3b82f6" opacity={0.8} />
      ))}
    </svg>
  );
}

function CheckerboardSvg() {
  const cells = [];
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      if ((row + col) % 2 === 0) {
        cells.push(
          <rect key={`${row}-${col}`} x={col * 10} y={row * 10} width="10" height="10" fill="#3b82f6" opacity={0.7} />
        );
      }
    }
  }
  return (
    <svg className={styles.patternSvg} viewBox="0 0 40 40" fill="none">
      <rect width="40" height="40" rx="3" fill="#0d1424" />
      {cells}
    </svg>
  );
}

function AssStripesSvg() {
  return (
    <svg className={styles.patternSvg} viewBox="0 0 40 40" fill="none">
      <rect width="40" height="40" rx="3" fill="#0d1424" />
      <line x1="-4" y1="4" x2="44" y2="52" stroke="#3b82f6" strokeWidth="5" opacity="0.6" />
      <line x1="-16" y1="4" x2="32" y2="52" stroke="#3b82f6" strokeWidth="5" opacity="0.6" />
      <line x1="8" y1="4" x2="56" y2="52" stroke="#3b82f6" strokeWidth="5" opacity="0.6" />
    </svg>
  );
}

// ── Accordion Section ────────────────────────────────────────────────────────

interface SectionProps {
  title: string;
  icon: React.ReactNode;
  badge?: string;
  badgeVariant?: 'blue' | 'green';
  defaultOpen?: boolean;
  children: React.ReactNode;
}

function Section({ title, icon, badge, badgeVariant = 'blue', defaultOpen = false, children }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={styles.section}>
      <div
        className={styles.sectionHeader}
        onClick={() => setOpen((o) => !o)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && setOpen((o) => !o)}
        aria-expanded={open}
      >
        <div className={styles.sectionHeaderLeft}>
          <span className={styles.sectionIcon}>{icon}</span>
          <span className={styles.sectionTitle}>{title}</span>
          {badge && (
            <span
              className={`${styles.sectionBadge} ${badgeVariant === 'green' ? styles.sectionBadgeGreen : ''}`}
            >
              {badge}
            </span>
          )}
        </div>
        <svg
          className={`${styles.chevron} ${open ? styles.chevronOpen : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>
      {open && <div className={styles.sectionBody}>{children}</div>}
    </div>
  );
}

// ── Toggle Switch ────────────────────────────────────────────────────────────

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <div className={styles.toggleRow}>
      <span className={styles.toggleLabel}>{label}</span>
      <label className={styles.toggleSwitch} aria-label={label}>
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className={styles.toggleSlider} />
      </label>
    </div>
  );
}

// ── Tooltip Icon ─────────────────────────────────────────────────────────────

function TooltipIcon({ text }: { text: string }) {
  return (
    <svg
      className={styles.tooltipIcon}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      aria-label={text}
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

// ── Slider Field ─────────────────────────────────────────────────────────────

function SliderField({
  label,
  value,
  min,
  max,
  step,
  formatValue,
  hint,
  tooltip,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  formatValue?: (v: number) => string;
  hint?: string;
  tooltip?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className={styles.sliderRow}>
      <div className={styles.sliderHeader}>
        <span className={styles.fieldLabel}>
          {label}
          {tooltip && <TooltipIcon text={tooltip} />}
        </span>
        <span className={styles.sliderValue}>
          {formatValue ? formatValue(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {hint && <span className={styles.sliderHint}>{hint}</span>}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────

type OutputFormat = 'npy' | 'hdf5' | 'csv' | 'vtk';
const OUTPUT_FORMATS: OutputFormat[] = ['npy', 'hdf5', 'csv', 'vtk'];

export function ProcessingConfig() {
  const config = useAnalysisStore((s) => s.config);
  const updateConfig = useAnalysisStore((s) => s.updateConfig);
  const stage = useAnalysisStore((s) => s.stage);
  const setJobId = useAnalysisStore((s) => s.setJobId);
  const setProgress = useAnalysisStore((s) => s.setProgress);
  const setError = useAnalysisStore((s) => s.setError);
  const { startProcessing } = useProcessing();

  const [patternType, setPatternType] = useState<'dots' | 'checkerboard' | 'stripes'>('dots');

  const isRunning = stage !== 'idle' && stage !== 'done' && stage !== 'error';

  // Computed sensitivity: S = focalLength / (ZD - ZA) * pixelPitch
  const sensitivity = config.focalLengthMm > 0 && config.zdMm > config.zaMm
    ? ((config.focalLengthMm / (config.zdMm - config.zaMm)) * (config.pixelPitchUm / 1000)).toFixed(4)
    : '—';

  // Overlap density label
  const overlapDensityLabel = useCallback((v: number): string => {
    const factor = Math.round(1 / (1 - v));
    return `${factor}× denser`;
  }, []);

  const toggleFormat = useCallback(
    (fmt: OutputFormat) => {
      const current = config.outputFormats;
      const next = current.includes(fmt)
        ? current.filter((f) => f !== fmt)
        : [...current, fmt];
      if (next.length > 0) updateConfig({ outputFormats: next });
    },
    [config.outputFormats, updateConfig]
  );

  const handleRun = useCallback(async () => {
    if (isRunning) return;
    setError(null);
    setProgress('loading', 0, 'Starting…');
    try {
      const jobId = await startProcessing(config);
      setJobId(jobId);
    } catch (err) {
      setProgress('error', 0, 'Failed to start');
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [isRunning, config, startProcessing, setJobId, setProgress, setError]);

  return (
    <div className={styles.container}>
      {/* ══ Section 1: Background Pattern ══ */}
      <Section
        title="Background Pattern"
        badge="ASS-BOS"
        defaultOpen
        icon={
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="12" r="3" />
            <path d="M3 12h1M20 12h1M12 3v1M12 20v1M5.64 5.64l.71.71M17.66 17.66l.71.71M17.66 6.34l-.71.71M5.64 18.36l.71-.71" />
          </svg>
        }
      >
        <div className={styles.patternGrid}>
          {(
            [
              { key: 'dots', label: 'Random Dots', Svg: RandomDotsSvg },
              { key: 'checkerboard', label: 'Checkerboard', Svg: CheckerboardSvg },
              { key: 'stripes', label: 'ASS Stripes', Svg: AssStripesSvg },
            ] as const
          ).map(({ key, label, Svg }) => (
            <div
              key={key}
              className={`${styles.patternCard} ${patternType === key ? styles.patternCardActive : ''}`}
              onClick={() => setPatternType(key)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && setPatternType(key)}
            >
              <Svg />
              <span className={styles.patternName}>{label}</span>
            </div>
          ))}
        </div>
        <div className={styles.metricsRow}>
          <div className={styles.metricCell}>
            <span className={styles.metricCellLabel}>SNR</span>
            <span className={styles.metricCellValue}>—</span>
          </div>
          <div className={styles.metricCell}>
            <span className={styles.metricCellLabel}>Contrast</span>
            <span className={styles.metricCellValue}>—</span>
          </div>
        </div>
      </Section>

      {/* ══ Section 2: Displacement Algorithm ══ */}
      <Section
        title="Displacement Algorithm"
        defaultOpen
        icon={
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        }
      >
        <div className={styles.fieldRow}>
          <span className={styles.fieldLabel}>Method</span>
          <div className={styles.btnGroup}>
            {(
              [
                { value: 'cross_correlation', label: 'X-Corr' },
                { value: 'lucas_kanade', label: 'Lucas-Kanade' },
                { value: 'farneback', label: 'Farneback' },
              ] as const
            ).map(({ value, label }) => (
              <button
                key={value}
                className={`${styles.btnGroupBtn} ${config.method === value ? styles.btnGroupBtnActive : ''}`}
                onClick={() => updateConfig({ method: value })}
                type="button"
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <SliderField
          label="Window Size"
          value={config.windowSize}
          min={8}
          max={256}
          step={8}
          formatValue={(v) => `${v} px`}
          hint={`Grid: ~${Math.floor(512 / (config.windowSize * (1 - config.overlap)))} × ${Math.floor(512 / (config.windowSize * (1 - config.overlap)))} vectors`}
          onChange={(v) => updateConfig({ windowSize: v })}
        />

        <SliderField
          label="Overlap"
          value={config.overlap}
          min={0}
          max={0.95}
          step={0.05}
          formatValue={(v) => `${Math.round(v * 100)}%`}
          hint={overlapDensityLabel(config.overlap)}
          onChange={(v) => updateConfig({ overlap: v })}
        />

        <SliderField
          label="Pre-filter σ"
          value={config.sigma}
          min={0}
          max={5}
          step={0.1}
          formatValue={(v) => `${v.toFixed(1)} px`}
          onChange={(v) => updateConfig({ sigma: v })}
        />

        <Toggle
          checked={config.ensembleAveraging}
          onChange={(v) => updateConfig({ ensembleAveraging: v })}
          label="Ensemble Averaging"
        />
        {config.ensembleAveraging && (
          <div className={styles.fieldHint} style={{ marginTop: -6, marginLeft: 2 }}>
            Meinhart 1999 — avg correlation maps before peak detection
          </div>
        )}

        <Toggle
          checked={config.multiPass}
          onChange={(v) => updateConfig({ multiPass: v })}
          label="Multi-pass"
        />
        {config.multiPass && (
          <div className={styles.fieldRow}>
            <span className={styles.fieldLabel}>Pass count</span>
            <input
              type="number"
              min={2}
              max={8}
              value={config.multiPassCount}
              onChange={(e) => updateConfig({ multiPassCount: parseInt(e.target.value, 10) })}
              style={{ width: '80px' }}
            />
          </div>
        )}
      </Section>

      {/* ══ Section 3: Physical Calibration ══ */}
      <Section
        title="Physical Calibration"
        icon={
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
        }
      >
        <div className={styles.inlineRow}>
          <div className={styles.fieldRow}>
            <label className={styles.fieldLabel}>
              Z<sub>D</sub> [mm]
              <TooltipIcon text="Camera-to-background distance" />
            </label>
            <input
              type="number"
              value={config.zdMm}
              onChange={(e) => updateConfig({ zdMm: parseFloat(e.target.value) })}
            />
          </div>
          <div className={styles.fieldRow}>
            <label className={styles.fieldLabel}>
              Z<sub>A</sub> [mm]
              <TooltipIcon text="Camera-to-object distance" />
            </label>
            <input
              type="number"
              value={config.zaMm}
              onChange={(e) => updateConfig({ zaMm: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className={styles.inlineRow}>
          <div className={styles.fieldRow}>
            <label className={styles.fieldLabel}>Focal length [mm]</label>
            <input
              type="number"
              value={config.focalLengthMm}
              onChange={(e) => updateConfig({ focalLengthMm: parseFloat(e.target.value) })}
            />
          </div>
          <div className={styles.fieldRow}>
            <label className={styles.fieldLabel}>Pixel pitch [µm]</label>
            <input
              type="number"
              value={config.pixelPitchUm}
              onChange={(e) => updateConfig({ pixelPitchUm: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className={styles.fieldRow}>
          <label className={styles.fieldLabel}>Pixel scale [mm/px]</label>
          <input
            type="number"
            step={0.001}
            value={config.pixelScaleMm}
            onChange={(e) => updateConfig({ pixelScaleMm: parseFloat(e.target.value) })}
          />
        </div>

        <div className={styles.sensitivityBox}>
          <span className={styles.sensitivityLabel}>Computed Sensitivity</span>
          <span className={styles.sensitivityValue}>
            S = {sensitivity} px/rad
          </span>
        </div>
      </Section>

      {/* ══ Section 4: Abel Inversion ══ */}
      <Section
        title="Abel Inversion"
        icon={
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="2" x2="12" y2="22" />
            <path d="M12 2 C 16 7, 16 17, 12 22" />
          </svg>
        }
      >
        <Toggle
          checked={config.abelEnabled}
          onChange={(v) => updateConfig({ abelEnabled: v })}
          label="Enable Abel Inversion"
        />

        {config.abelEnabled && (
          <>
            <div className={styles.fieldRow}>
              <span className={styles.fieldLabel}>Method</span>
              <div className={styles.btnGroup}>
                {(
                  [
                    { value: 'three_point', label: 'Three-Point' },
                    { value: 'hansenlaw', label: 'Hansen-Law' },
                    { value: 'basex', label: 'BASEX' },
                  ] as const
                ).map(({ value, label }) => (
                  <button
                    key={value}
                    className={`${styles.btnGroupBtn} ${config.abelMethod === value ? styles.btnGroupBtnActive : ''}`}
                    onClick={() => updateConfig({ abelMethod: value })}
                    type="button"
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.fieldRow}>
              <span className={styles.fieldLabel}>Axis Mode</span>
              <div className={styles.btnGroup}>
                {(['auto', 'manual'] as const).map((m) => (
                  <button
                    key={m}
                    className={`${styles.btnGroupBtn} ${config.abelAxisMode === m ? styles.btnGroupBtnActive : ''}`}
                    onClick={() => updateConfig({ abelAxisMode: m })}
                    type="button"
                  >
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {config.abelAxisMode === 'manual' && (
              <div className={styles.fieldRow}>
                <label className={styles.fieldLabel}>Axis Position [px]</label>
                <input
                  type="number"
                  value={config.abelAxisPosition}
                  onChange={(e) =>
                    updateConfig({ abelAxisPosition: parseInt(e.target.value, 10) })
                  }
                />
              </div>
            )}
          </>
        )}
      </Section>

      {/* ══ Section 5: Velocity Analysis ══ */}
      <Section
        title="Velocity Analysis"
        badge="NEW"
        badgeVariant="green"
        icon={
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
        }
      >
        <Toggle
          checked={config.velocityEnabled}
          onChange={(v) => updateConfig({ velocityEnabled: v })}
          label="Enable Velocity Analysis"
        />

        {config.velocityEnabled && (
          <>
            <div className={styles.fieldRow}>
              <span className={styles.fieldLabel}>Method</span>
              <div className={styles.btnGroup}>
                {(
                  [
                    { value: 'frame_to_frame', label: 'Frame-to-Frame' },
                    { value: 'kymography', label: 'Kymography' },
                  ] as const
                ).map(({ value, label }) => (
                  <button
                    key={value}
                    className={`${styles.btnGroupBtn} ${config.velocityMethod === value ? styles.btnGroupBtnActive : ''}`}
                    onClick={() => updateConfig({ velocityMethod: value })}
                    type="button"
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {config.velocityMethod === 'kymography' && (
              <>
                <div className={styles.fieldRow}>
                  <span className={styles.fieldLabel}>Axis</span>
                  <div className={styles.btnGroup}>
                    {(['horizontal', 'vertical'] as const).map((ax) => (
                      <button
                        key={ax}
                        className={`${styles.btnGroupBtn} ${config.kymoAxis === ax ? styles.btnGroupBtnActive : ''}`}
                        onClick={() => updateConfig({ kymoAxis: ax })}
                        type="button"
                      >
                        {ax.charAt(0).toUpperCase() + ax.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>

                <div className={styles.fieldRow}>
                  <label className={styles.fieldLabel}>Line Position [px]</label>
                  <input
                    type="number"
                    value={config.kymoLinePos}
                    onChange={(e) =>
                      updateConfig({ kymoLinePos: parseInt(e.target.value, 10) })
                    }
                  />
                </div>
              </>
            )}
          </>
        )}
      </Section>

      {/* ══ Section 6: Export ══ */}
      <Section
        title="Export"
        icon={
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        }
      >
        <div className={styles.fieldRow}>
          <span className={styles.fieldLabel}>Output Formats</span>
          <div className={styles.formatGrid}>
            {OUTPUT_FORMATS.map((fmt) => {
              const active = config.outputFormats.includes(fmt);
              return (
                <div
                  key={fmt}
                  className={`${styles.formatCheckbox} ${active ? styles.formatCheckboxActive : ''}`}
                  onClick={() => toggleFormat(fmt)}
                  role="checkbox"
                  aria-checked={active}
                  tabIndex={0}
                  onKeyDown={(e) => e.key === 'Enter' && toggleFormat(fmt)}
                >
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={() => toggleFormat(fmt)}
                    tabIndex={-1}
                    aria-hidden
                  />
                  <span className={styles.formatCheckboxLabel}>
                    {fmt.toUpperCase()}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div className={styles.fieldRow}>
          <label className={styles.fieldLabel}>Output Directory</label>
          <input
            type="text"
            placeholder="/path/to/output"
            value={config.outputPath}
            onChange={(e) => updateConfig({ outputPath: e.target.value })}
            style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}
          />
        </div>
      </Section>

      {/* ══ Run Button ══ */}
      <button
        className={`${styles.runBtn} ${isRunning ? styles.runBtnRunning : ''}`}
        onClick={handleRun}
        disabled={!config.inputPath || isRunning}
        type="button"
      >
        {isRunning ? (
          <>
            <svg
              width="16" height="16" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth={2}
              style={{ animation: 'spin 1s linear infinite' }}
            >
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
            Processing…
          </>
        ) : (
          <>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Run Analysis
          </>
        )}
      </button>
    </div>
  );
}
