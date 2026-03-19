import { useState, useCallback, useEffect } from 'react';
import { useAnalysisStore } from '../../store/analysisStore';
import { useProcessing } from '../../hooks/useProcessing';
import type { ResultField, ColormapName } from '../../types/analysis';
import { PlotPanel } from './PlotPanel';
import { Kymograph } from '../Kymograph/Kymograph';
import styles from './ResultsViewer.module.css';

interface Tab {
  id: ResultField | 'kymograph';
  label: string;
  requiresConcentration?: boolean;
  requiresVelocity?: boolean;
}

const TABS: Tab[] = [
  { id: 'magnitude', label: 'Displacement' },
  { id: 'dx', label: 'dx' },
  { id: 'dy', label: 'dy' },
  { id: 'concentration', label: 'Concentration', requiresConcentration: true },
  { id: 'u', label: 'Velocity U', requiresVelocity: true },
  { id: 'v', label: 'Velocity V', requiresVelocity: true },
  { id: 'vorticity', label: 'Vorticity', requiresVelocity: true },
  { id: 'kymograph', label: 'Kymograph', requiresVelocity: true },
];

const COLORMAPS: ColormapName[] = ['viridis', 'plasma', 'turbo', 'RdBu', 'seismic', 'inferno'];

interface InspectorState {
  x: number;
  y: number;
  value: number;
}

export function ResultsViewer() {
  const jobId = useAnalysisStore((s) => s.jobId);
  const stage = useAnalysisStore((s) => s.stage);
  const results = useAnalysisStore((s) => s.results);
  const selectedFrame = useAnalysisStore((s) => s.selectedFrame);
  const selectedField = useAnalysisStore((s) => s.selectedField);
  const selectedColormap = useAnalysisStore((s) => s.selectedColormap);
  const showQuiver = useAnalysisStore((s) => s.showQuiver);
  const fieldData = useAnalysisStore((s) => s.fieldData);
  const metadata = useAnalysisStore((s) => s.metadata);
  const config = useAnalysisStore((s) => s.config);

  const setSelectedFrame = useAnalysisStore((s) => s.setSelectedFrame);
  const setSelectedField = useAnalysisStore((s) => s.setSelectedField);
  const setSelectedColormap = useAnalysisStore((s) => s.setSelectedColormap);
  const toggleQuiver = useAnalysisStore((s) => s.toggleQuiver);
  const setFieldData = useAnalysisStore((s) => s.setFieldData);

  const [activeTab, setActiveTab] = useState<ResultField | 'kymograph'>('magnitude');
  const [showColorbar, setShowColorbar] = useState(true);
  const [inspector, setInspector] = useState<InspectorState | null>(null);

  const { fetchFieldData } = useProcessing();

  const hasResults = stage === 'done' && results.length > 0;

  const currentSummary = results.find((r) => r.frameIdx === selectedFrame);
  const hasConcentration = currentSummary?.hasConcentration ?? false;
  const hasVelocity = currentSummary?.hasVelocity ?? false;

  const frameCount = metadata?.frameCount ?? 0;

  // Fetch field data when tab / frame changes
  useEffect(() => {
    if (!jobId || !hasResults || activeTab === 'kymograph') {
      setFieldData(null);
      return;
    }

    const field = activeTab as ResultField;
    let cancelled = false;

    fetchFieldData(jobId, selectedFrame, field)
      .then((data) => {
        if (!cancelled) setFieldData(data);
      })
      .catch(() => {
        if (!cancelled) setFieldData(null);
      });

    return () => {
      cancelled = true;
    };
  }, [jobId, hasResults, selectedFrame, activeTab, fetchFieldData, setFieldData]);

  const handleTabChange = useCallback(
    (tab: ResultField | 'kymograph') => {
      setActiveTab(tab);
      if (tab !== 'kymograph') {
        setSelectedField(tab as ResultField);
      }
    },
    [setSelectedField]
  );

  const handlePrevFrame = useCallback(() => {
    setSelectedFrame(Math.max(0, selectedFrame - 1));
  }, [selectedFrame, setSelectedFrame]);

  const handleNextFrame = useCallback(() => {
    setSelectedFrame(Math.min(Math.max(0, frameCount - 1), selectedFrame + 1));
  }, [selectedFrame, frameCount, setSelectedFrame]);

  const handleHover = useCallback((x: number, y: number, value: number) => {
    setInspector({ x, y, value });
  }, []);

  const handleResetZoom = useCallback(() => {
    // Trigger Plotly relayout via a key change — simplest reset strategy
    setFieldData(fieldData ? { ...fieldData } : null);
  }, [fieldData, setFieldData]);

  return (
    <div className={styles.container}>
      {/* ── Tab Bar ── */}
      <div className={styles.tabBar}>
        {TABS.map((tab) => {
          const disabled =
            (tab.requiresConcentration && !hasConcentration) ||
            (tab.requiresVelocity && !hasVelocity);
          return (
            <button
              key={tab.id}
              className={`${styles.tab} ${activeTab === tab.id ? styles.tabActive : ''} ${disabled ? styles.tabDisabled : ''}`}
              onClick={() => !disabled && handleTabChange(tab.id)}
              disabled={disabled}
              type="button"
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* ── Toolbar ── */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarGroup}>
          <span className={styles.toolbarLabel}>Colormap</span>
          <select
            className={styles.toolbarSelect}
            value={selectedColormap}
            onChange={(e) => setSelectedColormap(e.target.value as ColormapName)}
          >
            {COLORMAPS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.toolbarSep} />

        <div className={styles.toolbarGroup}>
          <button
            className={`${styles.toolbarBtn} ${showQuiver ? styles.toolbarBtnActive : ''}`}
            onClick={toggleQuiver}
            type="button"
            title="Toggle quiver overlay"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="12 5 19 12 12 19" />
            </svg>
            Quiver
          </button>

          <button
            className={`${styles.toolbarBtn} ${showColorbar ? styles.toolbarBtnActive : ''}`}
            onClick={() => setShowColorbar((v) => !v)}
            type="button"
            title="Toggle colorbar"
          >
            Colorbar
          </button>

          <button
            className={styles.toolbarBtn}
            onClick={handleResetZoom}
            type="button"
            title="Reset zoom"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
            Reset
          </button>
        </div>

        {/* Pixel Inspector */}
        <div className={styles.inspectorReadout}>
          <span>x: <span className={styles.inspectorVal}>{inspector ? inspector.x.toFixed(0) : '—'}</span></span>
          <div className={styles.inspectorSep} />
          <span>y: <span className={styles.inspectorVal}>{inspector ? inspector.y.toFixed(0) : '—'}</span></span>
          <div className={styles.inspectorSep} />
          <span>
            val: <span className={styles.inspectorVal}>
              {inspector ? inspector.value.toFixed(4) : '—'}
            </span>
            {fieldData ? ` ${fieldData.unit}` : ''}
          </span>
        </div>
      </div>

      {/* ── Frame Selector ── */}
      {activeTab !== 'kymograph' && (
        <div className={styles.frameSelector}>
          <span className={styles.frameSelectorLabel}>Frame</span>
          <button
            className={styles.frameArrow}
            onClick={handlePrevFrame}
            disabled={selectedFrame <= 0}
            type="button"
            aria-label="Previous frame"
          >
            ‹
          </button>
          <input
            type="number"
            className={styles.frameInput}
            min={0}
            max={Math.max(0, frameCount - 1)}
            value={selectedFrame}
            onChange={(e) => setSelectedFrame(parseInt(e.target.value, 10) || 0)}
          />
          <button
            className={styles.frameArrow}
            onClick={handleNextFrame}
            disabled={selectedFrame >= frameCount - 1}
            type="button"
            aria-label="Next frame"
          >
            ›
          </button>
          <span className={styles.frameCount}>/ {Math.max(0, frameCount - 1)}</span>
        </div>
      )}

      {/* ── Plot Area ── */}
      <div className={styles.plotArea}>
        {!hasResults ? (
          <div className={styles.emptyState}>
            <svg
              className={styles.emptyIcon}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={1}
            >
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <path d="M3 9h18M9 21V9" />
              <circle cx="15" cy="15" r="2" />
            </svg>
            <div className={styles.emptyTitle}>
              {stage === 'idle' ? 'No results yet' : 'Processing…'}
            </div>
            <div className={styles.emptySubtitle}>
              {stage === 'idle'
                ? 'Load a file and run analysis to see results'
                : 'Results will appear when processing completes'}
            </div>
          </div>
        ) : activeTab === 'kymograph' ? (
          <Kymograph
            jobId={jobId ?? ''}
            frameCount={frameCount}
            frameRate={metadata?.frameRate ?? 1}
            axis={config.kymoAxis}
            linePos={config.kymoLinePos}
            field={selectedField}
          />
        ) : (
          <PlotPanel
            fieldData={fieldData}
            colormap={selectedColormap}
            showQuiver={showQuiver}
            field={selectedField}
            showColorbar={showColorbar}
            onHover={handleHover}
          />
        )}
      </div>
    </div>
  );
}
