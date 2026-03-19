import React, { useCallback, useEffect, useRef, useState } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import type { Data, Layout } from 'plotly.js';
const Plot = createPlotlyComponent(Plotly);
import { useAnalysisStore } from '../../store/analysisStore';
import { useProcessing } from '../../hooks/useProcessing';
import type { ResultField } from '../../types/analysis';
import styles from './Kymograph.module.css';

interface KymographProps {
  jobId: string;
  frameCount: number;
  frameRate: number;
  axis: 'horizontal' | 'vertical';
  linePos: number;
  field: ResultField;
}

export function Kymograph({
  jobId,
  frameCount,
  frameRate,
  axis,
  linePos: initialLinePos,
  field,
}: KymographProps) {
  const updateConfig = useAnalysisStore((s) => s.updateConfig);
  const kymographResult = useAnalysisStore((s) => s.kymographResult);
  const setKymographResult = useAnalysisStore((s) => s.setKymographResult);
  const { fetchKymograph } = useProcessing();

  const [linePos, setLinePos] = useState(initialLinePos);
  const [isDragging, setIsDragging] = useState(false);
  const [isLineHovered, setIsLineHovered] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const previewRef = useRef<HTMLDivElement>(null);

  // Sync linePos to store config
  useEffect(() => {
    updateConfig({ kymoLinePos: linePos });
  }, [linePos, updateConfig]);

  // Fetch kymograph when linePos changes (debounced)
  useEffect(() => {
    if (!jobId || frameCount === 0) return;
    setIsLoading(true);
    const timer = setTimeout(() => {
      fetchKymograph({
        jobId,
        frameIdxStart: 0,
        frameIdxEnd: frameCount - 1,
        axis,
        linePos,
        field,
      })
        .then((result) => {
          setKymographResult(result);
        })
        .catch(() => {
          // Silently fail — leave previous result visible
        })
        .finally(() => {
          setIsLoading(false);
        });
    }, 400);
    return () => clearTimeout(timer);
  }, [jobId, frameCount, axis, linePos, field, fetchKymograph, setKymographResult]);

  // ── Draggable line logic ─────────────────────────
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !previewRef.current) return;
      const rect = previewRef.current.getBoundingClientRect();
      if (axis === 'horizontal') {
        const relY = e.clientY - rect.top;
        const px = Math.round((relY / rect.height) * 512);
        setLinePos(Math.max(0, Math.min(512, px)));
      } else {
        const relX = e.clientX - rect.left;
        const px = Math.round((relX / rect.width) * 512);
        setLinePos(Math.max(0, Math.min(512, px)));
      }
    },
    [isDragging, axis]
  );

  const handleMouseUp = useCallback(() => setIsDragging(false), []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // ── Build Plotly traces for kymograph ────────────
  const kymoTraces = React.useMemo((): Data[] => {
    if (!kymographResult) return [];

    const timeAxis = Array.from(
      { length: kymographResult.kymograph[0]?.length ?? 0 },
      (_, i) => (i / frameRate) * 1000
    );

    const heatmap: Data = {
      type: 'heatmap',
      z: kymographResult.kymograph,
      x: timeAxis,
      colorscale: 'RdBu',
      showscale: true,
      colorbar: {
        title: {
          text: 'px',
          side: 'right',
          font: { color: '#94a3b8', size: 11 },
        },
        tickfont: { color: '#94a3b8', size: 10, family: 'JetBrains Mono, monospace' },
        bgcolor: 'rgba(17,24,39,0.8)',
        bordercolor: '#1e293b',
        borderwidth: 1,
        thickness: 12,
        len: 0.85,
      },
      hovertemplate: 'time: %{x:.2f} ms<br>pos: %{y} px<br>disp: %{z:.4f}<extra></extra>',
    } as Data;

    const traces: Data[] = [heatmap];

    // Detected velocity lines overlay
    if (kymographResult.velocityProfile.length > 0 && timeAxis.length > 0) {
      const xStart = timeAxis[0] ?? 0;
      const xEnd = timeAxis[timeAxis.length - 1] ?? 0;
      const v = kymographResult.convectiveVelocity;
      const rows = kymographResult.kymograph.length;
      const midY = rows / 2;
      const slope = v > 0 ? v : 0;
      const yEnd = midY + slope * (xEnd - xStart);

      const velocityLine: Data = {
        type: 'scatter',
        mode: 'lines',
        x: [xStart, xEnd],
        y: [midY, yEnd],
        line: { color: '#ef4444', width: 2, dash: 'dash' },
        name: `v = ${v.toFixed(2)} m/s`,
        showlegend: true,
        hovertemplate: `Convective v = ${v.toFixed(3)} m/s<extra></extra>`,
      } as Data;

      traces.push(velocityLine);
    }

    return traces;
  }, [kymographResult, frameRate]);

  const kymoLayout: Partial<Layout> = React.useMemo(
    () => ({
      margin: { t: 10, b: 50, l: 55, r: 80 },
      paper_bgcolor: '#0a0f1e',
      plot_bgcolor: '#0a0f1e',
      xaxis: {
        title: { text: 'Time [ms]', font: { color: '#64748b', size: 11 } },
        tickfont: { color: '#64748b', size: 10, family: 'JetBrains Mono, monospace' },
        gridcolor: '#1e293b',
        zerolinecolor: '#1e293b',
        showgrid: true,
        automargin: true,
      },
      yaxis: {
        title: { text: 'Position [px]', font: { color: '#64748b', size: 11 } },
        tickfont: { color: '#64748b', size: 10, family: 'JetBrains Mono, monospace' },
        gridcolor: '#1e293b',
        zerolinecolor: '#1e293b',
        showgrid: true,
        automargin: true,
        autorange: 'reversed',
      },
      legend: {
        bgcolor: 'rgba(17,24,39,0.8)',
        bordercolor: '#1e293b',
        borderwidth: 1,
        font: { color: '#94a3b8', size: 10 },
        x: 0,
        y: 1,
        xanchor: 'left',
        yanchor: 'top',
      },
      font: { color: '#f1f5f9' },
    }),
    []
  );

  // Normalized line position (0–1) for SVG overlay
  const lineFraction = linePos / 512;

  return (
    <div className={styles.container}>
      {/* ── Frame Preview with draggable line ── */}
      <div className={styles.previewPane} ref={previewRef}>
        <div className={styles.previewPlaceholder}>
          <svg
            className={styles.previewPlaceholderIcon}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <rect x="2" y="2" width="20" height="20" rx="3" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21,15 16,10 5,21" />
          </svg>
          <span className={styles.previewPlaceholderText}>Frame preview</span>
        </div>

        <svg
          className={styles.svgOverlay}
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          <g
            className={`${styles.lineOverlayGroup} ${axis === 'vertical' ? styles.lineOverlayGroupV : ''}`}
            onMouseDown={handleMouseDown}
            onMouseEnter={() => setIsLineHovered(true)}
            onMouseLeave={() => !isDragging && setIsLineHovered(false)}
          >
            {axis === 'horizontal' ? (
              <>
                {/* Invisible wide hit target */}
                <line
                  x1="0" y1={lineFraction * 100}
                  x2="100" y2={lineFraction * 100}
                  stroke="transparent"
                  strokeWidth="8"
                />
                <line
                  className={`${styles.dragLine} ${isLineHovered || isDragging ? styles.dragLineHover : ''}`}
                  x1="0" y1={lineFraction * 100}
                  x2="100" y2={lineFraction * 100}
                />
                <text
                  className={styles.lineLabel}
                  x="2"
                  y={lineFraction * 100 - 2}
                  style={{ fontSize: '3px' }}
                >
                  y = {linePos} px
                </text>
              </>
            ) : (
              <>
                <line
                  x1={lineFraction * 100} y1="0"
                  x2={lineFraction * 100} y2="100"
                  stroke="transparent"
                  strokeWidth="8"
                />
                <line
                  className={`${styles.dragLine} ${isLineHovered || isDragging ? styles.dragLineHover : ''}`}
                  x1={lineFraction * 100} y1="0"
                  x2={lineFraction * 100} y2="100"
                />
                <text
                  className={styles.lineLabel}
                  x={lineFraction * 100 + 1}
                  y="5"
                  style={{ fontSize: '3px' }}
                >
                  x = {linePos} px
                </text>
              </>
            )}
          </g>
        </svg>
      </div>

      {/* ── Kymograph Plot ── */}
      <div className={styles.kymoPane}>
        {isLoading && (
          <div className={styles.loadingOverlay}>
            <svg
              className={styles.loadingSpinner}
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          </div>
        )}

        <div className={styles.kymoPanelInner}>
          {kymographResult && kymographResult.kymograph.length > 0 ? (
            <Plot
              data={kymoTraces}
              layout={kymoLayout}
              config={{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                scrollZoom: true,
                modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
            />
          ) : (
            <div className={styles.previewPlaceholder} style={{ height: '100%' }}>
              <svg
                className={styles.previewPlaceholderIcon}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
              <span className={styles.previewPlaceholderText}>
                Kymograph will appear here
              </span>
            </div>
          )}
        </div>

        {/* ── Convective Velocity Readout ── */}
        <div className={styles.velocityReadout}>
          <span className={styles.velocityLabel}>Convective velocity</span>
          {kymographResult ? (
            <>
              <span className={styles.velocityValue}>
                {kymographResult.convectiveVelocity.toFixed(2)}
              </span>
              <span className={styles.velocityUnit}>m/s</span>
            </>
          ) : (
            <span className={styles.velocityNA}>—</span>
          )}
        </div>
      </div>
    </div>
  );
}
