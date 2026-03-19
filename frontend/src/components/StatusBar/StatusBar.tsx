import { useEffect, useRef, useState } from 'react';
import { useAnalysisStore } from '../../store/analysisStore';
import { useWebSocket } from '../../hooks/useWebSocket';
import type { PipelineStage } from '../../types/analysis';
import styles from './StatusBar.module.css';

interface PipelineStep {
  stage: PipelineStage;
  label: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  { stage: 'loading',      label: 'Load' },
  { stage: 'preprocessing', label: 'Preprocess' },
  { stage: 'displacement',  label: 'Correlate' },
  { stage: 'abel',          label: 'Abel' },
  { stage: 'velocity',      label: 'Velocity' },
  { stage: 'exporting',     label: 'Export' },
  { stage: 'done',          label: 'Done' },
];

const STAGE_ORDER: PipelineStage[] = [
  'idle', 'loading', 'preprocessing', 'displacement',
  'abel', 'concentration', 'velocity', 'exporting', 'done', 'error',
];

function stageIndex(s: PipelineStage): number {
  return STAGE_ORDER.indexOf(s);
}

function getStepClass(
  step: PipelineStep,
  currentStage: PipelineStage
): string {
  if (currentStage === 'error') {
    // All steps before current are still "done" visually
    return styles.stageError;
  }
  const stepIdx = stageIndex(step.stage);
  const curIdx = stageIndex(currentStage);
  if (stepIdx < curIdx) return styles.stageDone;
  if (stepIdx === curIdx) return styles.stageCurrent;
  return '';
}

export function StatusBar() {
  const jobId = useAnalysisStore((s) => s.jobId);
  const stage = useAnalysisStore((s) => s.stage);
  const progress = useAnalysisStore((s) => s.progress);
  const progressMessage = useAnalysisStore((s) => s.progressMessage);
  const error = useAnalysisStore((s) => s.error);

  const { connected } = useWebSocket(jobId);

  // Processing speed counter
  const [fps, setFps] = useState<number | null>(null);
  const lastProgressRef = useRef<{ p: number; t: number } | null>(null);

  useEffect(() => {
    if (stage === 'displacement' && progress > 0) {
      const now = Date.now();
      const last = lastProgressRef.current;
      if (last && progress > last.p) {
        const dt = (now - last.t) / 1000;
        const dp = progress - last.p;
        if (dt > 0) {
          const framesPerSec = (dp / 100) * 1 / dt;
          setFps(Math.round(framesPerSec * 10) / 10);
        }
      }
      lastProgressRef.current = { p: progress, t: now };
    } else if (stage === 'idle' || stage === 'done') {
      setFps(null);
      lastProgressRef.current = null;
    }
  }, [stage, progress]);

  const progressBarClass = [
    styles.progressFill,
    stage === 'done' ? styles.progressFillDone : '',
    stage === 'error' ? styles.progressFillError : '',
  ]
    .filter(Boolean)
    .join(' ');

  const messageClass = [
    styles.messageArea,
    error ? styles.messageError : '',
    progressMessage && !error ? styles.messageActive : '',
  ]
    .filter(Boolean)
    .join(' ');

  const currentIdxInPipeline = stageIndex(stage);

  return (
    <div className={styles.bar}>
      {/* ── Pipeline Steps ── */}
      <div className={styles.pipeline}>
        {PIPELINE_STEPS.map((step, i) => {
          const stepClass = getStepClass(step, stage);
          const prevDone =
            i > 0 && stageIndex(PIPELINE_STEPS[i - 1].stage) < currentIdxInPipeline;

          return (
            <div key={step.stage} className={styles.stageItem}>
              {i > 0 && (
                <div
                  className={`${styles.stageSep} ${prevDone ? styles.stageSepDone : ''}`}
                />
              )}
              <div className={`${stepClass}`}>
                <button className={styles.stageBtn} type="button" tabIndex={-1}>
                  <span className={styles.stageDot} />
                  {step.label}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Progress Message ── */}
      <div className={messageClass}>
        {error ?? (progressMessage || (stage === 'idle' ? 'Ready' : ''))}
      </div>

      {/* ── Right Section ── */}
      <div className={styles.rightSection}>
        {/* Progress bar + % */}
        {stage !== 'idle' && (
          <>
            <div className={styles.progressSection}>
              <div className={styles.progressBar}>
                <div
                  className={progressBarClass}
                  style={{ width: `${progress}%` }}
                />
              </div>
              <span className={styles.progressPct}>
                {(progress ?? 0).toFixed(0)}%
              </span>
            </div>
            <div className={styles.sep} />
          </>
        )}

        {/* Speed */}
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Speed</span>
          <span className={`${styles.statValue} ${fps ? styles.statValueGreen : ''}`}>
            {fps !== null ? `${fps} fps` : '—'}
          </span>
        </div>

        <div className={styles.sep} />

        {/* Memory placeholder */}
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Mem</span>
          <span className={styles.statValue}>—</span>
        </div>

        <div className={styles.sep} />

        {/* WebSocket connection dot */}
        <div className={styles.statItem}>
          <div
            className={`${styles.connDot} ${connected ? styles.connDotOnline : styles.connDotOffline}`}
            title={connected ? 'Connected' : 'Disconnected'}
          />
          <span className={styles.statLabel}>{connected ? 'Live' : 'Offline'}</span>
        </div>
      </div>
    </div>
  );
}
