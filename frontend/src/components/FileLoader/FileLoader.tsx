import React, { useCallback, useRef, useState } from 'react';
import { useAnalysisStore } from '../../store/analysisStore';
import { useProcessing } from '../../hooks/useProcessing';
import type { ProcessingConfig } from '../../types/analysis';
import styles from './FileLoader.module.css';

type CameraType = ProcessingConfig['cameraType'];

const CAMERA_TYPES: { value: CameraType; label: string }[] = [
  { value: 'photron', label: 'Photron (.mraw)' },
  { value: 'photron_avi', label: 'Photron (.avi)' },
  { value: 'dalsa', label: 'DALSA' },
  { value: 'tiff_sequence', label: 'TIFF Sequence' },
];

function detectCameraType(filename: string): CameraType {
  const lower = filename.toLowerCase();
  if (lower.endsWith('.mraw') || lower.endsWith('.cihx')) return 'photron';
  if (lower.endsWith('.avi')) return 'photron_avi';
  if (lower.endsWith('.tiff') || lower.endsWith('.tif')) return 'tiff_sequence';
  return 'dalsa';
}

function getCameraBadgeClass(type: CameraType): string {
  if (type === 'photron' || type === 'photron_avi') return styles.cameraBadgePhotron;
  if (type === 'dalsa') return styles.cameraBadgeDalsa;
  return styles.cameraBadgeTiff;
}

function getCameraBadgeLabel(type: CameraType): string {
  switch (type) {
    case 'photron': return 'PHOTRON';
    case 'photron_avi': return 'PHOTRON AVI';
    case 'dalsa': return 'DALSA';
    case 'tiff_sequence': return 'TIFF SEQ';
  }
}

export function FileLoader() {
  const { probeFile } = useProcessing();
  const metadata = useAnalysisStore((s) => s.metadata);
  const config = useAnalysisStore((s) => s.config);
  const selectedFrame = useAnalysisStore((s) => s.selectedFrame);
  const setMetadata = useAnalysisStore((s) => s.setMetadata);
  const updateConfig = useAnalysisStore((s) => s.updateConfig);
  const setSelectedFrame = useAnalysisStore((s) => s.setSelectedFrame);
  const jobId = useAnalysisStore((s) => s.jobId);

  const [isDragging, setIsDragging] = useState(false);
  const [isProbing, setIsProbing] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleProbe = useCallback(async () => {
    if (!config.inputPath) return;
    setIsProbing(true);
    setProbeError(null);
    try {
      const meta = await probeFile(config.inputPath, config.cameraType);
      setMetadata(meta);
      setSelectedFrame(0);
    } catch (err) {
      setProbeError(err instanceof Error ? err.message : 'Probe failed');
      setMetadata(null);
    } finally {
      setIsProbing(false);
    }
  }, [config.inputPath, config.cameraType, probeFile, setMetadata, setSelectedFrame]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) {
        const path = (file as File & { path?: string }).path ?? file.name;
        const detected = detectCameraType(file.name);
        updateConfig({ inputPath: path, cameraType: detected });
      }
    },
    [updateConfig]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        const path = (file as File & { path?: string }).path ?? file.name;
        const detected = detectCameraType(file.name);
        updateConfig({ inputPath: path, cameraType: detected });
      }
    },
    [updateConfig]
  );

  const dropZoneClass = [
    styles.dropZone,
    isDragging ? styles.dropZoneActive : '',
    config.inputPath && !isDragging ? styles.dropZoneHover : '',
  ]
    .filter(Boolean)
    .join(' ');

  const frameCount = metadata?.frameCount ?? 1;

  return (
    <div className={styles.container}>
      {/* ── Drop Zone ── */}
      <div
        className={dropZoneClass}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current?.click()}
        aria-label="Drop file or click to browse"
      >
        <input
          ref={fileInputRef}
          type="file"
          className={styles.hiddenInput}
          accept=".mraw,.avi,.cihx,.tiff,.tif"
          onChange={handleFileChange}
        />
        <svg className={styles.dropIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
        </svg>
        <div className={styles.dropLabel}>
          <span className={styles.dropLabelHighlight}>Drop file</span> or click to browse
        </div>
        <div className={styles.dropFormats}>.mraw · .avi · .cihx · .tiff</div>
      </div>

      {/* ── Path Input ── */}
      <div className={styles.pathRow}>
        <input
          type="text"
          className={styles.pathInput}
          placeholder="/path/to/recording.mraw"
          value={config.inputPath}
          onChange={(e) => updateConfig({ inputPath: e.target.value })}
          spellCheck={false}
        />
        <button
          className={styles.browseBtn}
          onClick={() => fileInputRef.current?.click()}
          type="button"
        >
          Browse
        </button>
      </div>

      {/* ── Camera Type Row ── */}
      <div className={styles.cameraRow}>
        <span className={styles.cameraLabel}>Camera type</span>
        <select
          className={styles.cameraSelect}
          value={config.cameraType}
          onChange={(e) =>
            updateConfig({ cameraType: e.target.value as CameraType })
          }
        >
          {CAMERA_TYPES.map((ct) => (
            <option key={ct.value} value={ct.value}>
              {ct.label}
            </option>
          ))}
        </select>
        <span className={`${styles.cameraBadge} ${getCameraBadgeClass(config.cameraType)}`}>
          <span className={styles.dotLive} />
          {getCameraBadgeLabel(config.cameraType)}
        </span>
      </div>

      {/* ── Probe Button ── */}
      <button
        className={styles.probeBtn}
        onClick={handleProbe}
        disabled={!config.inputPath || isProbing}
        type="button"
      >
        {isProbing ? (
          <>
            <svg
              width="14" height="14" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth={2}
              style={{ animation: 'spin 1s linear infinite' }}
            >
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
            Probing…
          </>
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
            Probe File
          </>
        )}
      </button>

      {/* ── Error ── */}
      {probeError && (
        <div className={styles.errorMsg}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          {probeError}
        </div>
      )}

      {/* ── Metadata Panel ── */}
      {metadata && (
        <div className={styles.metaPanel}>
          <div className={styles.metaPanelHeader}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <rect x="2" y="3" width="20" height="14" rx="2" />
              <line x1="8" y1="21" x2="16" y2="21" />
              <line x1="12" y1="17" x2="12" y2="21" />
            </svg>
            Frame Metadata
          </div>
          <div className={styles.metaGrid}>
            <div className={styles.metaItem}>
              <span className={styles.metaItemLabel}>Camera</span>
              <span className={styles.metaItemValue}>{metadata.cameraType}</span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaItemLabel}>FPS</span>
              <span className={`${styles.metaItemValue} ${styles.metaItemValueGreen}`}>
                {metadata.frameRate.toLocaleString()}
              </span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaItemLabel}>Resolution</span>
              <span className={styles.metaItemValue}>
                {metadata.width} × {metadata.height}
              </span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaItemLabel}>Bit Depth</span>
              <span className={styles.metaItemValue}>{metadata.bitDepth}-bit</span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaItemLabel}>Frames</span>
              <span className={styles.metaItemValue}>
                {metadata.frameCount.toLocaleString()}
              </span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaItemLabel}>Trigger</span>
              <span className={styles.metaItemValue}>
                {metadata.triggerFrame !== null
                  ? `#${metadata.triggerFrame}`
                  : '—'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── Timeline Scrubber ── */}
      {metadata && (
        <div className={styles.timelineSection}>
          <div className={styles.timelineHeader}>
            <span className={styles.timelineLabel}>Reference Frame</span>
            <span className={styles.timelineValue}>
              {selectedFrame.toString().padStart(4, '0')} / {(frameCount - 1).toString().padStart(4, '0')}
            </span>
          </div>
          <input
            type="range"
            className={styles.timelineScrubber}
            min={0}
            max={Math.max(0, frameCount - 1)}
            value={selectedFrame}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              setSelectedFrame(v);
              updateConfig({ referenceFrame: v });
            }}
          />
        </div>
      )}

      {/* ── Thumbnail ── */}
      {metadata && (
        <div className={styles.thumbnailArea}>
          {jobId ? (
            <img
              className={styles.thumbnailImg}
              src={`/api/results/${jobId}/thumbnail?frame=${selectedFrame}`}
              alt={`Frame ${selectedFrame}`}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = 'none';
              }}
            />
          ) : (
            <div className={styles.thumbnailPlaceholder}>
              <svg
                className={styles.thumbnailPlaceholderIcon}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <rect x="2" y="2" width="20" height="20" rx="3" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21,15 16,10 5,21" />
              </svg>
              <span className={styles.thumbnailPlaceholderText}>
                No preview — run probe
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
