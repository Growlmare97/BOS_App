import type {
  FrameMetadata,
  ProcessingConfig,
  ResultField,
  FieldData,
  KymographResult,
  KymographRequest,
} from '../types/analysis';

async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Use default error message
    }
    throw new Error(detail);
  }

  return response.json() as Promise<T>;
}

export function useProcessing() {
  const probeFile = async (
    path: string,
    cameraType: string
  ): Promise<FrameMetadata> => {
    return apiFetch<FrameMetadata>('/api/probe', {
      method: 'POST',
      body: JSON.stringify({ path, camera_type: cameraType }),
    });
  };

  const startProcessing = async (
    config: ProcessingConfig
  ): Promise<string> => {
    const payload = {
      camera_type: config.cameraType,
      input_path: config.inputPath,
      metadata_path: config.metadataPath,
      reference_frame: config.referenceFrame,
      measurement_frames: config.measurementFrames,
      method: config.method,
      window_size: config.windowSize,
      overlap: config.overlap,
      sigma: config.sigma,
      ensemble_averaging: config.ensembleAveraging,
      multi_pass: config.multiPass,
      multi_pass_count: config.multiPassCount,
      abel_enabled: config.abelEnabled,
      abel_method: config.abelMethod,
      abel_axis_mode: config.abelAxisMode,
      abel_axis_position: config.abelAxisPosition,
      concentration_enabled: config.concentrationEnabled,
      gas_type: config.gasType,
      ambient_gas: config.ambientGas,
      temperature_k: config.temperatureK,
      pressure_pa: config.pressurePa,
      velocity_enabled: config.velocityEnabled,
      velocity_method: config.velocityMethod,
      kymo_axis: config.kymoAxis,
      kymo_line_pos: config.kymoLinePos,
      pixel_scale_mm: config.pixelScaleMm,
      pixel_pitch_um: config.pixelPitchUm,
      zd_mm: config.zdMm,
      za_mm: config.zaMm,
      focal_length_mm: config.focalLengthMm,
      output_path: config.outputPath,
      output_formats: config.outputFormats,
    };

    const result = await apiFetch<{ job_id: string }>('/api/process', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    return result.job_id;
  };

  const fetchFieldData = async (
    jobId: string,
    frameIdx: number,
    field: ResultField
  ): Promise<FieldData> => {
    return apiFetch<FieldData>(
      `/api/results/${jobId}/field?frame=${frameIdx}&field=${field}`
    );
  };

  const fetchKymograph = async (
    request: KymographRequest
  ): Promise<KymographResult> => {
    return apiFetch<KymographResult>('/api/kymograph', {
      method: 'POST',
      body: JSON.stringify({
        job_id: request.jobId,
        frame_idx_start: request.frameIdxStart,
        frame_idx_end: request.frameIdxEnd,
        axis: request.axis,
        line_pos: request.linePos,
        field: request.field,
      }),
    });
  };

  return { probeFile, startProcessing, fetchFieldData, fetchKymograph };
}
