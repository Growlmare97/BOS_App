export interface FrameMetadata {
  cameraType: string;
  frameCount: number;
  frameRate: number;
  width: number;
  height: number;
  bitDepth: number;
  triggerFrame: number | null;
}

export interface ProcessingConfig {
  cameraType: 'photron' | 'photron_avi' | 'dalsa' | 'tiff_sequence';
  inputPath: string;
  metadataPath: string | null;
  referenceFrame: number;
  measurementFrames: string;
  method: 'cross_correlation' | 'lucas_kanade' | 'farneback';
  windowSize: number;
  overlap: number;
  sigma: number;
  ensembleAveraging: boolean;
  multiPass: boolean;
  multiPassCount: number;
  abelEnabled: boolean;
  abelMethod: 'three_point' | 'hansenlaw' | 'basex';
  abelAxisMode: 'auto' | 'manual';
  abelAxisPosition: number;
  concentrationEnabled: boolean;
  gasType: string;
  ambientGas: string;
  temperatureK: number;
  pressurePa: number;
  velocityEnabled: boolean;
  velocityMethod: 'frame_to_frame' | 'kymography';
  kymoAxis: 'horizontal' | 'vertical';
  kymoLinePos: number;
  pixelScaleMm: number;
  pixelPitchUm: number;
  zdMm: number;
  zaMm: number;
  focalLengthMm: number;
  outputPath: string;
  outputFormats: Array<'npy' | 'hdf5' | 'csv' | 'vtk'>;
}

export interface ResultSummary {
  frameIdx: number;
  maxDisplacement: number;
  meanDisplacement: number;
  bgNoise: number;
  snrDb: number;
  hasConcentration: boolean;
  hasVelocity: boolean;
}

export type PipelineStage =
  | 'idle'
  | 'loading'
  | 'preprocessing'
  | 'displacement'
  | 'abel'
  | 'concentration'
  | 'velocity'
  | 'exporting'
  | 'done'
  | 'error';

export interface ProgressMessage {
  jobId: string;
  stage: PipelineStage;
  progress: number;
  message: string;
  timestamp: string;
}

export type ResultField =
  | 'magnitude'
  | 'dx'
  | 'dy'
  | 'concentration'
  | 'u'
  | 'v'
  | 'vorticity';

export type ColormapName =
  | 'viridis'
  | 'plasma'
  | 'turbo'
  | 'RdBu'
  | 'seismic'
  | 'inferno';

export interface FieldData {
  data: number[][];
  shape: [number, number];
  vmin: number;
  vmax: number;
  unit: string;
}

export interface KymographResult {
  kymograph: number[][];
  velocityProfile: number[];
  convectiveVelocity: number;
}

export interface KymographRequest {
  jobId: string;
  frameIdxStart: number;
  frameIdxEnd: number;
  axis: 'horizontal' | 'vertical';
  linePos: number;
  field: ResultField;
}
