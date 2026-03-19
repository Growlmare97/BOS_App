import { create } from 'zustand';
import type {
  FrameMetadata,
  ProcessingConfig,
  PipelineStage,
  ResultSummary,
  ResultField,
  ColormapName,
  FieldData,
  KymographResult,
} from '../types/analysis';

const defaultConfig: ProcessingConfig = {
  cameraType: 'photron',
  inputPath: '',
  metadataPath: null,
  referenceFrame: 0,
  measurementFrames: '1-end',
  method: 'cross_correlation',
  windowSize: 32,
  overlap: 0.5,
  sigma: 1.0,
  ensembleAveraging: false,
  multiPass: false,
  multiPassCount: 2,
  abelEnabled: false,
  abelMethod: 'three_point',
  abelAxisMode: 'auto',
  abelAxisPosition: 0,
  concentrationEnabled: false,
  gasType: 'CO2',
  ambientGas: 'air',
  temperatureK: 293.15,
  pressurePa: 101325,
  velocityEnabled: false,
  velocityMethod: 'frame_to_frame',
  kymoAxis: 'horizontal',
  kymoLinePos: 128,
  pixelScaleMm: 0.1,
  pixelPitchUm: 20,
  zdMm: 1000,
  zaMm: 500,
  focalLengthMm: 50,
  outputPath: '',
  outputFormats: ['npy'],
};

interface AnalysisState {
  metadata: FrameMetadata | null;
  config: ProcessingConfig;
  jobId: string | null;
  stage: PipelineStage;
  progress: number;
  progressMessage: string;
  results: ResultSummary[];
  selectedFrame: number;
  selectedField: ResultField;
  selectedColormap: ColormapName;
  showQuiver: boolean;
  fieldData: FieldData | null;
  kymographResult: KymographResult | null;
  error: string | null;

  setMetadata: (metadata: FrameMetadata | null) => void;
  updateConfig: (partial: Partial<ProcessingConfig>) => void;
  setJobId: (jobId: string | null) => void;
  setProgress: (stage: PipelineStage, progress: number, message: string) => void;
  setResults: (results: ResultSummary[]) => void;
  setSelectedFrame: (frame: number) => void;
  setSelectedField: (field: ResultField) => void;
  setSelectedColormap: (colormap: ColormapName) => void;
  toggleQuiver: () => void;
  setFieldData: (data: FieldData | null) => void;
  setKymographResult: (result: KymographResult | null) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  metadata: null,
  config: defaultConfig,
  jobId: null,
  stage: 'idle',
  progress: 0,
  progressMessage: '',
  results: [],
  selectedFrame: 0,
  selectedField: 'magnitude',
  selectedColormap: 'viridis',
  showQuiver: false,
  fieldData: null,
  kymographResult: null,
  error: null,

  setMetadata: (metadata) => set({ metadata }),

  updateConfig: (partial) =>
    set((state) => ({ config: { ...state.config, ...partial } })),

  setJobId: (jobId) => set({ jobId }),

  setProgress: (stage, progress, message) =>
    set({ stage, progress, progressMessage: message }),

  setResults: (results) => set({ results }),

  setSelectedFrame: (selectedFrame) => set({ selectedFrame }),

  setSelectedField: (selectedField) => set({ selectedField }),

  setSelectedColormap: (selectedColormap) => set({ selectedColormap }),

  toggleQuiver: () => set((state) => ({ showQuiver: !state.showQuiver })),

  setFieldData: (fieldData) => set({ fieldData }),

  setKymographResult: (kymographResult) => set({ kymographResult }),

  setError: (error) => set({ error }),

  reset: () =>
    set({
      metadata: null,
      config: defaultConfig,
      jobId: null,
      stage: 'idle',
      progress: 0,
      progressMessage: '',
      results: [],
      selectedFrame: 0,
      selectedField: 'magnitude',
      selectedColormap: 'viridis',
      showQuiver: false,
      fieldData: null,
      kymographResult: null,
      error: null,
    }),
}));
