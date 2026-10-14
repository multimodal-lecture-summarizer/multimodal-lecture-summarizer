export const UserRole = {
  ADMIN: "admin",
  USER: "user"
} as const;
export type UserRole = typeof UserRole[keyof typeof UserRole];

export const VideoStatus = {
  PENDING: "pending",
  PROCESSING: "processing",
  DONE: "done",
  FAILED: "failed"
} as const;
export type VideoStatus = typeof VideoStatus[keyof typeof VideoStatus];

export const JobType = {
  SUMMARIZE: "summarize",
  QA: "qa"
} as const;
export type JobType = typeof JobType[keyof typeof JobType];

export const JobStatus = {
  PENDING: "pending",
  RUNNING: "running",
  COMPLETED: "completed",
  FAILED: "failed"
} as const;
export type JobStatus = typeof JobStatus[keyof typeof JobStatus];

export interface WordToken {
  word: string;
  start: number;
  end: number;
  time: string;
}

export interface TranscriptLine {
  speaker?: string;
  start: number;
  end: number;
  text: string;
  words?: WordToken[];
}

export interface Chapter {
  start: string;
  end: string;
  title: string;
  summary: string;
}

export interface JobStatusData {
  id: string;
  email: string;
  file: string;
  duration: string;
  status: JobStatus;
  pipeline: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  duration: string;
  date: string;
  rawDate: number;
  status: VideoStatus;
  thumbnailUrl?: string;
  youtubeUrl?: string;
  progress?: number;
  stage?: string;
  logs?: string[];
  jobId?: string;
}

export type TakeawayItem =
  | string
  | {
      timestamp?: number;
      time_str?: string;
      title?: string;
      description: string;
    };

export interface PerformanceSummary {
  avgRtf: number;
  avgProcessingTimeSec: number;
  totalProcessedHours: number;
  totalJobsEvaluated: number;
}

export interface ScatterDataPoint {
  videoId: string;
  title: string;
  durationSec: number;
  processingSec: number;
  rtf: number;
  resolution: string;
}

export interface DurationBucket {
  bucket: 'short' | 'medium' | 'long' | string;
  label: string;
  count: number;
  avgDurationSec: number;
  avgProcessingSec: number;
  avgRtf: number;
  successRate: number;
}

export interface PerformanceAnalyticsData {
  summaryMetrics: PerformanceSummary;
  scatterData: ScatterDataPoint[];
  durationBuckets: DurationBucket[];
}

export interface ASRBenchmarkItem {
  model: string;
  params: string;
  wer: number;
  cer: number;
  rtf: number;
  relativeSpeed: string;
  isProduction: boolean;
  notes: string;
}

export interface SummarizationBenchmarkItem {
  methodId: string;
  name: string;
  rouge1: number;
  rouge2: number;
  rougeL: number;
  bertscoreF1: number;
  avgLatencySec: number;
  isProposed: boolean;
  description: string;
}

export interface VisionBenchmarkItem {
  model: string;
  clipScore: number;
  latencyPerFrameSec: number;
  genericRate: number;
  isProduction: boolean;
}

export interface BenchmarksData {
  asrBenchmarks: ASRBenchmarkItem[];
  summarizationBenchmarks: SummarizationBenchmarkItem[];
  visionBenchmarks: VisionBenchmarkItem[];
  summaryConclusions: string[];
}

export interface ASREvalMetrics {
  wer: number;
  cer: number;
  werPct: number;
  cerPct: number;
  wordAccuracy: number;
}

export interface SummaryEvalMetrics {
  rouge1: number;
  rouge2: number;
  rougeL: number;
  bertscoreF1: number;
}

export interface VideoEvaluationResult {
  evalId: string;
  videoId: string;
  videoTitle: string;
  hasTranscript: boolean;
  hasSummary: boolean;
  asrMetrics?: ASREvalMetrics | null;
  summaryMetrics?: SummaryEvalMetrics | null;
  referenceTranscriptPreview?: string | null;
  referenceSummaryPreview?: string | null;
  evaluatedAt: string;
}

