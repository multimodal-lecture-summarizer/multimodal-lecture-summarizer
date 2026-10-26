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

export interface KeyframeBBox {
  track_id?: number | null;
  trackId?: number | null;
  label: string;
  confidence: number;
  bbox_norm?: [number, number, number, number];
  bboxNorm?: [number, number, number, number]; // [x1, y1, x2, y2] trong khoảng 0..1
}

export interface ObjectTrackSummary {
  track_id?: number;
  trackId?: number;
  label: string;
  avg_confidence?: number;
  avgConfidence?: number;
  first_seen_sec?: number;
  firstSeenSec?: number;
  last_seen_sec?: number;
  lastSeenSec?: number;
  frames_appeared?: number;
  framesAppeared?: number;
  sample_bbox_norm?: [number, number, number, number] | null;
  sampleBboxNorm?: [number, number, number, number] | null;
  in_keyframe?: boolean;
  inKeyframe?: boolean;
  best_timestamp_sec?: number;
  bestTimestampSec?: number;
}

export interface SceneDetectedObjects {
  keyframe_bboxes?: KeyframeBBox[];
  keyframeBboxes?: KeyframeBBox[];
  keyframe_detections?: KeyframeBBox[];
  keyframeDetections?: KeyframeBBox[];
  tracks_summary?: ObjectTrackSummary[];
  tracksSummary?: ObjectTrackSummary[];
}

export interface VideoObjectsSummary {
  total_unique_tracks?: number;
  totalUniqueTracks?: number;
  label_distribution?: Record<string, number>;
  labelDistribution?: Record<string, number>;
  most_frequent_objects?: string[];
  mostFrequentObjects?: string[];
  sample_fps_used?: number | null;
  sampleFpsUsed?: number | null;
}

export interface VideoScene {
  sceneId?: string;
  scene_id?: string;
  videoId?: string;
  video_id?: string;
  sceneIndex: number;
  scene_index?: number;
  startTime?: string;
  start_time?: string;
  endTime?: string;
  end_time?: string;
  startSeconds: number;
  start_seconds?: number;
  endSeconds: number;
  end_seconds?: number;
  keyframeUrl?: string;
  keyframe_url?: string;
  caption?: string;
  script?: string;
  ocrText?: string;
  ocr_text?: string;
  detectedObjectsJson?: SceneDetectedObjects | null;
  detected_objects_json?: SceneDetectedObjects | null;
  objectCountsJson?: Record<string, number> | null;
  object_counts_json?: Record<string, number> | null;
}

export interface ObjectDetectionBenchmarkItem {
  model: string;
  params: string;
  map50: number;
  map50_95?: number;
  map5095?: number;
  precision: number;
  recall: number;
  latencyMs?: number;
  latency_ms?: number;
  fps: number;
  isProduction?: boolean;
  is_production?: boolean;
  notes: string;
}

export interface BenchmarksData {
  asrBenchmarks: ASRBenchmarkItem[];
  summarizationBenchmarks: SummarizationBenchmarkItem[];
  visionBenchmarks: VisionBenchmarkItem[];
  objectDetectionBenchmarks?: ObjectDetectionBenchmarkItem[];
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

export interface ODEvalMetrics {
  odPrecision?: number | null;
  odRecall?: number | null;
  odMap50?: number | null;
  odMap50_95?: number | null;
  odMap5095?: number | null;
}

export interface PerClassMetricsItem {
  precision: number;
  recall: number;
  ap50: number;
  ap50_95?: number;
  ap5095?: number;
  gt_count?: number;
  gtCount?: number;
  pred_count?: number;
  predCount?: number;
}

export interface VideoEvaluationResult {
  evalId: string;
  videoId: string;
  videoTitle: string;
  hasTranscript: boolean;
  hasSummary: boolean;
  hasObjectDetection?: boolean;
  asrMetrics?: ASREvalMetrics | null;
  summaryMetrics?: SummaryEvalMetrics | null;
  odMetrics?: ODEvalMetrics | null;
  odPrecision?: number | null;
  odRecall?: number | null;
  odMap50?: number | null;
  odMap50_95?: number | null;
  odMap5095?: number | null;
  odDetails?: {
    od_precision?: number;
    od_recall?: number;
    od_map50?: number;
    od_map50_95?: number;
    per_class_metrics?: Record<string, PerClassMetricsItem>;
    perClassMetrics?: Record<string, PerClassMetricsItem>;
    tracking_metrics?: {
      total_unique_tracks?: number;
      totalUniqueTracks?: number;
      avg_track_duration_sec?: number;
      avgTrackDurationSec?: number;
      avg_confidence?: number;
      avgConfidence?: number;
    };
    trackingMetrics?: {
      total_unique_tracks?: number;
      totalUniqueTracks?: number;
      avg_track_duration_sec?: number;
      avgTrackDurationSec?: number;
      avg_confidence?: number;
      avgConfidence?: number;
    };
    evaluation_mode?: string;
    evaluationMode?: string;
  } | null;
  referenceTranscriptPreview?: string | null;
  referenceSummaryPreview?: string | null;
  evaluatedAt: string;
}

export interface UserDTO {
  userId: string;
  email: string;
  role: string;
  createdAt?: string;
  lastLogin?: string;
  isActive: boolean;
  outputLanguage?: string;
}

export type SummaryLength = 'short' | 'medium' | 'detailed';

export interface SummaryVariantItem {
  summary_text?: string;
  summaryText?: string;
  key_takeaways_json?: any[];
  keyTakeawaysJson?: any[];
  word_count?: number;
  wordCount?: number;
  reading_time_sec?: number;
  readingTimeSec?: number;
  model_used?: string;
  modelUsed?: string;
  generated_at?: string;
  generatedAt?: string;
}

export type SummaryVariantsMap = Partial<Record<SummaryLength, SummaryVariantItem>>;

export interface Summary {
  summaryId?: string;
  summary_id?: string;
  videoId?: string;
  video_id?: string;
  summaryText?: string;
  summary_text?: string;
  summaryLength?: SummaryLength | null;
  summary_length?: SummaryLength | null;
  summaryVariantsJson?: SummaryVariantsMap | null;
  summary_variants_json?: SummaryVariantsMap | null;
  wordCount?: number | null;
  word_count?: number | null;
  readingTimeSec?: number | null;
  reading_time_sec?: number | null;
  transcriptText?: string;
  transcript_text?: string;
  transcriptSegments?: TranscriptLine[];
  transcript_segments?: TranscriptLine[];
  modelUsed?: string;
  model_used?: string;
  processingTime?: number;
  processing_time?: number;
  chapters?: Chapter[];
  keyframes?: any[];
  keyTakeaways?: TakeawayItem[];
  key_takeaways?: TakeawayItem[];
  objectsSummaryJson?: any;
  objects_summary_json?: any;
}



