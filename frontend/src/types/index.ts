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
  status: VideoStatus;
  thumbnailUrl?: string;
  youtubeUrl?: string;
  progress?: number;
  stage?: string;
  logs?: string[];
  jobId?: string;
}
