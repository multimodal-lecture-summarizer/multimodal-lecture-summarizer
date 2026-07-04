export enum UserRole {
  ADMIN = "admin",
  USER = "user"
}

export enum VideoStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  DONE = "done",
  FAILED = "failed"
}

export enum JobType {
  SUMMARIZE = "summarize",
  QA = "qa"
}

export enum JobStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed"
}

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
}
