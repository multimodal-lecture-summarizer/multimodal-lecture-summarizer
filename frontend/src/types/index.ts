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

export interface JobStatus {
  id: string;
  email: string;
  file: string;
  duration: string;
  status: 'done' | 'processing' | 'failed';
  pipeline: string;
}

export interface HistoryItem {
  id: string;
  title: string;
  duration: string;
  date: string;
  status: 'done' | 'processing';
  thumbnailUrl?: string;
  youtubeUrl?: string;
}
