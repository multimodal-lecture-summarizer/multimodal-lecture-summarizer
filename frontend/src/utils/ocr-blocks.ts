import type { OcrBlock, VideoScene } from '../types';

/** Block with snake/camel variants resolved. */
export interface NormBlock {
  lineIndex: number;
  text: string;
  confidence: number;
  bbox: [number, number, number, number];
  polygon: [number, number][];
}

export interface OcrScene {
  index: number;
  startSec: number;
  endSec: number;
  keyframeUrl: string;
  ocrText: string;
  blocks: NormBlock[];
}

export type ConfidenceBand = 'high' | 'mid' | 'low';

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

export const normalizeBlock = (b: OcrBlock, fallbackIdx: number): NormBlock => {
  const raw = b.bboxNorm ?? b.bbox_norm ?? [0, 0, 0, 0];
  const x1 = clamp01(Math.min(raw[0], raw[2]));
  const y1 = clamp01(Math.min(raw[1], raw[3]));
  const x2 = clamp01(Math.max(raw[0], raw[2]));
  const y2 = clamp01(Math.max(raw[1], raw[3]));
  const poly = b.polygonNorm ?? b.polygon_norm ?? [];
  return {
    lineIndex: b.lineIndex ?? b.line_index ?? fallbackIdx,
    text: b.text ?? '',
    confidence: b.confidence ?? 0,
    bbox: [x1, y1, x2, y2],
    polygon: poly.length === 4 ? poly : [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
  };
};

export const normalizeOcrScene = (s: VideoScene): OcrScene => {
  const blocks = (s.ocrBlocksJson ?? s.ocr_blocks_json ?? []).map(normalizeBlock);
  return {
    index: s.sceneIndex ?? s.scene_index ?? 1,
    startSec: s.startSeconds ?? s.start_seconds ?? 0,
    endSec: s.endSeconds ?? s.end_seconds ?? 0,
    keyframeUrl: s.keyframeUrl ?? s.keyframe_url ?? '',
    ocrText: s.ocrText ?? s.ocr_text ?? '',
    blocks,
  };
};

export const confidenceBand = (c: number): ConfidenceBand => (c >= 0.9 ? 'high' : c >= 0.75 ? 'mid' : 'low');

/** Split text into parts, flagging the ones matching the query (case-insensitive). */
export const highlightParts = (text: string, query: string): { text: string; match: boolean }[] => {
  const q = query.trim();
  if (!q) return [{ text, match: false }];
  const lower = text.toLowerCase();
  const ql = q.toLowerCase();
  const parts: { text: string; match: boolean }[] = [];
  let pos = 0;
  while (pos < text.length) {
    const hit = lower.indexOf(ql, pos);
    if (hit < 0) {
      parts.push({ text: text.slice(pos), match: false });
      break;
    }
    if (hit > pos) parts.push({ text: text.slice(pos, hit), match: false });
    parts.push({ text: text.slice(hit, hit + ql.length), match: true });
    pos = hit + ql.length;
  }
  return parts;
};

export const countWords = (text: string): number => text.split(/\s+/).filter(Boolean).length;

export const buildSceneText = (blocks: NormBlock[]): string => blocks.map((b) => b.text).join('\n');

export interface OcrStats {
  kept: number;
  hidden: number;
  chars: number;
  words: number;
  avg: number;
  min: number;
  max: number;
  /** buckets 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0 */
  histogram: number[];
  /** Share of the frame covered by text boxes (0..1, overlaps counted once per box, capped at 1). */
  coverage: number;
}

export const computeOcrStats = (all: NormBlock[], minConf: number): OcrStats => {
  const kept = all.filter((b) => b.confidence >= minConf);
  const histogram = [0, 0, 0, 0, 0];
  let sum = 0;
  let area = 0;
  let min = 1;
  let max = 0;
  let chars = 0;
  let words = 0;
  kept.forEach((b) => {
    sum += b.confidence;
    min = Math.min(min, b.confidence);
    max = Math.max(max, b.confidence);
    chars += b.text.length;
    words += countWords(b.text);
    area += (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    histogram[Math.min(4, Math.max(0, Math.floor((b.confidence - 0.5) * 10)))] += 1;
  });
  return {
    kept: kept.length,
    hidden: all.length - kept.length,
    chars,
    words,
    avg: kept.length ? sum / kept.length : 0,
    min: kept.length ? min : 0,
    max: kept.length ? max : 0,
    histogram,
    coverage: Math.min(1, area),
  };
};
