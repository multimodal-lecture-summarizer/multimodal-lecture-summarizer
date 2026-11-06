import { CONFIG } from '../config';

/** mm:ss label for a second offset. */
export const formatSeconds = (secs: number): string => {
  if (isNaN(secs) || secs < 0) return '00:00';
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
};

/** Absolute keyframe URL; relative paths are served by the API host. */
export const resolveKeyframeUrl = (rawUrl?: string | null): string => {
  if (!rawUrl) return '';
  if (rawUrl.startsWith('http://') || rawUrl.startsWith('https://')) return rawUrl;
  return `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${rawUrl}`;
};
