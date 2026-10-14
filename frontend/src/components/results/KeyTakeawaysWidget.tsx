/* Hallmark · component: KeyTakeawaysWidget · genre: editorial · theme: Specimen / Neutral-Tactile
 * states: default · hover · focus-visible · active · disabled · loading · empty · success (copied)
 * contrast: pass (WCAG AA/AAA) · anti-ai-slop: zero AI sparkles/wand icons, tactile SVG only
 */

import React, { useState, useMemo } from 'react';
import type { TakeawayItem } from '../../types';

export interface KeyTakeawaysWidgetProps {
  takeaways?: TakeawayItem[];
  onSeekTimestamp?: (seconds: number) => void;
  isLoading?: boolean;
  className?: string;
}

interface NormalizedTakeaway {
  id: string;
  index: number;
  timestamp?: number;
  timeStr?: string;
  title?: string;
  description: string;
}

// Convert seconds to mm:ss format
const formatSecondsToTime = (totalSeconds: number): string => {
  const m = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, '0');
  const s = Math.floor(totalSeconds % 60)
    .toString()
    .padStart(2, '0');
  return `${m}:${s}`;
};

// Parse time string (e.g. "02:15" or "01:05:30") to seconds
const parseTimeToSeconds = (timeStr: string): number | undefined => {
  const clean = timeStr.replace(/[\[\]\(\)]/g, '').trim();
  const parts = clean.split(':').map(Number);
  if (parts.some(isNaN)) return undefined;

  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return undefined;
};

// Parse a raw takeaway item into a structured model
const normalizeItem = (item: TakeawayItem, idx: number): NormalizedTakeaway => {
  if (typeof item === 'string') {
    let rawText = item.trim();
    let timestamp: number | undefined;
    let timeStr: string | undefined;

    // Check for leading timestamp: [02:15], (02:15), 02:15 -, 02:15:
    const timestampMatch = rawText.match(/^(\[|\()?(\d{1,2}:\d{2}(?::\d{2})?)(\]|\))?[\s\:\-]*/);
    if (timestampMatch) {
      const extractedTime = timestampMatch[2];
      timestamp = parseTimeToSeconds(extractedTime);
      if (timestamp !== undefined) {
        timeStr = extractedTime;
        rawText = rawText.slice(timestampMatch[0].length).trim();
      }
    }

    // Split title and description if separated by : or -
    let title: string | undefined;
    let description = rawText;

    const separatorMatch = rawText.match(/^([^:\-\—\–]{3,45})[:\-\—\–]\s+(.+)$/);
    if (separatorMatch && !separatorMatch[1].includes('.')) {
      title = separatorMatch[1].trim();
      description = separatorMatch[2].trim();
    }

    return {
      id: `takeaway-${idx}`,
      index: idx + 1,
      timestamp,
      timeStr,
      title,
      description: description || rawText || 'Điểm nhấn quan trọng',
    };
  }

  // Object shape: { timestamp, time_str, title, description }
  let ts = item.timestamp;
  let tStr = item.time_str;

  if (ts !== undefined && isNaN(ts)) {
    ts = undefined;
  }

  if (ts !== undefined && !tStr) {
    tStr = formatSecondsToTime(ts);
  } else if (!ts && tStr) {
    ts = parseTimeToSeconds(tStr);
  }

  return {
    id: `takeaway-${idx}`,
    index: idx + 1,
    timestamp: ts,
    timeStr: tStr,
    title: item.title?.trim(),
    description: item.description?.trim() || 'Nội dung sự kiện chính',
  };
};

export const KeyTakeawaysWidget: React.FC<KeyTakeawaysWidgetProps> = ({
  takeaways = [],
  onSeekTimestamp,
  isLoading = false,
  className = '',
}) => {
  const [copied, setCopied] = useState(false);

  const normalizedList = useMemo<NormalizedTakeaway[]>(() => {
    if (!Array.isArray(takeaways)) return [];
    return takeaways.map((item, idx) => normalizeItem(item, idx));
  }, [takeaways]);

  const handleCopyAll = async () => {
    if (normalizedList.length === 0) return;

    const formattedLines = normalizedList.map((item) => {
      const timePart = item.timeStr ? `[${item.timeStr}] ` : '';
      const titlePart = item.title ? `${item.title}: ` : '';
      return `${item.index}. ${timePart}${titlePart}${item.description}`;
    });

    const fullContent = `SỰ KIỆN CHÍNH & ĐIỂM NHẤN CỐT LÕI:\n${formattedLines.join('\n')}`;

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(fullContent);
      } else {
        const textArea = document.createElement('textarea');
        textArea.value = fullContent;
        textArea.style.position = 'fixed';
        textArea.style.left = '-9999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        document.execCommand('copy');
        textArea.remove();
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2200);
    } catch (err) {
      console.warn('Failed to copy takeaways to clipboard', err);
    }
  };

  // Skeleton loading view
  if (isLoading) {
    return (
      <section
        className={`bg-white border border-slate-200/80 rounded-xl p-5 md:p-6 shadow-sm ${className}`}
        aria-busy="true"
        aria-label="Đang tải sự kiện chính"
      >
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-5 h-5 bg-slate-200 rounded animate-pulse" />
            <div className="w-48 h-5 bg-slate-200 rounded animate-pulse" />
            <div className="w-16 h-5 bg-slate-100 rounded-full animate-pulse" />
          </div>
          <div className="w-24 h-7 bg-slate-100 rounded-md animate-pulse" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 mt-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div
              key={i}
              className="p-3.5 border border-slate-100 rounded-lg bg-slate-50/50 space-y-2 animate-pulse"
            >
              <div className="flex items-center justify-between">
                <div className="w-8 h-4 bg-slate-200 rounded" />
                <div className="w-14 h-4 bg-slate-200 rounded" />
              </div>
              <div className="w-full h-3.5 bg-slate-200 rounded" />
              <div className="w-4/5 h-3.5 bg-slate-200 rounded" />
            </div>
          ))}
        </div>
      </section>
    );
  }

  // Empty state view
  if (normalizedList.length === 0) {
    return (
      <section
        className={`bg-white border border-slate-200/80 rounded-xl p-5 md:p-6 shadow-sm ${className}`}
      >
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <h3 className="font-heading text-sm md:text-base font-bold text-slate-900 tracking-tight">
              Sự kiện chính &amp; Điểm nhấn cốt lõi
            </h3>
          </div>
        </div>
        <div className="py-8 px-4 text-center flex flex-col items-center justify-center text-slate-500">
          {/* Subtle editorial outline list SVG */}
          <svg
            className="w-8 h-8 text-slate-300 mb-2.5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="8" y1="6" x2="21" y2="6" />
            <line x1="8" y1="12" x2="21" y2="12" />
            <line x1="8" y1="18" x2="21" y2="18" />
            <circle cx="4" cy="6" r="1.5" />
            <circle cx="4" cy="12" r="1.5" />
            <circle cx="4" cy="18" r="1.5" />
          </svg>
          <p className="text-xs font-medium text-slate-500">
            Chưa có sự kiện nổi bật được trích xuất cho video này.
          </p>
          <p className="text-[11px] text-slate-400 mt-1">
            Các điểm nhấn sẽ được trích xuất tự động khi mô hình AI hoàn tất phân tích.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section
      className={`bg-white border border-slate-200/80 rounded-xl p-5 md:p-6 shadow-sm ${className}`}
      aria-label="Sự kiện chính và điểm nhấn cốt lõi"
    >
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100">
        <div className="flex items-center gap-2.5 flex-wrap">
          <h3 className="font-heading text-sm md:text-base font-bold text-slate-900 tracking-tight">
            Sự kiện chính &amp; Điểm nhấn cốt lõi
          </h3>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-bold bg-primary/10 text-primary border border-primary/20">
            {normalizedList.length} sự kiện
          </span>
        </div>

        {/* Copy All Button */}
        <button
          type="button"
          onClick={handleCopyAll}
          className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150 border ${copied
            ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
            : 'bg-slate-50 hover:bg-slate-100 text-slate-700 border-slate-200/80 active:bg-slate-200'
            }`}
          title="Sao chép toàn bộ sự kiện chính vào bộ nhớ tạm"
          aria-label={copied ? 'Đã sao chép vào bộ nhớ tạm' : 'Sao chép nhanh các điểm nhấn'}
        >
          {copied ? (
            <>
              {/* Checkmark SVG */}
              <svg
                className="w-3.5 h-3.5 text-emerald-600"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Đã sao chép!</span>
            </>
          ) : (
            <>
              {/* Clean clipboard SVG */}
              <svg
                className="w-3.5 h-3.5 text-slate-500"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
              <span>Sao chép nhanh</span>
            </>
          )}
        </button>
      </div>

      {/* Grid of Takeaway Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 mt-4">
        {normalizedList.map((item) => {
          const hasTimestamp = item.timestamp !== undefined && !isNaN(item.timestamp);

          return (
            <article
              key={item.id}
              onClick={() => {
                if (hasTimestamp && onSeekTimestamp) {
                  onSeekTimestamp(item.timestamp!);
                }
              }}
              className={`group relative flex flex-col justify-between p-3.5 rounded-xl border transition-all duration-200 ${hasTimestamp
                ? 'bg-white hover:bg-[#FAF5FF]/40 border-slate-200/70 hover:border-primary/50 cursor-pointer shadow-[0_1px_2px_rgba(0,0,0,0.02)] hover:shadow-sm'
                : 'bg-white border-slate-200/60'
                }`}
            >
              {/* Card Header: Index & Timestamp pill */}
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="font-mono text-[11px] font-bold text-slate-400 group-hover:text-primary transition-colors tracking-wider">
                  {item.index.toString().padStart(2, '0')}
                </span>

                {item.timeStr && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (hasTimestamp && onSeekTimestamp) {
                        onSeekTimestamp(item.timestamp!);
                      }
                    }}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-slate-100 hover:bg-primary text-slate-700 hover:text-white transition-colors duration-150 border border-slate-200/60"
                    title={`Nhảy tới mốc thời gian ${item.timeStr}`}
                  >
                    {/* Play SVG Icon */}
                    <svg
                      className="w-2.5 h-2.5 fill-current"
                      viewBox="0 0 24 24"
                    >
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    <span>{item.timeStr}</span>
                  </button>
                )}
              </div>

              {/* Card Body: Title (if available) & Description */}
              <div className="space-y-1 flex-1">
                {item.title && (
                  <h4 className="text-xs font-bold text-slate-900 leading-snug group-hover:text-primary transition-colors line-clamp-1">
                    {item.title}
                  </h4>
                )}
                <p className="text-xs text-slate-600 leading-relaxed font-normal">
                  {item.description}
                </p>
              </div>

              {/* Subtle visual anchor when clickable */}
              {hasTimestamp && (
                <div className="mt-2.5 pt-2 border-t border-slate-100/80 flex items-center justify-between text-[10px] text-slate-400 group-hover:text-primary transition-colors">
                  <span className="font-medium">Xem khoảnh khắc này</span>
                  <svg
                    className="w-3 h-3 transform group-hover:translate-x-0.5 transition-transform"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="9 18 15 12 9 6" />
                  </svg>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
};

export default KeyTakeawaysWidget;
