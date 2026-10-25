import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Play, RotateCw, Copy, Check, FileText, Clock, Layers } from 'lucide-react';
import type {
  TranscriptLine,
  Chapter,
  TakeawayItem,
  SummaryLength,
  SummaryVariantsMap,
  SummaryVariantItem
} from '../types';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';
import { KeyTakeawaysWidget } from '../components/results/KeyTakeawaysWidget';
import { SceneObjectExplorer } from '../components/SceneObjectExplorer';
import { useDevMode } from '../context/DevModeContext';

interface ChapterDTO {
  title: string;
  startTime: number;
  endTime: number;
  summary: string;
}

interface KeyframeDTO {
  timestamp: number;
  imageUrl: string;
  description: string;
  transcript?: string;
  importanceScore: number;
}

interface SummaryDTO {
  summaryId: string;
  videoId: string;
  summaryText: string;
  summary_text?: string;
  transcriptText: string;
  transcript_text?: string;
  transcriptSegments?: TranscriptLine[];
  transcript_segments?: TranscriptLine[];
  modelUsed: string;
  model_used?: string;
  processingTime: number;
  processing_time?: number;
  chapters: ChapterDTO[];
  keyframes: KeyframeDTO[];
  keyTakeaways?: TakeawayItem[];
  key_takeaways?: TakeawayItem[];
  summaryLength?: SummaryLength | null;
  summary_length?: SummaryLength | null;
  summaryVariantsJson?: SummaryVariantsMap | null;
  summary_variants_json?: SummaryVariantsMap | null;
  wordCount?: number | null;
  word_count?: number | null;
  readingTimeSec?: number | null;
  reading_time_sec?: number | null;
}

interface VideoDTO {
  videoId: string;
  originalUrl?: string;
  filePath?: string;
  duration?: number;
  status: string;
  title?: string;
}

const isConversationalQuote = (s: string) => {
  const lower = s.toLowerCase();
  if (/^(?:it\s+(?:inspires|is|seems|looks|feels|helps|keeps)|i\s+(?:think|like|want|believe|guess|love|feel|have|am|get|know)|we\s+(?:call|have|are|can|want|started|progressed|train)|you\s+(?:can|know|see|want)|they\s+(?:have|are|can)|so\s+|and\s+|well\s+|just\s+|now\s+|basically\s+|actually\s+)/i.test(lower)) {
    return true;
  }
  if (/\b(?:i|me|my|mine|we|us|our|ours|you|your|yours)\b/i.test(lower) && s.split(/\s+/).length <= 9) {
    return true;
  }
  if (/\b(?:as well|kind of|sort of|like this|you know|teeth fit)\b/i.test(lower)) {
    return true;
  }
  return false;
};

const cleanDisplayTitle = (rawTitle?: string, fallbackChapterTitle?: string): string => {
  if (!rawTitle) return '';
  let t = rawTitle.trim().replace(/^["'*`#]+|["'*`#]+$/g, '');

  // Handle speaker banner / pipe formats (e.g., "STEVEN ERDMANCZYK | CHARLOTTE PARKOUR" -> "Charlotte Parkour")
  if (t.includes('|')) {
    const parts = t.split('|').map(p => p.trim()).filter(Boolean);
    if (parts.length > 1) {
      const isSpeakerCredit = (s: string) => {
        if (/^(?:bởi|tác giả|diễn giả|giảng viên|người hướng dẫn|speaker|presenter|by|dr\.|prof\.)\b/i.test(s)) return true;
        // Upper-case name like "STEVEN ERDMANCZYK"
        if (/^[A-Z\s]{4,}$/.test(s) && s.split(/\s+/).length <= 4) return true;
        return false;
      };
      const topicParts = parts.filter(p => !isSpeakerCredit(p));
      t = topicParts.length > 0 ? topicParts.join(' - ') : parts[parts.length - 1];
    }
  }

  // If title is a casual spoken sentence from transcript (e.g. "It inspires me as well"), fall back to Chapter 1 topic
  if (isConversationalQuote(t) && fallbackChapterTitle) {
    let chClean = fallbackChapterTitle.replace(/^(?:chương\s*\d+|phần\s*\d+|chapter\s*\d+)[\:\-—\s]*/i, '').trim();
    chClean = chClean.replace(/^(?:giới\s*thiệu\s*(?:về|dự\s*án)?|tìm\s*hiểu\s*(?:về)?|tổng\s*quan\s*(?:về)?|hướng\s*dẫn\s*(?:về)?|bài\s*học\s*(?:về)?|những\s+lợi\s+ích\s+của\s+(?:việc)?|introduction\s*(?:to)?|overview\s*(?:of)?)\s*/i, '').trim();
    if (chClean.length >= 4) {
      return chClean.charAt(0).toUpperCase() + chClean.slice(1);
    }
  }

  // Convert shouting all-caps titles (e.g. "CHARLOTTE PARKOUR") to natural Title Case
  if (t === t.toUpperCase() && /[A-Z]/.test(t) && t.length > 3) {
    t = t.toLowerCase().replace(/(?:^|\s|\b)[a-z]/g, c => c.toUpperCase());
  }

  const narrativePatterns = [
    /^(?:video\s+(?:này\s+)?(?:bắt\s*đầu|nói\s*về|giới\s*thiệu\s*về|hướng\s*dẫn\s*về|trình\s*bày\s*về|chia\s*sẻ\s*về)\s*(?:bằng\s*cách\s*)?)/i,
    /^(?:video\s+(?:này\s+)?(?:bắt\s*đầu\s*với|mở\s*đầu\s*bằng|mở\s*đầu\s*với|tập\s*trung\s*vào)\s*)/i,
    /^(?:video\s+này\s+là\s+)/i,
    /^(?:đây\s+là\s+video\s+(?:về|nói\s*về|hướng\s*dẫn\s*về|giới\s*thiệu\s*về)\s*)/i,
    /^(?:the\s+video\s+(?:begins\s+with|starts\s+with|is\s+about|introduces|covers|demonstrates|shows)\s*)/i,
    /^(?:this\s+video\s+(?:begins\s+with|starts\s+with|is\s+about|introduces|covers|demonstrates|shows)\s*)/i,
    /^(?:a\s+video\s+(?:about|introducing|showing|demonstrating)\s*)/i,
  ];
  for (const pat of narrativePatterns) {
    t = t.replace(pat, '').trim();
  }
  t = t.replace(/[\.,;:…]+$/, '').trim();
  if (!t || t.length < 3) return rawTitle;
  return t.charAt(0).toUpperCase() + t.slice(1);
};



// Formatter seconds -> mm:ss
const formatTime = (secs: number) => {
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
};

// Parse time string mm:ss -> seconds
const parseTimeText = (timeStr: string) => {
  const parts = timeStr.trim().split(':').map(Number);
  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return 0;
};

// Format reading duration in humanized Vietnamese
const formatReadingTime = (totalSeconds: number): string => {
  if (totalSeconds < 60) {
    return `${Math.max(10, totalSeconds)} giây`;
  }
  const mins = Math.floor(totalSeconds / 60);
  const remSecs = totalSeconds % 60;
  if (remSecs === 0) {
    return `${mins} phút`;
  }
  return `${mins} phút ${remSecs} giây`;
};

const LENGTH_OPTIONS: { id: SummaryLength; label: string; badge: string; description: string }[] = [
  {
    id: 'short',
    label: 'Ngắn gọn',
    badge: 'Tóm tắt ngắn gọn',
    description: 'Nắm nhanh thông điệp & hành động cốt lõi'
  },
  {
    id: 'medium',
    label: 'Tiêu chuẩn',
    badge: 'Tổng quan tiêu chuẩn',
    description: 'Nội dung cân bằng, đầy đủ'
  },
  {
    id: 'detailed',
    label: 'Chi tiết',
    badge: 'Phân tích chi tiết',
    description: 'Nghiên cứu & phân tích chuyên sâu'
  }
];

// Ergonomic markdown formatter with clickable timestamp badges and bold styling
const renderFormattedSummary = (text: string, onSeek: (seconds: number) => void) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];

  const renderInlineText = (inlineText: string, keyPrefix: string) => {
    const boldParts = inlineText.split(/(\*\*[^*]+\*\*)/g);

    return boldParts.map((part, pIdx) => {
      if (!part) return null;
      const isBold = part.startsWith('**') && part.endsWith('**') && part.length > 4;
      const content = isBold ? part.slice(2, -2) : part;

      const tsParts = content.split(/(\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?|\(\b\d{1,2}:\d{2}(?::\d{2})?\b\))/g);

      const renderedInner = tsParts.map((sub, sIdx) => {
        if (!sub) return null;
        const subKey = `${keyPrefix}-${pIdx}-${sIdx}`;
        const cleanTime = sub.replace(/[\[\]\(\)]/g, '').trim();

        if (/^\d{1,2}:\d{2}(?::\d{2})?$/.test(cleanTime)) {
          const secs = parseTimeText(cleanTime);
          return (
            <button
              key={subKey}
              type="button"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onSeek(secs);
              }}
              className="inline-flex items-center gap-1 px-1.5 py-0.5 mx-0.5 text-xs font-mono font-medium text-primary bg-primary/10 hover:bg-primary hover:text-white rounded transition-all duration-150 cursor-pointer align-baseline select-none active:scale-95 group/ts"
              title={`Chuyển tới mốc ${cleanTime} và phát video`}
            >
              <Play className="w-2.5 h-2.5 fill-current shrink-0 opacity-80 group-hover/ts:opacity-100" />
              <span>{cleanTime}</span>
            </button>
          );
        }

        return <span key={subKey}>{sub}</span>;
      });

      if (isBold) {
        return (
          <strong key={`${keyPrefix}-${pIdx}-bold`} className="font-semibold text-slate-900">
            {renderedInner}
          </strong>
        );
      }
      return <React.Fragment key={`${keyPrefix}-${pIdx}`}>{renderedInner}</React.Fragment>;
    });
  };

  let listBuffer: { key: string; text: string }[] = [];

  const flushList = () => {
    if (listBuffer.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} className="my-2 space-y-1.5 pl-1">
          {listBuffer.map((item) => (
            <li key={item.key} className="flex items-start gap-2.5 text-sm text-slate-700 leading-relaxed">
              <span className="w-1.5 h-1.5 rounded-full bg-primary/60 mt-2 shrink-0" />
              <div className="flex-1 min-w-0">
                {renderInlineText(item.text, item.key)}
              </div>
            </li>
          ))}
        </ul>
      );
      listBuffer = [];
    }
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }

    if (trimmed.startsWith('#')) {
      flushList();
      const level = (trimmed.match(/^#+/) || ['#'])[0].length;
      const headingContent = trimmed.replace(/^#+\s*/, '');
      elements.push(
        <div key={`h-${index}`} className="mt-4 mb-2 first:mt-1">
          <div className="flex items-center gap-2">
            <span className="w-1 h-4 rounded-full bg-primary shrink-0" />
            <h3 className={`font-semibold text-slate-900 ${level <= 2 ? 'text-base' : 'text-sm'}`}>
              {renderInlineText(headingContent, `h-${index}`)}
            </h3>
          </div>
        </div>
      );
      return;
    }

    const bulletMatch = trimmed.match(/^[-*•]\s+(.*)$/);
    if (bulletMatch) {
      listBuffer.push({ key: `bullet-${index}`, text: bulletMatch[1] });
      return;
    }

    const numberedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
    if (numberedMatch) {
      flushList();
      elements.push(
        <div key={`num-${index}`} className="flex items-start gap-2.5 my-1.5 text-sm text-slate-700 leading-relaxed pl-1">
          <span className="font-mono text-xs font-bold text-primary bg-primary/10 rounded px-1.5 py-0.5 mt-0.5 shrink-0">
            {numberedMatch[1]}
          </span>
          <div className="flex-1 min-w-0">
            {renderInlineText(numberedMatch[2], `num-text-${index}`)}
          </div>
        </div>
      );
      return;
    }

    flushList();
    elements.push(
      <p key={`p-${index}`} className="text-sm text-slate-700 leading-relaxed my-2 first:mt-0 last:mb-0">
        {renderInlineText(trimmed, `p-${index}`)}
      </p>
    );
  });

  flushList();

  return <div className="space-y-1">{elements}</div>;
};

export const ResultsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get('videoId') || searchParams.get('id');
  const toast = useToast();
  const { t } = useTranslation();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [videoData, setVideoData] = useState<VideoDTO | null>(null);
  const [summaryData, setSummaryData] = useState<SummaryDTO | null>(null);

  // Summary Length and Variants State
  const [activeSummaryLength, setActiveSummaryLength] = useState<SummaryLength>('medium');
  const [isRegeneratingSummary, setIsRegeneratingSummary] = useState(false);
  const [isCopiedSummary, setIsCopiedSummary] = useState(false);

  const [transcriptList, setTranscriptList] = useState<TranscriptLine[]>([]);
  const [chaptersList, setChaptersList] = useState<Chapter[]>([]);

  const { isDevMode } = useDevMode();
  const videoRef = useRef<HTMLVideoElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const activeLineRef = useRef<HTMLDivElement>(null);
  const transcriptContainerRef = useRef<HTMLDivElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); 
  const [duration, setDuration] = useState(0); 
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedChapterIdx, setExpandedChapterIdx] = useState<number | null>(null);
  useEffect(() => {
    if (!videoId) {
      setError("Không tìm thấy ID video bài giảng. Vui lòng quay lại lịch sử.");
      return;
    }

    setLoading(true);
    setError(null);
    
    Promise.all([
      api.getVideo(videoId),
      api.getSummary(videoId)
    ])
    .then(([videoRes, summaryRes]) => {
      if (!videoRes.success || !videoRes.data) {
        throw new Error(videoRes.message || "Không thể tải chi tiết video.");
      }
      setVideoData(videoRes.data);
      setDuration(videoRes.data.duration || 0);

      if (!summaryRes.success || !summaryRes.data) {
        throw new Error(summaryRes.message || "Không thể tải tóm tắt video.");
      }

      const loadedData = summaryRes.data;
      const initialLen: SummaryLength = (loadedData.summaryLength || (loadedData as any).summary_length || 'medium') as SummaryLength;
      setActiveSummaryLength(initialLen);

      const loadedVariants: SummaryVariantsMap = {
        ...(loadedData.summaryVariantsJson || (loadedData as any).summary_variants_json || {})
      };

      // Backward compatibility fallback for legacy summaries without variant cache
      if (!loadedVariants[initialLen] && (loadedData.summaryText || (loadedData as any).summary_text)) {
        loadedVariants[initialLen] = {
          summary_text: loadedData.summaryText || (loadedData as any).summary_text,
          key_takeaways_json: loadedData.keyTakeaways || (loadedData as any).key_takeaways,
          word_count: loadedData.wordCount || (loadedData as any).word_count,
          reading_time_sec: loadedData.readingTimeSec || (loadedData as any).reading_time_sec,
          model_used: loadedData.modelUsed || (loadedData as any).model_used,
          generated_at: new Date().toISOString()
        };
      }

      setSummaryData({
        ...loadedData,
        summaryLength: initialLen,
        summary_length: initialLen,
        summaryVariantsJson: loadedVariants,
        summary_variants_json: loadedVariants
      });
      
      if (summaryRes.data.chapters && summaryRes.data.chapters.length > 0) {
        const mappedChapters = summaryRes.data.chapters.map((c: any) => ({
          title: c.title,
          start: formatTime(c.startTime !== undefined ? c.startTime : (c.start_time || 0)),
          end: formatTime(c.endTime !== undefined ? c.endTime : (c.end_time || 0)),
          summary: c.summary
        }));
        setChaptersList(mappedChapters);
      } else {
        setChaptersList([]);
      }

      const segments = summaryRes.data.transcriptSegments || (summaryRes.data as any).transcript_segments;
      const transcript = summaryRes.data.transcriptText || summaryRes.data.transcript_text;
      
      if (segments && Array.isArray(segments) && segments.length > 0) {
        const mappedTranscript = segments.map((seg: any) => ({
          speaker: seg.speaker || "SPEAKER_01",
          start: seg.start,
          end: seg.end,
          text: seg.text,
          words: (seg.words || []).map((w: any) => ({
            word: w.word,
            start: w.start,
            end: w.end,
            time: formatTime(w.start)
          }))
        }));
        setTranscriptList(mappedTranscript);
      } else if (transcript) {
        let mappedTranscript;
        try {
          const parsed = JSON.parse(transcript);
          if (Array.isArray(parsed)) {
            mappedTranscript = parsed.map((seg: any) => ({
              speaker: seg.speaker || "SPEAKER_01",
              start: seg.start,
              end: seg.end,
              text: seg.text,
              words: (seg.words || []).map((w: any) => ({
                word: w.word,
                start: w.start,
                end: w.end,
                time: formatTime(w.start)
              }))
            }));
          } else {
            throw new Error("Not an array");
          }
        } catch (e) {
          const sentences = transcript
            .split(/(?<=[.!?])\s+/)
            .filter((s: string) => s.trim().length > 0);
          
          const totalDuration = videoRes.data?.duration || 0;
          const sentenceDuration = sentences.length > 0 ? totalDuration / sentences.length : 10;
          
          mappedTranscript = sentences.map((sentence: string, idx: number) => {
            const start = idx * sentenceDuration;
            const end = (idx + 1) * sentenceDuration;
            const words = sentence.split(/\s+/).map((w, wIdx, arr) => {
              const wordDuration = sentenceDuration / arr.length;
              const wStart = start + wIdx * wordDuration;
              return {
                word: w,
                start: wStart,
                end: wStart + wordDuration,
                time: formatTime(wStart)
              };
            });
            return {
              speaker: "SPEAKER_01",
              start,
              end,
              text: sentence,
              words
            };
          });
        }
        setTranscriptList(mappedTranscript);
      } else {
        setTranscriptList([]);
      }
      setLoading(false);
    })
    .catch(err => {
      console.error("Failed to load video/summary:", err);
      setError(err.message || "Lỗi tải dữ liệu bài giảng từ máy chủ.");
      setLoading(false);
    });
  }, [videoId]);

  // Handle Play/Pause
  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  };

  // Sync state on video timeupdate
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  // Handle video loaded metadata
  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration || 2710);
    }
  };

  // Scroll smoothly to video player container with sticky header clearance
  const scrollToVideoPlayer = () => {
    const playerContainer = document.getElementById('videoContainer');
    if (playerContainer) {
      const headerOffset = 80; // 64px sticky header + 16px buffer
      const elementPosition = playerContainer.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
      window.scrollTo({
        top: Math.max(0, offsetPosition),
        behavior: 'smooth'
      });
    }
  };

  // Pure seek video without scrolling or forced autoplay (used for dragging seekbar, volume, sync)
  const seekTo = (secs: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = secs;
      setCurrentTime(secs);
    }
  };

  // Hallmark UX Navigation: Seek to timestamp, scroll smoothly up to video player, and start playback immediately
  const seekAndPlayTo = (secs: number, shouldScroll: boolean = true) => {
    if (videoRef.current) {
      videoRef.current.currentTime = secs;
      setCurrentTime(secs);
      
      const playPromise = videoRef.current.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            setIsPlaying(true);
          })
          .catch((err) => {
            console.warn("Autoplay was prevented by browser policy:", err);
            setIsPlaying(!videoRef.current?.paused);
          });
      }
    }

    if (shouldScroll) {
      scrollToVideoPlayer();
    }
  };

  // Drag seek bar
  const handleSeekChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    seekTo(val);
  };

  // Volume Change
  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    if (videoRef.current) {
      videoRef.current.volume = val;
      videoRef.current.muted = val === 0;
      setIsMuted(val === 0);
    }
  };

  // Mute toggle
  const toggleMute = () => {
    if (!videoRef.current) return;
    const nextMute = !isMuted;
    setIsMuted(nextMute);
    videoRef.current.muted = nextMute;
  };

  // Speed selection
  const selectSpeed = (speed: number) => {
    setPlaybackRate(speed);
    setIsSpeedMenuOpen(false);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  };

  // Fullscreen trigger
  const toggleFullscreen = () => {
    const playerContainer = document.getElementById('videoContainer');
    if (!playerContainer) return;
    if (!document.fullscreenElement) {
      playerContainer.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  // Switch summary length with 0ms instant cache or on-demand generation
  const handleSwitchSummaryLength = async (targetLength: SummaryLength, forceRegenerate = false) => {
    if (!videoId || isRegeneratingSummary) return;

    const currentVariants: SummaryVariantsMap = {
      ...(summaryData?.summaryVariantsJson || summaryData?.summary_variants_json || {})
    };

    const cachedVariant = currentVariants[targetLength];
    const hasCache = !forceRegenerate && Boolean(cachedVariant && (cachedVariant.summary_text || cachedVariant.summaryText));

    if (hasCache && cachedVariant) {
      // 0ms Instant Cache Switch
      setActiveSummaryLength(targetLength);
      const newSummaryText = cachedVariant.summary_text || cachedVariant.summaryText || '';
      const newTakeaways = (cachedVariant.key_takeaways_json && cachedVariant.key_takeaways_json.length > 0)
        ? cachedVariant.key_takeaways_json
        : (cachedVariant.keyTakeawaysJson || summaryData?.keyTakeaways || (summaryData as any)?.key_takeaways);

      setSummaryData(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          summaryLength: targetLength,
          summary_length: targetLength,
          summaryText: newSummaryText,
          summary_text: newSummaryText,
          wordCount: cachedVariant.word_count ?? cachedVariant.wordCount ?? prev.wordCount,
          word_count: cachedVariant.word_count ?? cachedVariant.wordCount ?? prev.word_count,
          readingTimeSec: cachedVariant.reading_time_sec ?? cachedVariant.readingTimeSec ?? prev.readingTimeSec,
          reading_time_sec: cachedVariant.reading_time_sec ?? cachedVariant.readingTimeSec ?? prev.reading_time_sec,
          modelUsed: cachedVariant.model_used || cachedVariant.modelUsed || prev.modelUsed,
          keyTakeaways: newTakeaways,
          key_takeaways: newTakeaways
        };
      });

      // Background silent sync with backend without blocking UI
      try {
        api.resummarizeVideo(videoId, targetLength, false).catch(() => {});
      } catch (_) {}
      return;
    }

    // On-demand path (cache miss or explicit refresh)
    setActiveSummaryLength(targetLength);
    setIsRegeneratingSummary(true);

    try {
      const res = await api.resummarizeVideo(videoId, targetLength, forceRegenerate);
      if (res.success && res.data) {
        const updatedDto = res.data;
        const updatedVariants: SummaryVariantsMap = {
          ...currentVariants,
          ...(updatedDto.summaryVariantsJson || updatedDto.summary_variants_json || {})
        };

        setSummaryData(prev => {
          if (!prev) return prev;
          return {
            ...prev,
            ...updatedDto,
            summaryLength: targetLength,
            summary_length: targetLength,
            summaryText: updatedDto.summaryText || updatedDto.summary_text || prev.summaryText,
            summary_text: updatedDto.summaryText || updatedDto.summary_text || prev.summary_text,
            summaryVariantsJson: updatedVariants,
            summary_variants_json: updatedVariants,
            keyTakeaways: updatedDto.keyTakeaways || updatedDto.key_takeaways || prev.keyTakeaways,
          };
        });

        const newVideoTitle = (updatedDto as any).video_title || (updatedDto as any).videoTitle;
        if (newVideoTitle) {
          setVideoData(prev => prev ? ({ ...prev, title: newVideoTitle }) : prev);
        }


        const lengthLabels: Record<SummaryLength, string> = {
          short: 'Ngắn gọn (TL;DR)',
          medium: 'Tiêu chuẩn',
          detailed: 'Chi tiết'
        };
        toast.success(`Đã hoàn tất bản tóm tắt cấp độ ${lengthLabels[targetLength]}.`, 'Cập nhật tóm tắt');
      } else {
        throw new Error(res.message || 'Không thể tạo bản tóm tắt mới.');
      }
    } catch (err: any) {
      console.error('Error resummarizing:', err);
      toast.error(err.message || 'Có lỗi xảy ra khi tạo bản tóm tắt theo độ dài yêu cầu.', 'Tạo tóm tắt thất bại');
    } finally {
      setIsRegeneratingSummary(false);
    }
  };

  // Copy summary content
  const handleCopySummary = async () => {
    const textToCopy = summaryData?.summaryText || (summaryData as any)?.summary_text;
    if (!textToCopy) return;

    try {
      await navigator.clipboard.writeText(textToCopy);
      setIsCopiedSummary(true);
      toast.info('Nội dung tóm tắt đã được lưu vào bộ nhớ tạm.', 'Đã sao chép');
      setTimeout(() => setIsCopiedSummary(false), 2000);
    } catch (err) {
      toast.error('Không thể truy cập bộ nhớ tạm trên thiết bị.', 'Sao chép thất bại');
    }
  };

  // Active takeaways synchronized with selected summary length
  const activeKeyTakeaways = useMemo(() => {
    const variants = summaryData?.summaryVariantsJson || summaryData?.summary_variants_json;
    const currentVariant = variants?.[activeSummaryLength];
    const variantTakeaways = currentVariant?.key_takeaways_json || currentVariant?.keyTakeawaysJson;
    if (variantTakeaways && Array.isArray(variantTakeaways) && variantTakeaways.length > 0) {
      return variantTakeaways;
    }
    return summaryData?.keyTakeaways || (summaryData as any)?.key_takeaways;
  }, [summaryData, activeSummaryLength]);

  const currentSummaryText = summaryData?.summaryText || (summaryData as any)?.summary_text || '';
  const currentVariants: SummaryVariantsMap = summaryData?.summaryVariantsJson || summaryData?.summary_variants_json || {};
  const activeVariant: SummaryVariantItem | undefined = currentVariants[activeSummaryLength];

  const liveWordCount = useMemo(() => {
    if (activeVariant?.word_count) return activeVariant.word_count;
    if (activeVariant?.wordCount) return activeVariant.wordCount;
    if (summaryData?.wordCount) return summaryData.wordCount;
    if (summaryData?.word_count) return summaryData.word_count;
    if (!currentSummaryText) return 0;
    return currentSummaryText.trim().split(/\s+/).filter(Boolean).length;
  }, [activeVariant, summaryData, currentSummaryText]);

  const liveReadingTimeSec = useMemo(() => {
    if (activeVariant?.reading_time_sec) return activeVariant.reading_time_sec;
    if (activeVariant?.readingTimeSec) return activeVariant.readingTimeSec;
    if (summaryData?.readingTimeSec) return summaryData.readingTimeSec;
    if (summaryData?.reading_time_sec) return summaryData.reading_time_sec;
    return Math.max(15, Math.round((liveWordCount / 200) * 60));
  }, [activeVariant, summaryData, liveWordCount]);

  const liveModelUsed = useMemo(() => {
    return activeVariant?.model_used || activeVariant?.modelUsed || summaryData?.modelUsed || summaryData?.model_used || '';
  }, [activeVariant, summaryData]);

  // Compute active transcript line and active chapter
  const getActiveLineIndex = () => {
    let activeIdx = -1;
    for (let i = transcriptList.length - 1; i >= 0; i--) {
      if (currentTime >= transcriptList[i].start) {
        activeIdx = i;
        break;
      }
    }
    return activeIdx === -1 ? 0 : activeIdx;
  };

  const activeLineIdx = getActiveLineIndex();

  const getActiveChapterIndex = () => {
    let activeIdx = -1;
    const parsedChapters = chaptersList.map(c => ({
      time: parseTimeText(c.start),
    }));

    for (let i = parsedChapters.length - 1; i >= 0; i--) {
      if (currentTime >= parsedChapters[i].time) {
        activeIdx = i;
        break;
      }
    }
    return activeIdx === -1 ? 0 : activeIdx;
  };

  const activeChapterIdx = getActiveChapterIndex();

  // Scroll active line into view strictly within transcript container, without scrolling window
  useEffect(() => {
    if (
      transcriptContainerRef.current &&
      activeLineRef.current &&
      document.activeElement !== searchInputRef.current
    ) {
      const container = transcriptContainerRef.current;
      const activeEl = activeLineRef.current;
      const containerRect = container.getBoundingClientRect();
      const activeRect = activeEl.getBoundingClientRect();
      const relativeTop = activeRect.top - containerRect.top + container.scrollTop;
      container.scrollTo({
        top: Math.max(0, relativeTop - container.clientHeight / 2 + activeEl.clientHeight / 2),
        behavior: 'smooth'
      });
    }
  }, [activeLineIdx]);

  // Export actions
  const handleExport = async (format: 'txt' | 'pdf' | 'srt' | 'docx') => {
    if (!videoId) return;
    try {
      const blob = await api.exportSummary(videoId, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `summary_${videoId}.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      toast.error(`${t('common.error')}: ${err.message}`, t('common.error'));
    }
  };

  // Progress Bar Percentage
  const progressPercent = (currentTime / (duration || 2710)) * 100;
  const seekSliderStyle = {
    background: `linear-gradient(to right, var(--primary) 0%, var(--primary) ${progressPercent}%, rgba(255,255,255,0.3) ${progressPercent}%, rgba(255,255,255,0.3) 100%)`
  };



  if (loading) {
    return (
      <div className="w-full max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-8 min-h-screen text-slate-900">
        {/* Top actions bar skeleton */}
        <div className="flex justify-between items-center py-4 border-b border-slate-200 animate-pulse">
          <div className="flex gap-2">
            <div className="h-8 w-24 bg-slate-200 rounded-lg" />
            <div className="h-8 w-24 bg-slate-200 rounded-lg" />
          </div>
          <div className="h-8 w-32 bg-slate-200 rounded-lg" />
        </div>

        {/* Two-column layout skeleton */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Video player skeleton */}
          <div className="lg:col-span-7 min-w-0 space-y-6">
            <div className="aspect-video bg-slate-200 animate-pulse rounded-2xl w-full" />
            <div className="space-y-3">
              <div className="h-6 w-3/4 bg-slate-200 animate-pulse rounded-md" />
              <div className="h-4 w-1/3 bg-slate-200 animate-pulse rounded-md" />
            </div>
            <div className="bg-white border border-slate-100 rounded-xl p-4 space-y-2">
              <div className="h-4 w-1/4 bg-slate-200 animate-pulse rounded-md" />
              <div className="h-4 w-full bg-slate-200 animate-pulse rounded-md" />
              <div className="h-4 w-5/6 bg-slate-200 animate-pulse rounded-md" />
            </div>
          </div>

          {/* Right Column: Summarization panels skeleton */}
          <div className="lg:col-span-5 min-w-0 space-y-6">
            <div className="bg-white border border-slate-100 rounded-2xl p-6 shadow-sm space-y-6">
              {/* Tab selectors skeleton */}
              <div className="flex border-b border-slate-100 pb-3 gap-4">
                <div className="h-6 w-20 bg-slate-200 animate-pulse rounded-md" />
                <div className="h-6 w-20 bg-slate-200 animate-pulse rounded-md" />
                <div className="h-6 w-20 bg-slate-200 animate-pulse rounded-md" />
              </div>
              
              {/* Summary text skeleton */}
              <Skeleton.Text lines={8} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center h-[60vh] bg-[#FAF5FF] p-6 text-center">
        <span className="material-symbols-outlined text-[48px] text-danger mb-4">warning</span>
        <p className="text-danger font-bold max-w-md mb-6 text-sm">{error}</p>
        <Link 
          to="/history" 
          className="px-6 py-2 bg-slate-900 text-white rounded hover:bg-primary font-semibold text-sm transition-all"
        >
          Quay lại Lịch sử
        </Link>
      </div>
    );
  }

  return (
    <div className="w-full max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6 min-h-screen text-slate-900">
      {/* Top Nav/Actions Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 py-3 border-b border-slate-200/60">
        <div className="flex items-center gap-2">
          <Link to="/" className="px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-100 text-xs font-semibold transition-all flex items-center gap-1.5 text-slate-700">
            <span className="material-symbols-outlined text-sm">home</span> {t('results.home')}
          </Link>
          <Link to="/history" className="px-3 py-1.5 border border-slate-200 rounded-lg hover:bg-slate-100 text-xs font-semibold transition-all flex items-center gap-1.5 text-slate-700">
            <span className="material-symbols-outlined text-sm">history</span> {t('results.history')}
          </Link>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-600 flex items-center gap-1 select-none">
              <span className="material-symbols-outlined text-sm text-slate-400">download</span>
              Xuất tài liệu:
            </span>
            <div className="inline-flex items-center p-1 bg-slate-100 rounded-lg border border-slate-200 text-xs font-medium gap-0.5">
              <button 
                onClick={() => handleExport('docx')} 
                title="Xuất tài liệu Microsoft Word (.docx)"
                className="px-2.5 py-1 rounded-md hover:bg-white hover:text-slate-900 hover:shadow-2xs text-slate-700 transition-all flex items-center gap-1 font-medium"
              >
                <span className="material-symbols-outlined text-sm text-blue-600">article</span> Word
              </button>
              <button 
                onClick={() => handleExport('pdf')} 
                title="Xuất tài liệu PDF (.pdf)"
                className="px-2.5 py-1 rounded-md hover:bg-white hover:text-slate-900 hover:shadow-2xs text-slate-700 transition-all flex items-center gap-1 font-medium"
              >
                <span className="material-symbols-outlined text-sm text-red-500">picture_as_pdf</span> PDF
              </button>
              <button 
                onClick={() => handleExport('srt')} 
                title="Xuất tệp phụ đề (.srt)"
                className="px-2.5 py-1 rounded-md hover:bg-white hover:text-slate-900 hover:shadow-2xs text-slate-700 transition-all flex items-center gap-1 font-medium"
              >
                <span className="material-symbols-outlined text-sm text-emerald-600">subtitles</span> SRT
              </button>
              <button 
                onClick={() => handleExport('txt')} 
                title="Xuất bản ghi văn bản thuần (.txt)"
                className="px-2.5 py-1 rounded-md hover:bg-white hover:text-slate-900 hover:shadow-2xs text-slate-700 transition-all flex items-center gap-1 font-medium"
              >
                <span className="material-symbols-outlined text-sm text-slate-500">description</span> TXT
              </button>
            </div>
          </div>
          <Link to={`/qa?videoId=${videoId}`} className="px-3.5 py-1.5 bg-slate-900 text-white rounded-lg hover:bg-primary text-xs font-semibold transition-all flex items-center gap-1.5 shadow-2xs">
            <span className="material-symbols-outlined text-sm">forum</span> {t('results.qa_rag')}
          </Link>
        </div>
      </div>

      {/* Main Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Column: Video Player & Transcripts */}
        <div className="lg:col-span-7 min-w-0 bg-white border border-slate-200 rounded-xl p-5 md:p-6 flex flex-col justify-between shadow-xs space-y-4">
          <div>
            <div className="mb-4">
              <span className="text-[11px] font-bold uppercase tracking-wider text-primary bg-primary/10 px-2.5 py-0.5 rounded-md inline-block mb-1.5">
                {t('results.original_video')}
              </span>
              <h1 className="font-headline-md text-xl md:text-2xl font-bold text-slate-900 leading-snug break-words">
                {videoData?.title ? cleanDisplayTitle(videoData.title, summaryData?.chapters?.[0]?.title) : t('results.original_video')}
              </h1>

            </div>
            
            {/* Player Container */}
            <div className="w-full mb-4">
              <div className="relative group aspect-video max-h-[460px] bg-black rounded-lg overflow-hidden border border-slate-200 shadow-xs" id="videoContainer">
                <video 
                  ref={videoRef}
                  src={videoData?.filePath ? (videoData.filePath.startsWith('http') ? videoData.filePath : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${videoData.filePath}`) : ''} 
                  poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
                  className="w-full h-full object-contain"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onClick={togglePlay}
                  playsInline
                  loop
                />
                
                {/* Center Play button */}
                {!isPlaying && (
                  <button onClick={togglePlay} className="absolute inset-0 m-auto w-14 h-14 bg-primary text-white rounded-full flex items-center justify-center hover:scale-110 transition-transform shadow-lg z-20">
                    <span className="material-symbols-outlined text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>play_arrow</span>
                  </button>
                )}

                {/* HTML Video Custom Controls Overlay */}
                <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/85 to-transparent flex flex-col gap-2 z-20 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <div className="w-full flex items-center">
                    <input 
                      type="range" 
                      className="w-full h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-primary" 
                      min={0} 
                      max={duration || 2710} 
                      step={0.1}
                      value={currentTime} 
                      onChange={handleSeekChange}
                      style={seekSliderStyle}
                    />
                  </div>
                  <div className="flex justify-between items-center text-white text-xs">
                    <div className="flex items-center gap-3">
                      <button onClick={togglePlay} className="hover:text-primary transition-colors">
                        <span className="material-symbols-outlined text-sm">{isPlaying ? "pause" : "play_arrow"}</span>
                      </button>
                      <div className="flex items-center gap-1.5">
                        <button onClick={toggleMute} className="hover:text-primary transition-colors">
                          <span className="material-symbols-outlined text-sm">{isMuted || volume === 0 ? "volume_off" : "volume_up"}</span>
                        </button>
                        <input 
                          type="range" 
                          className="w-12 h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-primary volume-slider" 
                          min={0} 
                          max={1} 
                          step={0.05} 
                          value={isMuted ? 0 : volume} 
                          onChange={handleVolumeChange}
                        />
                      </div>
                      <span className="font-mono-data text-[10px]">{formatTime(currentTime)} / {formatTime(duration)}</span>
                    </div>
                    
                    <div className="flex items-center gap-3 relative">
                      <div className="relative">
                        <button onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)} className="hover:text-primary transition-colors px-1 bg-slate-800 rounded font-mono-data text-[10px]">
                          {playbackRate}x
                        </button>
                        {isSpeedMenuOpen && (
                          <div className="absolute bottom-6 right-0 bg-slate-900 border border-slate-700 rounded shadow-md flex flex-col overflow-hidden text-[10px] z-50">
                            {[0.5, 1.0, 1.25, 1.5, 2.0].map((rate) => (
                              <button 
                                key={rate} 
                                onClick={() => selectSpeed(rate)}
                                className={`px-3 py-1 text-left hover:bg-primary hover:text-slate-900 font-semibold ${playbackRate === rate ? 'text-primary' : 'text-white'}`}
                              >
                                {rate}x
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                      <button onClick={toggleFullscreen} className="hover:text-primary transition-colors">
                        <span className="material-symbols-outlined text-sm">fullscreen</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            </div>
          </div>
        {/* Right Column: Transcripts */}
        <div className="lg:col-span-5 min-w-0 flex flex-col">
          <div className="bg-white border border-slate-200 rounded-xl p-4 flex flex-col h-full min-h-[380px] lg:max-h-[580px] shadow-xs">
            {/* Transcript Sync Area */}
            <div className="border border-slate-200/60 rounded-xl overflow-hidden bg-[#FAF5FF] flex flex-col flex-1 min-h-0">
              <div className="flex items-center justify-between px-4 py-3 bg-white border-b border-slate-200/60">
                <span className="font-label-md text-xs font-bold uppercase tracking-wider text-slate-400">Timestamps &amp; Transcript</span>
                <div className="flex items-center glass-panel px-3 py-1.5 rounded-full border border-slate-200/60 focus-within:border-primary transition-all max-w-[200px] sm:max-w-xs bg-white">
                  <span className="material-symbols-outlined text-slate-400 text-sm">search</span>
                  <input 
                    ref={searchInputRef}
                    className="bg-transparent border-none focus:ring-0 text-xs w-full px-2 outline-none" 
                    placeholder={t('results.search_keyword')}
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    type="text"
                  />
                </div>
              </div>

              <div 
                ref={transcriptContainerRef}
                className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar min-h-0"
              >
                {transcriptList.map((line, idx) => {
                  const isLineActive = idx === activeLineIdx;
                  const matchesSearch = searchQuery.trim() === '' || line.text.toLowerCase().includes(searchQuery.toLowerCase());
                  
                  if (!matchesSearch) return null;

                  return (
                    <div 
                      key={idx}
                      ref={isLineActive ? activeLineRef : null}
                      onClick={() => seekAndPlayTo(line.start, false)}
                      className={`flex gap-3 items-start p-2 rounded-lg cursor-pointer transition-colors duration-150 ${
                        isLineActive ? 'bg-primary/10 border-l-2 border-primary' : 'hover:bg-slate-50'
                      }`}
                    >
                      <span className={`font-mono-data text-[10px] px-1.5 py-0.5 rounded font-bold shrink-0 mt-0.5 ${
                        isLineActive ? 'bg-primary text-white' : 'bg-slate-200/50 text-slate-900'
                      }`}>
                        {formatTime(line.start)}
                      </span>
                      <p className="text-xs text-slate-900 leading-relaxed flex-1 flex flex-wrap">
                        {line.words && line.words.length > 0 ? (
                          line.words.map((word, wIdx) => {
                            const isWordActive = isLineActive && currentTime >= word.start && currentTime <= word.end;
                            const isSearchMatch = searchQuery.trim() !== '' && word.word.toLowerCase().includes(searchQuery.toLowerCase());

                            return (
                              <span 
                                key={wIdx}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  seekAndPlayTo(word.start, false);
                                }}
                                className={`mr-1 px-0.5 rounded cursor-pointer transition-all ${
                                  isWordActive 
                                    ? 'bg-primary text-white font-bold' 
                                    : isSearchMatch 
                                      ? 'bg-warning-bg text-slate-900 border-b border-warning-border' 
                                      : 'hover:text-primary'
                                }`}
                              >
                                {word.word}
                              </span>
                            );
                          })
                        ) : (
                          <span className="italic text-slate-500">{line.text}</span>
                        )}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Full Information Section */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6 mt-8">
        <h2 className="font-headline-md text-lg font-bold text-slate-900 flex items-center gap-2">
           <span className="material-symbols-outlined text-primary">insights</span>
           {t('results.lecture_overview', 'Tổng quan & Phân tích bài giảng')}
        </h2>

        {/* Key Takeaways & Highlights Component synchronized with active summary length */}
        <KeyTakeawaysWidget
          takeaways={activeKeyTakeaways}
          onSeekTimestamp={(ts) => seekAndPlayTo(ts, true)}
          isLoading={loading || isRegeneratingSummary}
        />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 bg-white border border-slate-200 rounded-xl p-5 md:p-6 shadow-xs">
          {/* Ergonomic Interactive Summary Reader */}
          <div className="lg:col-span-2 min-w-0 flex flex-col">
            <div className="flex flex-col flex-1 min-h-0 border border-slate-200/80 rounded-xl overflow-hidden bg-white shadow-2xs">
              {/* Header with Segmented Control & Actions */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-4 bg-slate-50/70 border-b border-slate-200/80">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="material-symbols-outlined text-primary text-xl">subject</span>
                  <h3 className="font-bold text-slate-900 text-sm md:text-base truncate">
                    {t('results.content_summary', 'Tóm tắt nội dung bài giảng')}
                  </h3>
                  <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium bg-primary/10 text-primary border border-primary/20 shrink-0">
                    {LENGTH_OPTIONS.find(o => o.id === activeSummaryLength)?.badge}
                  </span>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  {/* Segmented Control */}
                  <div className="inline-flex items-center p-0.5 bg-slate-200/60 rounded-lg border border-slate-200 text-xs">
                    {LENGTH_OPTIONS.map((option) => {
                      const isActive = activeSummaryLength === option.id;
                      const isCached = Boolean(currentVariants[option.id]?.summary_text || currentVariants[option.id]?.summaryText);

                      return (
                        <button
                          key={option.id}
                          type="button"
                          onClick={() => handleSwitchSummaryLength(option.id, false)}
                          disabled={isRegeneratingSummary}
                          className={`relative inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs transition-all duration-150 cursor-pointer ${
                            isActive
                              ? 'bg-white text-slate-900 shadow-2xs font-semibold'
                              : 'text-slate-600 hover:text-slate-900 hover:bg-white/50'
                          } ${isRegeneratingSummary ? 'opacity-60 cursor-not-allowed' : 'active:scale-95'}`}
                          title={`${option.label}: ${option.description} ${isCached ? '• Đã lưu sẵn sàng (chuyển đổi 0ms)' : '• Tạo nội dung theo yêu cầu'}`}
                        >
                          <span>{option.label}</span>
                          {isCached && (
                            <span
                              className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0"
                              title="Bản tóm tắt đã có sẵn trong bộ nhớ đệm (chuyển đổi 0ms)"
                            />
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1 border-l border-slate-200 pl-2">
                    <button
                      type="button"
                      onClick={() => handleSwitchSummaryLength(activeSummaryLength, true)}
                      disabled={isRegeneratingSummary}
                      className={`p-1.5 rounded-md text-slate-500 hover:text-primary hover:bg-slate-200/50 transition-colors border border-slate-200/80 ${
                        isRegeneratingSummary ? 'cursor-not-allowed opacity-50' : 'cursor-pointer active:scale-95'
                      }`}
                      title="Tạo lại bản tóm tắt cho chế độ hiện tại"
                    >
                      <RotateCw className={`w-3.5 h-3.5 ${isRegeneratingSummary ? 'animate-spin text-primary' : ''}`} />
                    </button>
                    <button
                      type="button"
                      onClick={handleCopySummary}
                      className="p-1.5 rounded-md text-slate-500 hover:text-primary hover:bg-slate-200/50 transition-colors border border-slate-200/80 cursor-pointer active:scale-95"
                      title={isCopiedSummary ? "Đã sao chép vào bộ nhớ tạm" : "Sao chép toàn bộ nội dung"}
                    >
                      {isCopiedSummary ? (
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Reading Stats Bar */}
              <div className="flex flex-wrap items-center gap-3 py-2 px-4 bg-slate-50/40 border-b border-slate-200/60 text-xs text-slate-500">
                <div className="flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  <span>
                    Độ dài: <strong className="font-semibold text-slate-700">{liveWordCount} từ</strong>
                  </span>
                </div>
                <span className="text-slate-300">•</span>
                <div className="flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  <span>
                    Thời gian đọc: <strong className="font-semibold text-slate-700">~{formatReadingTime(liveReadingTimeSec)}</strong>
                  </span>
                </div>
                {liveModelUsed && (
                  <>
                    <span className="text-slate-300">•</span>
                    <div className="flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                      <span className="truncate max-w-[200px]" title={liveModelUsed}>
                        Mô hình: <strong className="font-semibold text-slate-700">{liveModelUsed}</strong>
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Reader Body */}
              {isRegeneratingSummary ? (
                <div className="flex-1 p-6 bg-[#FAF5FF]/70 flex flex-col justify-center min-h-[240px]">
                  <div className="flex items-center gap-2.5 p-3.5 rounded-lg bg-primary/5 border border-primary/20 text-xs font-medium text-primary">
                    <RotateCw className="w-4 h-4 animate-spin shrink-0 text-primary" />
                    <span>
                      Đang tổng hợp bản tóm tắt {activeSummaryLength === 'short' ? 'ngắn gọn' : activeSummaryLength === 'detailed' ? 'chi tiết' : 'tiêu chuẩn'} từ dữ liệu bài giảng...
                    </span>
                  </div>
                  <div className="space-y-3 mt-5">
                    <div className="h-4 bg-primary/10 rounded w-3/4 animate-pulse" />
                    <div className="h-4 bg-slate-200/80 rounded w-full animate-pulse" />
                    <div className="h-4 bg-slate-200/80 rounded w-5/6 animate-pulse" />
                    <div className="h-4 bg-slate-200/80 rounded w-2/3 animate-pulse" />
                  </div>
                </div>
              ) : (
                <div className="flex-1 p-5 bg-[#FAF5FF] text-slate-700 overflow-y-auto custom-scrollbar max-h-[520px]">
                  {currentSummaryText ? (
                    renderFormattedSummary(currentSummaryText, (secs) => seekAndPlayTo(secs, true))
                  ) : (
                    <div className="py-8 text-center text-sm text-slate-400 italic">
                      Chưa có nội dung tóm tắt cho bài giảng này.
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Box 2: Chapters */}
          <div className="lg:col-span-1 min-w-0 flex flex-col">
            <div className="flex flex-col flex-1 min-h-0">
              <h2 className="font-headline-md text-md font-bold text-slate-900 mb-6 flex items-center gap-2 shrink-0">
                <span className="material-symbols-outlined text-primary">list</span>
                {t('results.auto_chapters')}
              </h2>
              <div className="flex-1 space-y-2 overflow-y-auto custom-scrollbar pr-2">
                {chaptersList.map((chapter, idx) => {
                  const isChapterActive = idx === activeChapterIdx;
                  const isExpanded = idx === expandedChapterIdx;
                  const chStartSecs = parseTimeText(chapter.start);

                  return (
                    <div key={idx} className="flex flex-col border border-slate-200/50 rounded-lg overflow-hidden shadow-sm transition-all duration-300">
                      <div 
                        onClick={() => seekAndPlayTo(chStartSecs, true)}
                        className={`group/ch flex justify-between items-center p-3 cursor-pointer transition-all duration-150 select-none active:scale-[0.99] ${
                          isChapterActive 
                            ? 'border-l-4 border-l-primary bg-primary/10 text-slate-900 font-bold shadow-xs' 
                            : 'border-l-4 border-l-transparent bg-[#FAF5FF] text-slate-600 hover:bg-primary/5 hover:text-slate-900'
                        }`}
                        title={`Phát video từ phân chương: ${chapter.title} (${chapter.start})`}
                      >
                        <div className="flex flex-col gap-1 pr-2 flex-1 min-w-0">
                          <span className="text-xs truncate font-medium group-hover/ch:text-primary transition-colors" title={chapter.title}>{chapter.title}</span>
                          <span className="font-mono-data text-[10px] text-slate-400 font-medium flex items-center gap-1">
                            <span className="material-symbols-outlined text-[12px] group-hover/ch:text-primary">play_arrow</span>
                            {chapter.start}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setExpandedChapterIdx(isExpanded ? null : idx);
                            }}
                            className={`w-6 h-6 flex justify-center items-center rounded-full hover:bg-slate-200/50 transition-colors ${isExpanded ? 'text-primary bg-primary/10' : 'text-slate-400'}`}
                          >
                            <span className="material-symbols-outlined text-sm transition-transform duration-300" style={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                              expand_more
                            </span>
                          </button>
                        </div>
                      </div>
                      
                      <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}`}>
                        <div className="px-3 pb-3 pt-1 text-[11px] text-slate-500 bg-[#FAF5FF]/50 border-t border-slate-100 italic leading-relaxed">
                          {chapter.summary || "Chưa có nội dung tóm tắt."}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Dev Mode Sections: Object Detection & Tracking + Keyframes Gallery */}
        {isDevMode && (
          <>
            {videoId && (
              <SceneObjectExplorer
                videoId={videoId}
                onSeekVideo={seekTo}
                currentTime={currentTime}
              />
            )}

            {/* Keyframes Gallery Section */}
            {summaryData?.keyframes && summaryData.keyframes.length > 0 && (
              <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
                <div className="flex flex-col gap-1">
                  <h2 className="font-headline-md text-md font-bold text-slate-900 flex items-center gap-2">
                    <span className="material-symbols-outlined text-primary">image_search</span>
                    {t('results.keyframes')}
                  </h2>
                  <p className="text-[11px] text-slate-500 italic mt-1">
                    {t('results.importance_desc', '* Độ quan trọng: Mức độ tương đồng ngữ nghĩa, hành động và chất lượng hình ảnh do AI đánh giá')}
                  </p>
                </div>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                  {summaryData.keyframes.map((kf, idx) => {
                    const imgUrl = kf.imageUrl || (kf as any).image_url || "";
                    const importance = kf.importanceScore !== undefined ? kf.importanceScore : ((kf as any).importance_score || 0);
                    
                    return (
                      <div 
                        key={idx} 
                        onClick={() => seekAndPlayTo(kf.timestamp, true)}
                        className="group border border-slate-200/60 rounded-xl overflow-hidden hover:border-primary transition-all duration-300 cursor-pointer active:scale-[0.99] bg-[#FAF5FF] flex flex-col justify-between shadow-xs hover:shadow-md"
                        title={`Xem và phát video từ mốc ${formatTime(kf.timestamp)}`}
                      >
                        <div className="relative aspect-video w-full bg-black overflow-hidden shrink-0">
                          <img 
                            src={imgUrl.startsWith('http') ? imgUrl : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${imgUrl}`} 
                            alt={kf.description}
                            className="absolute inset-0 w-full h-full object-contain transform group-hover:scale-105 transition-transform duration-500"
                          />
                          <div className="absolute top-2 left-2 z-10 px-1.5 py-0.5 bg-slate-900/85 text-white font-mono-data text-[9px] rounded font-bold">
                            {formatTime(kf.timestamp)}
                          </div>
                        </div>
                        
                        <div className="p-4 flex-1 flex flex-col justify-between space-y-2">
                          <div>
                            <p className="text-xs text-slate-900 font-semibold leading-normal line-clamp-2" title={kf.description}>
                              <span className="material-symbols-outlined text-[14px] align-text-bottom mr-1 text-primary">image</span>
                              {kf.description}
                            </p>
                            {kf.transcript && (
                              <p className="text-[10px] text-slate-500 italic leading-tight line-clamp-2 mt-2" title={kf.transcript}>
                                <span className="material-symbols-outlined text-[12px] align-text-bottom mr-1 text-slate-400">record_voice_over</span>
                                "{kf.transcript}"
                              </p>
                            )}
                          </div>
                          <div className="flex justify-between items-center pt-2 border-t border-slate-200/50 text-[10px] text-slate-500 mt-2">
                            <span className="font-bold text-primary">{t('results.importance')}</span>
                            <span className="font-mono-data font-bold bg-slate-200/50 px-1.5 py-0.5 rounded text-slate-900">
                              {Math.round(importance * 100)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
