import React, { useRef, useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import type { TranscriptLine, Chapter } from '../types';
import { api } from '../services/api';
import { CONFIG } from '../config';
import './ResultsPage.css';

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
  importanceScore: number;
}

interface SummaryDTO {
  summaryId: string;
  videoId: string;
  summaryText: string;
  transcriptText: string;
  modelUsed: string;
  processingTime: number;
  chapters: ChapterDTO[];
  keyframes: KeyframeDTO[];
}

interface VideoDTO {
  videoId: string;
  originalUrl?: string;
  filePath?: string;
  duration?: number;
  status: string;
}

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

// Mock data
export const _mockTranscript: TranscriptLine[] = [
  {
    speaker: "SPEAKER_00",
    start: 0,
    end: 14,
    text: "Chào mừng các bạn đến với bài giảng về Trí tuệ Nhân tạo.",
    words: [
      { word: "Chào", start: 0, end: 0.3, time: "00:00.0" },
      { word: "mừng", start: 0.3, end: 0.6, time: "00:00.3" },
      { word: "các", start: 0.6, end: 0.8, time: "00:00.6" },
      { word: "bạn", start: 0.8, end: 1.0, time: "00:00.8" },
      { word: "đến", start: 1.0, end: 1.2, time: "00:01.0" },
      { word: "với", start: 1.2, end: 1.4, time: "00:01.2" },
      { word: "bài", start: 1.4, end: 1.6, time: "00:01.4" },
      { word: "giảng", start: 1.6, end: 1.9, time: "00:01.6" },
      { word: "về", start: 1.9, end: 2.1, time: "00:01.9" },
      { word: "Trí", start: 2.1, end: 2.4, time: "00:02.1" },
      { word: "tuệ", start: 2.4, end: 2.6, time: "00:02.4" },
      { word: "Nhân", start: 2.6, end: 2.9, time: "00:02.6" },
      { word: "tạo.", start: 2.9, end: 3.5, time: "00:02.9" },
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 15,
    end: 311,
    text: "Hôm nay chúng ta sẽ đi sâu vào cấu trúc của mạng nơ-ron nhân tạo.",
    words: [
      { word: "Hôm", start: 15.0, end: 15.3, time: "00:15.0" },
      { word: "nay", start: 15.3, end: 15.6, time: "00:15.3" },
      { word: "chúng", start: 15.6, end: 15.8, time: "00:15.6" },
      { word: "ta", start: 15.8, end: 16.0, time: "00:16.0" },
      { word: "sẽ", start: 16.0, end: 16.2, time: "00:16.2" },
      { word: "đi", start: 16.2, end: 16.5, time: "00:16.2" },
      { word: "sâu", start: 16.5, end: 16.8, time: "00:16.5" },
      { word: "vào", start: 16.8, end: 17.1, time: "00:16.8" },
      { word: "cấu", start: 17.1, end: 17.4, time: "00:17.1" },
      { word: "trúc", start: 17.4, end: 17.7, time: "00:17.4" },
      { word: "của", start: 17.7, end: 17.9, time: "00:17.7" },
      { word: "mạng", start: 17.9, end: 18.2, time: "00:17.9" },
      { word: "nơ-ron", start: 18.2, end: 18.7, time: "00:18.2" },
      { word: "nhân", start: 18.7, end: 19.0, time: "00:18.7" },
      { word: "tạo.", start: 19.0, end: 19.5, time: "00:19.0" },
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 312,
    end: 339,
    text: "Khái niệm quan trọng nhất ở đây là Backpropagation (Lan truyền ngược). Nó giúp điều chỉnh trọng số...",
    words: [
      { word: "Khái", start: 312.0, end: 312.4, time: "05:12.0" },
      { word: "niệm", start: 312.4, end: 312.7, time: "05:12.4" },
      { word: "quan", start: 312.7, end: 313.0, time: "05:12.7" },
      { word: "trọng", start: 313.0, end: 313.3, time: "05:13.0" },
      { word: "nhất", start: 313.3, end: 313.6, time: "05:13.3" },
      { word: "ở", start: 313.6, end: 313.8, time: "05:13.6" },
      { word: "đây", start: 313.8, end: 314.1, time: "05:13.8" },
      { word: "là", start: 314.1, end: 314.3, time: "05:14.1" },
      { word: "Backpropagation", start: 314.3, end: 315.2, time: "05:14.3" },
      { word: "(Lan", start: 315.2, end: 315.5, time: "05:15.2" },
      { word: "truyền", start: 315.5, end: 315.8, time: "05:15.5" },
      { word: "ngược).", start: 315.8, end: 316.2, time: "05:15.8" },
      { word: "Nó", start: 316.2, end: 316.4, time: "05:16.2" },
      { word: "giúp", start: 316.4, end: 316.7, time: "05:16.4" },
      { word: "điều", start: 316.7, end: 317.0, time: "05:16.7" },
      { word: "chỉnh", start: 317.0, end: 317.3, time: "05:17.0" },
      { word: "trọng", start: 317.3, end: 317.6, time: "05:17.3" },
      { word: "số...", start: 317.6, end: 318.5, time: "05:17.6" },
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 340,
    end: 749,
    text: "Hãy nhìn vào đạo hàm của hàm mất mát. Chúng ta tính gradient để tìm điểm cực tiểu.",
    words: [
      { word: "Hãy", start: 340.0, end: 340.3, time: "05:40.0" },
      { word: "nhìn", start: 340.3, end: 340.6, time: "05:40.3" },
      { word: "vào", start: 340.6, end: 340.8, time: "05:40.6" },
      { word: "đạo", start: 340.8, end: 341.1, time: "05:40.8" },
      { word: "hàm", start: 341.1, end: 341.4, time: "05:41.1" },
      { word: "của", start: 341.4, end: 341.6, time: "05:41.4" },
      { word: "hàm", start: 341.6, end: 341.9, time: "05:41.6" },
      { word: "mất", start: 341.9, end: 342.2, time: "05:41.9" },
      { word: "mát.", start: 342.2, end: 342.6, time: "05:42.2" },
      { word: "Chúng", start: 342.6, end: 342.9, time: "05:42.6" },
      { word: "ta", start: 342.9, end: 343.1, time: "05:42.9" },
      { word: "tính", start: 343.1, end: 343.4, time: "05:43.1" },
      { word: "gradient", start: 343.4, end: 344.0, time: "05:43.4" },
      { word: "để", start: 344.0, end: 344.2, time: "05:44.0" },
      { word: "tìm", start: 344.2, end: 344.5, time: "05:44.2" },
      { word: "điểm", start: 344.5, end: 344.8, time: "05:44.5" },
      { word: "cực", start: 344.8, end: 345.1, time: "05:44.8" },
      { word: "tiểu.", start: 345.1, end: 346.0, time: "05:45.1" },
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 750,
    end: 919,
    text: "Đối với các chuỗi dài, mạng RNN gặp hiện tượng tiêu biến gradient.",
    words: [
      { word: "Đối", start: 750.0, end: 750.4, time: "12:30.0" },
      { word: "với", start: 750.4, end: 750.7, time: "12:30.4" },
      { word: "các", start: 750.7, end: 751.0, time: "12:30.7" },
      { word: "chuỗi", start: 751.0, end: 751.3, time: "12:31.0" },
      { word: "dài,", start: 751.3, end: 751.6, time: "12:31.3" },
      { word: "mạng", start: 751.6, end: 751.9, time: "12:31.6" },
      { word: "RNN", start: 751.9, end: 752.5, time: "12:31.9" },
      { word: "gặp", start: 752.5, end: 752.8, time: "12:32.5" },
      { word: "hiện", start: 752.8, end: 753.1, time: "12:32.8" },
      { word: "tượng", start: 753.1, end: 753.4, time: "12:33.1" },
      { word: "tiêu", start: 753.4, end: 753.7, time: "12:33.4" },
      { word: "biến", start: 753.7, end: 754.0, time: "12:33.7" },
      { word: "gradient.", start: 754.0, end: 755.0, time: "12:34.0" },
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 920,
    end: 2710,
    text: "Tiếp theo, hãy chuyển sang các kiến trúc hiện đại hơn như mô hình Transformer.",
    words: [
      { word: "Tiếp", start: 920.0, end: 920.3, time: "15:20.0" },
      { word: "theo,",
        start: 920.3,
        end: 920.6,
        time: "15:20.3"
      },
      { word: "hãy", start: 920.6, end: 920.8, time: "15:20.6" },
      { word: "chuyển", start: 920.8, end: 921.1, time: "15:20.8" },
      { word: "sang", start: 921.1, end: 921.4, time: "15:21.1" },
      { word: "các", start: 921.4, end: 921.6, time: "15:21.4" },
      { word: "kiến", start: 921.6, end: 921.9, time: "15:21.6" },
      { word: "trúc", start: 921.9, end: 922.2, time: "15:21.9" },
      { word: "hiện", start: 922.2, end: 922.5, time: "15:22.2" },
      { word: "đại", start: 922.5, end: 922.8, time: "15:22.5" },
      { word: "hơn", start: 922.8, end: 923.0, time: "15:22.8" },
      { word: "như", start: 923.0, end: 923.3, time: "15:23.0" },
      { word: "mô", start: 923.3, end: 923.5, time: "15:23.3" },
      { word: "hình", start: 923.5, end: 923.8, time: "15:23.5" },
      { word: "Transformer.", start: 923.8, end: 925.0, time: "15:23.8" },
    ]
  }
];

export const _mockChapters: Chapter[] = [
  { start: "00:00", end: "05:12", title: "1. Giới thiệu Deep Learning", summary: "Phần này giới thiệu bài giảng." },
  { start: "05:12", end: "12:30", title: "2. Thuật toán Backpropagation & Gradient Descent", summary: "Giới thiệu lan truyền ngược." },
  { start: "12:30", end: "15:20", title: "3. Hạn chế của mạng RNN", summary: "Giải thích lý do RNN gặp lỗi với chuỗi dài." },
  { start: "15:20", end: "45:10", title: "4. Kiến trúc Transformer & Self-Attention", summary: "Tìm hiểu kiến trúc mới." },
];

// Helper to parse transcript string into TranscriptLine format
const parseTranscriptText = (text: string): TranscriptLine[] => {
  if (!text) return [];
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
  let currentStart = 0;
  
  return sentences.map((sentence) => {
    const trimmed = sentence.trim();
    const words = trimmed.split(/\s+/);
    const duration = Math.max(words.length * 0.4, 3.0);
    const end = currentStart + duration;
    
    const wordTokens = words.map((w, wIdx) => {
      const wDur = duration / words.length;
      const wStart = currentStart + wIdx * wDur;
      const wEnd = wStart + wDur;
      return {
        word: w,
        start: wStart,
        end: wEnd,
        time: formatTime(wStart),
      };
    });
    
    const line: TranscriptLine = {
      speaker: "SPEAKER_00",
      start: currentStart,
      end: end,
      text: trimmed,
      words: wordTokens,
    };
    
    currentStart = end;
    return line;
  });
};

export const ResultsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const videoId = searchParams.get('videoId') || searchParams.get('id');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [videoData, setVideoData] = useState<VideoDTO | null>(null);
  const [summaryData, setSummaryData] = useState<SummaryDTO | null>(null);

  const [transcriptList, setTranscriptList] = useState<TranscriptLine[]>([]);
  const [chaptersList, setChaptersList] = useState<Chapter[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const activeLineRef = useRef<HTMLDivElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); 
  const [duration, setDuration] = useState(0); 
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const [isSpeedMenuOpen, setIsSpeedMenuOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    if (!videoId) {
      // Fallback to fetch first video
      api.getVideos().then(res => {
        if (res.success && res.data && res.data.length > 0) {
          const firstDone = res.data.find((v: any) => v.status === 'done');
          if (firstDone) {
            setSearchParams({ videoId: firstDone.videoId });
          }
        }
      });
      return;
    }

    setLoading(true);
    setError(null);

    Promise.all([
      api.getVideo(videoId).catch(() => null),
      api.getSummary(videoId).catch(() => null)
    ])
      .then(([videoRes, summaryRes]) => {
        if (videoRes && videoRes.success) {
          setVideoData(videoRes.data);
          setDuration(videoRes.data.duration || 0);
        }
        if (summaryRes && summaryRes.success && summaryRes.data) {
          setSummaryData(summaryRes.data);
          
          // Parse chapters
          if (summaryRes.data.chapters) {
            const mappedChapters = summaryRes.data.chapters.map((c: any) => ({
              start: formatTime(c.startTime),
              end: formatTime(c.endTime),
              title: c.title,
              summary: c.summary,
            }));
            setChaptersList(mappedChapters);
          }

          // Parse transcript
          if (summaryRes.data.transcriptText) {
            const parsedLines = parseTranscriptText(summaryRes.data.transcriptText);
            setTranscriptList(parsedLines);
          }
        } else {
          setError("Không tìm thấy kết quả tóm tắt cho video này. Tiến trình xử lý có thể đang diễn ra.");
        }
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || "Lỗi khi lấy thông tin kết quả.");
        setLoading(false);
      });
  }, [videoId, setSearchParams]);


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

  // Seek video
  const seekTo = (secs: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = secs;
      setCurrentTime(secs);
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

  // Set start time to video element on initial mount
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.currentTime = currentTime;
    }
  }, []);

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

  // Scroll active line into view
  useEffect(() => {
    if (activeLineRef.current && document.activeElement !== searchInputRef.current) {
      activeLineRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      });
    }
  }, [activeLineIdx]);

  // Export actions
  const handleExport = async (format: 'txt' | 'pdf' | 'srt') => {
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
      alert(`Xuất thất bại: ${err.message}`);
    }
  };

  // Progress Bar Percentage
  const progressPercent = (currentTime / (duration || 2710)) * 100;
  const seekSliderStyle = {
    background: `linear-gradient(to right, var(--primary) 0%, var(--primary) ${progressPercent}%, rgba(255,255,255,0.3) ${progressPercent}%, rgba(255,255,255,0.3) 100%)`
  };

  const isYouTube = !!(videoData?.originalUrl && (videoData.originalUrl.includes('youtube.com') || videoData.originalUrl.includes('youtu.be')));

  const getYouTubeEmbedUrl = (url: string) => {
    let videoId = '';
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
    const match = url.match(regExp);
    if (match && match[2].length === 11) {
      videoId = match[2];
    }
    return videoId ? `https://www.youtube-nocookie.com/embed/${videoId}?enablejsapi=1&autoplay=1` : '';
  };

  if (loading) {
    return (
      <div className="results-page flex-center" style={{ minHeight: '60vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: '3rem', color: 'var(--primary)', marginBottom: '20px' }}></i>
        <p style={{ color: 'var(--text-muted)' }}>Đang tải kết quả tóm tắt từ Database...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-page flex-center animate-fade-in" style={{ minHeight: '60vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '20px' }}>
        <i className="fa-solid fa-triangle-exclamation" style={{ fontSize: '3rem', color: '#ef4444', marginBottom: '20px' }}></i>
        <p style={{ color: '#ef4444', fontWeight: 500, textAlign: 'center', maxWidth: '500px' }}>{error}</p>
        <Link to="/history" className="btn primary" style={{ marginTop: '20px' }}>Quay lại Lịch sử</Link>
      </div>
    );
  }

  return (
    <div className="results-page animate-fade-in">
      <div className="nav-actions">
        <Link to="/" className="btn"><i className="fa-solid fa-house"></i> Trang chủ</Link>
        <Link to="/history" className="btn"><i className="fa-solid fa-clock-rotate-left"></i> Lịch sử</Link>
        <button className="btn" onClick={() => handleExport('pdf')}><i className="fa-solid fa-file-pdf"></i> Xuất PDF</button>
        <button className="btn" onClick={() => handleExport('txt')}><i className="fa-solid fa-file-lines"></i> Xuất TXT</button>
        <Link to={`/qa?videoId=${videoId}`} className="btn primary"><i className="fa-solid fa-comments"></i> Hỏi Đáp RAG</Link>
      </div>

      <div className="layout-grid">
        {/* Left Column: Video & Transcript */}
        <div className="card left-column">
          <h2><i className="fa-brands fa-youtube" style={{ color: '#ef4444' }}></i> Bài Giảng Gốc & Transcript (WhisperX)</h2>
          
          {/* Custom Video Player / YouTube Embed */}
          <div className="video-player-wrapper" style={{ marginBottom: '20px' }}>
            {isYouTube ? (
              <div className="iframe-container" style={{ position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', borderRadius: '8px', boxShadow: 'var(--shadow-md)' }}>
                <iframe
                  src={getYouTubeEmbedUrl(videoData!.originalUrl!)}
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
            ) : (
              <div className={`video-container ${isPlaying ? 'playing' : ''}`} id="videoContainer">
                <video 
                  ref={videoRef}
                  src={videoData?.filePath ? (videoData.filePath.startsWith('http') ? videoData.filePath : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${videoData.filePath}`) : ''} 
                  poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onClick={togglePlay}
                  playsInline
                  loop
                />
                
                {/* Center Play Button Overlay */}
                <button className={`play-overlay-btn ${isPlaying ? 'playing' : ''}`} onClick={togglePlay}>
                  <i className="fa-solid fa-play"></i>
                </button>

                {/* Video Controls Overlay */}
                <div className="controls-overlay">
                  <div className="progress-container">
                    <input 
                      type="range" 
                      className="seek-slider" 
                      min={0} 
                      max={duration || 2710} 
                      step={0.1}
                      value={currentTime} 
                      onChange={handleSeekChange}
                      style={seekSliderStyle}
                    />
                  </div>

                  <div className="controls-row">
                    <div className="controls-left">
                      <button className="control-btn" onClick={togglePlay}>
                        <i className={isPlaying ? "fa-solid fa-pause" : "fa-solid fa-play"}></i>
                      </button>
                      <div className="volume-container">
                        <button className="control-btn" onClick={toggleMute}>
                          <i className={isMuted || volume === 0 ? "fa-solid fa-volume-xmark" : volume < 0.5 ? "fa-solid fa-volume-low" : "fa-solid fa-volume-high"}></i>
                        </button>
                        <input 
                          type="range" 
                          className="volume-slider" 
                          min={0} 
                          max={1} 
                          step={0.05} 
                          value={isMuted ? 0 : volume} 
                          onChange={handleVolumeChange}
                        />
                      </div>
                      <span className="time-display">{formatTime(currentTime)} / {formatTime(duration)}</span>
                    </div>

                    <div className="controls-right">
                      {/* Speed Adjustment */}
                      <div className="speed-container">
                        <button className="control-btn" onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)}>
                          {playbackRate}x
                        </button>
                        <div className={`speed-menu ${isSpeedMenuOpen ? 'show' : ''}`}>
                          {[0.5, 1.0, 1.25, 1.5, 2.0].map((rate) => (
                            <div 
                              key={rate} 
                              className={`speed-option ${playbackRate === rate ? 'active' : ''}`}
                              onClick={() => selectSpeed(rate)}
                            >
                              {rate === 1.0 ? '1.0x (Standard)' : `${rate}x`}
                            </div>
                          ))}
                        </div>
                      </div>
                      <button className="control-btn" onClick={toggleFullscreen}>
                        <i className="fa-solid fa-expand"></i>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Transcript Section */}
          <div className="transcript">
            <h3 className="transcript-header">
              <span>Word-level Timestamps</span>
              <div className="search-wrapper">
                <i className="fa-solid fa-magnifying-glass search-icon"></i>
                <input 
                  ref={searchInputRef}
                  type="text" 
                  placeholder="Tìm kiếm nội dung bài giảng..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </h3>
            
            <div className="transcript-body">
              {transcriptList.map((line, idx) => {
                const isLineActive = idx === activeLineIdx;
                const matchesSearch = searchQuery.trim() === '' || line.text.toLowerCase().includes(searchQuery.toLowerCase());
                
                if (!matchesSearch) return null;

                return (
                  <div 
                    key={idx}
                    ref={isLineActive ? activeLineRef : null}
                    className={`transcript-line ${isLineActive ? 'active' : ''}`}
                    onClick={() => seekTo(line.start)}
                  >
                    <span className="ts">{formatTime(line.start)}</span>
                    <span className="words-container">
                      {line.words?.map((word, wIdx) => {
                        const isWordActive = isLineActive && currentTime >= word.start && currentTime <= word.end;
                        const isSearchMatch = searchQuery.trim() !== '' && word.word.toLowerCase().includes(searchQuery.toLowerCase());

                        return (
                          <span 
                            key={wIdx}
                            className={`word ${isWordActive ? 'word-active' : ''} ${isSearchMatch ? 'search-match' : ''}`}
                            data-start={word.start}
                            data-end={word.end}
                            data-time={word.time}
                            onClick={(e) => {
                              e.stopPropagation(); // Prevent line seek
                              seekTo(word.start);
                            }}
                          >
                            {word.word}{' '}
                          </span>
                        );
                      })}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Summarization & Partition */}
        <div className="card right-column">
          <h2><i className="fa-solid fa-brain" style={{ color: 'var(--primary)' }}></i> Kết Quả Tổng Hợp (Multimodal Fusion)</h2>
          
          <h3 className="section-title-sm"><i className="fa-solid fa-align-left"></i> Tóm Tắt Abstractive (GPT/Gemini)</h3>
          <div className="summary-box" style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
            {summaryData?.summaryText}
          </div>

          <h3 className="section-title-sm"><i className="fa-solid fa-list"></i> Phân Chương Tự Động</h3>
          <div className="chapters">
            {chaptersList.map((chapter, idx) => {
              const isChapterActive = idx === activeChapterIdx;
              const chStartSecs = parseTimeText(chapter.start);

              return (
                <div 
                  key={idx} 
                  className={`chapter-item ${isChapterActive ? 'active' : ''}`}
                  onClick={() => seekTo(chStartSecs)}
                >
                  <span>{chapter.title}</span>
                  <span className="ts">{chapter.start}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Keyframes Gallery (Multimodal Visual Pipeline) */}
      {summaryData?.keyframes && summaryData.keyframes.length > 0 && (
        <div className="card keyframes-card-section" style={{ marginTop: '30px' }}>
          <h2><i className="fa-solid fa-images" style={{ color: 'var(--primary)', marginRight: '10px' }}></i> Keyframes Phân Tích Hình Ảnh (CLIP & BLIP-2)</h2>
          <div className="keyframes-gallery" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px', marginTop: '20px' }}>
            {summaryData.keyframes.map((kf, idx) => (
              <div 
                key={idx} 
                className="keyframe-item-card" 
                style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)', overflow: 'hidden', cursor: 'pointer', transition: 'transform 0.2s' }}
                onClick={() => seekTo(kf.timestamp)}
                onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.02)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
              >
                <div style={{ position: 'relative', paddingBottom: '56.25%', background: '#000' }}>
                  <img 
                    src={`${CONFIG.API_BASE_URL.replace('/api/v1', '')}${kf.imageUrl}`} 
                     alt={kf.description}
                     style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }}
                   />
                 </div>
                 <div style={{ padding: '15px' }}>
                   <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.85rem' }}>
                     <span style={{ color: 'var(--primary)', fontWeight: 500 }}><i className="fa-regular fa-clock"></i> {formatTime(kf.timestamp)}</span>
                     <span style={{ color: 'var(--text-muted)' }}>Độ quan trọng: {Math.round(kf.importanceScore * 100)}%</span>
                   </div>
                   <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-main)', lineHeight: '1.4' }}>{kf.description}</p>
                 </div>
               </div>
             ))}
           </div>
         </div>
       )}
    </div>
  );
};
