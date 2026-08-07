import React, { useRef, useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import type { TranscriptLine, Chapter } from '../types';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';

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
  transcriptText: string;
  transcriptSegments?: TranscriptLine[];
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





export const ResultsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get('videoId') || searchParams.get('id');
  const toast = useToast();
  const { t } = useTranslation();

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
      setSummaryData(summaryRes.data);
      
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
      <div className="bg-[#FAF5FF] text-slate-900 p-6 md:p-margin-desktop max-w-container-max mx-auto space-y-8 min-h-screen">
        {/* Top actions bar skeleton */}
        <div className="flex justify-between items-center py-4 border-b border-slate-200 animate-pulse">
          <div className="flex gap-2">
            <div className="h-8 w-24 bg-slate-200 rounded-lg" />
            <div className="h-8 w-24 bg-slate-200 rounded-lg" />
          </div>
          <div className="h-8 w-32 bg-slate-200 rounded-lg" />
        </div>

        {/* Two-column layout skeleton */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Video player skeleton */}
          <div className="lg:col-span-7 space-y-6">
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
          <div className="lg:col-span-5 space-y-6">
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
    <div className="bg-[#FAF5FF] text-slate-900 p-6 md:p-margin-desktop max-w-container-max mx-auto space-y-8 min-h-screen">
      {/* Top Nav/Actions Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 py-4 border-b border-slate-200/50">
        <div className="flex items-center gap-2">
          <Link to="/" className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-200/50 text-xs font-bold transition-all flex items-center gap-1.5 text-slate-900">
            <span className="material-symbols-outlined text-sm">home</span> {t('results.home')}
          </Link>
          <Link to="/history" className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-200/50 text-xs font-bold transition-all flex items-center gap-1.5 text-slate-900">
            <span className="material-symbols-outlined text-sm">history</span> {t('results.history')}
          </Link>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => handleExport('docx')} className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-200/50 text-xs font-bold transition-all flex items-center gap-1.5 text-slate-900">
            <span className="material-symbols-outlined text-sm">article</span> {t('results.export_docx')}
          </button>
          <button onClick={() => handleExport('pdf')} className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-200/50 text-xs font-bold transition-all flex items-center gap-1.5 text-slate-900">
            <span className="material-symbols-outlined text-sm">picture_as_pdf</span> {t('results.export_pdf')}
          </button>
          <button onClick={() => handleExport('srt')} className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-200/50 text-xs font-bold transition-all flex items-center gap-1.5 text-slate-900">
            <span className="material-symbols-outlined text-sm">subtitles</span> {t('results.export_srt')}
          </button>
          <button onClick={() => handleExport('txt')} className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-200/50 text-xs font-bold transition-all flex items-center gap-1.5 text-slate-900">
            <span className="material-symbols-outlined text-sm">description</span> {t('results.export_txt')}
          </button>
          <Link to={`/qa?videoId=${videoId}`} className="px-4 py-2 bg-slate-900 text-white rounded-lg hover:opacity-90 text-xs font-bold transition-all flex items-center gap-1.5">
            <span className="material-symbols-outlined text-sm">forum</span> {t('results.qa_rag')}
          </Link>
        </div>
      </div>

      {/* Main Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left Column: Video Player & Transcripts */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between shadow-sm space-y-6">
          <div>
            <h2 className="font-headline-md text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary">play_circle</span>
              {t('results.original_video')}
            </h2>
            
            {/* Player Container */}
            <div className="w-full mb-6">
              <div className="relative group aspect-video bg-black rounded-lg overflow-hidden border border-slate-200 shadow-sm" id="videoContainer">
                <video 
                  ref={videoRef}
                  src={videoData?.filePath ? (videoData.filePath.startsWith('http') ? videoData.filePath : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${videoData.filePath}`) : ''} 
                  poster="https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&auto=format&fit=crop&q=60"
                  className="w-full h-full object-cover"
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
        <div className="lg:col-span-5 relative min-h-[400px] lg:min-h-0">
          <div className="absolute inset-0 bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
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

              <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar min-h-0">
                {transcriptList.map((line, idx) => {
                  const isLineActive = idx === activeLineIdx;
                  const matchesSearch = searchQuery.trim() === '' || line.text.toLowerCase().includes(searchQuery.toLowerCase());
                  
                  if (!matchesSearch) return null;

                  return (
                    <div 
                      key={idx}
                      ref={isLineActive ? activeLineRef : null}
                      onClick={() => seekTo(line.start)}
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
                                  seekTo(word.start);
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
           {t('results.ai_insights', 'AI Insights & Analysis')}
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 bg-white border border-slate-200 rounded-xl p-6 shadow-sm ">
          {/* Summary */}
          <div className="lg:col-span-2 flex flex-col">
            <div className="flex flex-col flex-1 min-h-0">
              <h2 className="font-headline-md text-md font-bold text-slate-900 mb-6 flex items-center gap-2 shrink-0">
                <span className="material-symbols-outlined text-primary">subject</span>
                {t('results.content_summary')}
              </h2>
              <div className="flex-1 p-4 bg-[#FAF5FF] border border-slate-200/60 rounded-xl text-sm text-slate-500 leading-relaxed overflow-y-auto custom-scrollbar whitespace-pre-wrap">
                {summaryData?.summaryText}
              </div>
            </div>
          </div>

          {/* Box 2: Chapters */}
          <div className="lg:col-span-1 flex flex-col">
            <div className="flex flex-col flex-1 min-h-0">
              <h2 className="font-headline-md text-md font-bold text-slate-900 mb-6 flex items-center gap-2 shrink-0">
                <span className="material-symbols-outlined text-primary">list</span>
                {t('results.auto_chapters')}
              </h2>
              <div className="flex-1 space-y-2 overflow-y-auto custom-scrollbar pr-2">
                {chaptersList.map((chapter, idx) => {
                  const isChapterActive = idx === activeChapterIdx;
                  const chStartSecs = parseTimeText(chapter.start);

                  return (
                    <div 
                      key={idx} 
                      onClick={() => seekTo(chStartSecs)}
                      className={`flex justify-between items-center p-3 rounded-lg border cursor-pointer transition-colors duration-150 ${
                        isChapterActive 
                          ? 'border-primary bg-primary/10 text-slate-900 font-bold' 
                          : 'border-slate-200/50 bg-[#FAF5FF] text-slate-500 hover:border-primary'
                      }`}
                    >
                      <span className="text-xs truncate max-w-[200px] sm:max-w-xs">{chapter.title}</span>
                      <span className="font-mono-data text-[10px] px-1.5 py-0.5 bg-slate-200/50 rounded text-slate-900 font-bold">
                        {chapter.start}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

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
                  onClick={() => seekTo(kf.timestamp)}
                  className="group border border-slate-200/60 rounded-xl overflow-hidden hover:border-primary transition-all duration-300 cursor-pointer bg-[#FAF5FF] flex flex-col justify-between"
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
      </div>
    </div>
  );
};
