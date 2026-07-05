import React, { useRef, useState, useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import type { TranscriptLine, Chapter } from '../types';
import { api } from '../services/api';
import { CONFIG } from '../config';

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



const mockVideoData: VideoDTO = {
  videoId: "mock-deep-learning-1",
  originalUrl: "https://www.youtube.com/watch?v=5HnS3__1v-o",
  duration: 2710,
  status: "done"
};

const mockSummaryData: SummaryDTO = {
  summaryId: "mock-sum-1",
  videoId: "mock-deep-learning-1",
  summaryText: `Bài giảng giới thiệu tổng quan về học sâu (Deep Learning), bắt đầu từ các khái niệm cơ bản về Perceptron, mạng nơ-ron đa tầng (MLP), cơ chế lan truyền ngược (Backpropagation) và thuật toán tối ưu Gradient Descent.

Tiếp theo, giảng viên phân tích các điểm nghẽn của mạng Recurrent Neural Network (RNN) khi xử lý chuỗi dữ liệu dài (biến mất/bùng nổ gradient).

Cuối cùng, bài giảng đi sâu vào giải pháp cách mạng: kiến trúc Transformer và cơ chế Self-Attention, giải thích tại sao nó cho phép tính toán song song hiệu quả và nắm bắt ngữ cảnh tốt hơn.`,
  transcriptText: "",
  modelUsed: "Gemini 1.5 Flash",
  processingTime: 4.2,
  chapters: [
    {
      startTime: 0,
      endTime: 312,
      title: "1. Giới thiệu học sâu & Cấu trúc Perceptron",
      summary: "Giảng viên giới thiệu cấu trúc cơ bản của một nơ-ron nhân tạo (Perceptron), cách kết hợp các ngõ vào và hàm kích hoạt."
    },
    {
      startTime: 312,
      endTime: 750,
      title: "2. Thuật toán Lan truyền ngược (Backpropagation)",
      summary: "Tìm hiểu cách mạng nơ-ron cập nhật trọng số thông qua đạo hàm chuỗi (chain rule) và tối ưu hóa hàm mất mát."
    },
    {
      startTime: 750,
      endTime: 920,
      title: "3. Hạn chế của RNN & Sự cần thiết của Transformer",
      summary: "Phân tích lỗi mất mát thông tin đối với chuỗi dài trong mạng RNN và giới thiệu giải pháp tự chú ý (self-attention)."
    },
    {
      startTime: 920,
      endTime: 2710,
      title: "4. Kiến trúc Transformer & Cơ chế Self-Attention",
      summary: "Giải thích chi tiết ma trận Query, Key, Value trong cơ chế Self-Attention giúp mô hình tính toán song song."
    }
  ],
  keyframes: [
    {
      timestamp: 15,
      imageUrl: "https://images.unsplash.com/photo-1507668077129-56e32842fceb?w=500",
      description: "Slide mô tả cấu trúc toán học của mạng nơ-ron kết nối đầy đủ (Dense Layer) với hàm kích hoạt ReLU.",
      importanceScore: 0.98
    },
    {
      timestamp: 312,
      imageUrl: "https://images.unsplash.com/photo-1639762681485-074b7f938ba0?w=500",
      description: "Sơ đồ đồ thị tính toán của thuật toán Backpropagation và quy tắc đạo hàm hàm hợp.",
      importanceScore: 0.94
    },
    {
      timestamp: 750,
      imageUrl: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=500",
      description: "Đồ thị so sánh độ dốc thông tin biến mất (Vanishing Gradient) giữa hàm Sigmoid và hàm ReLU.",
      importanceScore: 0.88
    },
    {
      timestamp: 920,
      imageUrl: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=500",
      description: "Kiến trúc chi tiết của khối Encoder trong mô hình Transformer ban đầu (Attention Is All You Need).",
      importanceScore: 0.96
    }
  ]
};

const mockChaptersList: Chapter[] = [
  { start: "00:00", end: "05:12", title: "1. Giới thiệu học sâu & Cấu trúc Perceptron", summary: "Giảng viên giới thiệu cấu trúc cơ bản của một nơ-ron nhân tạo." },
  { start: "05:12", end: "12:30", title: "2. Thuật toán Lan truyền ngược (Backpropagation)", summary: "Cập nhật trọng số mạng nơ-ron." },
  { start: "12:30", end: "15:20", title: "3. Hạn chế của RNN & Sự cần thiết của Transformer", summary: "Phân tích lỗi biến mất gradient." },
  { start: "15:20", end: "45:10", title: "4. Kiến trúc Transformer & Cơ chế Self-Attention", summary: "Giải thích cơ chế Self-Attention." }
];

const mockTranscriptList: TranscriptLine[] = [
  {
    speaker: "SPEAKER_00",
    start: 0,
    end: 15,
    text: "Chào mừng các bạn đến với bài giảng giới thiệu về Học Sâu và Trí tuệ Nhân tạo đa phương thức.",
    words: [
      { word: "Chào", start: 0.2, end: 0.8, time: "00:00" },
      { word: "mừng", start: 0.8, end: 1.2, time: "00:00" },
      { word: "các", start: 1.2, end: 1.5, time: "00:01" },
      { word: "bạn", start: 1.5, end: 1.8, time: "00:01" },
      { word: "đến", start: 1.8, end: 2.1, time: "00:01" },
      { word: "với", start: 2.1, end: 2.4, time: "00:02" },
      { word: "bài", start: 2.4, end: 2.7, time: "00:02" },
      { word: "giảng", start: 2.7, end: 3.2, time: "00:02" },
      { word: "giới", start: 3.2, end: 3.6, time: "00:03" },
      { word: "thiệu", start: 3.6, end: 4.0, time: "00:03" },
      { word: "về", start: 4.0, end: 4.3, time: "00:04" },
      { word: "Học", start: 4.3, end: 4.8, time: "00:04" },
      { word: "Sâu", start: 4.8, end: 5.4, time: "00:04" },
      { word: "và", start: 5.4, end: 5.8, time: "00:05" },
      { word: "Trí", start: 5.8, end: 6.2, time: "00:05" },
      { word: "tuệ", start: 6.2, end: 6.6, time: "00:06" },
      { word: "Nhân", start: 6.6, end: 7.0, time: "00:06" },
      { word: "tạo.", start: 7.0, end: 7.8, time: "00:07" }
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 15,
    end: 312,
    text: "Hôm nay chúng ta sẽ đi sâu vào cấu trúc cơ bản của mạng nơ-ron nhân tạo thông qua mô hình Perceptron.",
    words: [
      { word: "Hôm", start: 15.1, end: 15.5, time: "00:15" },
      { word: "nay", start: 15.5, end: 15.8, time: "00:15" },
      { word: "chúng", start: 15.8, end: 16.1, time: "00:15" },
      { word: "ta", start: 16.1, end: 16.4, time: "00:16" },
      { word: "sẽ", start: 16.4, end: 16.8, time: "00:16" },
      { word: "đi", start: 16.8, end: 17.1, time: "00:16" },
      { word: "sâu", start: 17.1, end: 17.5, time: "00:17" },
      { word: "vào", start: 17.5, end: 17.9, time: "00:17" },
      { word: "cấu", start: 17.9, end: 18.2, time: "00:17" },
      { word: "trúc", start: 18.2, end: 18.6, time: "00:18" },
      { word: "cơ", start: 18.6, end: 18.9, time: "00:18" },
      { word: "bản", start: 18.9, end: 19.3, time: "00:18" },
      { word: "của", start: 19.3, end: 19.6, time: "00:19" },
      { word: "mạng", start: 19.6, end: 20.0, time: "00:19" },
      { word: "nơ-ron", start: 20.0, end: 20.6, time: "00:20" },
      { word: "nhân", start: 20.6, end: 21.0, time: "00:20" },
      { word: "tạo.", start: 21.0, end: 21.6, time: "00:21" }
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 312,
    end: 750,
    text: "Cơ chế cốt lõi để cập nhật trọng số trong mạng nơ-ron là thuật toán lan truyền ngược Backpropagation.",
    words: [
      { word: "Cơ", start: 312.2, end: 312.6, time: "05:12" },
      { word: "chế", start: 312.6, end: 312.9, time: "05:12" },
      { word: "cốt", start: 312.9, end: 313.2, time: "05:12" },
      { word: "lõi", start: 313.2, end: 313.5, time: "05:13" },
      { word: "để", start: 313.5, end: 313.8, time: "05:13" },
      { word: "cập", start: 313.8, end: 314.1, time: "05:13" },
      { word: "nhật", start: 314.1, end: 314.4, time: "05:14" },
      { word: "trọng", start: 314.4, end: 314.8, time: "05:14" },
      { word: "số", start: 314.8, end: 315.2, time: "05:14" },
      { word: "trong", start: 315.2, end: 315.5, time: "05:15" },
      { word: "mạng", start: 315.5, end: 315.8, time: "05:15" },
      { word: "nơ-ron", start: 315.8, end: 316.3, time: "05:15" },
      { word: "là", start: 316.3, end: 316.6, time: "05:16" },
      { word: "thuật", start: 316.6, end: 317.0, time: "05:16" },
      { word: "toán", start: 317.0, end: 317.4, time: "05:17" },
      { word: "lan", start: 317.4, end: 317.8, time: "05:17" },
      { word: "truyền", start: 317.8, end: 318.2, time: "05:17" },
      { word: "ngược.", start: 318.2, end: 319.0, time: "05:18" }
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 750,
    end: 920,
    text: "Đối với chuỗi dữ liệu rất dài, mạng tuần tự RNN thường gặp lỗi triệt tiêu hoặc bùng nổ gradient.",
    words: [
      { word: "Đối", start: 750.1, end: 750.5, time: "12:30" },
      { word: "với", start: 750.5, end: 750.8, time: "12:30" },
      { word: "chuỗi", start: 750.8, end: 751.2, time: "12:30" },
      { word: "dữ", start: 751.2, end: 751.5, time: "12:31" },
      { word: "liệu", start: 751.5, end: 751.8, time: "12:31" },
      { word: "rất", start: 751.8, end: 752.2, time: "12:32" },
      { word: "dài,", start: 752.2, end: 752.6, time: "12:32" },
      { word: "mạng", start: 752.6, end: 753.0, time: "12:33" },
      { word: "tuần", start: 753.0, end: 753.4, time: "12:33" },
      { word: "tự", start: 753.4, end: 753.8, time: "12:33" },
      { word: "RNN", start: 753.8, end: 754.4, time: "12:34" },
      { word: "thường", start: 754.4, end: 754.8, time: "12:34" },
      { word: "gặp", start: 754.8, end: 755.2, time: "12:34" },
      { word: "lỗi", start: 755.2, end: 755.5, time: "12:35" },
      { word: "triệt", start: 755.5, end: 755.9, time: "12:35" },
      { word: "tiêu", start: 755.9, end: 756.3, time: "12:36" },
      { word: "hoặc", start: 756.3, end: 756.7, time: "12:36" },
      { word: "bùng", start: 756.7, end: 757.1, time: "12:37" },
      { word: "nổ.", start: 757.1, end: 757.8, time: "12:37" }
    ]
  },
  {
    speaker: "SPEAKER_00",
    start: 920,
    end: 2710,
    text: "Để giải quyết vấn đề này, kiến trúc Transformer đã ra đời sử dụng hoàn toàn cơ chế tự chú ý Self-Attention.",
    words: [
      { word: "Để", start: 920.1, end: 920.5, time: "15:20" },
      { word: "giải", start: 920.5, end: 920.8, time: "15:20" },
      { word: "quyết", start: 920.8, end: 921.2, time: "15:21" },
      { word: "vấn", start: 921.2, end: 921.5, time: "15:21" },
      { word: "đề", start: 921.5, end: 921.8, time: "15:21" },
      { word: "này,", start: 921.8, end: 922.2, time: "15:22" },
      { word: "kiến", start: 922.2, end: 922.6, time: "15:22" },
      { word: "trúc", start: 922.6, end: 923.0, time: "15:23" },
      { word: "Transformer", start: 923.0, end: 923.8, time: "15:23" },
      { word: "đã", start: 923.8, end: 924.1, time: "15:24" },
      { word: "ra", start: 924.1, end: 924.4, time: "15:24" },
      { word: "đời", start: 924.4, end: 924.8, time: "15:24" },
      { word: "sử", start: 924.8, end: 925.1, time: "15:25" },
      { word: "dụng", start: 925.1, end: 925.4, time: "15:25" },
      { word: "hoàn", start: 925.4, end: 925.8, time: "15:25" },
      { word: "toàn", start: 925.8, end: 926.2, time: "15:26" },
      { word: "cơ", start: 926.2, end: 926.5, time: "15:26" },
      { word: "chế", start: 926.5, end: 926.8, time: "15:26" },
      { word: "tự", start: 926.8, end: 927.1, time: "15:27" },
      { word: "chú", start: 927.1, end: 927.4, time: "15:27" },
      { word: "ý.", start: 927.4, end: 928.0, time: "15:27" }
    ]
  }
];

export const ResultsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const videoId = searchParams.get('videoId') || searchParams.get('id');

  const [loading, setLoading] = useState(false);
  const [error] = useState<string | null>(null);

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
      setVideoData(mockVideoData);
      setDuration(mockVideoData.duration || 2710);
      setSummaryData(mockSummaryData);
      setChaptersList(mockChaptersList);
      setTranscriptList(mockTranscriptList);
      return;
    }

    setLoading(true);
    
    Promise.all([
      api.getVideo(videoId),
      api.getSummary(videoId)
    ])
    .then(([videoRes, summaryRes]) => {
      if (videoRes.success && videoRes.data) {
        setVideoData(videoRes.data);
        setDuration(videoRes.data.duration || 2710);
      } else {
        setVideoData(mockVideoData);
        setDuration(mockVideoData.duration || 2710);
      }

      if (summaryRes.success && summaryRes.data) {
        setSummaryData(summaryRes.data);
        
        if (summaryRes.data.chapters && summaryRes.data.chapters.length > 0) {
          const mappedChapters = summaryRes.data.chapters.map((c: any) => ({
            start: formatTime(c.startTime),
            end: formatTime(c.endTime),
            title: c.title,
            summary: c.summary
          }));
          setChaptersList(mappedChapters);
        } else {
          setChaptersList(mockChaptersList);
        }

        if (summaryRes.data.transcriptText) {
          const sentences = summaryRes.data.transcriptText
            .split(/(?<=[.!?])\s+/)
            .filter((s: string) => s.trim().length > 0);
          
          const totalDuration = videoRes.data?.duration || 2710;
          const sentenceDuration = sentences.length > 0 ? totalDuration / sentences.length : 10;
          
          const mappedTranscript = sentences.map((sentence: string, idx: number) => {
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
          setTranscriptList(mappedTranscript);
        } else {
          setTranscriptList(mockTranscriptList);
        }
      } else {
        setSummaryData(mockSummaryData);
        setChaptersList(mockChaptersList);
        setTranscriptList(mockTranscriptList);
      }
      setLoading(false);
    })
    .catch(err => {
      console.warn("Failed to load real video/summary, falling back to mock data.", err);
      setVideoData(mockVideoData);
      setDuration(mockVideoData.duration || 2710);
      setSummaryData(mockSummaryData);
      setChaptersList(mockChaptersList);
      setTranscriptList(mockTranscriptList);
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
    let yid = '';
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*/;
    const match = url.match(regExp);
    if (match && match[2].length === 11) {
      yid = match[2];
    }
    return yid ? `https://www.youtube-nocookie.com/embed/${yid}?enablejsapi=1&autoplay=1` : '';
  };

  if (loading) {
    return (
      <div className="flex flex-col justify-center items-center h-[60vh] bg-background text-on-surface">
        <span className="material-symbols-outlined text-[48px] text-vibrant-cyan animate-spin mb-4">autorenew</span>
        <p className="text-secondary text-sm font-semibold">Đang tải kết quả tóm tắt từ Database...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center h-[60vh] bg-background p-6 text-center">
        <span className="material-symbols-outlined text-[48px] text-error mb-4">warning</span>
        <p className="text-error font-bold max-w-md mb-6 text-sm">{error}</p>
        <Link 
          to="/history" 
          className="px-6 py-2 bg-deep-navy text-white rounded hover:bg-primary font-semibold text-sm transition-all"
        >
          Quay lại Lịch sử
        </Link>
      </div>
    );
  }

  return (
    <div className="bg-background text-on-surface p-6 md:p-margin-desktop max-w-container-max mx-auto space-y-8 min-h-screen">
      {/* Top Nav/Actions Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 py-4 border-b border-outline-variant/30">
        <div className="flex items-center gap-2">
          <Link to="/" className="px-4 py-2 border border-outline-variant rounded-lg hover:bg-surface-container-high text-xs font-bold transition-all flex items-center gap-1.5 text-deep-navy">
            <span className="material-symbols-outlined text-sm">home</span> Trang chủ
          </Link>
          <Link to="/history" className="px-4 py-2 border border-outline-variant rounded-lg hover:bg-surface-container-high text-xs font-bold transition-all flex items-center gap-1.5 text-deep-navy">
            <span className="material-symbols-outlined text-sm">history</span> Lịch sử
          </Link>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => handleExport('pdf')} className="px-4 py-2 border border-outline-variant rounded-lg hover:bg-surface-container-high text-xs font-bold transition-all flex items-center gap-1.5 text-deep-navy">
            <span className="material-symbols-outlined text-sm">picture_as_pdf</span> Xuất PDF
          </button>
          <button onClick={() => handleExport('txt')} className="px-4 py-2 border border-outline-variant rounded-lg hover:bg-surface-container-high text-xs font-bold transition-all flex items-center gap-1.5 text-deep-navy">
            <span className="material-symbols-outlined text-sm">description</span> Xuất TXT
          </button>
          <Link to={`/qa?videoId=${videoId}`} className="px-4 py-2 bg-deep-navy text-white rounded-lg hover:opacity-90 text-xs font-bold transition-all flex items-center gap-1.5">
            <span className="material-symbols-outlined text-sm">forum</span> Hỏi Đáp RAG
          </Link>
        </div>
      </div>

      {/* Main Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Video Player & Transcripts */}
        <div className="bg-surface border border-outline-variant rounded-xl p-6 flex flex-col justify-between shadow-sm space-y-6">
          <div>
            <h2 className="font-headline-md text-lg font-bold text-deep-navy mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-error">play_circle</span>
              Bài Giảng Gốc &amp; Transcript (WhisperX)
            </h2>
            
            {/* Player Container */}
            <div className="w-full mb-6">
              <div className="relative group aspect-video bg-video-background rounded-lg overflow-hidden border border-outline-variant shadow-sm" id="videoContainer">
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
                  <button onClick={togglePlay} className="absolute inset-0 m-auto w-14 h-14 bg-vibrant-cyan text-white rounded-full flex items-center justify-center hover:scale-110 transition-transform shadow-lg z-20">
                    <span className="material-symbols-outlined text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>play_arrow</span>
                  </button>
                )}

                {/* HTML Video Custom Controls Overlay */}
                <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/85 to-transparent flex flex-col gap-2 z-20 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <div className="w-full flex items-center">
                    <input 
                      type="range" 
                      className="w-full h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-vibrant-cyan" 
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
                      <button onClick={togglePlay} className="hover:text-vibrant-cyan transition-colors">
                        <span className="material-symbols-outlined text-sm">{isPlaying ? "pause" : "play_arrow"}</span>
                      </button>
                      <div className="flex items-center gap-1.5">
                        <button onClick={toggleMute} className="hover:text-vibrant-cyan transition-colors">
                          <span className="material-symbols-outlined text-sm">{isMuted || volume === 0 ? "volume_off" : "volume_up"}</span>
                        </button>
                        <input 
                          type="range" 
                          className="w-12 h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-vibrant-cyan volume-slider" 
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
                        <button onClick={() => setIsSpeedMenuOpen(!isSpeedMenuOpen)} className="hover:text-vibrant-cyan transition-colors px-1 bg-slate-800 rounded font-mono-data text-[10px]">
                          {playbackRate}x
                        </button>
                        {isSpeedMenuOpen && (
                          <div className="absolute bottom-6 right-0 bg-slate-900 border border-slate-700 rounded shadow-md flex flex-col overflow-hidden text-[10px] z-50">
                            {[0.5, 1.0, 1.25, 1.5, 2.0].map((rate) => (
                              <button 
                                key={rate} 
                                onClick={() => selectSpeed(rate)}
                                className={`px-3 py-1 text-left hover:bg-vibrant-cyan hover:text-deep-navy font-semibold ${playbackRate === rate ? 'text-vibrant-cyan' : 'text-white'}`}
                              >
                                {rate}x
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                      <button onClick={toggleFullscreen} className="hover:text-vibrant-cyan transition-colors">
                        <span className="material-symbols-outlined text-sm">fullscreen</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Transcript Sync Area */}
            <div className="border border-outline-variant/60 rounded-xl overflow-hidden bg-background">
              <div className="flex items-center justify-between px-4 py-3 bg-surface border-b border-outline-variant/60">
                <span className="font-label-md text-xs font-bold uppercase tracking-wider text-outline">Timestamps &amp; Transcript</span>
                <div className="flex items-center glass-panel px-3 py-1.5 rounded-full border border-outline-variant/60 focus-within:border-vibrant-cyan transition-all max-w-[200px] sm:max-w-xs bg-white">
                  <span className="material-symbols-outlined text-outline text-sm">search</span>
                  <input 
                    ref={searchInputRef}
                    className="bg-transparent border-none focus:ring-0 text-xs w-full px-2 outline-none" 
                    placeholder="Tìm kiếm từ khóa..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    type="text"
                  />
                </div>
              </div>

              <div className="h-[250px] overflow-y-auto p-4 space-y-3 custom-scrollbar">
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
                        isLineActive ? 'bg-secondary-container/20 border-l-2 border-vibrant-cyan' : 'hover:bg-surface-container-low'
                      }`}
                    >
                      <span className={`font-mono-data text-[10px] px-1.5 py-0.5 rounded font-bold shrink-0 mt-0.5 ${
                        isLineActive ? 'bg-vibrant-cyan text-white' : 'bg-surface-container-high text-deep-navy'
                      }`}>
                        {formatTime(line.start)}
                      </span>
                      <p className="text-xs text-deep-navy leading-relaxed flex-1 flex flex-wrap">
                        {line.words?.map((word, wIdx) => {
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
                                  ? 'bg-vibrant-cyan text-white font-bold' 
                                  : isSearchMatch 
                                    ? 'bg-status-warning/20 text-deep-navy border-b border-status-warning' 
                                    : 'hover:text-vibrant-cyan'
                              }`}
                            >
                              {word.word}
                            </span>
                          );
                        })}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Summarization & Partition */}
        <div className="bg-surface border border-outline-variant rounded-xl p-6 flex flex-col justify-between shadow-sm space-y-6">
          <div className="space-y-6">
            <h2 className="font-headline-md text-lg font-bold text-deep-navy flex items-center gap-2">
              <span className="material-symbols-outlined text-vibrant-cyan">hub</span>
              Kết Quả Tóm Tắt (AI Fusion)
            </h2>
            
            <div>
              <h3 className="text-sm font-bold text-deep-navy mb-2 flex items-center gap-1">
                <span className="material-symbols-outlined text-vibrant-cyan text-base">subject</span>
                Tóm tắt nội dung bài giảng
              </h3>
              <div className="p-4 bg-background border border-outline-variant/60 rounded-xl text-sm text-secondary leading-relaxed max-h-[220px] overflow-y-auto custom-scrollbar whitespace-pre-wrap">
                {summaryData?.summaryText}
              </div>
            </div>

            <div>
              <h3 className="text-sm font-bold text-deep-navy mb-2 flex items-center gap-1">
                <span className="material-symbols-outlined text-vibrant-cyan text-base">list</span>
                Phân chương bài học tự động
              </h3>
              <div className="space-y-2 max-h-[220px] overflow-y-auto custom-scrollbar pr-2">
                {chaptersList.map((chapter, idx) => {
                  const isChapterActive = idx === activeChapterIdx;
                  const chStartSecs = parseTimeText(chapter.start);

                  return (
                    <div 
                      key={idx} 
                      onClick={() => seekTo(chStartSecs)}
                      className={`flex justify-between items-center p-3 rounded-lg border cursor-pointer transition-colors duration-150 ${
                        isChapterActive 
                          ? 'border-vibrant-cyan bg-secondary-container/20 text-deep-navy font-bold' 
                          : 'border-outline-variant/50 bg-background text-secondary hover:border-vibrant-cyan'
                      }`}
                    >
                      <span className="text-xs truncate max-w-[200px] sm:max-w-xs">{chapter.title}</span>
                      <span className="font-mono-data text-[10px] px-1.5 py-0.5 bg-surface-container-high rounded text-deep-navy font-bold">
                        {chapter.start}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Keyframes Gallery Section */}
      {summaryData?.keyframes && summaryData.keyframes.length > 0 && (
        <div className="bg-surface border border-outline-variant rounded-xl p-6 shadow-sm space-y-6">
          <h2 className="font-headline-md text-lg font-bold text-deep-navy flex items-center gap-2">
            <span className="material-symbols-outlined text-vibrant-cyan">image_search</span>
            Keyframes trích xuất từ Video bài giảng (CLIP)
          </h2>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {summaryData.keyframes.map((kf, idx) => (
              <div 
                key={idx} 
                onClick={() => seekTo(kf.timestamp)}
                className="group border border-outline-variant/60 rounded-xl overflow-hidden hover:border-vibrant-cyan transition-all duration-300 cursor-pointer bg-background flex flex-col justify-between"
              >
                <div className="relative aspect-video w-full bg-black overflow-hidden shrink-0">
                  <img 
                    src={`${CONFIG.API_BASE_URL.replace('/api/v1', '')}${kf.imageUrl}`} 
                    alt={kf.description}
                    className="absolute inset-0 w-full h-full object-contain transform group-hover:scale-105 transition-transform duration-500"
                  />
                  <div className="absolute top-2 left-2 z-10 px-1.5 py-0.5 bg-deep-navy/85 text-white font-mono-data text-[9px] rounded font-bold">
                    {formatTime(kf.timestamp)}
                  </div>
                </div>
                
                <div className="p-4 flex-1 flex flex-col justify-between space-y-2">
                  <p className="text-xs text-deep-navy leading-normal line-clamp-3">
                    {kf.description}
                  </p>
                  <div className="flex justify-between items-center pt-2 border-t border-outline-variant/30 text-[10px] text-secondary">
                    <span className="font-bold text-vibrant-cyan">Độ quan trọng</span>
                    <span className="font-mono-data font-bold bg-surface-container-high px-1.5 py-0.5 rounded text-deep-navy">
                      {Math.round(kf.importanceScore * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
