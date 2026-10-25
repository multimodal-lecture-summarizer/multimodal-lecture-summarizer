import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import Chart from 'chart.js/auto';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { VideoStatus } from '../types';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';
import { PerformanceCorrelationWidget } from '../components/PerformanceCorrelationWidget';
import { EvaluationBenchmarkView } from '../components/EvaluationBenchmarkView';
import { useTranslation } from 'react-i18next';
import { parseUTCDate } from '../utils/dateUtils';

const PaginationControl: React.FC<{
  currentPage: number;
  totalItems: number;
  limit: number;
  onPageChange: (page: number) => void;
}> = ({ currentPage, totalItems, limit, onPageChange }) => {
  const { t } = useTranslation();
  const totalPages = Math.max(1, Math.ceil(totalItems / limit));
  if (totalPages <= 1) return null;

  // Compute page numbers to display
  const pageNumbers: (number | string)[] = [];
  const addPage = (p: number) => pageNumbers.push(p);

  addPage(1);
  if (currentPage > 3) pageNumbers.push('...');

  for (let i = Math.max(2, currentPage - 1); i <= Math.min(totalPages - 1, currentPage + 1); i++) {
    addPage(i);
  }

  if (currentPage < totalPages - 2) pageNumbers.push('...');
  if (totalPages > 1) addPage(totalPages);

  // Filter out duplicates and format clean pages list
  const uniquePages = pageNumbers.filter((v, i, a) => a.indexOf(v) === i);

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t-2 border-slate-100 text-xs font-bold text-slate-900 shrink-0">
      <span className="text-slate-500">
        {t('admin.showing')} {Math.min(totalItems, (currentPage - 1) * limit + 1)} - {Math.min(currentPage * limit, totalItems)} {t('admin.of')} {totalItems} {t('admin.total')}
      </span>
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          disabled={currentPage === 1}
          onClick={() => onPageChange(currentPage - 1)}
          className="px-2.5 py-1.5 border-2 border-slate-200 rounded-lg hover:bg-slate-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed text-slate-900"
        >
          {t('admin.prev')}
        </button>
        {uniquePages.map((page, idx) => {
          if (page === '...') {
            return (
              <span key={`dots-${idx}`} className="px-2 text-slate-400">
                ...
              </span>
            );
          }
          return (
            <button
              key={`page-${page}`}
              type="button"
              onClick={() => onPageChange(page as number)}
              className={`px-3 py-1.5 rounded-lg transition-colors border-2 ${
                currentPage === page
                  ? 'bg-slate-900 border-slate-900 text-white'
                  : 'border-slate-200 hover:bg-slate-100 text-slate-900'
              }`}
            >
              {page}
            </button>
          );
        })}
        <button
          type="button"
          disabled={currentPage === totalPages}
          onClick={() => onPageChange(currentPage + 1)}
          className="px-2.5 py-1.5 border-2 border-slate-200 rounded-lg hover:bg-slate-100 transition-colors disabled:opacity-40 disabled:cursor-not-allowed text-slate-900"
        >
          {t('admin.next')}
        </button>
      </div>
    </div>
  );
};

const formatTime = (secs: number) => {
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
};

const getSystemLogs = (job: any) => {
  const startStr = job.startedAt ? parseUTCDate(job.startedAt)!.toISOString() : new Date().toISOString();
  let logs = `[INFO] ${startStr} - Khởi chạy pipeline phân tích đa phương tiện cho Video ${job.videoId || 'N/A'}.\n[INFO] Khởi tạo Faster-Whisper để nhận dạng giọng nói...`;
  
  const isCompleted = job.status === 'completed' || job.status === 'done' || job.status === 'SUCCESS';
  const isFailed = job.status === 'failed' || job.status === 'FAILED';
  
  if (isCompleted) {
    const endStr = job.completedAt ? parseUTCDate(job.completedAt)!.toISOString() : new Date().toISOString();
    return logs + `\n[INFO] Hoàn thành nhận dạng giọng nói.` +
                  `\n[INFO] Khởi chạy trích xuất Keyframes bằng CLIP...` +
                  `\n[INFO] Đọc dữ liệu ảnh và xếp hạng độ quan trọng slide...` +
                  `\n[INFO] Gửi văn bản sang Gemini 1.5 để tạo tóm tắt chương...` +
                  `\n[INFO] Lưu trữ vector embeddings thành công vào ChromaDB.` +
                  `\n[INFO] ${endStr} - Tác vụ hoàn thành thành công.`;
  }
  
  if (isFailed) {
    return logs + `\n[ERROR] Tiến trình gặp sự cố khi xử lý video.\n[ERROR] Chi tiết lỗi: ${job.errorLog || 'Không xác định'}`;
  }
  
  if (job.startedAt) {
    const elapsed = (Date.now() - parseUTCDate(job.startedAt)!.getTime()) / 1000;
    if (elapsed > 2) {
      logs += `\n[INFO] Đang nhận dạng giọng nói sang văn bản...`;
    }
    if (elapsed > 6) {
      logs += `\n[INFO] Hoàn thành bóc băng lời thoại. Bắt đầu trích xuất Keyframes bằng CLIP...`;
    }
    if (elapsed > 12) {
      logs += `\n[INFO] Đang phân tích mức độ quan trọng các khung hình slide...`;
    }
    if (elapsed > 18) {
      logs += `\n[INFO] Gửi văn bản và hình ảnh sang Gemini để tạo tóm tắt thông minh...`;
    }
    if (elapsed > 24) {
      logs += `\n[INFO] Tiến trình RAG: Lưu trữ vector embeddings vào ChromaDB...`;
    }
  } else {
    logs += `\n[INFO] Đang chạy Faster-Whisper và CLIP...`;
  }
  
  return logs;
};

interface UserItem {
  id: string;
  email: string;
  role: string;
  active: boolean;
  joined: string;
}

const AVAILABLE_FORMATS = ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv'];

const AVAILABLE_CODECS = [
  { id: 'h264', label: 'H.264 / AVC', desc: 'Tương thích cao nhất' },
  { id: 'hevc', label: 'H.265 / HEVC', desc: 'Hiệu suất cao 4K / HD' },
  { id: 'av1', label: 'AV1', desc: 'Thế hệ mới, mở' },
  { id: 'vp9', label: 'VP9', desc: 'Chuẩn Web / YouTube' },
  { id: 'vp8', label: 'VP8', desc: 'WebM kế thừa' },
  { id: 'mpeg4', label: 'MPEG-4', desc: 'Tương thích truyền thống' },
];

const RESOLUTION_PRESETS = [
  { id: '360p', label: '360p (SD - 640 × 360)', width: 640, height: 360 },
  { id: '480p', label: '480p (SD - 854 × 480)', width: 854, height: 480 },
  { id: '720p', label: '720p (HD - 1280 × 720)', width: 1280, height: 720 },
  { id: '1080p', label: '1080p (Full HD - 1920 × 1080)', width: 1920, height: 1080 },
  { id: '1440p', label: '1440p (2K QHD - 2560 × 1440)', width: 2560, height: 1440 },
  { id: '2160p', label: '2160p (4K UHD - 3840 × 2160)', width: 3840, height: 2160 },
];

const DURATION_PRESETS = [
  { label: '15 phút (900s)', value: 900 },
  { label: '30 phút (1.800s)', value: 1800 },
  { label: '45 phút (2.700s)', value: 2700 },
  { label: '60 phút / 1 giờ (3.600s)', value: 3600 },
  { label: '120 phút / 2 giờ (7.200s)', value: 7200 },
  { label: '180 phút / 3 giờ (10.800s)', value: 10800 },
];

const FILE_SIZE_PRESETS = [
  { label: '100 MB', value: 100 },
  { label: '250 MB', value: 250 },
  { label: '500 MB (Tiêu chuẩn)', value: 500 },
  { label: '1024 MB (1 GB)', value: 1024 },
  { label: '2048 MB (2 GB)', value: 2048 },
];

export const AdminPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'stats' | 'users' | 'videos' | 'celery' | 'system-videos' | 'system-jobs' | 'evaluation'>('stats');
  const toast = useToast();
  const { t } = useTranslation();

  // Chart refs
  const barChartRef = useRef<HTMLCanvasElement>(null);
  const doughnutChartRef = useRef<HTMLCanvasElement>(null);
  const barInstance = useRef<Chart | null>(null);
  const doughnutInstance = useRef<Chart | null>(null);

  // Video Standards state
  const [maxDuration, setMaxDuration] = useState(3600);
  const [allowedFormats, setAllowedFormats] = useState('mp4,avi,mkv');
  const [allowedCodecs, setAllowedCodecs] = useState('h264,hevc,av1,vp9');

  const getFormatsList = (val: string | string[]): string[] => {
    if (Array.isArray(val)) return val.map(f => String(f).trim().toLowerCase().replace(/^\./, '')).filter(Boolean);
    return String(val || '').split(',').map(f => f.trim().toLowerCase().replace(/^\./, '')).filter(Boolean);
  };

  const getCodecsList = (val: string | string[]): string[] => {
    if (Array.isArray(val)) return val.map(c => String(c).trim().toLowerCase()).filter(Boolean);
    return String(val || '').split(',').map(c => c.trim().toLowerCase()).filter(Boolean);
  };

  const [maxFileSize, setMaxFileSize] = useState(500);
  const [minAudioQuality, setMinAudioQuality] = useState(0.0);
  const [minWidth, setMinWidth] = useState<number>(640);
  const [minHeight, setMinHeight] = useState<number>(360);
  const [maxWidth, setMaxWidth] = useState<number>(3840);
  const [maxHeight, setMaxHeight] = useState<number>(2160);
  const [standardsLoading, setStandardsLoading] = useState(false);
  const [standardsMsg, setStandardsMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Users state
  const [users, setUsers] = useState<UserItem[]>([]);
  const [usersLoading, setUsersLoading] = useState(false);
  const [usersPage, setUsersPage] = useState(1);
  const [usersTotal, setUsersTotal] = useState<number | null>(null);

  // Celery queue tasks
  const [queueJobs, setQueueJobs] = useState<any[]>([]);
  const [queueLoading, setQueueLoading] = useState(false);
  const [queuePage, setQueuePage] = useState(1);
  const [queueTotal, setQueueTotal] = useState<number | null>(null);

  // System Videos and Jobs
  const [allVideos, setAllVideos] = useState<any[]>([]);
  const [videoLoading, setVideoLoading] = useState(false);
  const [systemVideosPage, setSystemVideosPage] = useState(1);
  const [systemVideosTotal, setSystemVideosTotal] = useState<number | null>(null);
  
  const [videosCompleted, setVideosCompleted] = useState<number | null>(null);
  const [videosFailed, setVideosFailed] = useState<number | null>(null);
  const [videosProcessing, setVideosProcessing] = useState<number | null>(null);

  const [allJobs, setAllJobs] = useState<any[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [systemJobsPage, setSystemJobsPage] = useState(1);
  const [systemJobsTotal, setSystemJobsTotal] = useState<number | null>(null);
  const [jobsTrigger, setJobsTrigger] = useState(0);

  const [jobsCompleted, setJobsCompleted] = useState<number | null>(null);
  const [jobsFailed, setJobsFailed] = useState<number | null>(null);
  const [jobsProcessing, setJobsProcessing] = useState<number | null>(null);

  // Dashboard Stats Videos State
  const [statsVideos, setStatsVideos] = useState<any[]>([]);
  const [statsLoading, setStatsLoading] = useState(false);
  const [dashboardStats, setDashboardStats] = useState<any>(null);

  // Selection and Modal States for Detail Views
  const [selectedVideo, setSelectedVideo] = useState<any | null>(null);
  const [videoSummary, setVideoSummary] = useState<any | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [videoModalOpen, setVideoModalOpen] = useState(false);

  const [selectedJob, setSelectedJob] = useState<any | null>(null);
  const [jobModalOpen, setJobModalOpen] = useState(false);

  const openVideoDetail = async (video: any) => {
    setSelectedVideo(video);
    setVideoModalOpen(true);
    if (video.status?.toLowerCase() === 'done') {
      setLoadingSummary(true);
      try {
        const res = await api.getSummary(video.videoId);
        if (res.success && res.data) {
          setVideoSummary(res.data);
        } else {
          setVideoSummary(null);
        }
      } catch (err) {
        console.warn("Failed to load video summary details", err);
        setVideoSummary(null);
      } finally {
        setLoadingSummary(false);
      }
    } else {
      setVideoSummary(null);
    }
  };

  const openJobDetail = async (jobOrVideo: any) => {
    if (jobOrVideo && jobOrVideo.videoId && !jobOrVideo.jobId) {
      // If it's a video object from celery queue tab, try to fetch its active job
      try {
        const res = await api.getJobStatus(jobOrVideo.videoId);
        if (res.success && res.data) {
          setSelectedJob(res.data);
        } else {
          // Fallback to a mock job constructed from video properties
          setSelectedJob({
            jobId: 'N/A',
            videoId: jobOrVideo.videoId,
            jobType: 'summarize',
            status: jobOrVideo.status === 'done' ? 'completed' : jobOrVideo.status === 'failed' ? 'failed' : 'running',
            startedAt: jobOrVideo.uploadedAt,
            completedAt: jobOrVideo.status === 'done' ? jobOrVideo.uploadedAt : null,
            errorLog: null
          });
        }
      } catch (err) {
        console.warn("Failed to fetch job status, using fallback", err);
        setSelectedJob({
          jobId: 'N/A',
          videoId: jobOrVideo.videoId,
          jobType: 'summarize',
          status: jobOrVideo.status === 'done' ? 'completed' : jobOrVideo.status === 'failed' ? 'failed' : 'running',
          startedAt: jobOrVideo.uploadedAt,
          completedAt: jobOrVideo.status === 'done' ? jobOrVideo.uploadedAt : null,
          errorLog: null
        });
      }
    } else {
      setSelectedJob(jobOrVideo);
    }
    setJobModalOpen(true);
  };

  // Chart initialization (only when on stats tab)
  useEffect(() => {
    if (activeTab !== 'stats' || !dashboardStats) {
      if (barInstance.current) barInstance.current.destroy();
      if (doughnutInstance.current) doughnutInstance.current.destroy();
      return;
    }

    const timer = setTimeout(() => {
      if (barChartRef.current) {
        if (barInstance.current) barInstance.current.destroy();

        const weeklyLabels = dashboardStats.weeklyVolume?.map((w: any) => w.weekLabel) || ['Tuần 1', 'Tuần 2', 'Tuần 3', 'Tuần 4', 'Tuần 5', 'Tuần 6'];
        const weeklyData = dashboardStats.weeklyVolume?.map((w: any) => w.count) || [0, 0, 0, 0, 0, 0];

        barInstance.current = new Chart(barChartRef.current, {
          type: 'bar',
          data: {
            labels: weeklyLabels,
            datasets: [{
              label: 'Số Video Xử Lý',
              data: weeklyData,
              backgroundColor: '#06B6D4',
              borderRadius: 6,
              barThickness: 15
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              y: { grid: { color: '#eceef0' }, beginAtZero: true },
              x: { grid: { display: false } }
            }
          }
        });
      }

      if (doughnutChartRef.current) {
        if (doughnutInstance.current) doughnutInstance.current.destroy();

        const modelLabels = dashboardStats.modelDistribution?.map((m: any) => m.modelName) || [];
        const modelData = dashboardStats.modelDistribution?.map((m: any) => m.count) || [];

        const finalLabels = modelLabels.length > 0 ? modelLabels : ['Chưa có dữ liệu'];
        const finalData = modelData.length > 0 ? modelData : [1];
        const finalColors = modelLabels.length > 0 
          ? ['#06B6D4', '#d5e3fd', '#0F172A', '#A855F7', '#EAB308', '#EC4899'].slice(0, modelLabels.length)
          : ['#E2E8F0'];

        doughnutInstance.current = new Chart(doughnutChartRef.current, {
          type: 'doughnut',
          data: {
            labels: finalLabels,
            datasets: [{
              data: finalData,
              backgroundColor: finalColors,
              borderWidth: 0,
              hoverOffset: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'bottom',
                labels: { padding: 15, usePointStyle: true, font: { size: 10 } }
              }
            },
            cutout: '70%'
          }
        });
      }
    }, 50);

    return () => {
      clearTimeout(timer);
      if (barInstance.current) barInstance.current.destroy();
      if (doughnutInstance.current) doughnutInstance.current.destroy();
    };
  }, [activeTab, dashboardStats]);

  // Fetch dynamic stats counts and videos list for General Dashboard
  useEffect(() => {
    if (activeTab === 'stats') {
      const fetchStats = () => {
        Promise.all([
          api.getUsers(), 
          api.getAllVideosAdmin(10),
          api.getAllJobsAdmin(1),
          api.getVideos(undefined, 1),
          api.getAdminStats().catch(err => {
            console.warn("Failed to fetch admin stats dashboard", err);
            return { success: false, data: null };
          })
        ])
          .then(([usersRes, videosRes, jobsRes, queueRes, statsRes]) => {
            const emailMap: Record<string, string> = {};
            if (usersRes.success && usersRes.data) {
              const mappedUsers = usersRes.data.map((u: any) => ({
                id: u.userId,
                email: u.email,
                role: u.role.toUpperCase(),
                active: u.isActive,
                joined: parseUTCDate(u.createdAt)!.toLocaleDateString('vi-VN')
              }));
              setUsers(mappedUsers);
              setUsersTotal(usersRes.metadata?.totalResults || usersRes.metadata?.total || usersRes.data.length);
              
              usersRes.data.forEach((u: any) => {
                emailMap[u.userId] = u.email;
              });
            }

            if (videosRes.success && videosRes.data) {
              setAllVideos(videosRes.data);
              setSystemVideosTotal(videosRes.metadata?.totalResults || videosRes.metadata?.total || videosRes.data.length);
              setVideosCompleted(videosRes.metadata?.completed || 0);
              setVideosFailed(videosRes.metadata?.failed || 0);
              setVideosProcessing(videosRes.metadata?.processing || 0);
              const mappedVideos = videosRes.data.map((v: any) => ({
                ...v,
                email: emailMap[v.userId] || "Unknown User",
                title: v.title || (v.originalUrl ? "YouTube Video" : (v.r2Url ? (v.r2Url.split('?')[0].split('/').pop() || "Video File") : (v.filePath && !v.filePath.includes('/stream') ? (v.filePath.split('?')[0].split('/').pop() || "Video File") : "Uploaded Video"))),
              }));
              setStatsVideos(mappedVideos);
            }

            if (jobsRes.success) {
              setSystemJobsTotal(jobsRes.metadata?.totalResults || jobsRes.metadata?.total || 0);
              setJobsCompleted(jobsRes.metadata?.completed || 0);
              setJobsFailed(jobsRes.metadata?.failed || 0);
              setJobsProcessing(jobsRes.metadata?.processing || 0);
            }

            if (queueRes.success) {
              setQueueTotal(queueRes.metadata?.totalResults || queueRes.metadata?.total || 0);
            }

            if (statsRes && statsRes.success && statsRes.data) {
              setDashboardStats(statsRes.data);
            }
          })
          .catch(err => {
            console.warn("Failed to load dashboard recent tasks", err);
          })
          .finally(() => {
            setStatsLoading(false);
          });
      };

      setStatsLoading(true);
      fetchStats();
    }
  }, [activeTab]);

  // Fetch video standards from API when entering standard config tab
  useEffect(() => {
    if (activeTab === 'videos') {
      setStandardsLoading(true);
      setStandardsMsg(null);
      api.getStandards()
        .then(res => {
          if (res.success && res.data) {
            setMaxDuration(res.data.maxDuration);
            const fmt = res.data.allowedFormats;
            setAllowedFormats(Array.isArray(fmt) ? fmt.join(',') : String(fmt || 'mp4,avi,mkv'));
            const cdc = res.data.allowedCodecs;
            if (cdc) {
              setAllowedCodecs(Array.isArray(cdc) ? cdc.join(',') : String(cdc));
            }
            setMaxFileSize(res.data.maxFileSize);
            setMinAudioQuality(res.data.minAudioQuality);
            if (res.data.minWidth !== undefined) setMinWidth(res.data.minWidth);
            if (res.data.minHeight !== undefined) setMinHeight(res.data.minHeight);
            if (res.data.maxWidth !== undefined) setMaxWidth(res.data.maxWidth);
            if (res.data.maxHeight !== undefined) setMaxHeight(res.data.maxHeight);
          }
        })
        .catch(err => {
          console.warn("Failed to fetch real standards from backend, using default presets.", err);
        })
        .finally(() => {
          setStandardsLoading(false);
        });
    }
  }, [activeTab]);

  // Fetch Celery Jobs/Videos list when entering Queue tab
  useEffect(() => {
    if (activeTab === 'celery') {
      const fetchQueue = () => {
        api.getVideos(undefined, 10, (queuePage - 1) * 10)
          .then(res => {
            if (res.success && res.data) {
              setQueueJobs(res.data);
              setQueueTotal(res.metadata?.totalResults || res.metadata?.total || res.data.length);
            }
          })
          .catch(err => {
            console.error("Failed to load queue videos:", err);
            setQueueJobs([]);
            setQueueTotal(0);
          })
          .finally(() => {
            setQueueLoading(false);
          });
      };

      setQueueLoading(true);
      fetchQueue();
    }
  }, [activeTab, queuePage]);

  // Fetch registered users list from API
  useEffect(() => {
    if (activeTab === 'users') {
      const fetchUsersList = () => {
        api.getUsers(10, (usersPage - 1) * 10)
          .then(res => {
            if (res.success && res.data) {
              const mappedUsers = res.data.map((u: any) => ({
                id: u.userId,
                email: u.email,
                role: u.role.toUpperCase(),
                active: u.isActive,
                joined: parseUTCDate(u.createdAt)!.toLocaleDateString('vi-VN')
              }));
              setUsers(mappedUsers);
              setUsersTotal(res.metadata?.totalResults || res.metadata?.total || res.data.length);
            }
          })
          .catch(err => {
            console.error("Failed to fetch users from backend:", err);
            setUsers([]);
            setUsersTotal(0);
          })
          .finally(() => {
            setUsersLoading(false);
          });
      };

      setUsersLoading(true);
      fetchUsersList();
    }
  }, [activeTab, usersPage]);

  // Fetch all system videos (Admin only)
  useEffect(() => {
    if (activeTab === 'system-videos') {
      const fetchSystemVideosList = () => {
        api.getAllVideosAdmin(10, (systemVideosPage - 1) * 10)
          .then(res => {
            if (res.success && res.data) {
              setAllVideos(res.data);
              setSystemVideosTotal(res.metadata?.totalResults || res.metadata?.total || res.data.length);
            }
          })
          .catch(err => {
            console.error("Failed to load system videos from API:", err);
            setAllVideos([]);
            setSystemVideosTotal(0);
          })
          .finally(() => {
            setVideoLoading(false);
          });
      };

      setVideoLoading(true);
      fetchSystemVideosList();
    }
  }, [activeTab, systemVideosPage]);

  // Fetch all system jobs (Admin only)
  useEffect(() => {
    if (activeTab === 'system-jobs') {
      const fetchSystemJobsList = () => {
        api.getAllJobsAdmin(10, (systemJobsPage - 1) * 10)
          .then(res => {
            if (res.success && res.data) {
              setAllJobs(res.data);
              setSystemJobsTotal(res.metadata?.totalResults || res.metadata?.total || res.data.length);
            }
          })
          .catch(err => {
            console.error("Failed to load system jobs from API:", err);
            setAllJobs([]);
            setSystemJobsTotal(0);
          })
          .finally(() => {
            setJobsLoading(false);
          });
      };

      setJobsLoading(true);
      fetchSystemJobsList();
    }
  }, [activeTab, systemJobsPage, jobsTrigger]);
  
  // Continuous Polling for Job details when Job Modal is open
  useEffect(() => {
    if (!jobModalOpen || !selectedJob || !selectedJob.videoId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.getJobStatus(selectedJob.videoId);
        if (res.success && res.data) {
          let updatedJob = null;
          if (Array.isArray(res.data)) {
            updatedJob = res.data.find((j: any) => {
              const type = (j.jobType || j.job_type || "").toUpperCase();
              return type === 'SUMMARIZE';
            }) || res.data[0];
          } else {
            updatedJob = res.data;
          }

          if (updatedJob) {
            setSelectedJob(updatedJob);

            // Update local tables
            setAllJobs(prev => prev.map(j => j.jobId === updatedJob.jobId ? updatedJob : j));

            const isCompleted = updatedJob.status === 'completed' || updatedJob.status === 'done' || updatedJob.status === 'SUCCESS';
            const isFailed = updatedJob.status === 'failed' || updatedJob.status === 'FAILED';
            const mappedVideoStatus = isCompleted ? 'done' : isFailed ? 'failed' : 'processing';

            setQueueJobs(prev => prev.map(job => 
              job.videoId === selectedJob.videoId ? { ...job, status: mappedVideoStatus } : job
            ));

            setAllVideos(prev => prev.map(v => 
              v.videoId === selectedJob.videoId ? { ...v, status: mappedVideoStatus } : v
            ));
          }
        }
      } catch (err) {
        console.warn("Failed to poll active job status in AdminPage", err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobModalOpen, selectedJob?.videoId]);

  // Video deletion action
  const handleDeleteVideo = (videoId: string) => {
    if (window.confirm("Bạn có chắc chắn muốn xóa video này và toàn bộ dữ liệu tóm tắt/jobs liên quan?")) {
      api.deleteVideoAdmin(videoId)
        .then(res => {
          if (res.success) {
            setAllVideos(prev => prev.filter(v => v.videoId !== videoId));
            toast.success("Xóa video thành công!", "Thành công");
          }
        })
        .catch(err => {
          toast.error(err.message || "Xóa video thất bại!", "Thất bại");
        });
    }
  };

  // User Actions
  const toggleUserActive = (id: string) => {
    api.toggleUserStatus(id)
      .then(res => {
        if (res.success && res.data) {
          setUsers(prev => prev.map(u => u.id === id ? { ...u, active: res.data.isActive } : u));
          toast.success("Đã thay đổi trạng thái hoạt động người dùng!", "Thành công");
        }
      })
      .catch(err => {
        toast.error(err.message || "Không thể thay đổi trạng thái hoạt động!", "Thất bại");
      });
  };

  const toggleUserRole = (id: string) => {
    const userObj = users.find(u => u.id === id);
    if (!userObj) return;
    const newRole = userObj.role.toLowerCase() === 'admin' ? 'user' : 'admin';
    
    api.changeUserRole(id, newRole)
      .then(res => {
        if (res.success && res.data) {
          setUsers(prev => prev.map(u => u.id === id ? { ...u, role: res.data.role.toUpperCase() } : u));
          toast.success("Đã thay đổi vai trò người dùng!", "Thành công");
        }
      })
      .catch(err => {
        toast.error(err.message || "Không thể thay đổi vai trò người dùng!", "Thất bại");
      });
  };

  // Save Standards handler
  const handleSaveStandards = (e: React.FormEvent) => {
    e.preventDefault();
    setStandardsLoading(true);
    setStandardsMsg(null);

    const cleanFormats = getFormatsList(allowedFormats).join(',');
    const cleanCodecs = getCodecsList(allowedCodecs).join(',');

    api.updateStandards({
      maxDuration: Number(maxDuration) || 3600,
      allowedFormats: cleanFormats || 'mp4,avi,mkv',
      allowedCodecs: cleanCodecs || 'h264,hevc',
      maxFileSize: Number(maxFileSize) || 500,
      minAudioQuality: Number(minAudioQuality) || 0.0,
      minWidth: Number(minWidth) || 640,
      minHeight: Number(minHeight) || 360,
      maxWidth: Number(maxWidth) || 3840,
      maxHeight: Number(maxHeight) || 2160,
    })
    .then(res => {
      if (res.success) {
        setStandardsMsg({
          type: 'success',
          text: 'Cấu hình giới hạn video đã được lưu vào cơ sở dữ liệu thành công!'
        });
        toast.success('Cấu hình giới hạn video đã được lưu thành công!', 'Thành công');
      }
    })
    .catch(err => {
      setStandardsMsg({
        type: 'error',
        text: err.message || 'Cập nhật cấu hình thất bại!'
      });
      toast.error(err.message || 'Cập nhật cấu hình thất bại!', 'Thất bại');
    })
    .finally(() => {
      setStandardsLoading(false);
    });
  };

  return (
    <div className="flex h-[calc(100vh-64px)] overflow-hidden bg-slate-50 text-slate-900">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r-2 border-slate-200 bg-white flex flex-col shrink-0 p-4 gap-2">
        <div className="px-2 py-4 border-b-2 border-slate-100 mb-4">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">admin_panel_settings</span>
            <span className="font-heading text-sm font-bold text-slate-900">{t('admin.title')}</span>
          </div>
        </div>

        <div className="flex-1 space-y-4">
          <div>
            <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider px-3 mb-1">{t('admin.analytics')}</div>
            <nav className="flex flex-col gap-1">
              <button 
                onClick={() => setActiveTab('stats')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left text-xs font-bold w-full ${
                  activeTab === 'stats' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <span className="material-symbols-outlined text-sm">bar_chart</span> {t('admin.stats')}
              </button>
              <button 
                onClick={() => setActiveTab('evaluation')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left text-xs font-bold w-full ${
                  activeTab === 'evaluation' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <span className="material-symbols-outlined text-sm">model_training</span> Đánh giá & Benchmark
              </button>
            </nav>
          </div>

          <div>
            <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider px-3 mb-1">{t('admin.management')}</div>
            <nav className="flex flex-col gap-1">
              <button 
                onClick={() => setActiveTab('users')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left text-xs font-bold w-full ${
                  activeTab === 'users' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <span className="material-symbols-outlined text-sm">group</span> {t('admin.users')}
              </button>
              <button 
                onClick={() => setActiveTab('system-videos')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left text-xs font-bold w-full ${
                  activeTab === 'system-videos' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <span className="material-symbols-outlined text-sm">video_library</span> {t('admin.videos')}
              </button>
              <button 
                onClick={() => setActiveTab('system-jobs')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left text-xs font-bold w-full ${
                  activeTab === 'system-jobs' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <span className="material-symbols-outlined text-sm">task</span> {t('admin.jobs')}
              </button>
              <button 
                onClick={() => setActiveTab('videos')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left text-xs font-bold w-full ${
                  activeTab === 'videos' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <span className="material-symbols-outlined text-sm">settings_suggest</span> {t('admin.settings')}
              </button>
              <button 
                onClick={() => setActiveTab('celery')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-left text-xs font-bold w-full ${
                  activeTab === 'celery' ? 'bg-primary/10 text-primary' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                }`}
              >
                <span className="material-symbols-outlined text-sm">dns</span> {t('admin.celery')}
              </button>
            </nav>
          </div>
        </div>
      </aside>
      
      {/* Main Panel Content Area */}
      <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-50 custom-scrollbar">
        <div className="max-w-7xl mx-auto space-y-6">
          
          {/* TAB: EVALUATION BENCHMARK */}
          {activeTab === 'evaluation' && (
            <EvaluationBenchmarkView />
          )}

          {/* TAB 1: GENERAL STATS */}
          {activeTab === 'stats' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h1 className="font-heading text-xl font-bold text-slate-900">Báo Cáo Thống Kê Tổng Hợp</h1>
                <div className="flex items-center gap-1.5 px-3 py-1.5 border-2 border-slate-200 bg-white text-xs font-bold rounded-lg text-slate-500 cursor-pointer hover:border-slate-300">
                  <span className="material-symbols-outlined text-sm">calendar_month</span> Hôm nay
                </div>
              </div>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none space-y-2">
                  <h3 className="text-xs text-slate-500 font-bold flex justify-between items-center">
                    Thành viên
                    <span className="material-symbols-outlined text-primary text-sm">group</span>
                  </h3>
                  {usersTotal === null ? (
                    <Skeleton className="h-7 w-16 my-1.5" />
                  ) : (
                    <div className="text-2xl font-bold text-slate-900">{usersTotal}</div>
                  )}
                  {usersTotal === null ? (
                    <Skeleton className="h-3.5 w-24" />
                  ) : (
                    <div className="text-[10px] text-emerald-500 font-bold flex items-center gap-0.5"><span className="material-symbols-outlined text-xs">trending_up</span> Hoạt động</div>
                  )}
                </div>
                <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none space-y-2">
                  <h3 className="text-xs text-slate-500 font-bold flex justify-between items-center">
                    Video Phân Tích
                    <span className="material-symbols-outlined text-primary text-sm">video_library</span>
                  </h3>
                  {systemVideosTotal === null ? (
                    <Skeleton className="h-7 w-16 my-1.5" />
                  ) : (
                    <div className="text-2xl font-bold text-slate-900">{systemVideosTotal}</div>
                  )}
                  {systemVideosTotal === null ? (
                    <Skeleton className="h-3.5 w-32" />
                  ) : (
                    <div className="text-[9px] text-slate-500 font-bold flex items-center gap-1 flex-wrap">
                      <span className="text-emerald-500">Đạt: {videosCompleted}</span>
                      <span className="text-slate-300">•</span>
                      <span className="text-amber-500">Chạy: {videosProcessing}</span>
                      <span className="text-slate-300">•</span>
                      <span className="text-red-500">Lỗi: {videosFailed}</span>
                    </div>
                  )}
                </div>
                <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none space-y-2">
                  <h3 className="text-xs text-slate-500 font-bold flex justify-between items-center">
                    Tổng tác vụ
                    <span className="material-symbols-outlined text-indigo-500 text-sm">task</span>
                  </h3>
                  {systemJobsTotal === null ? (
                    <Skeleton className="h-7 w-16 my-1.5" />
                  ) : (
                    <div className="text-2xl font-bold text-slate-900">{systemJobsTotal}</div>
                  )}
                  {systemJobsTotal === null ? (
                    <Skeleton className="h-3.5 w-32" />
                  ) : (
                    <div className="text-[9px] text-slate-500 font-bold flex items-center gap-1 flex-wrap">
                      <span className="text-emerald-500">Đạt: {jobsCompleted}</span>
                      <span className="text-slate-300">•</span>
                      <span className="text-amber-500">Chạy: {jobsProcessing}</span>
                      <span className="text-slate-300">•</span>
                      <span className="text-red-500">Lỗi: {jobsFailed}</span>
                    </div>
                  )}
                </div>
                <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none space-y-2">
                  <h3 className="text-xs text-slate-500 font-bold flex justify-between items-center">
                    Hàng đợi Celery
                    <span className="material-symbols-outlined text-primary text-sm">queue_play_next</span>
                  </h3>
                  {queueTotal === null ? (
                    <Skeleton className="h-7 w-16 my-1.5" />
                  ) : (
                    <div className="text-2xl font-bold text-slate-900">{queueTotal}</div>
                  )}
                  {queueTotal === null ? (
                    <Skeleton className="h-3.5 w-24" />
                  ) : (
                    <div className="text-[10px] text-emerald-500 font-bold flex items-center gap-0.5"><span className="material-symbols-outlined text-xs">check_circle</span> Đồng bộ</div>
                  )}
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none">
                  <h3 className="font-heading text-xs font-bold text-slate-900 mb-4">Lưu lượng xử lý video</h3>
                  <div className="h-64 relative">
                    <canvas ref={barChartRef}></canvas>
                  </div>
                </div>
                <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none">
                  <h3 className="font-heading text-xs font-bold text-slate-900 mb-4">Sử dụng Mô hình LLM</h3>
                  <div className="h-64 relative">
                    <canvas ref={doughnutChartRef}></canvas>
                  </div>
                </div>
              </div>
              
              {/* Performance Correlation Widget (RTF Analytics) */}
              <PerformanceCorrelationWidget />

              <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none">
                <h3 className="font-heading text-xs font-bold text-slate-900 mb-4">Tác vụ xử lý gần đây</h3>
                <div className="overflow-x-auto border-2 border-slate-200 rounded-xl">
                  {statsLoading ? (
                    <div className="p-12 text-center text-slate-500">
                      <span className="material-symbols-outlined text-2xl animate-spin text-primary">autorenew</span>
                      <p className="text-xs mt-2 font-body">Đang tải danh sách tác vụ...</p>
                    </div>
                  ) : statsVideos.length === 0 ? (
                    <div className="p-12 text-center text-slate-500">
                      <span className="material-symbols-outlined text-3xl">list_alt</span>
                      <p className="text-xs mt-2 font-body">Chưa có tác vụ nào được lưu trong hệ thống.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Video ID</th>
                          <th className="p-3">Email User</th>
                          <th className="p-3">Tên File / Nguồn</th>
                          <th className="p-3">Độ dài</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Thao tác</th>
                        </tr>
                      </thead>
                      <tbody>
                        {statsVideos.map(v => (
                          <tr key={v.videoId} className="border-b-2 border-slate-100 hover:bg-slate-50">
                            <td className="p-3 font-mono text-[10px]">#{v.videoId.substring(0, 8)}...</td>
                            <td className="p-3">{v.email}</td>
                            <td className="p-3 truncate max-w-[200px]" title={v.originalUrl || v.title}>
                              {v.title}
                            </td>
                            <td className="p-3">{Math.round(v.duration)}s</td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                                v.status === VideoStatus.DONE ? 'bg-emerald-50 text-emerald-600' : v.status === VideoStatus.FAILED ? 'bg-red-50 text-red-600' : 'bg-amber-50 text-amber-600'
                              }`}>
                                {v.status === VideoStatus.DONE ? 'Hoàn tất' : v.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3">
                              <button onClick={() => openVideoDetail(v)} className="px-2 py-1 bg-white rounded text-[10px] font-bold text-slate-900 border-2 border-slate-200 hover:border-slate-300 flex items-center gap-1">
                                <span className="material-symbols-outlined text-xs">info</span> Chi tiết
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: USER MANAGEMENT */}
          {activeTab === 'users' && (
            <div className="space-y-6">
              <h1 className="font-heading text-xl font-bold text-slate-900">Quản Lý Thành Viên Hệ Thống</h1>
              
              <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none">
                <h3 className="font-heading text-xs font-bold text-slate-900 mb-4">Danh sách thành viên đăng ký</h3>
                <div className="overflow-x-auto border-2 border-slate-200 rounded-xl">
                  {usersLoading ? (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Thành viên</th>
                          <th className="p-3">Email</th>
                          <th className="p-3">Quyền Hạn</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Ngày Đăng Ký</th>
                          <th className="p-3">Hành Động</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from({ length: 5 }).map((_, idx) => (
                          <Skeleton.TableRow key={idx} cells={6} />
                        ))}
                      </tbody>
                    </table>
                  ) : users.length === 0 ? (
                    <div className="p-12 text-center text-slate-500 font-body">
                      <span className="material-symbols-outlined text-3xl">people</span>
                      <p className="text-xs mt-2">Chưa có người dùng nào đăng ký.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Thành viên</th>
                          <th className="p-3">Email</th>
                          <th className="p-3">Quyền Hạn</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Ngày Đăng Ký</th>
                          <th className="p-3">Hành Động</th>
                        </tr>
                      </thead>
                      <tbody>
                        {users.map(u => (
                          <tr key={u.id} className="border-b-2 border-slate-100 hover:bg-slate-50">
                            <td className="p-3">
                              <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-slate-900 text-primary flex items-center justify-center font-bold text-xs uppercase shadow-none border-2 border-slate-200">
                                  {u.email.charAt(0)}
                                </div>
                                <span className="font-bold text-slate-900">{u.email.split('@')[0]}</span>
                              </div>
                            </td>
                            <td className="p-3 font-semibold text-slate-600">{u.email}</td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                                u.role === 'ADMIN' ? 'bg-amber-50 text-amber-600 border border-amber-200' : 'bg-primary/10 text-primary border border-primary/20'
                              }`}>
                                {u.role}
                              </span>
                            </td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                                u.active ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-600'
                              }`}>
                                {u.active ? 'Hoạt động' : 'Bị khóa'}
                              </span>
                            </td>
                            <td className="p-3 font-semibold text-slate-600">{u.joined}</td>
                            <td className="p-3">
                              <div className="flex gap-2">
                                <button onClick={() => toggleUserRole(u.id)} className="px-2 py-1 text-[10px] font-bold bg-white border-2 border-slate-200 rounded hover:bg-slate-100 text-slate-900 transition-colors">
                                  Đổi Vai Trò
                                </button>
                                <button 
                                  onClick={() => toggleUserActive(u.id)} 
                                  className={`px-2 py-1 text-[10px] font-bold border-2 rounded transition-colors ${
                                    u.active ? 'border-red-200 bg-red-50 hover:bg-red-100 text-red-600' : 'border-slate-200 bg-white hover:bg-slate-100 text-slate-900'
                                  }`}
                                >
                                  {u.active ? 'Khóa' : 'Mở khóa'}
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
                <div className="mt-4">
                  <PaginationControl
                    currentPage={usersPage}
                    totalItems={usersTotal || 0}
                    limit={10}
                    onPageChange={setUsersPage}
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: VIDEO STANDARD SETTINGS */}
          {activeTab === 'videos' && (
            <div className="space-y-6">
              <div>
                <h1 className="font-heading text-xl font-bold text-slate-900">Cấu Hình Tiêu Chuẩn Phân Tích Video</h1>
                <p className="text-xs text-slate-500 font-medium">Thiết lập các điều kiện kỹ thuật tự động kiểm định video trước và trong khi xử lý</p>
              </div>
              
              <div className="bg-white border-2 border-slate-200 rounded-xl p-6 shadow-none">
                <h3 className="font-heading text-xs font-bold text-slate-900 mb-6 pb-2 border-b-2 border-slate-100 flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-primary text-base">settings_suggest</span> Giới hạn tải lên &amp; Xác thực video
                </h3>
                {standardsLoading ? (
                  <div className="p-12 text-center text-slate-500 font-body">
                    <span className="material-symbols-outlined text-2xl animate-spin text-primary">autorenew</span>
                    <p className="text-xs mt-2">Đang tải cấu hình...</p>
                  </div>
                ) : (
                  <form onSubmit={handleSaveStandards} className="space-y-6 max-w-2xl text-xs text-slate-900 font-body">
                    
                    {/* 1. ALLOWED FORMATS */}
                    <div className="flex flex-col gap-2 p-4 bg-slate-50/70 border-2 border-slate-200/80 rounded-xl">
                      <div className="flex items-center justify-between">
                        <label className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span className="material-symbols-outlined text-primary text-sm">video_file</span>
                          Các định dạng cho phép
                        </label>
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => setAllowedFormats('mp4,mkv,avi,mov,webm')}
                            className="text-[10px] text-primary hover:underline font-bold px-1.5 py-0.5 rounded bg-primary/5 border border-primary/20"
                          >
                            Phổ biến
                          </button>
                          <button
                            type="button"
                            onClick={() => setAllowedFormats('mp4,webm')}
                            className="text-[10px] text-slate-600 hover:text-slate-900 font-semibold px-1.5 py-0.5 rounded bg-white border border-slate-200"
                          >
                            Web only
                          </button>
                          <button
                            type="button"
                            onClick={() => setAllowedFormats(AVAILABLE_FORMATS.join(','))}
                            className="text-[10px] text-slate-600 hover:text-slate-900 font-semibold px-1.5 py-0.5 rounded bg-white border border-slate-200"
                          >
                            Tất cả
                          </button>
                        </div>
                      </div>

                      {/* Selectable Format Pills */}
                      <div className="flex flex-wrap gap-2 pt-1">
                        {AVAILABLE_FORMATS.map((fmt) => {
                          const currentList = getFormatsList(allowedFormats);
                          const isSelected = currentList.includes(fmt);
                          return (
                            <button
                              type="button"
                              key={fmt}
                              onClick={() => {
                                let newList = [...currentList];
                                if (isSelected) {
                                  if (newList.length > 1) {
                                    newList = newList.filter(f => f !== fmt);
                                  } else {
                                    toast.info("Cần giữ ít nhất 1 định dạng video cho phép.", "Thông báo");
                                    return;
                                  }
                                } else {
                                  newList.push(fmt);
                                }
                                setAllowedFormats(newList.join(','));
                              }}
                              className={`px-3 py-1.5 rounded-lg font-bold text-xs flex items-center gap-1.5 transition-all border-2 ${
                                isSelected
                                  ? 'bg-primary text-white border-primary shadow-sm'
                                  : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                              }`}
                            >
                              <span className="material-symbols-outlined text-sm">
                                {isSelected ? 'check_circle' : 'add_circle'}
                              </span>
                              .{fmt.toUpperCase()}
                            </button>
                          );
                        })}
                      </div>
                      <span className="text-[10px] text-slate-500 font-mono pt-1">
                        Đang chọn: <strong className="text-slate-700 font-mono">{getFormatsList(allowedFormats).join(', ') || 'Chưa chọn'}</strong>
                      </span>
                    </div>

                    {/* 2. ALLOWED CODECS */}
                    <div className="flex flex-col gap-2 p-4 bg-slate-50/70 border-2 border-slate-200/80 rounded-xl">
                      <div className="flex items-center justify-between">
                        <label className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span className="material-symbols-outlined text-primary text-sm">developer_board</span>
                          Bộ giải mã video cho phép (Codecs)
                        </label>
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => setAllowedCodecs('h264,hevc,av1,vp9')}
                            className="text-[10px] text-primary hover:underline font-bold px-1.5 py-0.5 rounded bg-primary/5 border border-primary/20"
                          >
                            Khuyên dùng
                          </button>
                          <button
                            type="button"
                            onClick={() => setAllowedCodecs('h264,hevc')}
                            className="text-[10px] text-slate-600 hover:text-slate-900 font-semibold px-1.5 py-0.5 rounded bg-white border border-slate-200"
                          >
                            Tương thích cao
                          </button>
                          <button
                            type="button"
                            onClick={() => setAllowedCodecs(AVAILABLE_CODECS.map(c => c.id).join(','))}
                            className="text-[10px] text-slate-600 hover:text-slate-900 font-semibold px-1.5 py-0.5 rounded bg-white border border-slate-200"
                          >
                            Tất cả
                          </button>
                        </div>
                      </div>

                      {/* Selectable Codec Cards */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                        {AVAILABLE_CODECS.map((codec) => {
                          const currentList = getCodecsList(allowedCodecs);
                          const isSelected = currentList.includes(codec.id);
                          return (
                            <button
                              type="button"
                              key={codec.id}
                              onClick={() => {
                                let newList = [...currentList];
                                if (isSelected) {
                                  if (newList.length > 1) {
                                    newList = newList.filter(c => c !== codec.id);
                                  } else {
                                    toast.info("Cần giữ ít nhất 1 codec video cho phép.", "Thông báo");
                                    return;
                                  }
                                } else {
                                  newList.push(codec.id);
                                }
                                setAllowedCodecs(newList.join(','));
                              }}
                              className={`p-2.5 rounded-xl text-left border-2 transition-all flex items-start gap-2.5 ${
                                isSelected
                                  ? 'bg-primary/5 border-primary text-slate-900 shadow-sm'
                                  : 'bg-white border-slate-200 text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                              }`}
                            >
                              <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
                                isSelected ? 'bg-primary text-white' : 'bg-slate-200 text-slate-400'
                              }`}>
                                <span className="material-symbols-outlined text-xs">
                                  {isSelected ? 'check' : 'remove'}
                                </span>
                              </div>
                              <div className="min-w-0">
                                <div className="font-bold text-xs">{codec.label}</div>
                                <div className="text-[10px] text-slate-500 font-normal truncate">{codec.desc}</div>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                      <span className="text-[10px] text-slate-500 font-mono pt-1">
                        Đang chọn: <strong className="text-slate-700 font-mono">{getCodecsList(allowedCodecs).join(', ') || 'Chưa chọn'}</strong>
                      </span>
                    </div>

                    {/* 3. RESOLUTION DROPDOWNS */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 bg-slate-50/70 border-2 border-slate-200/80 rounded-xl">
                      {/* Min Resolution */}
                      <div className="flex flex-col gap-1.5">
                        <label className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span className="material-symbols-outlined text-primary text-sm">aspect_ratio</span>
                          Độ phân giải tối thiểu
                        </label>
                        <select
                          value={
                            RESOLUTION_PRESETS.find(p => p.width === minWidth && p.height === minHeight)?.id || 'custom'
                          }
                          onChange={(e) => {
                            const val = e.target.value;
                            if (val !== 'custom') {
                              const found = RESOLUTION_PRESETS.find(p => p.id === val);
                              if (found) {
                                setMinWidth(found.width);
                                setMinHeight(found.height);
                              }
                            }
                          }}
                          className="w-full px-3 py-2 bg-white border-2 border-slate-200 rounded-xl focus:border-primary focus:ring-0 outline-none text-xs font-bold text-slate-800"
                        >
                          {RESOLUTION_PRESETS.filter(p => p.height <= 1080).map((preset) => (
                            <option key={preset.id} value={preset.id}>
                              {preset.label}
                            </option>
                          ))}
                          <option value="custom">Tùy chỉnh kích thước...</option>
                        </select>

                        <div className="flex items-center gap-2 pt-1">
                          <div className="flex-1 flex items-center bg-white px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs">
                            <span className="text-slate-400 text-[10px] mr-1.5 font-bold">R:</span>
                            <input
                              type="number"
                              value={minWidth}
                              onChange={(e) => setMinWidth(parseInt(e.target.value) || 0)}
                              className="w-full font-mono font-bold text-slate-800 outline-none"
                            />
                            <span className="text-[10px] text-slate-400 font-mono">px</span>
                          </div>
                          <span className="text-slate-400 font-bold">×</span>
                          <div className="flex-1 flex items-center bg-white px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs">
                            <span className="text-slate-400 text-[10px] mr-1.5 font-bold">C:</span>
                            <input
                              type="number"
                              value={minHeight}
                              onChange={(e) => setMinHeight(parseInt(e.target.value) || 0)}
                              className="w-full font-mono font-bold text-slate-800 outline-none"
                            />
                            <span className="text-[10px] text-slate-400 font-mono">px</span>
                          </div>
                        </div>
                      </div>

                      {/* Max Resolution */}
                      <div className="flex flex-col gap-1.5">
                        <label className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span className="material-symbols-outlined text-primary text-sm">fullscreen</span>
                          Độ phân giải tối đa
                        </label>
                        <select
                          value={
                            RESOLUTION_PRESETS.find(p => p.width === maxWidth && p.height === maxHeight)?.id || 'custom'
                          }
                          onChange={(e) => {
                            const val = e.target.value;
                            if (val !== 'custom') {
                              const found = RESOLUTION_PRESETS.find(p => p.id === val);
                              if (found) {
                                setMaxWidth(found.width);
                                setMaxHeight(found.height);
                              }
                            }
                          }}
                          className="w-full px-3 py-2 bg-white border-2 border-slate-200 rounded-xl focus:border-primary focus:ring-0 outline-none text-xs font-bold text-slate-800"
                        >
                          {RESOLUTION_PRESETS.filter(p => p.height >= 720).map((preset) => (
                            <option key={preset.id} value={preset.id}>
                              {preset.label}
                            </option>
                          ))}
                          <option value="custom">Tùy chỉnh kích thước...</option>
                        </select>

                        <div className="flex items-center gap-2 pt-1">
                          <div className="flex-1 flex items-center bg-white px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs">
                            <span className="text-slate-400 text-[10px] mr-1.5 font-bold">R:</span>
                            <input
                              type="number"
                              value={maxWidth}
                              onChange={(e) => setMaxWidth(parseInt(e.target.value) || 0)}
                              className="w-full font-mono font-bold text-slate-800 outline-none"
                            />
                            <span className="text-[10px] text-slate-400 font-mono">px</span>
                          </div>
                          <span className="text-slate-400 font-bold">×</span>
                          <div className="flex-1 flex items-center bg-white px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs">
                            <span className="text-slate-400 text-[10px] mr-1.5 font-bold">C:</span>
                            <input
                              type="number"
                              value={maxHeight}
                              onChange={(e) => setMaxHeight(parseInt(e.target.value) || 0)}
                              className="w-full font-mono font-bold text-slate-800 outline-none"
                            />
                            <span className="text-[10px] text-slate-400 font-mono">px</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* 4. DURATION & FILE SIZE DROPDOWNS */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {/* Max Duration */}
                      <div className="flex flex-col gap-1.5">
                        <label className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span className="material-symbols-outlined text-primary text-sm">schedule</span>
                          Thời lượng video tối đa
                        </label>
                        <select
                          value={
                            DURATION_PRESETS.some(d => d.value === maxDuration) ? maxDuration : 'custom'
                          }
                          onChange={(e) => {
                            const val = e.target.value;
                            if (val !== 'custom') {
                              setMaxDuration(parseInt(val));
                            }
                          }}
                          className="w-full px-3 py-2 bg-white border-2 border-slate-200 rounded-xl focus:border-primary focus:ring-0 outline-none text-xs font-bold text-slate-800"
                        >
                          {DURATION_PRESETS.map((dur) => (
                            <option key={dur.value} value={dur.value}>
                              {dur.label}
                            </option>
                          ))}
                          <option value="custom">Tùy chỉnh số giây...</option>
                        </select>
                        <div className="flex items-center bg-white px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs">
                          <input
                            type="number"
                            value={maxDuration}
                            onChange={(e) => setMaxDuration(parseInt(e.target.value) || 0)}
                            className="w-full font-mono font-bold text-slate-800 outline-none"
                          />
                          <span className="text-[10px] text-slate-400 font-mono">giây (~{Math.round(maxDuration / 60)} phút)</span>
                        </div>
                      </div>

                      {/* Max File Size */}
                      <div className="flex flex-col gap-1.5">
                        <label className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span className="material-symbols-outlined text-primary text-sm">inventory_2</span>
                          Dung lượng tệp tối đa
                        </label>
                        <select
                          value={
                            FILE_SIZE_PRESETS.some(f => f.value === maxFileSize) ? maxFileSize : 'custom'
                          }
                          onChange={(e) => {
                            const val = e.target.value;
                            if (val !== 'custom') {
                              setMaxFileSize(parseInt(val));
                            }
                          }}
                          className="w-full px-3 py-2 bg-white border-2 border-slate-200 rounded-xl focus:border-primary focus:ring-0 outline-none text-xs font-bold text-slate-800"
                        >
                          {FILE_SIZE_PRESETS.map((item) => (
                            <option key={item.value} value={item.value}>
                              {item.label}
                            </option>
                          ))}
                          <option value="custom">Tùy chỉnh số MB...</option>
                        </select>
                        <div className="flex items-center bg-white px-2.5 py-1.5 rounded-lg border border-slate-200 text-xs">
                          <input
                            type="number"
                            value={maxFileSize}
                            onChange={(e) => setMaxFileSize(parseInt(e.target.value) || 0)}
                            className="w-full font-mono font-bold text-slate-800 outline-none"
                          />
                          <span className="text-[10px] text-slate-400 font-mono">MB</span>
                        </div>
                      </div>
                    </div>

                    {/* 5. AUDIO SNR QUALITY */}
                    <div className="flex flex-col gap-1 p-3 bg-slate-50 border border-slate-200 rounded-xl">
                      <div className="flex items-center justify-between">
                        <label className="font-bold text-slate-900 flex items-center gap-1.5">
                          <span className="material-symbols-outlined text-primary text-sm">graphic_eq</span>
                          Chất lượng âm thanh tối thiểu (SNR dB)
                        </label>
                        <span className="text-[10px] text-slate-500 font-semibold">Để 0.0 để không bắt buộc</span>
                      </div>
                      <input 
                        type="number" 
                        step="0.5"
                        value={minAudioQuality} 
                        onChange={(e) => setMinAudioQuality(parseFloat(e.target.value) || 0)}
                        className="w-full px-3 py-2 bg-white border-2 border-slate-200 rounded-xl focus:border-primary focus:ring-0 outline-none transition-colors font-mono font-bold"
                        required
                      />
                    </div>

                    {standardsMsg && (
                      <div className={`p-3 rounded-xl border text-xs font-bold flex items-center gap-2 ${
                        standardsMsg.type === 'success' ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-red-50 border-red-200 text-red-700'
                      }`}>
                        <span className="material-symbols-outlined text-base">
                          {standardsMsg.type === 'success' ? 'check_circle' : 'warning'}
                        </span>
                        {standardsMsg.text}
                      </div>
                    )}

                    <div className="pt-2">
                      <button type="submit" className="btn primary px-6 py-2.5 flex items-center gap-2 shadow-sm font-bold">
                        <span className="material-symbols-outlined text-base">save</span> Lưu cấu hình vào DB
                      </button>
                    </div>
                  </form>
                )}
              </div>
            </div>
          )}

          {/* TAB 5: CELERY JOB QUEUE */}
          {activeTab === 'celery' && (
            <div className="space-y-6">
              <h1 className="font-heading text-xl font-bold text-slate-900">Hệ Thống Hàng Đợi Celery Job Queue</h1>
              
              <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none space-y-4">
                <div className="flex justify-between items-center pb-3 border-b-2 border-slate-100 text-xs">
                  <h3 className="font-heading font-bold text-slate-900">Các tiến trình AI đang hoạt động trong hàng đợi</h3>
                  <span className="px-3 py-1 bg-emerald-50 text-emerald-600 border border-emerald-200 rounded-full font-bold">Máy chủ xử lý: Đang hoạt động (1)</span>
                </div>
                
                <div className="overflow-x-auto border-2 border-slate-200 rounded-xl">
                  {queueLoading ? (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Video ID</th>
                          <th className="p-3">Nguồn Video</th>
                          <th className="p-3">Ngôn Ngữ</th>
                          <th className="p-3">Trạng Thái Pipeline</th>
                          <th className="p-3">Thao Tác</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from({ length: 5 }).map((_, idx) => (
                          <Skeleton.TableRow key={idx} cells={5} />
                        ))}
                      </tbody>
                    </table>
                  ) : queueJobs.length === 0 ? (
                    <div className="p-12 text-center text-slate-500 font-body">
                      <span className="material-symbols-outlined text-3xl">queue_play_next</span>
                      <p className="text-xs mt-2">Hiện không có job nào đang chạy hoặc bị lỗi.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Video ID</th>
                          <th className="p-3">Nguồn Video</th>
                          <th className="p-3">Ngôn Ngữ</th>
                          <th className="p-3">Trạng Thái Pipeline</th>
                          <th className="p-3">Thao Tác</th>
                        </tr>
                      </thead>
                      <tbody>
                        {queueJobs.map(job => (
                          <tr key={job.videoId} className="border-b-2 border-slate-100 hover:bg-slate-50">
                            <td className="p-3 font-mono text-[10px]">#{job.videoId.substring(0, 8)}...</td>
                            <td className="p-3 truncate max-w-[200px]" title={job.title || job.originalUrl || (job.r2Url ? job.r2Url : (job.filePath && !job.filePath.includes('/stream') ? job.filePath : 'Video File'))}>
                              {job.title || (job.originalUrl ? 'YouTube Video' : (job.r2Url ? (job.r2Url.split('?')[0].split('/').pop() || 'video.mp4') : (job.filePath && !job.filePath.includes('/stream') ? (job.filePath.split('?')[0].split('/').pop() || 'video.mp4') : 'Video File')))}
                            </td>
                            <td className="p-3 font-bold text-slate-600">{job.language.toUpperCase()}</td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                                job.status === VideoStatus.DONE ? 'bg-emerald-50 text-emerald-600' : job.status === VideoStatus.FAILED ? 'bg-red-50 text-red-600' : 'bg-amber-50 text-amber-600'
                              }`}>
                                {job.status === VideoStatus.DONE ? 'Hoàn tất' : job.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3">
                              <div className="flex gap-2">
                                <button onClick={() => openJobDetail(job)} className="px-2 py-1 text-[10px] bg-white rounded border-2 border-slate-200 font-bold hover:bg-slate-100 text-slate-900 transition-colors">Xem chi tiết</button>
                                {job.status !== VideoStatus.DONE && job.status !== VideoStatus.FAILED && (
                                  <button className="px-2 py-1 text-[10px] border-2 border-red-200 bg-red-50 hover:bg-red-100 text-red-600 rounded font-bold transition-colors">Dừng</button>
                                )}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
                <div className="mt-4">
                  <PaginationControl
                    currentPage={queuePage}
                    totalItems={queueTotal || 0}
                    limit={10}
                    onPageChange={setQueuePage}
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB: SYSTEM VIDEOS MANAGEMENT */}
          {activeTab === 'system-videos' && (
            <div className="space-y-6">
              <h1 className="font-heading text-xl font-bold text-slate-900">Quản Lý Toàn Bộ Video Hệ Thống</h1>
              
              <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none">
                <h3 className="font-heading text-xs font-bold text-slate-900 mb-4">Danh sách video đã tải lên</h3>
                <div className="overflow-x-auto border-2 border-slate-200 rounded-xl">
                  {videoLoading ? (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Video ID</th>
                          <th className="p-3">User ID</th>
                          <th className="p-3">Nguồn / File</th>
                          <th className="p-3">Thời lượng (giây)</th>
                          <th className="p-3">Ngôn ngữ</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Ngày tải</th>
                          <th className="p-3">Hành động</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from({ length: 5 }).map((_, idx) => (
                          <Skeleton.TableRow key={idx} cells={8} />
                        ))}
                      </tbody>
                    </table>
                  ) : allVideos.length === 0 ? (
                    <div className="p-12 text-center text-slate-500 font-body">
                      <span className="material-symbols-outlined text-3xl">movie</span>
                      <p className="text-xs mt-2">Chưa có video nào được tải lên hệ thống.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Video ID</th>
                          <th className="p-3">User ID</th>
                          <th className="p-3">Nguồn / File</th>
                          <th className="p-3">Thời lượng (giây)</th>
                          <th className="p-3">Ngôn ngữ</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Ngày tải</th>
                          <th className="p-3">Hành động</th>
                        </tr>
                      </thead>
                      <tbody>
                        {allVideos.map((v: any) => (
                          <tr key={v.videoId} className="border-b-2 border-slate-100 hover:bg-slate-50">
                            <td className="p-3 font-mono text-[10px]" title={v.videoId}>#{v.videoId.substring(0, 8)}...</td>
                            <td className="p-3 font-mono text-[10px]" title={v.userId}>#{v.userId.substring(0, 8)}...</td>
                            <td className="p-3 truncate max-w-[200px]" title={v.title || v.originalUrl || (v.r2Url ? v.r2Url : (v.filePath && !v.filePath.includes('/stream') ? v.filePath : 'Video File'))}>
                              {v.title ? (
                                <span className="font-bold text-slate-900">{v.title}</span>
                              ) : v.originalUrl ? (
                                <a 
                                  href={v.originalUrl} 
                                  target="_blank" 
                                  rel="noopener noreferrer" 
                                  className="text-primary hover:underline flex items-center gap-1 w-fit font-semibold"
                                >
                                  <span className="material-symbols-outlined text-sm">open_in_new</span>
                                  YouTube Video
                                </a>
                              ) : (
                                <a 
                                  href={v.r2Url || v.filePath} 
                                  target="_blank" 
                                  rel="noopener noreferrer" 
                                  className="text-primary hover:underline flex items-center gap-1 truncate w-fit font-semibold"
                                  title="Bấm để tải về hoặc xem trực tiếp từ R2"
                                >
                                  <span className="material-symbols-outlined text-sm">download</span>
                                  {((v.r2Url || (v.filePath && !v.filePath.includes('/stream') ? v.filePath : '')).split('?')[0].split('/').pop()) || 'Video File'}
                                </a>
                              )}
                            </td>
                            <td className="p-3 font-semibold text-slate-600">{Math.round(v.duration)}s</td>
                            <td className="p-3 font-bold text-slate-600">{v.language.toUpperCase()}</td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                                v.status === VideoStatus.DONE ? 'bg-emerald-50 text-emerald-600' : v.status === VideoStatus.FAILED ? 'bg-red-50 text-red-600' : 'bg-amber-50 text-amber-600'
                              }`}>
                                {v.status === VideoStatus.DONE ? 'Hoàn tất' : v.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3 font-semibold text-slate-600">{parseUTCDate(v.uploadedAt)!.toLocaleDateString('vi-VN')}</td>
                            <td className="p-3">
                              <div className="flex gap-2">
                                <button onClick={() => openVideoDetail(v)} className="px-2 py-1 text-[10px] bg-white rounded border-2 border-slate-200 font-bold hover:bg-slate-100 text-slate-900 flex items-center gap-0.5 transition-colors">
                                  <span className="material-symbols-outlined text-xs">info</span> Chi tiết
                                </button>
                                <button onClick={() => handleDeleteVideo(v.videoId)} className="px-2 py-1 text-[10px] border-2 border-red-200 bg-red-50 hover:bg-red-100 text-red-600 rounded font-bold flex items-center gap-0.5 transition-colors">
                                  <span className="material-symbols-outlined text-xs">delete_outline</span> Xóa
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
                <div className="mt-4">
                  <PaginationControl
                    currentPage={systemVideosPage}
                    totalItems={systemVideosTotal || 0}
                    limit={10}
                    onPageChange={setSystemVideosPage}
                  />
                </div>
              </div>
            </div>
          )}

          {/* TAB: SYSTEM JOBS MANAGEMENT */}
          {activeTab === 'system-jobs' && (
            <div className="space-y-6">
              <h1 className="font-heading text-xl font-bold text-slate-900">Quản Lý Toàn Bộ Tác Vụ</h1>
              
              <div className="bg-white border-2 border-slate-200 rounded-xl p-5 shadow-none">
                <h3 className="font-heading text-xs font-bold text-slate-900 mb-4">Danh sách tiến trình xử lý nền</h3>
                <div className="overflow-x-auto border-2 border-slate-200 rounded-xl">
                  {jobsLoading ? (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Job ID</th>
                          <th className="p-3">Video ID</th>
                          <th className="p-3">Loại tác vụ</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Bắt đầu</th>
                          <th className="p-3">Hoàn thành</th>
                          <th className="p-3">Log lỗi</th>
                          <th className="p-3">Thao tác</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from({ length: 5 }).map((_, idx) => (
                          <Skeleton.TableRow key={idx} cells={8} />
                        ))}
                      </tbody>
                    </table>
                  ) : allJobs.length === 0 ? (
                    <div className="p-12 text-center text-slate-500 font-body">
                      <span className="material-symbols-outlined text-3xl">task</span>
                      <p className="text-xs mt-2">Chưa có tác vụ nào được chạy trên hệ thống.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse font-body">
                      <thead>
                        <tr className="bg-slate-50 border-b-2 border-slate-200 text-slate-900 font-bold">
                          <th className="p-3">Job ID</th>
                          <th className="p-3">Video ID</th>
                          <th className="p-3">Loại tác vụ</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Bắt đầu</th>
                          <th className="p-3">Hoàn thành</th>
                          <th className="p-3">Log lỗi</th>
                          <th className="p-3">Thao tác</th>
                        </tr>
                      </thead>
                      <tbody>
                        {allJobs.map((j: any) => (
                          <tr key={j.jobId} className="border-b-2 border-slate-100 hover:bg-slate-50">
                            <td className="p-3 font-mono text-[10px]" title={j.jobId}>#{j.jobId.substring(0, 8)}...</td>
                            <td className="p-3 font-mono text-[10px]" title={j.videoId}>#{j.videoId.substring(0, 8)}...</td>
                            <td className="p-3">
                              <span className="inline-block px-2 py-0.5 bg-indigo-50 text-indigo-600 font-bold rounded text-[9px] uppercase tracking-wider">
                                {j.jobType.toUpperCase()}
                              </span>
                            </td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                                j.status === 'done' || j.status === 'completed' ? 'bg-emerald-50 text-emerald-600' : j.status === 'failed' ? 'bg-red-50 text-red-600' : 'bg-amber-50 text-amber-600'
                              }`}>
                                {j.status === 'done' || j.status === 'completed' ? 'Hoàn tất' : j.status === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3 font-semibold text-slate-600">{j.startedAt ? parseUTCDate(j.startedAt)!.toLocaleTimeString('vi-VN') : '-'}</td>
                            <td className="p-3 font-semibold text-slate-600">{j.completedAt ? parseUTCDate(j.completedAt)!.toLocaleTimeString('vi-VN') : '-'}</td>
                            <td className="p-3 truncate max-w-[150px]" title={j.errorLog}>
                              {j.errorLog ? (
                                <span className="text-red-500 font-bold">{j.errorLog}</span>
                              ) : (
                                <span className="text-slate-400">-</span>
                              )}
                            </td>
                            <td className="p-3">
                              <button onClick={() => openJobDetail(j)} className="px-2 py-1 text-[10px] bg-white rounded border-2 border-slate-200 font-bold hover:bg-slate-100 text-slate-900 flex items-center gap-0.5 transition-colors">
                                <span className="material-symbols-outlined text-xs">info</span> Chi tiết
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
                <div className="mt-4">
                  <PaginationControl
                    currentPage={systemJobsPage}
                    totalItems={systemJobsTotal || 0}
                    limit={10}
                    onPageChange={setSystemJobsPage}
                  />
                </div>
              </div>
            </div>
          )}

        </div>
      </main>

      {/* Video Detail Modal */}
      {videoModalOpen && selectedVideo && (
        <div onClick={() => setVideoModalOpen(false)} className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4">
          <div onClick={(e) => e.stopPropagation()} className="w-full max-w-2xl bg-white border-2 border-slate-200 rounded-2xl shadow-none flex flex-col max-h-[90vh] overflow-hidden text-xs">
            <div className="flex justify-between items-center p-4 border-b-2 border-slate-100 shrink-0 bg-white">
              <h2 className="font-heading text-sm font-bold text-slate-900">Chi Tiết Video Hệ Thống</h2>
              <button onClick={() => setVideoModalOpen(false)} className="text-slate-400 hover:text-slate-900 text-xl shrink-0 font-bold leading-none">&times;</button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar text-slate-900 font-body">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Video ID:</span>
                  <span className="font-mono text-[10px] bg-slate-50 px-2 py-1 border-2 border-slate-200 rounded">{selectedVideo.videoId}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Mã người dùng sở hữu:</span>
                  <span className="font-mono text-[10px] bg-slate-50 px-2 py-1 border-2 border-slate-200 rounded">{selectedVideo.userId}</span>
                </div>
                <div className="flex flex-col gap-0.5 sm:col-span-2">
                  <span className="text-slate-500 font-bold">Tên File / Nguồn:</span>
                  <span className="font-mono text-[10px] bg-slate-50 px-2 py-1 border-2 border-slate-200 rounded break-all">
                    {selectedVideo.originalUrl ? 'YouTube Video' : (((selectedVideo.r2Url || selectedVideo.filePath).split('?')[0].split('/').pop()) || 'video.mp4')}
                  </span>
                </div>
                {(selectedVideo.originalUrl || selectedVideo.r2Url || selectedVideo.filePath) && (
                  <div className="flex flex-col gap-0.5 sm:col-span-2">
                    <span className="text-slate-500 font-bold">Liên kết xem video:</span>
                    <a 
                      href={selectedVideo.originalUrl || selectedVideo.r2Url || selectedVideo.filePath} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="text-primary hover:underline font-mono text-[10px] bg-slate-50 px-2 py-1 border-2 border-slate-200 rounded break-all flex items-center gap-1 w-fit font-bold"
                    >
                      <span className="material-symbols-outlined text-sm">open_in_new</span>
                      {selectedVideo.originalUrl || selectedVideo.r2Url || selectedVideo.filePath}
                    </a>
                  </div>
                )}
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Thời lượng:</span>
                  <span className="font-bold">{Math.round(selectedVideo.duration)}s (~{Math.round(selectedVideo.duration / 60)} phút)</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Ngôn ngữ:</span>
                  <span className="font-bold uppercase">{selectedVideo.language}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Trạng thái:</span>
                  <div>
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                      selectedVideo.status?.toLowerCase() === 'done' ? 'bg-emerald-50 text-emerald-600' : selectedVideo.status?.toLowerCase() === 'failed' ? 'bg-red-50 text-red-600' : 'bg-amber-50 text-amber-600'
                    }`}>
                      {selectedVideo.status?.toLowerCase() === 'done' ? 'Hoàn tất' : selectedVideo.status?.toLowerCase() === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                    </span>
                  </div>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Ngày tải lên:</span>
                  <span className="font-bold">{parseUTCDate(selectedVideo.uploadedAt)!.toLocaleString('vi-VN')}</span>
                </div>
              </div>

              {loadingSummary && (
                <div className="p-8 text-center text-slate-500 font-bold">
                  <span className="material-symbols-outlined text-2xl animate-spin text-primary mr-2">autorenew</span>
                  Đang tải tóm tắt &amp; phân tích AI...
                </div>
              )}

              {!loadingSummary && videoSummary && (
                <div className="pt-4 border-t-2 border-slate-100 space-y-4">
                  <div>
                    <h3 className="font-heading font-bold text-slate-900 mb-2 text-xs">Bản tóm tắt toàn văn</h3>
                    <div className="bg-slate-50 border-2 border-slate-200 p-4 rounded-xl text-slate-600 leading-relaxed max-h-40 overflow-y-auto custom-scrollbar font-semibold">
                      {videoSummary.summaryText}
                    </div>
                  </div>

                  {videoSummary.chaptersJson && videoSummary.chaptersJson.length > 0 && (
                    <div>
                      <h3 className="font-heading font-bold text-slate-900 mb-2 text-xs">Phân Chia Chương Bài Giảng ({videoSummary.chaptersJson.length})</h3>
                      <div className="space-y-2 max-h-44 overflow-y-auto custom-scrollbar">
                        {videoSummary.chaptersJson.map((ch: any, idx: number) => (
                          <div key={idx} className="flex gap-4 p-2.5 border-b-2 border-slate-100 last:border-0">
                            <span className="font-bold text-primary shrink-0 font-mono text-[10px] w-24">
                              {formatTime(ch.startTime)} - {formatTime(ch.endTime)}
                            </span>
                            <div>
                              <div className="font-bold text-slate-900">{ch.title}</div>
                              <div className="text-[10px] text-slate-500 mt-1 font-semibold">{ch.summary}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {videoSummary.keyframesJson && videoSummary.keyframesJson.length > 0 && (
                    <div>
                      <h3 className="font-heading font-bold text-slate-900 mb-2 text-xs">Slide Ảnh Keyframe Quan Trọng ({videoSummary.keyframesJson.length})</h3>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 max-h-44 overflow-y-auto p-1 custom-scrollbar">
                        {videoSummary.keyframesJson.map((kf: any, idx: number) => (
                          <div key={idx} className="border-2 border-slate-200 rounded-xl overflow-hidden bg-white">
                            <img src={kf.imageUrl?.startsWith('http') ? kf.imageUrl : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${kf.imageUrl}`} alt={kf.description} className="w-full h-20 object-cover" />
                            <div className="p-2 space-y-1">
                              <div className="font-bold text-primary text-[10px]">Mốc: {formatTime(kf.timestamp)}</div>
                              <div className="text-[10px] text-slate-500 font-semibold truncate" title={kf.description}>{kf.description}</div>
                              <div className="text-emerald-500 text-[10px] font-bold">CLIP: {Math.round(kf.importanceScore * 100)}%</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {!loadingSummary && !videoSummary && selectedVideo.status?.toLowerCase() === 'done' && (
                <div className="p-4 text-center text-red-500 font-bold">
                  <span className="material-symbols-outlined text-sm mr-1">warning</span>
                  Không tìm thấy tóm tắt cho video này.
                </div>
              )}
            </div>
            
            <div className="flex justify-end gap-2 p-4 border-t-2 border-slate-100 shrink-0 bg-white">
              <button onClick={() => setVideoModalOpen(false)} className="px-4 py-2 border-2 border-slate-200 hover:bg-slate-100 rounded-lg font-bold transition-colors text-slate-900">Đóng</button>
              {selectedVideo.status?.toLowerCase() === 'done' && (
                <Link to={`/results?videoId=${selectedVideo.videoId}`} onClick={() => setVideoModalOpen(false)} className="btn primary px-4 py-2 font-bold transition-colors flex items-center gap-1">
                  <span className="material-symbols-outlined text-xs">visibility</span> Xem Trang Client
                </Link>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Job Detail Modal */}
      {jobModalOpen && selectedJob && (
        <div onClick={() => setJobModalOpen(false)} className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4">
          <div onClick={(e) => e.stopPropagation()} className="w-full max-w-xl bg-white border-2 border-slate-200 rounded-2xl shadow-none flex flex-col max-h-[85vh] overflow-hidden text-xs">
            <div className="flex justify-between items-center p-4 border-b-2 border-slate-100 shrink-0 bg-white">
              <div className="flex items-center gap-2">
                <h2 className="font-heading text-sm font-bold text-slate-900">Chi Tiết Tác Vụ Celery</h2>
                {!(selectedJob.status === 'done' || selectedJob.status === 'completed' || selectedJob.status === 'failed') && (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[9px] font-bold bg-primary/10 text-primary animate-pulse border border-primary/20">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary"></span> Live Logs
                  </span>
                )}
              </div>
              <button onClick={() => setJobModalOpen(false)} className="text-slate-400 hover:text-slate-900 text-xl font-bold leading-none shrink-0">&times;</button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar text-slate-900 font-body">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Mã tác vụ:</span>
                  <span className="font-mono text-[10px] bg-slate-50 px-2 py-1 border-2 border-slate-200 rounded">{selectedJob.jobId || 'N/A'}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Video ID liên quan:</span>
                  <span className="font-mono text-[10px] bg-slate-50 px-2 py-1 border-2 border-slate-200 rounded">{selectedJob.videoId || 'N/A'}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Loại tác vụ:</span>
                  <div>
                    <span className="inline-block px-2.5 py-0.5 bg-indigo-50 text-indigo-600 border border-indigo-200 font-bold rounded text-[9px] uppercase tracking-wider">
                      {(selectedJob.jobType || 'summarize').toUpperCase()}
                    </span>
                  </div>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Trạng thái:</span>
                  <div>
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                      selectedJob.status === 'done' || selectedJob.status === 'completed' ? 'bg-emerald-50 text-emerald-600' : selectedJob.status === 'failed' ? 'bg-red-50 text-red-600' : 'bg-amber-50 text-amber-600'
                    }`}>
                      {selectedJob.status === 'done' || selectedJob.status === 'completed' ? 'Hoàn tất' : selectedJob.status === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                    </span>
                  </div>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Thời gian bắt đầu:</span>
                  <span className="font-bold">{selectedJob.startedAt ? parseUTCDate(selectedJob.startedAt)!.toLocaleString('vi-VN') : '-'}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-slate-500 font-bold">Thời gian hoàn thành:</span>
                  <span className="font-bold">{selectedJob.completedAt ? parseUTCDate(selectedJob.completedAt)!.toLocaleString('vi-VN') : '-'}</span>
                </div>
              </div>

              {selectedJob.errorLog && (
                <div className="pt-4 border-t-2 border-slate-100 space-y-2">
                  <h3 className="text-red-500 font-bold text-xs">Nhật ký báo lỗi chi tiết</h3>
                  <pre className="bg-red-50 text-red-600 border-2 border-red-200 p-4 rounded-xl text-[10px] font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed">
                    {selectedJob.errorLog}
                  </pre>
                </div>
              )}
              
              {!selectedJob.errorLog && (
                <div className="pt-4 border-t-2 border-slate-100 space-y-2">
                  <h3 className="text-emerald-500 font-bold text-xs">Nhật ký tiến trình hệ thống</h3>
                  <pre className="bg-slate-50 border-2 border-slate-200 p-4 rounded-xl text-[10px] font-mono overflow-x-auto whitespace-pre-wrap leading-relaxed text-slate-600 font-semibold">
                    {selectedJob.logs && selectedJob.logs.length > 0 
                      ? selectedJob.logs.join('\n') 
                      : getSystemLogs(selectedJob)}
                  </pre>
                </div>
              )}
            </div>
            
            <div className="flex justify-end gap-2 p-4 border-t-2 border-slate-100 shrink-0 bg-white">
              {(selectedJob.status === 'running' || selectedJob.status === 'pending' || selectedJob.status === 'RUNNING' || selectedJob.status === 'PENDING') && (
                <button 
                  onClick={async () => {
                    if (window.confirm("Bạn có chắc chắn muốn dừng tác vụ này không?")) {
                      try {
                        await api.cancelJob(selectedJob.jobId);
                        toast.success("Đã dừng tác vụ thành công!", "Thành công");
                        setJobModalOpen(false);
                        setJobsTrigger(prev => prev + 1);
                      } catch (err: any) {
                        toast.error(`Lỗi: ${err.message}`, "Thất bại");
                      }
                    }
                  }} 
                  className="px-4 py-2 bg-red-50 text-red-600 border-2 border-red-200 hover:bg-red-100 rounded-lg font-bold transition-colors"
                >
                  Dừng tác vụ
                </button>
              )}
              <button onClick={() => setJobModalOpen(false)} className="px-4 py-2 border-2 border-slate-200 hover:bg-slate-100 rounded-lg font-bold transition-colors text-slate-900">Đóng</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
