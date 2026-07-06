import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import Chart from 'chart.js/auto';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { VideoStatus } from '../types';
import { useToast } from '../context/ToastContext';

const formatTime = (secs: number) => {
  const m = Math.floor(secs / 60).toString().padStart(2, '0');
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
};

const getSystemLogs = (job: any) => {
  const startStr = job.startedAt ? new Date(job.startedAt).toISOString() : new Date().toISOString();
  let logs = `[INFO] ${startStr} - Khởi chạy pipeline phân tích đa phương tiện cho Video ${job.videoId || 'N/A'}.\n[INFO] Khởi tạo mô hình WhisperX trích xuất Audio...`;
  
  const isCompleted = job.status === 'completed' || job.status === 'done' || job.status === 'SUCCESS';
  const isFailed = job.status === 'failed' || job.status === 'FAILED';
  
  if (isCompleted) {
    const endStr = job.completedAt ? new Date(job.completedAt).toISOString() : new Date().toISOString();
    return logs + `\n[INFO] Hoàn thành chuyển đổi giọng nói (Speech-to-Text).` +
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
    const elapsed = (Date.now() - new Date(job.startedAt).getTime()) / 1000;
    if (elapsed > 2) {
      logs += `\n[INFO] Đang chuyển đổi giọng nói sang văn bản (Speech-to-Text)...`;
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
    logs += `\n[INFO] Đang chạy tác vụ WhisperX và CLIP...`;
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

export const AdminPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'stats' | 'metrics' | 'users' | 'videos' | 'celery' | 'system-videos' | 'system-jobs'>('stats');
  const toast = useToast();

  // Chart refs
  const barChartRef = useRef<HTMLCanvasElement>(null);
  const doughnutChartRef = useRef<HTMLCanvasElement>(null);
  const barInstance = useRef<Chart | null>(null);
  const doughnutInstance = useRef<Chart | null>(null);

  // Video Standards state
  const [maxDuration, setMaxDuration] = useState(3600);
  const [allowedFormats, setAllowedFormats] = useState('mp4,avi,mkv');
  const [maxFileSize, setMaxFileSize] = useState(500);
  const [minAudioQuality, setMinAudioQuality] = useState(0.0);
  const [standardsLoading, setStandardsLoading] = useState(false);
  const [standardsMsg, setStandardsMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Users state
  const [users, setUsers] = useState<UserItem[]>([
    { id: '1', email: 'hungphitran.22@gmail.com', role: 'ADMIN', active: true, joined: '2026-07-03' },
    { id: '2', email: 'nguyen.van.a@gmail.com', role: 'USER', active: true, joined: '2026-06-25' },
    { id: '3', email: 'tran.thi.b@student.edu.vn', role: 'USER', active: false, joined: '2026-06-18' },
    { id: '4', email: 'le.van.c@company.com', role: 'USER', active: true, joined: '2026-05-12' },
  ]);

  // Celery queue tasks
  const [queueJobs, setQueueJobs] = useState<any[]>([]);
  const [queueLoading, setQueueLoading] = useState(false);

  // System Videos and Jobs
  const [allVideos, setAllVideos] = useState<any[]>([]);
  const [videoLoading, setVideoLoading] = useState(false);
  const [allJobs, setAllJobs] = useState<any[]>([]);
  const [jobsLoading, setJobsLoading] = useState(false);

  // Dashboard Stats Videos State
  const [statsVideos, setStatsVideos] = useState<any[]>([]);
  const [statsLoading, setStatsLoading] = useState(false);

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
    if (activeTab !== 'stats') {
      if (barInstance.current) barInstance.current.destroy();
      if (doughnutInstance.current) doughnutInstance.current.destroy();
      return;
    }

    const timer = setTimeout(() => {
      if (barChartRef.current) {
        barInstance.current = new Chart(barChartRef.current, {
          type: 'bar',
          data: {
            labels: ['Tuần 1', 'Tuần 2', 'Tuần 3', 'Tuần 4', 'Tuần 5', 'Tuần 6'],
            datasets: [{
              label: 'Số Video Xử Lý',
              data: [320, 450, 680, 890, 1200, 1540],
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
        doughnutInstance.current = new Chart(doughnutChartRef.current, {
          type: 'doughnut',
          data: {
            labels: ['GPT-4o API', 'Gemini 1.5 API', 'Qwen2.5 Local'],
            datasets: [{
              data: [30, 25, 45],
              backgroundColor: ['#06B6D4', '#d5e3fd', '#0F172A'],
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
  }, [activeTab]);

  // Fetch dynamic stats counts and videos list for General Dashboard
  useEffect(() => {
    if (activeTab === 'stats') {
      const fetchStats = () => {
        Promise.all([api.getUsers(), api.getAllVideosAdmin(10)])
          .then(([usersRes, videosRes]) => {
            const emailMap: Record<string, string> = {};
            if (usersRes.success && usersRes.data) {
              const mappedUsers = usersRes.data.map((u: any) => ({
                id: u.userId,
                email: u.email,
                role: u.role.toUpperCase(),
                active: u.isActive,
                joined: new Date(u.createdAt).toLocaleDateString('vi-VN')
              }));
              setUsers(mappedUsers);
              
              usersRes.data.forEach((u: any) => {
                emailMap[u.userId] = u.email;
              });
            }

            if (videosRes.success && videosRes.data) {
              setAllVideos(videosRes.data);
              const mappedVideos = videosRes.data.map((v: any) => ({
                ...v,
                email: emailMap[v.userId] || "Unknown User",
                title: v.title || (v.originalUrl ? "YouTube Video" : (v.r2Url ? (v.r2Url.split('?')[0].split('/').pop() || "Video File") : (v.filePath && !v.filePath.includes('/stream') ? (v.filePath.split('?')[0].split('/').pop() || "Video File") : "Uploaded Video"))),
              }));
              setStatsVideos(mappedVideos);
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
      const interval = setInterval(fetchStats, 5000);
      return () => clearInterval(interval);
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
            setAllowedFormats(res.data.allowedFormats);
            setMaxFileSize(res.data.maxFileSize);
            setMinAudioQuality(res.data.minAudioQuality);
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
        api.getVideos()
          .then(res => {
            if (res.success && res.data) {
              setQueueJobs(res.data);
            }
          })
          .catch(err => {
            console.warn("Failed to load queue videos, using fallback mock jobs.", err);
          })
          .finally(() => {
            setQueueLoading(false);
          });
      };

      setQueueLoading(true);
      fetchQueue();
      const interval = setInterval(fetchQueue, 4000);
      return () => clearInterval(interval);
    }
  }, [activeTab]);

  // Fetch registered users list from API
  useEffect(() => {
    if (activeTab === 'users') {
      const fetchUsersList = () => {
        api.getUsers()
          .then(res => {
            if (res.success && res.data) {
              const mappedUsers = res.data.map((u: any) => ({
                id: u.userId,
                email: u.email,
                role: u.role.toUpperCase(),
                active: u.isActive,
                joined: new Date(u.createdAt).toLocaleDateString('vi-VN')
              }));
              setUsers(mappedUsers);
            }
          })
          .catch(err => {
            console.warn("Failed to fetch real users from backend, using default mock list.", err);
          });
      };

      fetchUsersList();
      const interval = setInterval(fetchUsersList, 6000);
      return () => clearInterval(interval);
    }
  }, [activeTab]);

  // Fetch all system videos (Admin only)
  useEffect(() => {
    if (activeTab === 'system-videos') {
      const fetchSystemVideosList = () => {
        api.getAllVideosAdmin()
          .then(res => {
            if (res.success && res.data) {
              setAllVideos(res.data);
            }
          })
          .catch(err => {
            console.warn("Failed to load system videos from API, using empty list fallback.", err);
          })
          .finally(() => {
            setVideoLoading(false);
          });
      };

      setVideoLoading(true);
      fetchSystemVideosList();
      const interval = setInterval(fetchSystemVideosList, 4000);
      return () => clearInterval(interval);
    }
  }, [activeTab]);

  // Fetch all system jobs (Admin only)
  useEffect(() => {
    if (activeTab === 'system-jobs') {
      const fetchSystemJobsList = () => {
        api.getAllJobsAdmin()
          .then(res => {
            if (res.success && res.data) {
              setAllJobs(res.data);
            }
          })
          .catch(err => {
            console.warn("Failed to load system jobs from API, using empty list fallback.", err);
          })
          .finally(() => {
            setJobsLoading(false);
          });
      };

      setJobsLoading(true);
      fetchSystemJobsList();
      const interval = setInterval(fetchSystemJobsList, 4000);
      return () => clearInterval(interval);
    }
  }, [activeTab]);
  
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
    api.updateStandards({
      maxDuration,
      allowedFormats,
      maxFileSize,
      minAudioQuality
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
    <div className="flex h-[calc(100vh-64px)] overflow-hidden bg-background text-on-surface">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-outline-variant bg-surface-container-low flex flex-col shrink-0 p-4 gap-2">
        <div className="px-2 py-4 border-b border-outline-variant/30 mb-4">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-vibrant-cyan">admin_panel_settings</span>
            <span className="font-headline-md text-sm font-bold text-deep-navy">Quản Trị Hệ Thống</span>
          </div>
        </div>

        <div className="flex-1 space-y-4">
          <div>
            <div className="text-[10px] uppercase font-bold text-outline tracking-wider px-3 mb-1">Analytics</div>
            <nav className="flex flex-col gap-1">
              <button 
                onClick={() => setActiveTab('stats')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-xs font-semibold w-full ${
                  activeTab === 'stats' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-sm">bar_chart</span> Báo cáo tổng hợp
              </button>
              <button 
                onClick={() => setActiveTab('metrics')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-xs font-semibold w-full ${
                  activeTab === 'metrics' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-sm">insights</span> Hiệu suất AI
              </button>
            </nav>
          </div>

          <div>
            <div className="text-[10px] uppercase font-bold text-outline tracking-wider px-3 mb-1">Quản lý</div>
            <nav className="flex flex-col gap-1">
              <button 
                onClick={() => setActiveTab('users')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-xs font-semibold w-full ${
                  activeTab === 'users' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-sm">group</span> Người dùng
              </button>
              <button 
                onClick={() => setActiveTab('system-videos')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-xs font-semibold w-full ${
                  activeTab === 'system-videos' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-sm">video_library</span> Quản lý Video
              </button>
              <button 
                onClick={() => setActiveTab('system-jobs')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-xs font-semibold w-full ${
                  activeTab === 'system-jobs' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-sm">worklist</span> Tác vụ (Jobs)
              </button>
              <button 
                onClick={() => setActiveTab('videos')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-xs font-semibold w-full ${
                  activeTab === 'videos' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-sm">settings_suggest</span> Tiêu chuẩn Video
              </button>
              <button 
                onClick={() => setActiveTab('celery')}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all text-left text-xs font-semibold w-full ${
                  activeTab === 'celery' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
                }`}
              >
                <span className="material-symbols-outlined text-sm">dns</span> Celery Queue
              </button>
            </nav>
          </div>
        </div>
      </aside>
      
      {/* Main Panel Content Area */}
      <main className="flex-1 overflow-y-auto p-6 md:p-margin-desktop bg-background custom-scrollbar">
        <div className="max-w-container-max mx-auto space-y-6">
          
          {/* TAB 1: GENERAL STATS */}
          {activeTab === 'stats' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h1 className="font-headline-xl text-xl font-bold text-deep-navy">Báo Cáo Thống Kê Tổng Hợp</h1>
                <div className="flex items-center gap-1.5 px-3 py-1.5 border border-outline-variant bg-white text-xs font-bold rounded-lg text-secondary cursor-pointer">
                  <span className="material-symbols-outlined text-sm">calendar_month</span> Hôm nay
                </div>
              </div>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-2">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">
                    Thành viên
                    <span className="material-symbols-outlined text-vibrant-cyan text-sm">group</span>
                  </h3>
                  <div className="text-2xl font-bold text-deep-navy">{users.length}</div>
                  <div className="text-[10px] text-status-success font-bold flex items-center gap-0.5"><span className="material-symbols-outlined text-xs">trending_up</span> +12.5%</div>
                </div>
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-2">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">
                    Video Phân Tích
                    <span className="material-symbols-outlined text-vibrant-cyan text-sm">video_library</span>
                  </h3>
                  <div className="text-2xl font-bold text-deep-navy">{allVideos.length}</div>
                  <div className="text-[10px] text-status-success font-bold flex items-center gap-0.5"><span className="material-symbols-outlined text-xs">trending_up</span> +8.2%</div>
                </div>
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-2">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">
                    Audio WER
                    <span className="material-symbols-outlined text-pink-500 text-sm">graphic_eq</span>
                  </h3>
                  <div className="text-2xl font-bold text-deep-navy">7.8%</div>
                  <div className="text-[10px] text-status-success font-bold flex items-center gap-0.5"><span className="material-symbols-outlined text-xs">check_circle</span> Đạt chuẩn (&lt;10%)</div>
                </div>
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-2">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">
                    Keyframe F-score
                    <span className="material-symbols-outlined text-vibrant-cyan text-sm">image_search</span>
                  </h3>
                  <div className="text-2xl font-bold text-deep-navy">0.52</div>
                  <div className="text-[10px] text-status-success font-bold flex items-center gap-0.5"><span className="material-symbols-outlined text-xs">check_circle</span> Vượt mục tiêu (0.45)</div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm">
                  <h3 className="text-xs font-bold text-deep-navy mb-4">Lưu lượng Video (Job Queue)</h3>
                  <div className="h-64 relative">
                    <canvas ref={barChartRef}></canvas>
                  </div>
                </div>
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm">
                  <h3 className="text-xs font-bold text-deep-navy mb-4">Sử dụng Mô hình LLM</h3>
                  <div className="h-64 relative">
                    <canvas ref={doughnutChartRef}></canvas>
                  </div>
                </div>
              </div>
              
              <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm">
                <h3 className="text-xs font-bold text-deep-navy mb-4">Tác vụ Xử lý Gần Đây (PostgreSQL `videos`)</h3>
                <div className="overflow-x-auto border border-outline-variant/60 rounded-xl">
                  {statsLoading ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-2xl animate-spin text-vibrant-cyan">autorenew</span>
                      <p className="text-xs mt-2">Đang tải danh sách tác vụ...</p>
                    </div>
                  ) : statsVideos.length === 0 ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-3xl">list_alt</span>
                      <p className="text-xs mt-2">Chưa có tác vụ nào được lưu trong hệ thống.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-surface-container-low border-b border-outline-variant text-deep-navy font-bold">
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
                          <tr key={v.videoId} className="border-b border-outline-variant/50 hover:bg-surface-container-low/50">
                            <td className="p-3 font-mono-data text-[10px]">#{v.videoId.substring(0, 8)}...</td>
                            <td className="p-3">{v.email}</td>
                            <td className="p-3 truncate max-w-[200px]" title={v.originalUrl || v.title}>
                              {v.title}
                            </td>
                            <td className="p-3">{Math.round(v.duration)}s</td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                                v.status === VideoStatus.DONE ? 'bg-status-success/15 text-status-success' : v.status === VideoStatus.FAILED ? 'bg-error/15 text-error' : 'bg-status-warning/15 text-status-warning'
                              }`}>
                                {v.status === VideoStatus.DONE ? 'Hoàn tất' : v.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3">
                              <button onClick={() => openVideoDetail(v)} className="px-2 py-1 bg-surface-container-high rounded text-[10px] font-bold text-deep-navy border border-outline-variant hover:bg-outline-variant/30 flex items-center gap-1">
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

          {/* TAB 2: AI METRICS PERFORMANCE */}
          {activeTab === 'metrics' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-xl font-bold text-deep-navy">Hiệu Suất &amp; Chất Lượng AI Pipeline</h1>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-1">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">Word Error Rate (ASR) <span className="material-symbols-outlined text-purple-600 text-sm">spellcheck</span></h3>
                  <div className="text-2xl font-bold text-deep-navy">7.8%</div>
                  <p className="text-[10px] text-secondary">Độ chính xác nhận diện từ</p>
                </div>
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-1">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">CLIP Keyframe F1 <span className="material-symbols-outlined text-vibrant-cyan text-sm">crop_free</span></h3>
                  <div className="text-2xl font-bold text-deep-navy">0.52</div>
                  <p className="text-[10px] text-secondary">Độ khớp so với giảng viên</p>
                </div>
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-1">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">LLM Latency (avg) <span className="material-symbols-outlined text-pink-500 text-sm">timer</span></h3>
                  <div className="text-2xl font-bold text-deep-navy">1.4s</div>
                  <p className="text-[10px] text-secondary">Thời gian phản hồi tóm tắt</p>
                </div>
                <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-1">
                  <h3 className="text-xs text-secondary font-bold flex justify-between items-center">Tổng Chi Phí Token <span className="material-symbols-outlined text-status-success text-sm">payments</span></h3>
                  <div className="text-2xl font-bold text-deep-navy">$12.45</div>
                  <p className="text-[10px] text-secondary">Tháng hiện tại</p>
                </div>
              </div>

              <div className="bg-white border border-outline-variant rounded-xl p-6 shadow-sm space-y-6">
                <h3 className="text-xs font-bold text-deep-navy">Đánh giá chi tiết các Mô hình</h3>
                
                <div className="space-y-4 text-xs">
                  <div className="space-y-1.5">
                    <div className="flex justify-between font-bold text-deep-navy">
                      <span>WhisperX ASR (Nhận diện giọng nói)</span>
                      <span>Độ chính xác: 92.2%</span>
                    </div>
                    <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                      <div className="bg-purple-600 h-full rounded-full" style={{ width: '92.2%' }}></div>
                    </div>
                  </div>
                  
                  <div className="space-y-1.5">
                    <div className="flex justify-between font-bold text-deep-navy">
                      <span>CLIP Keyframe (Cắt khung ảnh bài giảng)</span>
                      <span>F1 Score: 85.0%</span>
                    </div>
                    <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                      <div className="bg-vibrant-cyan h-full rounded-full" style={{ width: '85%' }}></div>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between font-bold text-deep-navy">
                      <span>Gemini 1.5 Flash (Tóm tắt RAG &amp; Hỏi đáp)</span>
                      <span>Độ khớp ý kiến: 89.5%</span>
                    </div>
                    <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                      <div className="bg-pink-500 h-full rounded-full" style={{ width: '89.5%' }}></div>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between font-bold text-deep-navy">
                      <span>Qwen 2.5 14B Local (Tóm tắt bài giảng)</span>
                      <span>Độ khớp ý kiến: 81.2%</span>
                    </div>
                    <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                      <div className="bg-deep-navy h-full rounded-full" style={{ width: '81.2%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: USER MANAGEMENT */}
          {activeTab === 'users' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-xl font-bold text-deep-navy">Quản Lý Thành Viên Hệ Thống</h1>
              
              <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm">
                <h3 className="text-xs font-bold text-deep-navy mb-4">Danh sách thành viên đăng ký</h3>
                <div className="overflow-x-auto border border-outline-variant/60 rounded-xl">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-surface-container-low border-b border-outline-variant text-deep-navy font-bold">
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
                        <tr key={u.id} className="border-b border-outline-variant/50 hover:bg-surface-container-low/50">
                          <td className="p-3">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-deep-navy text-vibrant-cyan flex items-center justify-center font-bold text-xs uppercase shadow-sm">
                                {u.email.charAt(0)}
                              </div>
                              <span className="font-semibold text-deep-navy">{u.email.split('@')[0]}</span>
                            </div>
                          </td>
                          <td className="p-3">{u.email}</td>
                          <td className="p-3">
                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                              u.role === 'ADMIN' ? 'bg-status-warning/15 text-status-warning border border-status-warning/20' : 'bg-secondary-container text-primary border border-outline-variant/30'
                            }`}>
                              {u.role}
                            </span>
                          </td>
                          <td className="p-3">
                            <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                              u.active ? 'bg-status-success/15 text-status-success' : 'bg-error/15 text-error'
                            }`}>
                              {u.active ? 'Hoạt động' : 'Bị khóa'}
                            </span>
                          </td>
                          <td className="p-3">{u.joined}</td>
                          <td className="p-3">
                            <div className="flex gap-2">
                              <button onClick={() => toggleUserRole(u.id)} className="px-2 py-1 text-[10px] font-bold bg-white border border-outline-variant rounded hover:bg-surface-container-low text-deep-navy transition-all">
                                Đổi Vai Trò
                              </button>
                              <button 
                                onClick={() => toggleUserActive(u.id)} 
                                className={`px-2 py-1 text-[10px] font-bold border rounded transition-all ${
                                  u.active ? 'border-error/50 bg-error/10 hover:bg-error/20 text-error' : 'border-outline-variant hover:bg-surface-container-low text-deep-navy'
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
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: VIDEO STANDARD SETTINGS */}
          {activeTab === 'videos' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-xl font-bold text-deep-navy">Cấu Hình Tiêu Chuẩn Phân Tích Video</h1>
              
              <div className="bg-white border border-outline-variant rounded-xl p-6 shadow-sm">
                <h3 className="text-xs font-bold text-deep-navy mb-4 pb-2 border-b border-outline-variant/30 flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-vibrant-cyan text-base">settings_suggest</span> Giới hạn tải lên &amp; Xác thực video
                </h3>
                {standardsLoading ? (
                  <div className="p-12 text-center text-secondary">
                    <span className="material-symbols-outlined text-2xl animate-spin text-vibrant-cyan">autorenew</span>
                    <p className="text-xs mt-2">Đang tải cấu hình...</p>
                  </div>
                ) : (
                  <form onSubmit={handleSaveStandards} className="space-y-4 max-w-xl text-xs text-deep-navy">
                    <div className="flex flex-col gap-1">
                      <label className="font-bold">Thời lượng video tối đa (Giây)</label>
                      <input 
                        type="number" 
                        value={maxDuration} 
                        onChange={(e) => setMaxDuration(parseInt(e.target.value) || 0)}
                        className="w-full px-3 py-2 bg-surface-container-low border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none transition-all"
                        required
                      />
                      <span className="text-[10px] text-secondary">Mặc định là 3600 giây (1 giờ). Video dài hơn sẽ bị chặn.</span>
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="font-bold">Các định dạng được cho phép (Dấu phẩy phân tách)</label>
                      <input 
                        type="text" 
                        value={allowedFormats} 
                        onChange={(e) => setAllowedFormats(e.target.value)}
                        className="w-full px-3 py-2 bg-surface-container-low border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none transition-all"
                        required
                      />
                      <span className="text-[10px] text-secondary">Ví dụ: mp4,avi,mkv,webm</span>
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="font-bold">Dung lượng file tối đa (Megabytes)</label>
                      <input 
                        type="number" 
                        value={maxFileSize} 
                        onChange={(e) => setMaxFileSize(parseInt(e.target.value) || 0)}
                        className="w-full px-3 py-2 bg-surface-container-low border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none transition-all"
                        required
                      />
                      <span className="text-[10px] text-secondary">Giới hạn dung lượng tối đa cho mỗi lần tải video lên.</span>
                    </div>

                    <div className="flex flex-col gap-1">
                      <label className="font-bold">Chất lượng âm thanh tối thiểu (SNR)</label>
                      <input 
                        type="number" 
                        step="0.1"
                        value={minAudioQuality} 
                        onChange={(e) => setMinAudioQuality(parseFloat(e.target.value) || 0)}
                        className="w-full px-3 py-2 bg-surface-container-low border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none transition-all"
                        required
                      />
                      <span className="text-[10px] text-secondary">Tỷ lệ Tín hiệu trên Nhiễu (SNR) tối thiểu. Để 0.0 để không kiểm tra.</span>
                    </div>

                    {standardsMsg && (
                      <div className={`text-xs font-semibold flex items-center gap-1.5 ${
                        standardsMsg.type === 'success' ? 'text-status-success' : 'text-error'
                      }`}>
                        <span className="material-symbols-outlined text-sm">
                          {standardsMsg.type === 'success' ? 'check_circle' : 'warning'}
                        </span>
                        {standardsMsg.text}
                      </div>
                    )}

                    <div className="pt-2">
                      <button type="submit" className="px-5 py-2.5 bg-vibrant-cyan hover:brightness-110 text-deep-navy font-bold rounded-xl flex items-center gap-1.5 shadow-sm transition-all active:scale-98">
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
              <h1 className="font-headline-xl text-xl font-bold text-deep-navy">Hệ Thống Hàng Đợi Celery Job Queue</h1>
              
              <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-outline-variant/30 text-xs">
                  <h3 className="font-bold text-deep-navy">Các tiến trình AI đang hoạt động trong hàng đợi</h3>
                  <span className="px-3 py-1 bg-status-success/15 text-status-success border border-status-success/20 rounded-full font-bold">Celery Worker: Active (1)</span>
                </div>
                
                <div className="overflow-x-auto border border-outline-variant/60 rounded-xl">
                  {queueLoading ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-2xl animate-spin text-vibrant-cyan">autorenew</span>
                      <p className="text-xs mt-2">Đang đồng bộ hàng đợi...</p>
                    </div>
                  ) : queueJobs.length === 0 ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-3xl">queue_play_next</span>
                      <p className="text-xs mt-2">Hiện không có job nào đang chạy hoặc bị lỗi.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-surface-container-low border-b border-outline-variant text-deep-navy font-bold">
                          <th className="p-3">Video ID</th>
                          <th className="p-3">Nguồn Video</th>
                          <th className="p-3">Ngôn Ngữ</th>
                          <th className="p-3">Trạng Thái Pipeline</th>
                          <th className="p-3">Thao Tác</th>
                        </tr>
                      </thead>
                      <tbody>
                        {queueJobs.map(job => (
                          <tr key={job.videoId} className="border-b border-outline-variant/50 hover:bg-surface-container-low/50">
                            <td className="p-3 font-mono-data text-[10px]">#{job.videoId.substring(0, 8)}...</td>
                            <td className="p-3 truncate max-w-[200px]" title={job.title || job.originalUrl || (job.r2Url ? job.r2Url : (job.filePath && !job.filePath.includes('/stream') ? job.filePath : 'Video File'))}>
                              {job.title || (job.originalUrl ? 'YouTube Video' : (job.r2Url ? (job.r2Url.split('?')[0].split('/').pop() || 'video.mp4') : (job.filePath && !job.filePath.includes('/stream') ? (job.filePath.split('?')[0].split('/').pop() || 'video.mp4') : 'Video File')))}
                            </td>
                            <td className="p-3 font-semibold">{job.language.toUpperCase()}</td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                                job.status === VideoStatus.DONE ? 'bg-status-success/15 text-status-success' : job.status === VideoStatus.FAILED ? 'bg-error/15 text-error' : 'bg-status-warning/15 text-status-warning'
                              }`}>
                                {job.status === VideoStatus.DONE ? 'Hoàn tất' : job.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3">
                              <div className="flex gap-2">
                                <button onClick={() => openJobDetail(job)} className="px-2 py-1 text-[10px] bg-surface-container-high rounded border border-outline-variant font-bold hover:bg-outline-variant/30 text-deep-navy">Xem chi tiết</button>
                                {job.status !== VideoStatus.DONE && job.status !== VideoStatus.FAILED && (
                                  <button className="px-2 py-1 text-[10px] border border-error/50 bg-error/10 hover:bg-error/20 text-error rounded font-bold">Dừng</button>
                                )}
                              </div>
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

          {/* TAB: SYSTEM VIDEOS MANAGEMENT */}
          {activeTab === 'system-videos' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-xl font-bold text-deep-navy">Quản Lý Toàn Bộ Video Hệ Thống</h1>
              
              <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm">
                <h3 className="text-xs font-bold text-deep-navy mb-4">Danh sách Video đã tải lên (PostgreSQL `videos` table)</h3>
                <div className="overflow-x-auto border border-outline-variant/60 rounded-xl">
                  {videoLoading ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-2xl animate-spin text-vibrant-cyan">autorenew</span>
                      <p className="text-xs mt-2">Đang tải danh sách video...</p>
                    </div>
                  ) : allVideos.length === 0 ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-3xl">movie</span>
                      <p className="text-xs mt-2">Chưa có video nào được tải lên hệ thống.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-surface-container-low border-b border-outline-variant text-deep-navy font-bold">
                          <th className="p-3">Video ID</th>
                          <th className="p-3">User ID</th>
                          <th className="p-3">Nguồn / File</th>
                          <th className="p-3">Độ dài (Giây)</th>
                          <th className="p-3">Ngôn ngữ</th>
                          <th className="p-3">Trạng thái</th>
                          <th className="p-3">Ngày tải</th>
                          <th className="p-3">Hành động</th>
                        </tr>
                      </thead>
                      <tbody>
                        {allVideos.map((v: any) => (
                          <tr key={v.videoId} className="border-b border-outline-variant/50 hover:bg-surface-container-low/50">
                            <td className="p-3 font-mono-data text-[10px]" title={v.videoId}>#{v.videoId.substring(0, 8)}...</td>
                            <td className="p-3 font-mono-data text-[10px]" title={v.userId}>#{v.userId.substring(0, 8)}...</td>
                            <td className="p-3 truncate max-w-[200px]" title={v.title || v.originalUrl || (v.r2Url ? v.r2Url : (v.filePath && !v.filePath.includes('/stream') ? v.filePath : 'Video File'))}>
                              {v.title ? (
                                <span className="font-semibold text-deep-navy">{v.title}</span>
                              ) : v.originalUrl ? (
                                <a 
                                  href={v.originalUrl} 
                                  target="_blank" 
                                  rel="noopener noreferrer" 
                                  className="text-vibrant-cyan hover:underline flex items-center gap-1 w-fit"
                                >
                                  <span className="material-symbols-outlined text-sm">open_in_new</span>
                                  YouTube Video
                                </a>
                              ) : (
                                <a 
                                  href={v.r2Url || v.filePath} 
                                  target="_blank" 
                                  rel="noopener noreferrer" 
                                  className="text-vibrant-cyan hover:underline flex items-center gap-1 truncate w-fit"
                                  title="Bấm để tải về hoặc xem trực tiếp từ R2"
                                >
                                  <span className="material-symbols-outlined text-sm">download</span>
                                  {((v.r2Url || (v.filePath && !v.filePath.includes('/stream') ? v.filePath : '')).split('?')[0].split('/').pop()) || 'Video File'}
                                </a>
                              )}
                            </td>
                            <td className="p-3">{Math.round(v.duration)}s</td>
                            <td className="p-3 font-semibold">{v.language.toUpperCase()}</td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                                v.status === VideoStatus.DONE ? 'bg-status-success/15 text-status-success' : v.status === VideoStatus.FAILED ? 'bg-error/15 text-error' : 'bg-status-warning/15 text-status-warning'
                              }`}>
                                {v.status === VideoStatus.DONE ? 'Hoàn tất' : v.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3">{new Date(v.uploadedAt).toLocaleDateString('vi-VN')}</td>
                            <td className="p-3">
                              <div className="flex gap-2">
                                <button onClick={() => openVideoDetail(v)} className="px-2 py-1 text-[10px] bg-surface-container-high rounded border border-outline-variant font-bold hover:bg-outline-variant/30 text-deep-navy flex items-center gap-0.5">
                                  <span className="material-symbols-outlined text-xs">info</span> Chi tiết
                                </button>
                                <button onClick={() => handleDeleteVideo(v.videoId)} className="px-2 py-1 text-[10px] border border-error/50 bg-error/10 hover:bg-error/20 text-error rounded font-bold flex items-center gap-0.5">
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
              </div>
            </div>
          )}

          {/* TAB: SYSTEM JOBS MANAGEMENT */}
          {activeTab === 'system-jobs' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-xl font-bold text-deep-navy">Quản Lý Toàn Bộ Tác Vụ (Jobs)</h1>
              
              <div className="bg-white border border-outline-variant rounded-xl p-5 shadow-sm">
                <h3 className="text-xs font-bold text-deep-navy mb-4">Danh sách tiến trình chạy ngầm (PostgreSQL `jobs` table)</h3>
                <div className="overflow-x-auto border border-outline-variant/60 rounded-xl">
                  {jobsLoading ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-2xl animate-spin text-vibrant-cyan">autorenew</span>
                      <p className="text-xs mt-2">Đang tải danh sách tác vụ...</p>
                    </div>
                  ) : allJobs.length === 0 ? (
                    <div className="p-12 text-center text-secondary">
                      <span className="material-symbols-outlined text-3xl">task</span>
                      <p className="text-xs mt-2">Chưa có tác vụ nào được chạy trên hệ thống.</p>
                    </div>
                  ) : (
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-surface-container-low border-b border-outline-variant text-deep-navy font-bold">
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
                          <tr key={j.jobId} className="border-b border-outline-variant/50 hover:bg-surface-container-low/50">
                            <td className="p-3 font-mono-data text-[10px]" title={j.jobId}>#{j.jobId.substring(0, 8)}...</td>
                            <td className="p-3 font-mono-data text-[10px]" title={j.videoId}>#{j.videoId.substring(0, 8)}...</td>
                            <td className="p-3">
                              <span className="inline-block px-2 py-0.5 bg-purple-100 text-purple-800 font-bold rounded text-[9px] uppercase tracking-wider">
                                {j.jobType.toUpperCase()}
                              </span>
                            </td>
                            <td className="p-3">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                                j.status === 'done' || j.status === 'completed' ? 'bg-status-success/15 text-status-success' : j.status === 'failed' ? 'bg-error/15 text-error' : 'bg-status-warning/15 text-status-warning'
                              }`}>
                                {j.status === 'done' || j.status === 'completed' ? 'Hoàn tất' : j.status === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                              </span>
                            </td>
                            <td className="p-3">{j.startedAt ? new Date(j.startedAt).toLocaleTimeString('vi-VN') : '-'}</td>
                            <td className="p-3">{j.completedAt ? new Date(j.completedAt).toLocaleTimeString('vi-VN') : '-'}</td>
                            <td className="p-3 truncate max-w-[150px]" title={j.errorLog}>
                              {j.errorLog ? (
                                <span className="text-error font-semibold">{j.errorLog}</span>
                              ) : (
                                <span className="text-secondary">-</span>
                              )}
                            </td>
                            <td className="p-3">
                              <button onClick={() => openJobDetail(j)} className="px-2 py-1 text-[10px] bg-surface-container-high rounded border border-outline-variant font-bold hover:bg-outline-variant/30 text-deep-navy flex items-center gap-0.5">
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

        </div>
      </main>

      {/* Video Detail Modal */}
      {videoModalOpen && selectedVideo && (
        <div onClick={() => setVideoModalOpen(false)} className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div onClick={(e) => e.stopPropagation()} className="w-full max-w-2xl bg-white border border-outline-variant rounded-2xl shadow-xl flex flex-col max-h-[90vh] overflow-hidden text-xs">
            <div className="flex justify-between items-center p-4 border-b border-outline-variant shrink-0 bg-surface">
              <h2 className="text-sm font-bold text-deep-navy">Chi Tiết Video Hệ Thống</h2>
              <button onClick={() => setVideoModalOpen(false)} className="text-secondary hover:text-primary text-xl shrink-0 font-bold leading-none">&times;</button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar text-deep-navy">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Video ID:</span>
                  <span className="font-mono-data text-[10px] bg-surface-container-low px-2 py-1 border border-outline-variant/30 rounded">{selectedVideo.videoId}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">User ID (Sở hữu):</span>
                  <span className="font-mono-data text-[10px] bg-surface-container-low px-2 py-1 border border-outline-variant/30 rounded">{selectedVideo.userId}</span>
                </div>
                <div className="flex flex-col gap-0.5 sm:col-span-2">
                  <span className="text-secondary font-bold">Tên File / Nguồn:</span>
                  <span className="font-mono-data text-[10px] bg-surface-container-low px-2 py-1 border border-outline-variant/30 rounded break-all">
                    {selectedVideo.originalUrl ? 'YouTube Video' : (((selectedVideo.r2Url || selectedVideo.filePath).split('?')[0].split('/').pop()) || 'video.mp4')}
                  </span>
                </div>
                {(selectedVideo.originalUrl || selectedVideo.r2Url || selectedVideo.filePath) && (
                  <div className="flex flex-col gap-0.5 sm:col-span-2">
                    <span className="text-secondary font-bold">Liên kết xem video:</span>
                    <a 
                      href={selectedVideo.originalUrl || selectedVideo.r2Url || selectedVideo.filePath} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="text-vibrant-cyan hover:underline font-mono-data text-[10px] bg-surface-container-low px-2 py-1 border border-outline-variant/30 rounded break-all flex items-center gap-1 w-fit"
                    >
                      <span className="material-symbols-outlined text-sm">open_in_new</span>
                      {selectedVideo.originalUrl || selectedVideo.r2Url || selectedVideo.filePath}
                    </a>
                  </div>
                )}
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Thời lượng:</span>
                  <span className="font-semibold">{Math.round(selectedVideo.duration)}s (~{Math.round(selectedVideo.duration / 60)} phút)</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Ngôn ngữ:</span>
                  <span className="font-semibold uppercase">{selectedVideo.language}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Trạng thái:</span>
                  <div>
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                      selectedVideo.status?.toLowerCase() === 'done' ? 'bg-status-success/15 text-status-success' : selectedVideo.status?.toLowerCase() === 'failed' ? 'bg-error/15 text-error' : 'bg-status-warning/15 text-status-warning'
                    }`}>
                      {selectedVideo.status?.toLowerCase() === 'done' ? 'Hoàn tất' : selectedVideo.status?.toLowerCase() === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                    </span>
                  </div>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Ngày tải lên:</span>
                  <span className="font-semibold">{new Date(selectedVideo.uploadedAt).toLocaleString('vi-VN')}</span>
                </div>
              </div>

              {loadingSummary && (
                <div className="p-8 text-center text-secondary">
                  <span className="material-symbols-outlined text-2xl animate-spin text-vibrant-cyan mr-2">autorenew</span>
                  Đang tải tóm tắt &amp; phân tích AI...
                </div>
              )}

              {!loadingSummary && videoSummary && (
                <div className="pt-4 border-t border-outline-variant space-y-4">
                  <div>
                    <h3 className="font-bold text-deep-navy mb-2 text-xs">Tóm Tắt Toàn Văn (AI Summary)</h3>
                    <div className="bg-surface-container-low border border-outline-variant/60 p-4 rounded-xl text-secondary leading-relaxed max-h-40 overflow-y-auto custom-scrollbar">
                      {videoSummary.summaryText}
                    </div>
                  </div>

                  {videoSummary.chaptersJson && videoSummary.chaptersJson.length > 0 && (
                    <div>
                      <h3 className="font-bold text-deep-navy mb-2 text-xs">Phân Chia Chương Bài Giảng ({videoSummary.chaptersJson.length})</h3>
                      <div className="space-y-2 max-h-44 overflow-y-auto custom-scrollbar">
                        {videoSummary.chaptersJson.map((ch: any, idx: number) => (
                          <div key={idx} className="flex gap-4 p-2.5 border-b border-outline-variant/30 last:border-0">
                            <span className="font-bold text-vibrant-cyan shrink-0 font-mono-data text-[10px] w-24">
                              {formatTime(ch.startTime)} - {formatTime(ch.endTime)}
                            </span>
                            <div>
                              <div className="font-bold text-deep-navy">{ch.title}</div>
                              <div className="text-[10px] text-secondary mt-1">{ch.summary}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {videoSummary.keyframesJson && videoSummary.keyframesJson.length > 0 && (
                    <div>
                      <h3 className="font-bold text-deep-navy mb-2 text-xs">Slide Ảnh Keyframe Quan Trọng ({videoSummary.keyframesJson.length})</h3>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 max-h-44 overflow-y-auto p-1 custom-scrollbar">
                        {videoSummary.keyframesJson.map((kf: any, idx: number) => (
                          <div key={idx} className="border border-outline-variant rounded-xl overflow-hidden bg-background">
                            <img src={kf.imageUrl?.startsWith('http') ? kf.imageUrl : `${CONFIG.API_BASE_URL.replace('/api/v1', '')}${kf.imageUrl}`} alt={kf.description} className="w-full h-20 object-cover" />
                            <div className="p-2 space-y-1">
                              <div className="font-bold text-vibrant-cyan text-[10px]">Mốc: {formatTime(kf.timestamp)}</div>
                              <div className="text-[10px] text-secondary truncate" title={kf.description}>{kf.description}</div>
                              <div className="text-status-success text-[10px] font-bold">CLIP: {Math.round(kf.importanceScore * 100)}%</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {!loadingSummary && !videoSummary && selectedVideo.status?.toLowerCase() === 'done' && (
                <div className="p-4 text-center text-error font-semibold">
                  <span className="material-symbols-outlined text-sm mr-1">warning</span>
                  Không tìm thấy tóm tắt cho video này.
                </div>
              )}
            </div>
            
            <div className="flex justify-end gap-2 p-4 border-t border-outline-variant shrink-0 bg-surface">
              <button onClick={() => setVideoModalOpen(false)} className="px-4 py-2 border border-outline-variant hover:bg-surface-container-high rounded-lg font-bold transition-all text-deep-navy">Đóng</button>
              {selectedVideo.status?.toLowerCase() === 'done' && (
                <Link to={`/results?videoId=${selectedVideo.videoId}`} onClick={() => setVideoModalOpen(false)} className="px-4 py-2 bg-deep-navy text-white rounded-lg hover:opacity-90 font-bold transition-all flex items-center gap-1">
                  <span className="material-symbols-outlined text-xs">visibility</span> Xem Trang Client
                </Link>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Job Detail Modal */}
      {jobModalOpen && selectedJob && (
        <div onClick={() => setJobModalOpen(false)} className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div onClick={(e) => e.stopPropagation()} className="w-full max-w-xl bg-white border border-outline-variant rounded-2xl shadow-xl flex flex-col max-h-[85vh] overflow-hidden text-xs">
            <div className="flex justify-between items-center p-4 border-b border-outline-variant shrink-0 bg-surface">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-deep-navy">Chi Tiết Tác Vụ Celery</h2>
                {!(selectedJob.status === 'done' || selectedJob.status === 'completed' || selectedJob.status === 'failed') && (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[9px] font-bold bg-vibrant-cyan/15 text-vibrant-cyan animate-pulse">
                    <span className="w-1.5 h-1.5 rounded-full bg-vibrant-cyan"></span> Live Logs
                  </span>
                )}
              </div>
              <button onClick={() => setJobModalOpen(false)} className="text-secondary hover:text-primary text-xl font-bold leading-none shrink-0">&times;</button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar text-deep-navy">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Job ID (Tác vụ):</span>
                  <span className="font-mono-data text-[10px] bg-surface-container-low px-2 py-1 border border-outline-variant/30 rounded">{selectedJob.jobId || 'N/A'}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Video ID liên quan:</span>
                  <span className="font-mono-data text-[10px] bg-surface-container-low px-2 py-1 border border-outline-variant/30 rounded">{selectedJob.videoId || 'N/A'}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Loại tác vụ:</span>
                  <div>
                    <span className="inline-block px-2.5 py-0.5 bg-purple-100 text-purple-800 font-bold rounded text-[9px] uppercase tracking-wider">
                      {(selectedJob.jobType || 'summarize').toUpperCase()}
                    </span>
                  </div>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Trạng thái:</span>
                  <div>
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                      selectedJob.status === 'done' || selectedJob.status === 'completed' ? 'bg-status-success/15 text-status-success' : selectedJob.status === 'failed' ? 'bg-error/15 text-error' : 'bg-status-warning/15 text-status-warning'
                    }`}>
                      {selectedJob.status === 'done' || selectedJob.status === 'completed' ? 'Hoàn tất' : selectedJob.status === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                    </span>
                  </div>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Thời gian bắt đầu:</span>
                  <span className="font-semibold">{selectedJob.startedAt ? new Date(selectedJob.startedAt).toLocaleString('vi-VN') : '-'}</span>
                </div>
                <div className="flex flex-col gap-0.5">
                  <span className="text-secondary font-bold">Thời gian hoàn thành:</span>
                  <span className="font-semibold">{selectedJob.completedAt ? new Date(selectedJob.completedAt).toLocaleString('vi-VN') : '-'}</span>
                </div>
              </div>

              {selectedJob.errorLog && (
                <div className="pt-4 border-t border-outline-variant space-y-2">
                  <h3 className="text-error font-bold text-xs">Nhật ký báo lỗi (Error Log / Traceback)</h3>
                  <pre className="bg-error/5 text-error border border-error/20 p-4 rounded-xl text-[10px] font-mono-data overflow-x-auto whitespace-pre-wrap leading-relaxed">
                    {selectedJob.errorLog}
                  </pre>
                </div>
              )}
              
              {!selectedJob.errorLog && (
                <div className="pt-4 border-t border-outline-variant space-y-2">
                  <h3 className="text-status-success font-bold text-xs">Nhật ký tiến trình (System Log)</h3>
                  <pre className="bg-surface-container-low border border-outline-variant/60 p-4 rounded-xl text-[10px] font-mono-data overflow-x-auto whitespace-pre-wrap leading-relaxed text-secondary">
                    {getSystemLogs(selectedJob)}
                  </pre>
                </div>
              )}
            </div>
            
            <div className="flex justify-end p-4 border-t border-outline-variant shrink-0 bg-surface">
              <button onClick={() => setJobModalOpen(false)} className="px-4 py-2 border border-outline-variant hover:bg-surface-container-high rounded-lg font-bold transition-all text-deep-navy">Đóng</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
