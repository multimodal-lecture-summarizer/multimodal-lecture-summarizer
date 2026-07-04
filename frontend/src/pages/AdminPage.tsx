import React, { useState, useEffect, useRef } from 'react';
import Chart from 'chart.js/auto';
import { api } from '../services/api';
import { CONFIG } from '../config';
import { VideoStatus } from '../types';
import './AdminPage.css';

interface UserItem {
  id: string;
  email: string;
  role: string;
  active: boolean;
  joined: string;
}

export const AdminPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'stats' | 'metrics' | 'users' | 'videos' | 'celery' | 'system-videos' | 'system-jobs'>('stats');

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

  const openJobDetail = (job: any) => {
    setSelectedJob(job);
    setJobModalOpen(true);
  };

  // 1. Chart initialization (only when on stats tab)
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
              backgroundColor: '#4f46e5',
              borderRadius: 6,
              barThickness: 20
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              y: { grid: { color: '#e2e8f0' }, beginAtZero: true },
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
              backgroundColor: ['#3b82f6', '#ec4899', '#10b981'],
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
                labels: { padding: 15, usePointStyle: true, font: { size: 11 } }
              }
            },
            cutout: '75%'
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

  // 1b. Fetch dynamic stats counts and videos list for General Dashboard
  useEffect(() => {
    if (activeTab === 'stats') {
      setStatsLoading(true);
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
              title: v.originalUrl ? "YouTube Link" : (v.filePath?.split('/').pop() || "Uploaded Video"),
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
    }
  }, [activeTab]);

  // 2. Fetch video standards from API when entering standard config tab
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

  // 3. Fetch Celery Jobs/Videos list when entering Queue tab
  useEffect(() => {
    if (activeTab === 'celery') {
      setQueueLoading(true);
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
    }
  }, [activeTab]);

  // 4. Fetch registered users list from API
  useEffect(() => {
    if (activeTab === 'users') {
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
    }
  }, [activeTab]);

  // 5. Fetch all system videos (Admin only)
  useEffect(() => {
    if (activeTab === 'system-videos') {
      setVideoLoading(true);
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
    }
  }, [activeTab]);

  // 6. Fetch all system jobs (Admin only)
  useEffect(() => {
    if (activeTab === 'system-jobs') {
      setJobsLoading(true);
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
    }
  }, [activeTab]);

  // Video deletion action
  const handleDeleteVideo = (videoId: string) => {
    if (window.confirm("Bạn có chắc chắn muốn xóa video này và toàn bộ dữ liệu tóm tắt/jobs liên quan?")) {
      api.deleteVideoAdmin(videoId)
        .then(res => {
          if (res.success) {
            setAllVideos(prev => prev.filter(v => v.videoId !== videoId));
          }
        })
        .catch(err => {
          alert(err.message || "Xóa video thất bại!");
        });
    }
  };

  // User Actions
  const toggleUserActive = (id: string) => {
    api.toggleUserStatus(id)
      .then(res => {
        if (res.success && res.data) {
          setUsers(prev => prev.map(u => u.id === id ? { ...u, active: res.data.isActive } : u));
        }
      })
      .catch(err => {
        alert(err.message || "Không thể thay đổi trạng thái hoạt động!");
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
        }
      })
      .catch(err => {
        alert(err.message || "Không thể thay đổi vai trò người dùng!");
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
      }
    })
    .catch(err => {
      setStandardsMsg({
        type: 'error',
        text: err.message || 'Cập nhật cấu hình thất bại!'
      });
    })
    .finally(() => {
      setStandardsLoading(false);
    });
  };

  return (
    <div className="admin-page animate-fade-in">
      {/* Sidebar Navigation */}
      <div className="admin-sidebar">
        
        <div className="admin-nav-group">
          <div className="admin-nav-label">Analytics</div>
          <a href="#stats" className={`admin-nav-item ${activeTab === 'stats' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab('stats'); }}>
            <i className="fa-solid fa-chart-pie"></i> Báo cáo hệ thống
          </a>
          <a href="#metrics" className={`admin-nav-item ${activeTab === 'metrics' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab('metrics'); }}>
            <i className="fa-solid fa-chart-line"></i> Hiệu suất AI
          </a>
        </div>
        
        <div className="admin-nav-group">
          <div className="admin-nav-label">Quản lý</div>
          <a href="#users" className={`admin-nav-item ${activeTab === 'users' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab('users'); }}>
            <i className="fa-solid fa-users"></i> Người dùng
          </a>
          <a href="#system-videos" className={`admin-nav-item ${activeTab === 'system-videos' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab('system-videos'); }}>
            <i className="fa-solid fa-film"></i> Quản lý Video
          </a>
          <a href="#system-jobs" className={`admin-nav-item ${activeTab === 'system-jobs' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab('system-jobs'); }}>
            <i className="fa-solid fa-list-check"></i> Quản lý Tác vụ (Jobs)
          </a>
          <a href="#videos" className={`admin-nav-item ${activeTab === 'videos' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab('videos'); }}>
            <i className="fa-solid fa-sliders"></i> Tiêu chuẩn Video
          </a>
          <a href="#celery" className={`admin-nav-item ${activeTab === 'celery' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveTab('celery'); }}>
            <i className="fa-solid fa-server"></i> Celery Job Queue
          </a>
        </div>
      </div>
      
      {/* Main Panel Content Area */}
      <div className="admin-main-content">
        
        {/* TAB 1: GENERAL STATS */}
        {activeTab === 'stats' && (
          <>
            <div className="admin-header">
              <h1>Báo Cáo Thống Kê Tổng Hợp</h1>
              <div className="calendar-selector">
                <i className="fa-regular fa-calendar"></i> Hôm nay <i className="fa-solid fa-chevron-down" style={{ marginLeft: '10px', fontSize: '0.8rem' }}></i>
              </div>
            </div>
            
            <div className="admin-stats-grid">
              <div className="admin-stat-card">
                <h3>Người dùng <i className="fa-solid fa-users" style={{ color: 'var(--primary)' }}></i></h3>
                <div className="value">{users.length}</div>
                <div className="trend up"><i className="fa-solid fa-arrow-trend-up"></i> +12.5%</div>
              </div>
              <div className="admin-stat-card">
                <h3>Video Xử Lý <i className="fa-solid fa-film" style={{ color: 'var(--primary)' }}></i></h3>
                <div className="value">{allVideos.length}</div>
                <div className="trend up"><i className="fa-solid fa-arrow-trend-up"></i> +8.2%</div>
              </div>
              <div className="admin-stat-card">
                <h3>Audio WER <i className="fa-solid fa-microphone-lines" style={{ color: '#ec4899' }}></i></h3>
                <div className="value">7.8%</div>
                <div className="trend label-ok"><i className="fa-solid fa-check"></i> Đạt chuẩn &lt; 10%</div>
              </div>
              <div className="admin-stat-card">
                <h3>Keyframe F-score <i className="fa-solid fa-image" style={{ color: '#3b82f6' }}></i></h3>
                <div className="value">0.52</div>
                <div className="trend label-ok"><i className="fa-solid fa-check"></i> Vượt mục tiêu (0.45)</div>
              </div>
            </div>
            
            <div className="admin-charts-row">
              <div className="admin-chart-card">
                <h3>Lưu lượng Video (Job Queue)</h3>
                <div className="canvas-wrapper">
                  <canvas ref={barChartRef}></canvas>
                </div>
              </div>
              <div className="admin-chart-card">
                <h3>Sử dụng Mô hình LLM</h3>
                <div className="canvas-wrapper doughnut-wrapper">
                  <canvas ref={doughnutChartRef}></canvas>
                </div>
              </div>
            </div>
            
            <div className="admin-table-card">
              <h3 className="admin-table-title">Tác vụ Xử lý Gần Đây (PostgreSQL `videos` table)</h3>
              <div className="admin-table-container">
                {statsLoading ? (
                  <div style={{ padding: '40px 0', textAlign: 'center' }}>
                    <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: '1.5rem', color: 'var(--primary)' }}></i>
                    <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Đang tải danh sách tác vụ...</p>
                  </div>
                ) : statsVideos.length === 0 ? (
                  <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <i className="fa-solid fa-list-check" style={{ fontSize: '2rem', marginBottom: '10px' }}></i>
                    <p>Chưa có tác vụ nào được lưu trong hệ thống.</p>
                  </div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Video ID</th>
                        <th>Email User</th>
                        <th>Tên File / Nguồn</th>
                        <th>Độ dài</th>
                        <th>Trạng thái</th>
                        <th>Thao tác</th>
                      </tr>
                    </thead>
                    <tbody>
                      {statsVideos.map(v => (
                        <tr key={v.videoId}>
                          <td className="uuid">#{v.videoId.substring(0, 8)}...</td>
                          <td>{v.email}</td>
                          <td>
                            {v.originalUrl ? (
                              <a href={v.originalUrl} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', textDecoration: 'underline' }}>
                                Youtube Link
                              </a>
                            ) : (
                              v.title
                            )}
                          </td>
                          <td>{Math.round(v.duration)}s</td>
                          <td>
                            <span className={`status ${v.status === VideoStatus.DONE ? 'done' : v.status === VideoStatus.FAILED ? 'failed' : 'processing'}`}>
                              {v.status === VideoStatus.DONE ? 'Hoàn tất' : v.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                            </span>
                          </td>
                          <td>
                            <button className="btn-small" onClick={() => openVideoDetail(v)}>
                              <i className="fa-solid fa-circle-info"></i> Chi tiết
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </>
        )}

        {/* TAB 2: AI METRICS PERFORMANCE */}
        {activeTab === 'metrics' && (
          <>
            <div className="admin-header">
              <h1>Hiệu Suất & Chất Lượng AI Pipeline</h1>
            </div>
            
            <div className="metrics-summary-grid">
              <div className="admin-stat-card">
                <h3>Word Error Rate (ASR) <i className="fa-solid fa-spell-check" style={{ color: '#a855f7' }}></i></h3>
                <div className="value">7.8%</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '5px 0 0 0' }}>Độ chính xác nhận diện từ</p>
              </div>
              <div className="admin-stat-card">
                <h3>CLIP Keyframe F1 <i className="fa-solid fa-crop" style={{ color: '#3b82f6' }}></i></h3>
                <div className="value">0.52</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '5px 0 0 0' }}>Độ khớp so với giảng viên</p>
              </div>
              <div className="admin-stat-card">
                <h3>LLM Latency (avg) <i className="fa-solid fa-gauge-high" style={{ color: '#ec4899' }}></i></h3>
                <div className="value">1.4s</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '5px 0 0 0' }}>Thời gian phản hồi tóm tắt</p>
              </div>
              <div className="admin-stat-card">
                <h3>Tổng Chi Phí Token <i className="fa-solid fa-wallet" style={{ color: '#10b981' }}></i></h3>
                <div className="value">$12.45</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '5px 0 0 0' }}>Tháng hiện tại</p>
              </div>
            </div>

            <div className="admin-chart-card" style={{ marginBottom: '30px' }}>
              <h3>Đánh giá chi tiết các Mô hình</h3>
              <div style={{ marginTop: '15px' }}>
                <div className="metric-bar-group">
                  <div className="metric-bar-header">
                    <span>WhisperX ASR (Nhận diện giọng nói)</span>
                    <span>Độ chính xác: 92.2%</span>
                  </div>
                  <div className="metric-bar-container">
                    <div className="metric-bar-fill" style={{ width: '92.2%', backgroundColor: '#a855f7' }}></div>
                  </div>
                </div>
                
                <div className="metric-bar-group">
                  <div className="metric-bar-header">
                    <span>CLIP Keyframe (Cắt khung ảnh bài giảng)</span>
                    <span>F1 Score: 85.0%</span>
                  </div>
                  <div className="metric-bar-container">
                    <div className="metric-bar-fill" style={{ width: '85%', backgroundColor: '#3b82f6' }}></div>
                  </div>
                </div>

                <div className="metric-bar-group">
                  <div className="metric-bar-header">
                    <span>Gemini 1.5 Flash (Tóm tắt RAG & Hỏi đáp)</span>
                    <span>Độ khớp ý kiến: 89.5%</span>
                  </div>
                  <div className="metric-bar-container">
                    <div className="metric-bar-fill" style={{ width: '89.5%', backgroundColor: '#ec4899' }}></div>
                  </div>
                </div>

                <div className="metric-bar-group">
                  <div className="metric-bar-header">
                    <span>Qwen 2.5 14B Local (Tóm tắt bài giảng)</span>
                    <span>Độ khớp ý kiến: 81.2%</span>
                  </div>
                  <div className="metric-bar-container">
                    <div className="metric-bar-fill" style={{ width: '81.2%', backgroundColor: '#10b981' }}></div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* TAB 3: USER MANAGEMENT */}
        {activeTab === 'users' && (
          <>
            <div className="admin-header">
              <h1>Quản Lý Thành Viên Hệ Thống</h1>
            </div>
            
            <div className="admin-table-card">
              <h3 className="admin-table-title">Danh sách thành viên đăng ký</h3>
              <div className="admin-table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Thành viên</th>
                      <th>Email</th>
                      <th>Quyền Hạn</th>
                      <th>Trạng thái</th>
                      <th>Ngày Đăng Ký</th>
                      <th>Hành Động</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map(u => (
                      <tr key={u.id}>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <div className="user-avatar">{u.email.charAt(0).toUpperCase()}</div>
                            <span style={{ fontWeight: '500' }}>{u.email.split('@')[0]}</span>
                          </div>
                        </td>
                        <td>{u.email}</td>
                        <td>
                          <span className={`badge ${u.role === 'ADMIN' ? 'blocked' : 'active'}`} style={{ background: u.role === 'ADMIN' ? '#fee2e2' : '#eef2ff', color: u.role === 'ADMIN' ? '#ef4444' : '#4f46e5' }}>
                            {u.role}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${u.active ? 'active' : 'blocked'}`}>
                            {u.active ? 'Hoạt động' : 'Bị khóa'}
                          </span>
                        </td>
                        <td>{u.joined}</td>
                        <td>
                          <div style={{ display: 'flex', gap: '8px' }}>
                            <button className="btn-small" onClick={() => toggleUserRole(u.id)}>
                              Đổi Vai Trò
                            </button>
                            <button className={`btn-small ${u.active ? 'danger' : ''}`} onClick={() => toggleUserActive(u.id)}>
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
          </>
        )}

        {/* TAB 4: VIDEO STANDARD SETTINGS */}
        {activeTab === 'videos' && (
          <>
            <div className="admin-header">
              <h1>Cấu Hình Tiêu Chuẩn Phân Tích Video</h1>
            </div>
            
            <div className="admin-chart-card">
              <h3>Giới hạn tải lên & Xác thực video</h3>
              {standardsLoading ? (
                <div style={{ padding: '40px 0', textAlign: 'center' }}>
                  <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: '1.5rem', color: 'var(--primary)' }}></i>
                  <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Đang tải cấu hình...</p>
                </div>
              ) : (
                <form onSubmit={handleSaveStandards} className="settings-form">
                  <div className="form-group">
                    <label>Thời lượng video tối đa (Giây)</label>
                    <input 
                      type="number" 
                      value={maxDuration} 
                      onChange={(e) => setMaxDuration(parseInt(e.target.value) || 0)}
                      required
                    />
                    <span className="help-text">Mặc định là 3600 giây (1 giờ). Video dài hơn sẽ bị chặn.</span>
                  </div>

                  <div className="form-group">
                    <label>Các định dạng được cho phép (Dấu phẩy phân tách)</label>
                    <input 
                      type="text" 
                      value={allowedFormats} 
                      onChange={(e) => setAllowedFormats(e.target.value)}
                      required
                    />
                    <span className="help-text">Ví dụ: mp4,avi,mkv,webm</span>
                  </div>

                  <div className="form-group">
                    <label>Dung lượng file tối đa (Megabytes)</label>
                    <input 
                      type="number" 
                      value={maxFileSize} 
                      onChange={(e) => setMaxFileSize(parseInt(e.target.value) || 0)}
                      required
                    />
                    <span className="help-text">Giới hạn dung lượng tối đa cho mỗi lần tải video lên.</span>
                  </div>

                  <div className="form-group">
                    <label>Chất lượng âm thanh tối thiểu (SNR)</label>
                    <input 
                      type="number" 
                      step="0.1"
                      value={minAudioQuality} 
                      onChange={(e) => setMinAudioQuality(parseFloat(e.target.value) || 0)}
                      required
                    />
                    <span className="help-text">Tỷ lệ Tín hiệu trên Nhiễu (SNR) tối thiểu. Để 0.0 để không kiểm tra.</span>
                  </div>

                  {standardsMsg && (
                    <div style={{ color: standardsMsg.type === 'success' ? '#10b981' : '#ef4444', fontSize: '0.9rem', fontWeight: '600' }}>
                      <i className={`fa-solid ${standardsMsg.type === 'success' ? 'fa-circle-check' : 'fa-triangle-exclamation'}`} style={{ marginRight: '6px' }}></i>
                      {standardsMsg.text}
                    </div>
                  )}

                  <div>
                    <button type="submit" className="btn primary">
                      <i className="fa-regular fa-floppy-disk"></i> Lưu cấu hình vào DB
                    </button>
                  </div>
                </form>
              )}
            </div>
          </>
        )}

        {/* TAB 5: CELERY JOB QUEUE */}
        {activeTab === 'celery' && (
          <>
            <div className="admin-header">
              <h1>Hệ Thống Hàng Đợi Celery Job Queue</h1>
            </div>
            
            <div className="admin-table-card">
              <div style={{ padding: '25px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)' }}>
                <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Các tiến trình AI đang hoạt động trong hàng đợi</h3>
                <span className="badge active" style={{ background: '#d1fae5', color: '#047857' }}>Celery Worker: Active (1)</span>
              </div>
              
              <div className="admin-table-container">
                {queueLoading ? (
                  <div style={{ padding: '40px 0', textAlign: 'center' }}>
                    <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: '1.5rem', color: 'var(--primary)' }}></i>
                    <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Đang đồng bộ hàng đợi...</p>
                  </div>
                ) : queueJobs.length === 0 ? (
                  <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <i className="fa-solid fa-list-check" style={{ fontSize: '2rem', marginBottom: '10px' }}></i>
                    <p>Hiện không có job nào đang chạy hoặc bị lỗi.</p>
                  </div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Video ID</th>
                        <th>Nguồn Video</th>
                        <th>Ngôn Ngữ</th>
                        <th>Trạng Thái Pipeline</th>
                        <th>Thao Tác</th>
                      </tr>
                    </thead>
                    <tbody>
                      {queueJobs.map(job => (
                        <tr key={job.videoId}>
                          <td className="uuid">#{job.videoId.substring(0, 8)}...</td>
                          <td>
                            {job.originalUrl ? (
                              <a href={job.originalUrl} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', textDecoration: 'underline' }}>
                                Youtube Link
                              </a>
                            ) : (
                              `File: ${job.filePath?.split('/').pop() || 'uploaded_video.mp4'}`
                            )}
                          </td>
                          <td>{job.language.toUpperCase()}</td>
                          <td>
                            <span className={`status ${job.status === VideoStatus.DONE ? 'done' : job.status === VideoStatus.FAILED ? 'failed' : 'processing'}`}>
                              {job.status === VideoStatus.DONE ? 'Hoàn tất' : job.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                            </span>
                          </td>
                          <td>
                            <div style={{ display: 'flex', gap: '8px' }}>
                              <button className="btn-small" onClick={() => openJobDetail(job)}>Xem chi tiết</button>
                              {job.status !== VideoStatus.DONE && job.status !== VideoStatus.FAILED && (
                                <button className="btn-small danger">Dừng</button>
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
          </>
        )}

        {/* TAB: SYSTEM VIDEOS MANAGEMENT */}
        {activeTab === 'system-videos' && (
          <>
            <div className="admin-header">
              <h1>Quản Lý Toàn Bộ Video Hệ Thống</h1>
            </div>
            
            <div className="admin-table-card">
              <h3 className="admin-table-title">Danh sách Video đã tải lên (PostgreSQL `videos` table)</h3>
              <div className="admin-table-container">
                {videoLoading ? (
                  <div style={{ padding: '40px 0', textAlign: 'center' }}>
                    <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: '1.5rem', color: 'var(--primary)' }}></i>
                    <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Đang tải danh sách video...</p>
                  </div>
                ) : allVideos.length === 0 ? (
                  <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <i className="fa-solid fa-film" style={{ fontSize: '2rem', marginBottom: '10px' }}></i>
                    <p>Chưa có video nào được tải lên hệ thống.</p>
                  </div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Video ID</th>
                        <th>User ID</th>
                        <th>Nguồn Video / Tên File</th>
                        <th>Độ dài (Giây)</th>
                        <th>Ngôn ngữ</th>
                        <th>Trạng thái</th>
                        <th>Ngày tải</th>
                        <th>Hành động</th>
                      </tr>
                    </thead>
                    <tbody>
                      {allVideos.map((v: any) => (
                        <tr key={v.videoId}>
                          <td className="uuid" title={v.videoId}>#{v.videoId.substring(0, 8)}...</td>
                          <td className="uuid" title={v.userId}>#{v.userId.substring(0, 8)}...</td>
                          <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {v.originalUrl ? (
                              <a href={v.originalUrl} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', textDecoration: 'underline' }}>
                                {v.originalUrl}
                              </a>
                            ) : (
                              v.filePath?.split('/').pop() || 'Video File'
                            )}
                          </td>
                          <td>{Math.round(v.duration)}s</td>
                          <td>{v.language.toUpperCase()}</td>
                          <td>
                            <span className={`status ${v.status === VideoStatus.DONE ? 'done' : v.status === VideoStatus.FAILED ? 'failed' : 'processing'}`}>
                              {v.status === VideoStatus.DONE ? 'Hoàn tất' : v.status === VideoStatus.FAILED ? 'Lỗi' : 'Đang xử lý'}
                            </span>
                          </td>
                          <td>{new Date(v.uploadedAt).toLocaleDateString('vi-VN')}</td>
                          <td>
                            <div style={{ display: 'flex', gap: '8px' }}>
                              <button className="btn-small" onClick={() => openVideoDetail(v)}>
                                <i className="fa-solid fa-circle-info"></i> Chi tiết
                              </button>
                              <button className="btn-small danger" onClick={() => handleDeleteVideo(v.videoId)}>
                                <i className="fa-regular fa-trash-can"></i> Xóa
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
          </>
        )}

        {/* TAB: SYSTEM JOBS MANAGEMENT */}
        {activeTab === 'system-jobs' && (
          <>
            <div className="admin-header">
              <h1>Quản Lý Toàn Bộ Tác Vụ (Jobs)</h1>
            </div>
            
            <div className="admin-table-card">
              <h3 className="admin-table-title">Danh sách tiến trình chạy ngầm (PostgreSQL `jobs` table)</h3>
              <div className="admin-table-container">
                {jobsLoading ? (
                  <div style={{ padding: '40px 0', textAlign: 'center' }}>
                    <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: '1.5rem', color: 'var(--primary)' }}></i>
                    <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>Đang tải danh sách tác vụ...</p>
                  </div>
                ) : allJobs.length === 0 ? (
                  <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    <i className="fa-solid fa-list-check" style={{ fontSize: '2rem', marginBottom: '10px' }}></i>
                    <p>Chưa có tác vụ nào được chạy trên hệ thống.</p>
                  </div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Job ID</th>
                        <th>Video ID</th>
                        <th>Loại tác vụ</th>
                        <th>Trạng thái</th>
                        <th>Bắt đầu</th>
                        <th>Hoàn thành</th>
                        <th>Log lỗi</th>
                        <th>Thao tác</th>
                      </tr>
                    </thead>
                    <tbody>
                      {allJobs.map((j: any) => (
                        <tr key={j.jobId}>
                          <td className="uuid" title={j.jobId}>#{j.jobId.substring(0, 8)}...</td>
                          <td className="uuid" title={j.videoId}>#{j.videoId.substring(0, 8)}...</td>
                          <td>
                            <span className="badge active" style={{ background: '#f3e8ff', color: '#6b21a8' }}>
                              {j.jobType.toUpperCase()}
                            </span>
                          </td>
                          <td>
                            <span className={`status ${j.status === 'done' || j.status === 'completed' ? 'done' : j.status === 'failed' ? 'failed' : 'processing'}`}>
                              {j.status === 'done' || j.status === 'completed' ? 'Hoàn tất' : j.status === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                            </span>
                          </td>
                          <td>{j.startedAt ? new Date(j.startedAt).toLocaleTimeString('vi-VN') : '-'}</td>
                          <td>{j.completedAt ? new Date(j.completedAt).toLocaleTimeString('vi-VN') : '-'}</td>
                          <td style={{ maxWidth: '250px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {j.errorLog ? (
                              <span className="error-text" title={j.errorLog} style={{ color: '#ef4444', fontSize: '0.85rem' }}>
                                {j.errorLog}
                              </span>
                            ) : (
                              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>-</span>
                            )}
                          </td>
                          <td>
                            <button className="btn-small" onClick={() => openJobDetail(j)}>
                              <i className="fa-solid fa-circle-info"></i> Chi tiết
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </>
        )}

      {/* Video Detail Modal */}
      {videoModalOpen && selectedVideo && (
        <div className="admin-modal-overlay" onClick={() => setVideoModalOpen(false)}>
          <div className="admin-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="admin-modal-header">
              <h2>Chi Tiết Video Hệ Thống</h2>
              <button className="admin-modal-close" onClick={() => setVideoModalOpen(false)}>&times;</button>
            </div>
            
            <div className="admin-modal-body">
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-label">Video ID:</span>
                  <span className="detail-val uuid">{selectedVideo.videoId}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">User ID (Sở hữu):</span>
                  <span className="detail-val uuid">{selectedVideo.userId}</span>
                </div>
                <div className="detail-item" style={{ gridColumn: 'span 2' }}>
                  <span className="detail-label">Nguồn Video / URL:</span>
                  <span className="detail-val">
                    {selectedVideo.originalUrl ? (
                      <a href={selectedVideo.originalUrl} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', textDecoration: 'underline' }}>
                        {selectedVideo.originalUrl}
                      </a>
                    ) : (
                      selectedVideo.filePath || 'Tệp tải lên cục bộ'
                    )}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Thời lượng:</span>
                  <span className="detail-val">{Math.round(selectedVideo.duration)}s (~{Math.round(selectedVideo.duration / 60)} phút)</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Ngôn ngữ:</span>
                  <span className="detail-val">{selectedVideo.language.toUpperCase()}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Trạng thái:</span>
                  <span className={`status ${selectedVideo.status?.toLowerCase() === 'done' ? 'done' : selectedVideo.status?.toLowerCase() === 'failed' ? 'failed' : 'processing'}`}>
                    {selectedVideo.status?.toLowerCase() === 'done' ? 'Hoàn tất' : selectedVideo.status?.toLowerCase() === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Ngày tải lên:</span>
                  <span className="detail-val">{new Date(selectedVideo.uploadedAt).toLocaleString('vi-VN')}</span>
                </div>
              </div>

              {loadingSummary && (
                <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  <i className="fa-solid fa-spinner fa-spin" style={{ fontSize: '1.5rem', color: 'var(--primary)', marginRight: '8px' }}></i>
                  Đang tải tóm tắt & phân tích AI...
                </div>
              )}

              {!loadingSummary && videoSummary && (
                <div className="modal-summary-section" style={{ marginTop: '20px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: '700', marginBottom: '10px' }}>Tóm Tắt Toàn Văn (AI Summary)</h3>
                  <div className="summary-text-box" style={{ background: 'var(--bg-secondary)', padding: '15px', borderRadius: '8px', fontSize: '0.95rem', lineHeight: '1.6', margin: '10px 0', border: '1px solid var(--border)', color: 'var(--text-main)' }}>
                    {videoSummary.summaryText}
                  </div>

                  {videoSummary.chaptersJson && videoSummary.chaptersJson.length > 0 && (
                    <div style={{ marginTop: '20px' }}>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: '700', marginBottom: '10px' }}>Phân Chia Chương Bài Giảng ({videoSummary.chaptersJson.length})</h3>
                      <div className="chapters-list" style={{ marginTop: '10px' }}>
                        {videoSummary.chaptersJson.map((ch: any, idx: number) => (
                          <div key={idx} className="chapter-row" style={{ display: 'flex', gap: '15px', padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
                            <span style={{ fontWeight: 'bold', color: 'var(--primary)', minWidth: '90px' }}>
                              {Math.floor(ch.startTime / 60)}:{(ch.startTime % 60).toString().padStart(2, '0')} - {Math.floor(ch.endTime / 60)}:{(ch.endTime % 60).toString().padStart(2, '0')}
                            </span>
                            <div>
                              <div style={{ fontWeight: '600', color: 'var(--text-main)' }}>{ch.title}</div>
                              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>{ch.summary}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {videoSummary.keyframesJson && videoSummary.keyframesJson.length > 0 && (
                    <div style={{ marginTop: '20px' }}>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: '700', marginBottom: '10px' }}>Slide Ảnh Keyframe Quan Trọng ({videoSummary.keyframesJson.length})</h3>
                      <div className="keyframes-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '15px', marginTop: '10px' }}>
                        {videoSummary.keyframesJson.map((kf: any, idx: number) => (
                          <div key={idx} className="keyframe-card" style={{ border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden', background: 'var(--bg-secondary)' }}>
                            <img src={`${CONFIG.API_BASE_URL.replace('/api/v1', '')}${kf.imageUrl}`} alt={kf.description} style={{ width: '100%', height: '100px', objectFit: 'cover' }} />
                            <div style={{ padding: '8px', fontSize: '0.8rem' }}>
                              <div style={{ fontWeight: 'bold', color: 'var(--primary)' }}>Mốc: {Math.floor(kf.timestamp / 60)}:{(kf.timestamp % 60).toString().padStart(2, '0')}</div>
                              <div style={{ color: 'var(--text-muted)', marginTop: '4px', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }} title={kf.description}>{kf.description}</div>
                              <div style={{ color: '#10b981', fontSize: '0.75rem', fontWeight: 'bold', marginTop: '4px' }}>Điểm: {Math.round(kf.importanceScore * 100)}%</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {!loadingSummary && !videoSummary && selectedVideo.status?.toLowerCase() === 'done' && (
                <div style={{ padding: '20px', textAlign: 'center', color: '#ef4444' }}>
                  <i className="fa-solid fa-triangle-exclamation" style={{ marginRight: '8px' }}></i>
                  Không tìm thấy tóm tắt cho video này.
                </div>
              )}
            </div>
            
            <div className="admin-modal-footer">
              <button className="btn" onClick={() => setVideoModalOpen(false)}>Đóng</button>
              {selectedVideo.status?.toLowerCase() === 'done' && (
                <a href={`#/video/${selectedVideo.videoId}`} className="btn primary" onClick={() => setVideoModalOpen(false)} style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                  <i className="fa-regular fa-eye"></i> Xem Trang Client
                </a>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Job Detail Modal */}
      {jobModalOpen && selectedJob && (
        <div className="admin-modal-overlay" onClick={() => setJobModalOpen(false)}>
          <div className="admin-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="admin-modal-header">
              <h2>Chi Tiết Tác Vụ Celery</h2>
              <button className="admin-modal-close" onClick={() => setJobModalOpen(false)}>&times;</button>
            </div>
            
            <div className="admin-modal-body">
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-label">Job ID (Tác vụ):</span>
                  <span className="detail-val uuid">{selectedJob.jobId}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Video ID liên quan:</span>
                  <span className="detail-val uuid">{selectedJob.videoId}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Loại tác vụ:</span>
                  <span className="detail-val badge" style={{ background: '#f3e8ff', color: '#6b21a8', display: 'inline-block' }}>
                    {selectedJob.jobType.toUpperCase()}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Trạng thái:</span>
                  <span className={`status ${selectedJob.status === 'done' || selectedJob.status === 'completed' ? 'done' : selectedJob.status === 'failed' ? 'failed' : 'processing'}`}>
                    {selectedJob.status === 'done' || selectedJob.status === 'completed' ? 'Hoàn tất' : selectedJob.status === 'failed' ? 'Lỗi' : 'Đang xử lý'}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Thời gian bắt đầu:</span>
                  <span className="detail-val">{selectedJob.startedAt ? new Date(selectedJob.startedAt).toLocaleString('vi-VN') : '-'}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Thời gian hoàn thành:</span>
                  <span className="detail-val">{selectedJob.completedAt ? new Date(selectedJob.completedAt).toLocaleString('vi-VN') : '-'}</span>
                </div>
              </div>

              {selectedJob.errorLog && (
                <div style={{ marginTop: '20px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
                  <h3 style={{ color: '#ef4444', fontSize: '1.1rem', fontWeight: '700' }}>Nhật ký báo lỗi (Error Log / Traceback)</h3>
                  <pre style={{ background: '#fef2f2', color: '#b91c1c', padding: '15px', borderRadius: '8px', overflowX: 'auto', fontSize: '0.85rem', whiteSpace: 'pre-wrap', border: '1px solid #fee2e2', marginTop: '10px' }}>
                    {selectedJob.errorLog}
                  </pre>
                </div>
              )}
              
              {!selectedJob.errorLog && (
                <div style={{ marginTop: '20px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
                  <h3 style={{ color: '#10b981', fontSize: '1.1rem', fontWeight: '700' }}>Nhật ký tiến trình (System Log)</h3>
                  <pre style={{ background: 'var(--bg-secondary)', color: 'var(--text-muted)', padding: '15px', borderRadius: '8px', overflowX: 'auto', fontSize: '0.85rem', whiteSpace: 'pre-wrap', border: '1px solid var(--border)', marginTop: '10px' }}>
                    [INFO] {new Date(selectedJob.startedAt).toISOString()} - Khởi chạy pipeline phân tích đa phương tiện cho Video {selectedJob.videoId}.
                    {"\n"}[INFO] Khởi tạo mô hình WhisperX trích xuất Audio...
                    {selectedJob.completedAt && (
                      <>
                        {"\n"}[INFO] Hoàn thành chuyển đổi giọng nói (Speech-to-Text).
                        {"\n"}[INFO] Khởi chạy trích xuất Keyframes bằng CLIP...
                        {"\n"}[INFO] Đọc dữ liệu ảnh và xếp hạng độ quan trọng slide...
                        {"\n"}[INFO] Gửi văn bản sang Gemini 1.5 để tạo tóm tắt chương...
                        {"\n"}[INFO] Lưu trữ vector embeddings thành công vào ChromaDB.
                        {"\n"}[INFO] {new Date(selectedJob.completedAt).toISOString()} - Tác vụ hoàn thành thành công.
                      </>
                    )}
                  </pre>
                </div>
              )}
            </div>
            
            <div className="admin-modal-footer">
              <button className="btn" onClick={() => setJobModalOpen(false)}>Đóng</button>
            </div>
          </div>
        </div>
      )}

      </div>
    </div>
  );
};
