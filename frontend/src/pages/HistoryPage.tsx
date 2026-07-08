import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import type { HistoryItem } from '../types';
import { VideoStatus } from '../types';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const toast = useToast();
  const { t } = useTranslation();

  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('All');
  const [sortBy, setSortBy] = useState('newest');

  const [currentPage, setCurrentPage] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const limit = 9;

  useEffect(() => {
    setLoading(true);
    const statusParam = selectedStatus === 'All' ? undefined : selectedStatus;
    api.getVideos(statusParam, limit, (currentPage - 1) * limit)
      .then(res => {
        if (res.success && res.data) {
          setTotalItems(res.metadata?.totalResults || res.metadata?.total || res.data.length);
          const items = res.data.map((video: any) => {
            const durationSec = video.duration || 0;
            const hours = Math.floor(durationSec / 3600);
            const minutes = Math.floor((durationSec % 3600) / 60);
            const seconds = Math.floor(durationSec % 60);
            
            const durationStr = hours > 0 
              ? `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
              : `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            const uploadDate = new Date(video.uploadedAt);
            const dateStr = uploadDate.toLocaleDateString('vi-VN', { month: 'short', day: 'numeric', year: 'numeric' });

            return {
              id: video.videoId,
              title: video.title || (video.originalUrl ? 'YouTube Video' : (video.r2Url ? (video.r2Url.split('?')[0].split('/').pop() || 'video.mp4') : (video.filePath && !video.filePath.includes('/stream') ? (video.filePath.split('?')[0].split('/').pop() || 'video.mp4') : 'Video bài giảng'))),
              duration: durationStr,
              date: dateStr,
              status: video.status as VideoStatus,
              progress: video.progress,
              stage: video.stage,
              logs: video.logs,
              jobId: video.jobId,
              thumbnailUrl: video.scenes && video.scenes.length > 0 ? video.scenes[0].keyframeUrl : undefined,
            };
          });
          setHistoryItems(items);
        }
      })
      .catch(err => {
        console.error("Failed to fetch real history from backend API:", err);
        toast.error(t('history.toast_fetch_err'), t('history.toast_conn_err'));
      })
      .finally(() => {
        setLoading(false);
      });
  }, [currentPage, selectedStatus]);

  const handleDelete = (id: string) => {
    if (window.confirm("Bạn có chắc chắn muốn xóa video này khỏi lịch sử?")) {
      setHistoryItems(prev => prev.filter(item => item.id !== id));
      toast.success(t('history.toast_del_success'), t('common.success'));
    }
  };

  const handleCancelJob = async (jobId: string) => {
    if (window.confirm("Bạn có chắc chắn muốn dừng tác vụ đang chạy này không?")) {
      try {
        await api.cancelJob(jobId);
        toast.success(t('history.toast_stop_success'), t('common.success'));
        // Reload list
        const statusParam = selectedStatus === 'All' ? undefined : selectedStatus;
        const res = await api.getVideos(statusParam, limit, (currentPage - 1) * limit);
        if (res.success && res.data) {
          setTotalItems(res.metadata?.totalResults || res.metadata?.total || res.data.length);
          const items = res.data.map((video: any) => {
            const durationSec = video.duration || 0;
            const hours = Math.floor(durationSec / 3600);
            const minutes = Math.floor((durationSec % 3600) / 60);
            const seconds = Math.floor(durationSec % 60);
            
            const durationStr = hours > 0 
              ? `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
              : `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            const uploadDate = new Date(video.uploadedAt);
            const dateStr = uploadDate.toLocaleDateString('vi-VN', { month: 'short', day: 'numeric', year: 'numeric' });

            return {
              id: video.videoId,
              title: video.title || (video.originalUrl ? 'YouTube Video' : (video.r2Url ? (video.r2Url.split('?')[0].split('/').pop() || 'video.mp4') : (video.filePath && !video.filePath.includes('/stream') ? (video.filePath.split('?')[0].split('/').pop() || 'video.mp4') : 'Video bài giảng'))),
              duration: durationStr,
              date: dateStr,
              status: video.status as VideoStatus,
              progress: video.progress,
              stage: video.stage,
              logs: video.logs,
              jobId: video.jobId,
              thumbnailUrl: video.scenes && video.scenes.length > 0 ? video.scenes[0].keyframeUrl : undefined,
            };
          });
          setHistoryItems(items);
        }
      } catch (err: any) {
        toast.error(`${t('history.toast_stop_err')}${err.message}`, t('history.toast_err'));
      }
    }
  };

  // Helper to assign card details
  const getCardDetails = (_title: string | undefined, index: number) => {
    const gradients = [
      'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
      'linear-gradient(135deg, #065f46 0%, #10b981 100%)',
      'linear-gradient(135deg, #581c87 0%, #8b5cf6 100%)',
      'linear-gradient(135deg, #7c2d12 0%, #f97316 100%)',
      'linear-gradient(135deg, #0f172a 0%, #475569 100%)',
    ];
    return {
      subject: 'Bài giảng',
      bgGradient: gradients[index % gradients.length],
      accuracy: 100
    };
  };

  // Filter & Sort
  const filteredItems = historyItems
    .filter(item => {
      const matchesSearch = item.title.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesSearch;
    })
    .sort((_a, _b) => {
      if (sortBy === 'newest') return 1;
      if (sortBy === 'oldest') return -1;
      return 0;
    });

  const queuedCount = historyItems.filter(item => item.status === VideoStatus.PROCESSING).length;
  const processedCount = historyItems.filter(item => item.status === VideoStatus.DONE).length;

  return (
    <div className="bg-surface text-on-surface font-body-md text-body-md antialiased min-h-screen">
      <style dangerouslySetInnerHTML={{__html: `
        .material-symbols-outlined {
          font-family: 'Material Symbols Outlined' !important;
          font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
          vertical-align: middle;
        }
      `}} />
      {/* Main Content Canvas */}
      <main className="flex-1 px-margin-mobile md:px-margin-desktop py-8 max-w-container-max mx-auto">
        {/* Page Header & Filtering Section */}
        <header className="mb-10">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
            <div>
              <h1 className="font-headline-xl text-3xl md:text-headline-xl text-deep-navy font-bold mb-2">{t('history.title')}</h1>
              <p className="text-secondary max-w-2xl font-body-md text-sm md:text-body-md">
                {t('history.subtitle')}
              </p>
            </div>
            <div className="flex items-center gap-4 text-mono-data font-mono-data text-xs bg-surface-container-high px-4 py-2 rounded-full border border-outline-variant shrink-0">
              <span className="flex items-center gap-1 font-bold">
                <span className="w-2 h-2 rounded-full bg-status-success"></span> {processedCount} {t('history.completed')}
              </span>
              <span className="text-outline-variant">|</span>
              <span className="flex items-center gap-1 font-bold">
                <span className="w-2 h-2 rounded-full bg-status-warning animate-pulse"></span> {queuedCount} {t('history.processing')}
              </span>
            </div>
          </div>

          {/* Bento-style Filter Bar */}
          <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-2 relative">
              <input 
                className="w-full pl-10 pr-4 py-3 bg-white border border-outline-variant rounded-xl focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none transition-all font-body-sm text-sm shadow-sm" 
                placeholder={t('history.search_placeholder')} 
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <span className="absolute left-3 top-1/2 -translate-y-1/2 material-symbols-outlined text-outline text-xl">manage_search</span>
            </div>
            
            <select 
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-white border border-slate-200 hover:border-slate-300 text-deep-navy font-semibold rounded-xl px-4 py-3 text-sm focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none cursor-pointer shadow-sm hover:shadow transition-all duration-200"
            >
              <option value="newest">{t('history.sort_newest')}</option>
              <option value="oldest">{t('history.sort_oldest')}</option>
            </select>

            <select 
              value={selectedStatus}
              onChange={(e) => {
                setSelectedStatus(e.target.value);
                setCurrentPage(1);
              }}
              className="bg-white border border-slate-200 hover:border-slate-300 text-deep-navy font-semibold rounded-xl px-4 py-3 text-sm focus:border-vibrant-cyan focus:ring-1 focus:ring-vibrant-cyan outline-none cursor-pointer shadow-sm hover:shadow transition-all duration-200"
            >
              <option value="All">{t('history.status_all')}</option>
              <option value="pending">{t('history.status_pending')}</option>
              <option value="processing">{t('history.status_processing')}</option>
              <option value="done">{t('history.status_done')}</option>
              <option value="failed">{t('history.status_failed')}</option>
            </select>
          </div>
        </header>

        {/* Analysis Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-gutter-desktop">
          {loading ? (
            Array.from({ length: 5 }).map((_, idx) => (
              <Skeleton.Card key={idx} />
            ))
          ) : (
            filteredItems.map((item, index) => {
            const details = getCardDetails(item.title, index);
            const isProcessing = item.status === VideoStatus.PROCESSING || item.status === VideoStatus.PENDING;
            const isFailed = item.status === VideoStatus.FAILED;

            return (
              <div 
                key={item.id} 
                onClick={() => {
                  if (item.status === VideoStatus.DONE) {
                    navigate(`/results?videoId=${item.id}`);
                  }
                }}
                className={`group bg-white border rounded-xl overflow-hidden transition-all duration-300 flex flex-col hover:shadow-md ${
                  isFailed 
                    ? 'border-error/25 hover:border-error bg-error/[0.01]' 
                    : isProcessing 
                      ? 'border-blue-100 hover:border-blue-400 bg-blue-50/[0.01]' 
                      : 'border-outline-variant hover:border-vibrant-cyan cursor-pointer'
                }`}
              >
                <div className="relative h-48 w-full overflow-hidden bg-slate-950 shrink-0">
                  <div className="absolute inset-0 bg-black/40 group-hover:bg-black/20 transition-colors z-10"></div>
                  
                  {/* Real-time Status Badges */}
                  {item.status === VideoStatus.DONE && (
                    <span className="absolute top-3 right-3 z-20 px-2 py-0.5 text-[10px] font-bold tracking-wide uppercase rounded-md bg-status-success/15 text-status-success border border-status-success/30 flex items-center gap-1 shadow-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-status-success"></span>
                      {t('history.status_done')}
                    </span>
                  )}
                  {item.status === VideoStatus.PROCESSING && (
                    <span className="absolute top-3 right-3 z-20 px-2 py-0.5 text-[10px] font-bold tracking-wide uppercase rounded-md bg-blue-50 text-blue-600 border border-blue-200/50 flex items-center gap-1 shadow-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-ping"></span>
                      {t('history.status_processing')}
                    </span>
                  )}
                  {item.status === VideoStatus.PENDING && (
                    <span className="absolute top-3 right-3 z-20 px-2 py-0.5 text-[10px] font-bold tracking-wide uppercase rounded-md bg-status-warning/15 text-status-warning border border-status-warning/30 flex items-center gap-1 shadow-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-status-warning"></span>
                      {t('history.status_pending')}
                    </span>
                  )}
                  {item.status === VideoStatus.FAILED && (
                    <span className="absolute top-3 right-3 z-20 px-2 py-0.5 text-[10px] font-bold tracking-wide uppercase rounded-md bg-error/15 text-error border border-error/30 flex items-center gap-1 shadow-sm">
                      <span className="w-1.5 h-1.5 rounded-full bg-error"></span>
                      {t('history.status_failed')}
                    </span>
                  )}

                  {item.status === VideoStatus.DONE && item.thumbnailUrl ? (
                    <img 
                      src={item.thumbnailUrl}
                      alt={item.title}
                      className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-500"
                    />
                  ) : isProcessing ? (
                    <div className="w-full h-full flex flex-col items-center justify-center text-white bg-slate-900 relative">
                      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-blue-500/10 to-transparent animate-pulse"></div>
                      <div className="relative flex flex-col items-center gap-2.5">
                        <div className="w-10 h-10 rounded-full border-4 border-blue-500/20 border-t-blue-500 animate-spin"></div>
                        <span className="text-[10px] font-mono tracking-widest text-blue-400 uppercase">
                          {item.stage || 'ANALYZING'}
                        </span>
                      </div>
                    </div>
                  ) : isFailed ? (
                    <div className="w-full h-full flex flex-col items-center justify-center text-white bg-error/[0.04] relative">
                      <span className="material-symbols-outlined text-4xl text-error">error</span>
                      <span className="text-[10px] font-mono tracking-widest text-error mt-2 uppercase">{t('history.status_failed')}</span>
                    </div>
                  ) : (
                    <div 
                      className="w-full h-full transform group-hover:scale-105 transition-transform duration-500 flex items-center justify-center text-white font-bold text-lg"
                      style={{ background: details.bgGradient }}
                    >
                      <span className="material-symbols-outlined text-4xl opacity-80">
                        sync
                      </span>
                    </div>
                  )}
                </div>

                <div className="p-5 flex-1 flex flex-col">
                  <div className="mb-4">
                    <h3 className="font-headline-md text-headline-md text-deep-navy mb-1 leading-tight group-hover:text-vibrant-cyan transition-colors line-clamp-2 font-bold" title={item.title}>
                      {item.title}
                    </h3>
                    <p className="text-secondary font-body-sm text-body-sm flex items-center gap-2 mb-2">
                      <span className="material-symbols-outlined text-sm">calendar_today</span> {item.date}
                      <span className="text-outline-variant">•</span>
                      <span className="material-symbols-outlined text-sm">schedule</span> {item.duration}
                    </p>

                    {/* Progress Bar overlay for processing stages */}
                    {item.status === VideoStatus.PROCESSING && item.progress !== undefined && (
                      <div className="mt-3 bg-blue-50/50 p-2 rounded-lg border border-blue-100/50">
                        <div className="flex justify-between text-[10px] font-mono text-blue-600 mb-1 font-bold">
                          <span className="truncate max-w-[70%]">{item.stage || 'Đang phân tích...'}</span>
                          <span>{item.progress}%</span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-1 mb-2">
                          <div 
                            className="bg-blue-500 h-1 rounded-full transition-all duration-300 animate-pulse" 
                            style={{ width: `${item.progress}%` }}
                          ></div>
                        </div>
                        {/* Rolling logs view */}
                        {item.logs && item.logs.length > 0 && (
                          <div className="mt-2 p-2 bg-slate-900 border border-slate-800 rounded font-mono text-[9px] text-slate-300 flex flex-col gap-0.5 max-h-[90px] overflow-y-auto shadow-inner select-text">
                            {item.logs.map((logLine, idx) => (
                              <div key={idx} className="truncate" title={logLine}>{logLine}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="mt-auto flex items-center justify-between pt-4 border-t border-outline-variant/30" onClick={(e) => e.stopPropagation()}>
                    <div className="flex gap-2">
                      {item.status === VideoStatus.DONE ? (
                        <div 
                          className="w-9 h-9 flex items-center justify-center text-secondary hover:text-vibrant-cyan hover:bg-surface-container-low border border-outline-variant/30 rounded-lg transition-all" 
                          title="Xem kết quả"
                        >
                          <span className="material-symbols-outlined text-[20px]">play_circle</span>
                        </div>
                      ) : isProcessing ? (
                        <button 
                          onClick={() => item.jobId && handleCancelJob(item.jobId)}
                          className="w-9 h-9 flex items-center justify-center text-error hover:text-white bg-error/10 hover:bg-error border border-error/20 rounded-lg transition-all" 
                          title="Dừng tác vụ"
                          disabled={!item.jobId}
                        >
                          <span className="material-symbols-outlined text-[20px]">stop_circle</span>
                        </button>
                      ) : (
                        <button 
                          disabled
                          className="w-9 h-9 flex items-center justify-center text-error/60 bg-error/5 border border-error/20 rounded-lg cursor-not-allowed" 
                          title="Xử lý thất bại"
                        >
                          <span className="material-symbols-outlined text-[20px] text-error">error_outline</span>
                        </button>
                      )}
                      
                      <button 
                        onClick={() => {
                          if (item.status === VideoStatus.DONE) {
                            api.exportSummary(item.id, 'pdf')
                              .then(blob => {
                                const url = window.URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `summary_${item.id}.pdf`;
                                document.body.appendChild(a);
                                a.click();
                                a.remove();
                              })
                              .catch(err => toast.error(`${t('history.toast_pdf_err')}${err.message}`, t('history.toast_fail')));
                          }
                        }}
                        className="w-9 h-9 flex items-center justify-center text-secondary hover:text-vibrant-cyan hover:bg-surface-container-low border border-outline-variant/30 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed" 
                        title="Tải báo cáo PDF"
                        disabled={item.status !== VideoStatus.DONE}
                      >
                        <span className="material-symbols-outlined text-[20px]">download</span>
                      </button>
                    </div>
                    
                    <button 
                      onClick={() => handleDelete(item.id)}
                      className="w-9 h-9 flex items-center justify-center hover:text-error hover:bg-error-container/30 border border-outline-variant/30 rounded-lg transition-all text-error" 
                      title="Xóa"
                    >
                      <span className="material-symbols-outlined text-[20px]">delete</span>
                    </button>
                  </div>
                </div>
              </div>
            );
          }))}

          {/* Empty State Suggestion Card */}
          <div 
            onClick={() => navigate('/upload')}
            className="border-2 border-dashed border-outline-variant rounded-xl flex flex-col items-center justify-center p-8 text-center bg-surface-container-lowest/50 hover:bg-surface-container-low transition-colors cursor-pointer group"
          >
            <div className="w-16 h-16 rounded-full bg-surface-container-high flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <span className="material-symbols-outlined text-outline group-hover:text-vibrant-cyan">cloud_upload</span>
            </div>
            <h4 className="font-headline-md text-headline-md text-deep-navy mb-2">{t('history.empty_state_title')}</h4>
            <p className="text-secondary font-body-sm text-body-sm mb-6 max-w-[200px]">{t('history.empty_state_desc')}</p>
            <button className="px-6 py-2 bg-deep-navy text-white rounded font-label-md text-label-md hover:bg-slate-800 transition-all">{t('history.upload_btn')}</button>
          </div>
        </div>

        {/* Pagination */}
        {(() => {
          const totalPages = Math.max(1, Math.ceil(totalItems / limit));
          if (totalPages <= 1) return null;

          const pageNumbers: (number | string)[] = [];
          const addPage = (p: number) => pageNumbers.push(p);
          addPage(1);
          if (currentPage > 3) pageNumbers.push('...');
          for (let i = Math.max(2, currentPage - 1); i <= Math.min(totalPages - 1, currentPage + 1); i++) {
            addPage(i);
          }
          if (currentPage < totalPages - 2) pageNumbers.push('...');
          if (totalPages > 1) addPage(totalPages);
          const uniquePages = pageNumbers.filter((v, i, a) => a.indexOf(v) === i);

          return (
            <div className="mt-12 flex items-center justify-center gap-2">
              <button
                onClick={() => setCurrentPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="w-10 h-10 flex items-center justify-center border border-outline-variant rounded-lg text-secondary hover:text-primary hover:bg-white transition-all disabled:opacity-30 active:scale-95 disabled:cursor-not-allowed disabled:hover:text-secondary disabled:hover:bg-transparent"
              >
                <span className="material-symbols-outlined text-xl">chevron_left</span>
              </button>
              {uniquePages.map((page, idx) => {
                if (page === '...') {
                  return (
                    <span key={`dots-${idx}`} className="px-2 text-outline-variant">
                      ...
                    </span>
                  );
                }
                return (
                  <button
                    key={`page-${page}`}
                    onClick={() => setCurrentPage(page as number)}
                    className={`w-10 h-10 flex items-center justify-center rounded-lg font-bold text-xs active:scale-95 transition-all ${
                      currentPage === page
                        ? 'bg-deep-navy text-white shadow-sm'
                        : 'border border-outline-variant text-secondary hover:text-primary hover:bg-white'
                    }`}
                  >
                    {page}
                  </button>
                );
              })}
              <button
                onClick={() => setCurrentPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="w-10 h-10 flex items-center justify-center border border-outline-variant rounded-lg text-secondary hover:text-primary hover:bg-white transition-all disabled:opacity-30 active:scale-95 disabled:cursor-not-allowed disabled:hover:text-secondary disabled:hover:bg-transparent"
              >
                <span className="material-symbols-outlined text-xl">chevron_right</span>
              </button>
            </div>
          );
        })()}
      </main>
      
      {/* Footer Shell */}
      <footer className="bg-surface-container-lowest flex flex-col md:flex-row justify-between items-center px-margin-desktop py-8 w-full border-t border-outline-variant mt-12 text-xs">
        <div className="mb-4 md:mb-0 text-center md:text-left space-y-1">
          <span className="font-label-md font-bold text-deep-navy">Lumina</span>
          <p className="text-secondary font-body-sm">© 2026 Lumina. All rights reserved.</p>
        </div>
        <div className="flex flex-wrap justify-center gap-6 font-semibold">
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#credits">Academic Credits</a>
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#privacy">Privacy Policy</a>
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#terms">Terms of Service</a>
          <a className="text-secondary hover:text-vibrant-cyan transition-colors" href="#docs">Research Documentation</a>
        </div>
      </footer>
    </div>
  );
};
