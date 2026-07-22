import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import type { HistoryItem } from '../types';
import { VideoStatus } from '../types';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';
import { Skeleton } from '../components/Skeleton';
import { parseUTCDate } from '../utils/dateUtils';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, Play, StopCircle, AlertCircle, Download, Trash2, 
  Calendar, Clock, CheckCircle2, Loader2, CloudUpload, Sparkles, ChevronDown
} from 'lucide-react';

const CustomSelect = ({ value, onChange, options, className }: any) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedOption = options.find((opt: any) => opt.value === value) || options[0];

  return (
    <div className={`relative ${className || ''}`} ref={dropdownRef}>
      <button 
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full h-full min-h-[48px] bg-white/50 border border-slate-100 hover:bg-white text-slate-700 font-semibold rounded-xl px-4 text-sm outline-none transition-colors flex items-center justify-between gap-2"
      >
        <span className="truncate">{selectedOption?.label}</span>
        <ChevronDown size={16} className={`text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 top-full mt-2 w-full left-0 bg-white rounded-xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden py-1"
          >
            {options.map((opt: any) => (
              <button
                key={opt.value}
                onClick={() => {
                  onChange(opt.value);
                  setIsOpen(false);
                }}
                className={`w-full text-left px-4 py-2.5 text-sm transition-colors ${value === opt.value ? 'bg-primary/10 text-primary font-bold' : 'text-slate-600 hover:bg-slate-50'}`}
              >
                {opt.label}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

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

  const loadData = () => {
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
            
            const uploadDate = parseUTCDate(video.uploadedAt)!;
            const dateStr = uploadDate.toLocaleDateString('vi-VN', { month: 'short', day: 'numeric', year: 'numeric' });

            return {
              id: video.videoId,
              title: video.title || (video.originalUrl ? 'YouTube Video' : (video.r2Url ? (video.r2Url.split('?')[0].split('/').pop() || 'video.mp4') : (video.filePath && !video.filePath.includes('/stream') ? (video.filePath.split('?')[0].split('/').pop() || 'video.mp4') : 'Video bài giảng'))),
              duration: durationStr,
              date: dateStr,
              rawDate: uploadDate.getTime(),
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
        console.error("Failed to fetch history:", err);
        toast.error(t('history.toast_fetch_err'), t('history.toast_conn_err'));
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    loadData();
  }, [currentPage, selectedStatus]);

  const handleDelete = async (id: string) => {
    if (window.confirm("Bạn có chắc chắn muốn xóa video này?")) {
      try {
        await api.deleteVideo(id);
        setHistoryItems(prev => prev.filter(item => item.id !== id));
        toast.success(t('history.toast_del_success'), t('common.success'));
      } catch (err: any) {
        toast.error(`Xóa video thất bại: ${err.message}`, t('common.error'));
      }
    }
  };

  const handleCancelJob = async (jobId: string) => {
    if (window.confirm("Bạn có chắc chắn muốn dừng tác vụ đang chạy này không?")) {
      try {
        await api.cancelJob(jobId);
        toast.success(t('history.toast_stop_success'), t('common.success'));
        loadData();
      } catch (err: any) {
        toast.error(`${t('history.toast_stop_err')}${err.message}`, t('history.toast_err'));
      }
    }
  };

  const getCardDetails = (_title: string | undefined, index: number) => {
    const gradients = [
      'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
      'linear-gradient(135deg, #059669 0%, #10b981 100%)',
      'linear-gradient(135deg, #7e22ce 0%, #a855f7 100%)',
      'linear-gradient(135deg, #ea580c 0%, #f97316 100%)',
      'linear-gradient(135deg, #334155 0%, #64748b 100%)',
    ];
    return { bgGradient: gradients[index % gradients.length] };
  };

  const filteredItems = historyItems
    .filter(item => item.title.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'newest') return b.rawDate - a.rawDate;
      if (sortBy === 'oldest') return a.rawDate - b.rawDate;
      return 0;
    });

  const queuedCount = historyItems.filter(item => item.status === VideoStatus.PROCESSING).length;
  const processedCount = historyItems.filter(item => item.status === VideoStatus.DONE).length;

  return (
    <div className="bg-[#FAF5FF] text-slate-900 font-body antialiased min-h-screen flex flex-col">
      <main className="flex-1 px-4 md:px-8 py-10 max-w-[1400px] mx-auto w-full">
        
        {/* Header & Stats */}
        <header className="mb-12">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-8">
            <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
              <h1 className="font-heading text-4xl text-slate-900 font-black mb-2 tracking-tight flex items-center gap-3">
                <Sparkles className="text-primary" size={32} />
                {t('history.title')}
              </h1>
              <p className="text-slate-600 text-lg">
                {t('history.subtitle')}
              </p>
            </motion.div>
            
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="flex items-center gap-4 bg-white/60 backdrop-blur-md px-6 py-3 rounded-[2rem] border border-slate-200 shadow-sm shrink-0">
              <div className="flex items-center gap-2 font-bold text-sm text-slate-700">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></span>
                {processedCount} {t('history.completed')}
              </div>
              <div className="w-px h-6 bg-slate-300"></div>
              <div className="flex items-center gap-2 font-bold text-sm text-slate-700">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.5)]"></span>
                {queuedCount} {t('history.processing')}
              </div>
            </motion.div>
          </div>

          {/* Filters Bar */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-2 rounded-xl grid grid-cols-1 md:grid-cols-4 gap-2 relative z-50">
            <div className="md:col-span-2 relative">
              <input 
                className="w-full h-full min-h-[48px] pl-12 pr-4 bg-white/50 border-none rounded-xl focus:ring-2 focus:ring-primary/20 outline-none transition-all font-body text-sm" 
                placeholder={t('history.search_placeholder')} 
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
            </div>
            
            <CustomSelect
              value={sortBy}
              onChange={(val: string) => setSortBy(val)}
              options={[
                { value: 'newest', label: t('history.sort_newest') },
                { value: 'oldest', label: t('history.sort_oldest') }
              ]}
              className="z-40"
            />

            <CustomSelect
              value={selectedStatus}
              onChange={(val: string) => {
                setSelectedStatus(val);
                setCurrentPage(1);
              }}
              options={[
                { value: 'All', label: t('history.status_all') },
                { value: 'pending', label: t('history.status_pending') },
                { value: 'processing', label: t('history.status_processing') },
                { value: 'done', label: t('history.status_done') },
                { value: 'failed', label: t('history.status_failed') }
              ]}
              className="z-30"
            />
          </motion.div>
        </header>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 lg:gap-8">
          {loading ? (
            Array.from({ length: 6 }).map((_, idx) => (
              <motion.div key={idx} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: idx * 0.1 }}>
                <Skeleton.Card className="rounded-2xl border-none shadow-sm" />
              </motion.div>
            ))
          ) : (
            <AnimatePresence>
              {filteredItems.map((item, index) => {
                const details = getCardDetails(item.title, index);
                const isProcessing = item.status === VideoStatus.PROCESSING || item.status === VideoStatus.PENDING;
                const isFailed = item.status === VideoStatus.FAILED;
                const isDone = item.status === VideoStatus.DONE;

                return (
                  <motion.div 
                    layout
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    key={item.id} 
                    onClick={() => {
                      if (isDone) navigate(`/results?videoId=${item.id}`);
                    }}
                    className={`group glass-panel rounded-2xl overflow-hidden transition-all duration-300 flex flex-col hover:-translate-y-1 hover:shadow-xl ${
                      isFailed ? 'border-red-200/50 hover:border-red-300 hover:shadow-red-500/10' : 
                      isProcessing ? 'border-blue-200/50 hover:border-blue-300 hover:shadow-blue-500/10' : 
                      'border-white/40 hover:border-primary/40 hover:shadow-primary/10 cursor-pointer'
                    }`}
                  >
                    {/* Thumbnail Area */}
                    <div className="relative h-56 w-full overflow-hidden bg-slate-900 shrink-0">
                      <div className="absolute inset-0 bg-black/30 group-hover:bg-transparent transition-colors duration-500 z-10" />
                      
                      {/* Status Badge */}
                      <div className="absolute top-4 right-4 z-20">
                        {isDone && (
                          <span className="px-3 py-1.5 text-[11px] font-black tracking-widest uppercase rounded-xl bg-emerald-500/90 text-white backdrop-blur-md flex items-center gap-1.5 shadow-lg">
                            <CheckCircle2 size={14} />
                            {t('history.status_done')}
                          </span>
                        )}
                        {item.status === VideoStatus.PROCESSING && (
                          <span className="px-3 py-1.5 text-[11px] font-black tracking-widest uppercase rounded-xl bg-blue-500/90 text-white backdrop-blur-md flex items-center gap-1.5 shadow-lg">
                            <Loader2 size={14} className="animate-spin" />
                            {t('history.status_processing')}
                          </span>
                        )}
                        {item.status === VideoStatus.PENDING && (
                          <span className="px-3 py-1.5 text-[11px] font-black tracking-widest uppercase rounded-xl bg-amber-500/90 text-white backdrop-blur-md flex items-center gap-1.5 shadow-lg">
                            <Clock size={14} />
                            {t('history.status_pending')}
                          </span>
                        )}
                        {isFailed && (
                          <span className="px-3 py-1.5 text-[11px] font-black tracking-widest uppercase rounded-xl bg-red-500/90 text-white backdrop-blur-md flex items-center gap-1.5 shadow-lg">
                            <AlertCircle size={14} />
                            {t('history.status_failed')}
                          </span>
                        )}
                      </div>

                      {isDone && item.thumbnailUrl ? (
                        <img 
                          src={item.thumbnailUrl}
                          alt={item.title}
                          className="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-700"
                        />
                      ) : isProcessing ? (
                        <div className="w-full h-full flex flex-col items-center justify-center text-white relative bg-gradient-to-br from-slate-800 to-slate-900">
                          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-500/20 via-transparent to-transparent animate-pulse"></div>
                          <Loader2 size={40} className="text-blue-400 animate-spin mb-3" />
                          <span className="text-[11px] font-mono font-bold tracking-[0.2em] text-blue-300 uppercase relative z-10">
                            {item.stage || 'ANALYZING'}
                          </span>
                        </div>
                      ) : isFailed ? (
                        <div className="w-full h-full flex flex-col items-center justify-center text-white bg-gradient-to-br from-red-900/40 to-slate-900">
                          <AlertCircle size={48} className="text-red-400 mb-2 opacity-80" />
                          <span className="text-xs font-mono font-bold tracking-widest text-red-300 uppercase opacity-80">{t('history.status_failed')}</span>
                        </div>
                      ) : (
                        <div className="w-full h-full flex items-center justify-center" style={{ background: details.bgGradient }}>
                          <Play size={48} className="text-white opacity-40 transform group-hover:scale-110 group-hover:opacity-60 transition-all duration-500" />
                        </div>
                      )}
                    </div>

                    {/* Content Area */}
                    <div className="p-6 flex-1 flex flex-col">
                      <div className="mb-4">
                        <h3 className="font-heading text-lg font-black text-slate-900 mb-2 leading-snug group-hover:text-primary transition-colors line-clamp-2" title={item.title}>
                          {item.title}
                        </h3>
                        <div className="flex items-center gap-4 text-xs font-semibold text-slate-500">
                          <span className="flex items-center gap-1.5 bg-slate-100 px-2 py-1 rounded-md">
                            <Calendar size={14} className="text-slate-400" /> {item.date}
                          </span>
                          <span className="flex items-center gap-1.5 bg-slate-100 px-2 py-1 rounded-md">
                            <Clock size={14} className="text-slate-400" /> {item.duration}
                          </span>
                        </div>

                        {item.status === VideoStatus.PROCESSING && item.progress !== undefined && (
                          <div className="mt-4 p-3 rounded-xl bg-blue-50 border border-blue-100">
                            <div className="flex justify-between text-[11px] font-bold text-blue-600 mb-2">
                              <span className="truncate pr-2">{item.stage || 'Đang xử lý...'}</span>
                              <span className="font-mono">{item.progress}%</span>
                            </div>
                            <div className="w-full bg-blue-100/50 rounded-full h-1.5 mb-2 overflow-hidden">
                              <motion.div 
                                className="bg-blue-500 h-full rounded-full relative" 
                                initial={{ width: 0 }}
                                animate={{ width: `${item.progress}%` }}
                                transition={{ duration: 0.5 }}
                              >
                                <div className="absolute top-0 right-0 bottom-0 left-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.2)_25%,rgba(255,255,255,0.2)_50%,transparent_50%,transparent_75%,rgba(255,255,255,0.2)_75%,rgba(255,255,255,0.2)_100%)] bg-[length:1rem_1rem] animate-[progress-stripes_1s_linear_infinite]"></div>
                              </motion.div>
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="mt-auto pt-4 flex items-center justify-start gap-2 border-t border-slate-100" onClick={(e) => e.stopPropagation()}>
                        {isProcessing ? (
                          <button 
                            onClick={() => item.jobId && handleCancelJob(item.jobId)}
                            className="w-10 h-10 flex items-center justify-center text-red-500 bg-red-50 hover:bg-red-500 hover:text-white rounded-xl transition-colors" 
                            title="Dừng tác vụ"
                            disabled={!item.jobId}
                          >
                            <StopCircle size={20} />
                          </button>
                        ) : isFailed ? (
                          <button disabled className="w-10 h-10 flex items-center justify-center text-red-300 bg-red-50/50 rounded-xl cursor-not-allowed">
                            <AlertCircle size={20} />
                          </button>
                        ) : null}
                        
                        <button 
                          onClick={() => {
                            if (isDone) {
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
                          className="w-10 h-10 flex items-center justify-center text-slate-500 bg-slate-100 hover:bg-primary hover:text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed" 
                          title="Tải báo cáo PDF"
                          disabled={!isDone}
                        >
                          <Download size={18} />
                        </button>
                        
                        <button 
                          onClick={() => handleDelete(item.id)}
                          className="w-10 h-10 flex items-center justify-center text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-colors" 
                          title="Xóa"
                        >
                          <Trash2 size={18} />
                        </button>
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              {filteredItems.length === 0 && !loading && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
                  onClick={() => navigate('/upload')}
                  className="col-span-full border-2 border-dashed border-primary/30 rounded-2xl flex flex-col items-center justify-center p-12 text-center bg-primary/5 hover:bg-primary/10 transition-colors cursor-pointer group min-h-[300px]"
                >
                  <div className="w-20 h-20 rounded-full bg-white flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-md shadow-primary/10">
                    <CloudUpload size={32} className="text-primary" />
                  </div>
                  <h4 className="font-heading text-2xl text-slate-900 mb-3 font-black">{t('history.empty_state_title')}</h4>
                  <p className="text-slate-500 text-sm mb-8 max-w-sm">{t('history.empty_state_desc')}</p>
                  <button className="btn primary !px-8">{t('history.upload_btn')}</button>
                </motion.div>
              )}
            </AnimatePresence>
          )}
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
          const uniquePages = Array.from(new Set(pageNumbers));

          return (
            <div className="mt-16 flex items-center justify-center gap-2">
              <button
                onClick={() => setCurrentPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="w-10 h-10 flex items-center justify-center glass-btn rounded-xl disabled:opacity-50"
              >
                &larr;
              </button>
              {uniquePages.map((page, idx) => {
                if (page === '...') return <span key={`dots-${idx}`} className="px-2 text-slate-400">...</span>;
                return (
                  <button
                    key={`page-${page}`}
                    onClick={() => setCurrentPage(page as number)}
                    className={`w-10 h-10 flex items-center justify-center rounded-xl font-bold text-sm transition-all ${
                      currentPage === page
                        ? 'bg-primary text-white shadow-lg shadow-primary/30 scale-110'
                        : 'glass-btn hover:text-primary hover:bg-white'
                    }`}
                  >
                    {page}
                  </button>
                );
              })}
              <button
                onClick={() => setCurrentPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="w-10 h-10 flex items-center justify-center glass-btn rounded-xl disabled:opacity-50"
              >
                &rarr;
              </button>
            </div>
          );
        })()}
      </main>
      
      <footer className="bg-white/50 backdrop-blur-md flex flex-col md:flex-row justify-between items-center px-4 md:px-8 py-8 w-full border-t border-slate-200 mt-auto text-xs">
        <div className="mb-4 md:mb-0 text-center md:text-left space-y-1">
          <span className="font-heading font-bold text-slate-900 text-sm">PrismVideo</span>
          <p className="text-slate-500 font-body">© 2026 PrismVideo. All rights reserved.</p>
        </div>
        <div className="flex flex-wrap justify-center gap-6 font-bold">
          <a className="text-slate-500 hover:text-primary transition-colors" href="#credits">Credits</a>
          <a className="text-slate-500 hover:text-primary transition-colors" href="#privacy">Privacy</a>
          <a className="text-slate-500 hover:text-primary transition-colors" href="#terms">Terms</a>
        </div>
      </footer>
    </div>
  );
};
