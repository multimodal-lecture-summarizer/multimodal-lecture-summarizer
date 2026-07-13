import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';
import { parseUTCDate } from '../utils/dateUtils';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  UploadCloud, 
  Link as LinkIcon, 
  ClipboardPaste, 
  DownloadCloud, 
  Film, 
  ArrowRight,
  Loader2,
  CheckCircle2,
  Mic,
  FileText,
  Image as ImageIcon,
  Sparkles,
  AlertCircle
} from 'lucide-react';

export const UploadPage: React.FC = () => {
  const toast = useToast();
  const { t } = useTranslation();
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [currentStage, setCurrentStage] = useState<string>('queued');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<any>(null);
  const isPollingRef = useRef(false);

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const handleGlobalPaste = (e: ClipboardEvent) => {
      const activeEl = document.activeElement;
      if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
        return;
      }
      const pastedText = e.clipboardData?.getData('text') || '';
      if (pastedText.trim() && (pastedText.includes('youtube.com') || pastedText.includes('youtu.be'))) {
        setYoutubeUrl(pastedText.trim());
        toast.info(t('upload.toast_yt_clip'), t('upload.toast_yt_received'));
        handleStartProcessing(null, pastedText.trim());
      }
    };

    window.addEventListener('paste', handleGlobalPaste);
    return () => {
      window.removeEventListener('paste', handleGlobalPaste);
    };
  }, []);

  const [recentVideos, setRecentVideos] = useState<any[]>([]);

  useEffect(() => {
    const fetchRecent = () => {
      api.getVideos(undefined, 5, 0)
        .then((res) => {
          if (res.success && res.data) {
            setRecentVideos(res.data);
            
            // Auto resume polling overlay if a video is still processing
            if (!isProcessing) {
              const activeVideo = res.data.find(
                (v: any) => v.status === 'PROCESSING' || v.status === 'PENDING'
              );
              if (activeVideo) {
                const videoId = activeVideo.videoId;
                setIsProcessing(true);
                startPolling(videoId);
              }
            }
          }
        })
        .catch((err) => console.error("Failed to load recent videos:", err));
    };

    fetchRecent();
    const interval = setInterval(fetchRecent, 4000);
    return () => clearInterval(interval);
  }, [isProcessing]);

  const getStageText = (stage: string) => {
    switch (stage) {
      case 'queued': return t('upload.stage_queued');
      case 'download': return t('upload.stage_download');
      case 'audio': return t('upload.stage_audio');
      case 'speaker': return t('upload.stage_speaker');
      case 'visual': return t('upload.stage_visual');
      case 'semantic': return t('upload.stage_semantic');
      case 'timeline': return t('upload.stage_timeline');
      case 'text': return t('upload.stage_text');
      case 'completed': return t('upload.stage_completed');
      case 'failed': return t('upload.stage_failed');
      default: return t('upload.stage_default');
    }
  };

  const startPolling = (videoId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    setProgressPercent(0);
    setCurrentStage('queued');
    
    pollIntervalRef.current = setInterval(async () => {
      if (pollIntervalRef.current === null) return;
      if (isPollingRef.current) return;
      
      isPollingRef.current = true;
      try {
        const res = await api.getJobStatus(videoId);
        if (res.success && res.data && res.data.length > 0) {
          const job = res.data.find((j: any) => {
            const type = (j.jobType || j.job_type || "").toUpperCase();
            return type === 'SUMMARIZE';
          });
          if (job) {
            if (job.progress !== undefined && job.progress !== null) {
              setProgressPercent(job.progress);
            }
            if (job.stage) {
              setCurrentStage(job.stage);
            }

            const jobStatusUpper = (job.status || "").toUpperCase();
            if (jobStatusUpper === 'COMPLETED' || jobStatusUpper === 'SUCCESS' || jobStatusUpper === 'DONE') {
              if (pollIntervalRef.current !== null) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
                toast.success(t('upload.toast_success_done'), t('common.success'));
                setCurrentStep(5);
                setProgressPercent(100);
                setCurrentStage('completed');
                setTimeout(() => {
                  navigate(`/results?videoId=${videoId}`);
                }, 1000);
              }
            } else if (jobStatusUpper === 'FAILED') {
              clearInterval(pollIntervalRef.current);
              setIsProcessing(false);
              setErrorMsg(job.errorLog || 'Đã xảy ra lỗi trong quá trình xử lý video.');
              toast.error(job.errorLog || t('upload.toast_err_process'), t('upload.toast_upload_failed'));
            } else {
              let step = 1;
              const stage = job.stage;
              if (stage === 'audio' || stage === 'speaker') {
                step = 2;
              } else if (stage === 'visual') {
                step = 3;
              } else if (stage === 'semantic' || stage === 'storage' || stage === 'timeline') {
                step = 4;
              } else if (stage === 'text' || stage === 'completed') {
                step = 5;
              }
              setCurrentStep(step);
            }
          } else {
            clearInterval(pollIntervalRef.current);
            setIsProcessing(false);
          }
        } else {
          clearInterval(pollIntervalRef.current);
          setIsProcessing(false);
        }
      } catch (err) {
        console.error("Error polling job status:", err);
        // Do not immediately clear interval on a network error/500, to allow recovery
        // clearInterval(pollIntervalRef.current);
        // setIsProcessing(false);
      } finally {
        isPollingRef.current = false;
      }
    }, 2000);
  };

  const handleStartProcessing = async (file: File | null, url: string | null) => {
    setIsProcessing(true);
    setCurrentStep(1);
    setErrorMsg('');
    
    try {
      const res = await api.uploadVideo(file, url);
      if (res.success && res.data) {
        const videoId = res.data.videoId || res.data.video_id;
        toast.info(t('upload.toast_upload_success'), t('upload.toast_uploaded'));
        startPolling(videoId);
      } else {
        throw new Error(res.message || "Tải lên video thất bại.");
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Không thể gửi video lên máy chủ.");
      toast.error(err.message || t('upload.toast_upload_err'), t('upload.toast_upload_failed'));
      setIsProcessing(false);
    }
  };

  const handleYoutubeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!youtubeUrl.trim()) return;
    handleStartProcessing(null, youtubeUrl);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleStartProcessing(e.target.files[0], null);
    }
  };

  const handleBrowseClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleStartProcessing(e.dataTransfer.files[0], null);
      return;
    }
    const droppedText = e.dataTransfer.getData('text');
    if (droppedText && (droppedText.includes('youtube.com') || droppedText.includes('youtu.be'))) {
      setYoutubeUrl(droppedText.trim());
      toast.info(t('upload.toast_yt_drop'), t('upload.toast_yt_received'));
      handleStartProcessing(null, droppedText.trim());
    }
  };

  // Pipeline Steps Configuration
  const pipelineSteps = [
    { id: 1, icon: DownloadCloud, title: t('upload.step1_title') },
    { id: 2, icon: Mic, title: t('upload.step2_title') },
    { id: 3, icon: FileText, title: t('upload.step3_title') },
    { id: 4, icon: ImageIcon, title: t('upload.step4_title') },
    { id: 5, icon: Sparkles, title: t('upload.step5_title') },
  ];

  return (
    <div className="bg-[#FAF5FF] min-h-[calc(100vh-64px)] overflow-hidden">
      <div className="grid grid-cols-1 lg:grid-cols-2 h-full min-h-[calc(100vh-64px)]">
        
        {/* Left Pane - Upload Zone */}
        <div className="p-8 md:p-12 lg:p-16 flex flex-col justify-center relative">
          {/* Background Blobs */}
          <div className="absolute top-0 left-0 w-[50%] h-[50%] bg-primary/10 rounded-br-full blur-[100px] pointer-events-none" />
          
          <div className="max-w-xl mx-auto w-full relative z-10">
            <header className="mb-10">
              <h1 className="font-heading text-4xl md:text-5xl text-slate-900 mb-4 font-black tracking-tight">
                {t('upload.title')}
              </h1>
              <p className="text-slate-600 text-lg">
                {t('upload.desc')}
              </p>
            </header>

            <AnimatePresence>
              {errorMsg && (
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mb-8 p-4 glass border-l-4 border-l-red-500 rounded-xl flex items-start gap-3"
                >
                  <AlertCircle className="text-red-500 mt-0.5 shrink-0" size={20} />
                  <span className="font-medium text-slate-700">{errorMsg}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Interactive Drag & Drop Area */}
            <motion.div 
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              className={`relative overflow-hidden cursor-pointer rounded-[2rem] border-2 transition-all duration-300 ${
                dragActive ? 'border-primary bg-primary/5 shadow-2xl shadow-primary/20 scale-[1.02]' : 'border-slate-200/50 glass hover:border-primary/40 hover:shadow-xl hover:bg-white/40'
              }`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={handleBrowseClick}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                accept="video/*" 
                style={{ display: 'none' }} 
                onClick={(e) => e.stopPropagation()}
              />
              <div className="flex flex-col items-center justify-center p-16 text-center">
                <div className="w-20 h-20 bg-gradient-to-br from-primary-light to-white rounded-full flex items-center justify-center mb-6 shadow-inner text-primary">
                  <UploadCloud size={40} />
                </div>
                <h3 className="font-heading text-2xl font-bold text-slate-900 mb-2">{t('upload.drag_drop')}</h3>
                <p className="text-slate-500 font-body mb-8">{t('upload.support_formats')}</p>
                <button 
                  onClick={handleBrowseClick}
                  disabled={isProcessing}
                  className="btn primary shadow-lg shadow-primary/20 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isProcessing ? <Loader2 className="animate-spin mx-auto" size={20} /> : t('upload.browse_files')}
                </button>
              </div>
            </motion.div>

            <div className="flex items-center gap-4 my-8">
              <div className="h-px bg-slate-200/80 flex-1"></div>
              <span className="text-sm font-bold text-slate-400 uppercase tracking-widest">HOẶC</span>
              <div className="h-px bg-slate-200/80 flex-1"></div>
            </div>

            {/* URL Input */}
            <div className="glass-panel p-6 rounded-[1.5rem]">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-blue-50 text-blue-500 rounded-xl flex items-center justify-center">
                  <LinkIcon size={20} />
                </div>
                <div>
                  <h3 className="font-heading text-lg font-bold text-slate-900">{t('upload.remote_import')}</h3>
                  <p className="text-slate-500 text-sm">
                    {t('upload.youtube_desc')}
                  </p>
                </div>
              </div>
              
              <form onSubmit={handleYoutubeSubmit} className="flex flex-col gap-3">
                <div className="relative flex items-center">
                  <input 
                    className="w-full bg-white/50 border border-slate-200 rounded-xl px-4 py-3.5 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all text-slate-700" 
                    placeholder={t('upload.youtube_placeholder')} 
                    type="url"
                    value={youtubeUrl}
                    onChange={(e) => setYoutubeUrl(e.target.value)}
                  />
                  <button
                    type="button"
                    onClick={async () => {
                      try {
                        const text = await navigator.clipboard.readText();
                        if (text && (text.includes('youtube.com') || text.includes('youtu.be'))) {
                          setYoutubeUrl(text.trim());
                          toast.success(t('upload.toast_paste_success'), t('common.success'));
                        } else if (text) {
                          setYoutubeUrl(text.trim());
                          toast.info(t('upload.toast_paste_warn'), t('upload.toast_warn'));
                        } else {
                          toast.error(t('upload.toast_paste_empty'), t('upload.toast_err_paste'));
                        }
                      } catch (err) {
                        toast.error(t('upload.toast_paste_err'), t('upload.toast_err_access'));
                      }
                    }}
                    className="absolute right-3 p-2 text-slate-400 hover:text-primary transition-colors hover:bg-primary/5 rounded-lg"
                    title="Dán từ Clipboard"
                  >
                    <ClipboardPaste size={18} />
                  </button>
                </div>
                <button 
                  type="submit" 
                  disabled={isProcessing}
                  className="btn bg-slate-900 text-white hover:bg-slate-800 border-none w-full flex justify-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isProcessing ? <Loader2 className="animate-spin" size={18} /> : <DownloadCloud size={18} />}
                  {isProcessing ? t('upload.status_processing') : t('upload.fetch_video')}
                </button>
              </form>
            </div>
          </div>
        </div>

        {/* Right Pane - Processing & History */}
        <div className="bg-white border-l border-slate-200/60 p-8 md:p-12 lg:p-16 overflow-y-auto relative">
          <div className="max-w-xl mx-auto w-full">
            
            {/* Pipeline Visualization */}
            <div className="mb-16">
              <h2 className="font-heading text-2xl font-bold text-slate-900 mb-8 flex items-center gap-3">
                {isProcessing ? (
                  <Loader2 className="text-primary animate-spin" size={24} />
                ) : (
                  <Sparkles className="text-primary" size={24} />
                )}
                {t('upload.pipeline_status')}
              </h2>
              
              <div className="space-y-6 relative before:absolute before:inset-0 before:ml-[1.15rem] before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-slate-200 before:to-transparent">
                {pipelineSteps.map((step) => {
                  const isCompleted = currentStep > step.id;
                  const isActive = currentStep === step.id && isProcessing;

                  const StepIcon = step.icon;

                  return (
                    <motion.div 
                      key={step.id} 
                      className={`relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active`}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: step.id * 0.1 }}
                    >
                      <div className="flex items-center justify-center w-10 h-10 rounded-full border-4 border-white bg-slate-100 text-slate-400 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2"
                        style={isActive ? { backgroundColor: '#7C3AED', color: 'white', borderColor: '#EFE7FC' } : isCompleted ? { backgroundColor: '#10B981', color: 'white' } : {}}
                      >
                        {isCompleted ? <CheckCircle2 size={18} /> : <StepIcon size={18} />}
                      </div>
                      <div className={`w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-xl border ${
                        isActive ? 'border-primary bg-primary/5 shadow-md shadow-primary/5' : 
                        isCompleted ? 'border-emerald-200 bg-emerald-50/30' : 
                        'border-slate-100 bg-white opacity-50'
                      }`}>
                        <div className="flex items-center justify-between mb-1">
                          <h4 className={`font-bold text-sm ${isActive ? 'text-primary' : isCompleted ? 'text-emerald-700' : 'text-slate-500'}`}>
                            {step.title}
                          </h4>
                        </div>
                        <div className={`text-xs font-mono font-medium ${isActive ? 'text-primary animate-pulse' : isCompleted ? 'text-emerald-500' : 'text-slate-400'}`}>
                          {isActive && step.id === 1 && currentStage !== 'queued' ? `${getStageText(currentStage)} ${progressPercent}%` : 
                           isActive ? t('upload.status_processing') : 
                           isCompleted ? t('upload.status_complete') : 
                           t('upload.status_pending')}
                        </div>
                        {isActive && (
                          <div className="w-full bg-slate-200 h-1.5 rounded-full mt-3 overflow-hidden">
                            <motion.div 
                              className="bg-primary h-full" 
                              initial={{ width: 0 }}
                              animate={{ width: `${progressPercent}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>

            {/* Recent Uploads */}
            <div>
              <div className="flex items-center justify-between mb-6">
                <h2 className="font-heading text-2xl font-bold text-slate-900">{t('upload.recent_videos')}</h2>
                <Link to="/history" className="text-primary hover:text-primary-hover text-sm font-bold flex items-center gap-1 group">
                  {t('upload.view_all')}
                  <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                </Link>
              </div>

              <div className="space-y-3">
                {recentVideos.length === 0 ? (
                  <div className="p-8 text-center glass border border-slate-200 rounded-2xl">
                    <Film className="mx-auto text-slate-300 mb-3" size={32} />
                    <p className="text-slate-500">{t('upload.no_videos')}</p>
                  </div>
                ) : (
                  recentVideos.map((video, idx) => {
                    const statusUpper = (video.status || "").toUpperCase();
                    const isDone = statusUpper === 'DONE' || statusUpper === 'COMPLETED' || statusUpper === 'SUCCESS';
                    const isFailed = statusUpper === 'FAILED';
                    
                    return (
                      <motion.div 
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        key={video.videoId} 
                        className="p-4 rounded-2xl border border-slate-200 bg-white hover:border-primary/30 hover:shadow-md transition-all flex items-center gap-4 group"
                      >
                        {video.scenes && video.scenes.length > 0 && video.scenes[0].keyframeUrl ? (
                          <div className="w-16 h-12 rounded-xl overflow-hidden border border-slate-200 shrink-0 relative">
                            <img src={video.scenes[0].keyframeUrl} alt="thumbnail" className="w-full h-full object-cover" />
                            <div className="absolute inset-0 bg-black/10 group-hover:bg-transparent transition-colors"></div>
                          </div>
                        ) : (
                          <div className="w-16 h-12 rounded-xl bg-slate-100 flex items-center justify-center text-slate-400 shrink-0 border border-slate-200">
                            <Film size={20} />
                          </div>
                        )}
                        <div className="flex-1 min-w-0">
                          <h4 className="font-bold text-slate-900 truncate text-sm">
                            {video.title || (video.originalUrl ? video.originalUrl : (video.r2Url ? (video.r2Url.split('?')[0].split('/').pop() || 'video.mp4') : (video.filePath && !video.filePath.includes('/stream') ? (video.filePath.split('?')[0].split('/').pop() || 'video.mp4') : 'Video bài giảng')))}
                          </h4>
                          <div className="flex items-center gap-3 text-xs text-slate-500 mt-1">
                            <span className="font-mono">{parseUTCDate(video.uploadedAt)!.toLocaleDateString()}</span>
                            <span>•</span>
                            <span>{video.duration ? `${Math.round(video.duration)}s` : 'N/A'}</span>
                          </div>
                        </div>
                        <div className="flex flex-col items-end gap-2 shrink-0">
                          <span className={`px-2.5 py-1 text-[10px] font-black rounded-lg uppercase tracking-wider ${
                            isDone ? 'bg-emerald-50 text-emerald-600' :
                            isFailed ? 'bg-red-50 text-red-600' :
                            'bg-primary/10 text-primary animate-pulse'
                          }`}>
                            {isDone ? t('history.status_done') :
                             isFailed ? t('history.status_failed') :
                             `${t('history.status_processing')} ${video.progress !== undefined && video.progress !== null ? `(${video.progress}%)` : ''}`}
                          </span>
                          {isDone && (
                            <Link 
                              to={`/results?videoId=${video.videoId}`} 
                              className="text-primary hover:text-primary-hover font-bold text-xs inline-flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                              {t('upload.view_results')}
                            </Link>
                          )}
                        </div>
                      </motion.div>
                    );
                  })
                )}
              </div>
            </div>

          </div>
        </div>
        
      </div>
    </div>
  );
};
