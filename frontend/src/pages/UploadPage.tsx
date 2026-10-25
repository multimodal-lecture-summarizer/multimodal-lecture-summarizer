import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';
import { parseUTCDate } from '../utils/dateUtils';
import { motion, AnimatePresence } from 'framer-motion';
import { VideoPreflightModal } from '../components/VideoPreflightModal';
import { 
  validateVideoFileClient, 
  buildClientPreflightData, 
  normalizeVideoStandards, 
  parseAllowedFormats,
  type VideoStandardConfig, 
  type ClientVideoMetadata,
  type ClientPreflightData
} from '../utils/videoValidator';
import type { SummaryLength } from '../types';
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
  Image as ImageIcon, 
  AlertCircle,
  FileCheck2,
  X,
  BookOpen,
  Layers,
  GraduationCap
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
  const [preflightOpen, setPreflightOpen] = useState(false);
  const [stagedFile, setStagedFile] = useState<File | null>(null);
  const [stagedUrl, setStagedUrl] = useState<string | null>(null);
  const [selectedSummaryLength, setSelectedSummaryLength] = useState<SummaryLength>('medium');
  const [sourceLanguage, setSourceLanguage] = useState<'auto' | 'vi' | 'en'>('auto');

  // Client-side Video Validation & Standards State
  const [standards, setStandards] = useState<VideoStandardConfig>(() => normalizeVideoStandards(null));
  const [isValidatingClient, setIsValidatingClient] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [clientMetadata, setClientMetadata] = useState<ClientVideoMetadata | null>(null);
  const [stagedPreflightData, setStagedPreflightData] = useState<ClientPreflightData | null>(null);

  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<any>(null);
  const isPollingRef = useRef(false);
  const isUploadingRef = useRef(false);
  const activeVideoIdRef = useRef<string | null>(null);

  // Fetch active system standards from Backend on mount
  useEffect(() => {
    let isMounted = true;
    api.getStandards()
      .then((res) => {
        if (isMounted && res.success && res.data) {
          setStandards(normalizeVideoStandards(res.data));
        }
      })
      .catch((err) => {
        console.warn("Failed to fetch dynamic video standards from backend, using presets:", err);
      });
    return () => {
      isMounted = false;
    };
  }, []);

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
        triggerPreflight(null, pastedText.trim());
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
            
            // Auto resume polling overlay if a video is processing and we are not currently polling
            if (!activeVideoIdRef.current) {
              const activeVideos = res.data.filter(
                (v: any) => v.status === 'PROCESSING' || v.status === 'PENDING'
              );
              if (activeVideos.length > 0) {
                // Prioritize video with highest progress or active run
                const target = activeVideos.sort((a: any, b: any) => (b.progress || 0) - (a.progress || 0))[0];
                setIsProcessing(true);
                startPolling(target.videoId);
              }
            }
          }
        })
        .catch((err) => console.error("Failed to load recent videos:", err));
    };

    fetchRecent();
    const interval = setInterval(fetchRecent, 4000);
    return () => clearInterval(interval);
  }, []);

  const mapStageToStep = (stage: string | null | undefined, progress?: number): number => {
    if (!stage && (progress === undefined || progress === null)) return 1;
    const s = (stage || '').toLowerCase().trim();

    if (
      s.includes('timeline') ||
      s.includes('quality') ||
      s.includes('storage') ||
      s.includes('intermediate') ||
      s.includes('summariz') ||
      s.includes('text') ||
      s.includes('rag') ||
      s.includes('complete') ||
      (progress !== undefined && progress >= 75)
    ) {
      return 5;
    }

    if (
      s.includes('semantic') ||
      s.includes('ocr') ||
      s.includes('clip') ||
      s.includes('florence') ||
      (progress !== undefined && progress >= 58)
    ) {
      return 4;
    }

    if (
      s.includes('visual') ||
      s.includes('scene') ||
      s.includes('slide') ||
      s.includes('object') ||
      s.includes('tracking') ||
      s.includes('yolo') ||
      (progress !== undefined && progress >= 38)
    ) {
      return 3;
    }

    if (
      s.includes('audio') ||
      s.includes('speaker') ||
      s.includes('whisper') ||
      s.includes('diariz') ||
      (progress !== undefined && progress >= 6)
    ) {
      return 2;
    }

    return 1;
  };

  const getStageText = (stage: string) => {
    const s = (stage || '').toLowerCase().trim();
    if (s.includes('queued')) return 'Đang trong hàng đợi xử lý...';
    if (s.includes('uploading')) return 'Đang gửi tệp video lên máy chủ...';
    if (s.includes('download')) return 'Đang chuẩn bị tệp video...';
    if (s.includes('preflight')) return 'Đang thẩm định thông số kỹ thuật...';
    if (s.includes('audio') || s.includes('whisper')) return 'Đang trích xuất & nhận dạng âm thanh...';
    if (s.includes('speaker') || s.includes('diariz')) return 'Đang phân biệt người nói theo thời gian...';
    if (s.includes('object') || s.includes('tracking') || s.includes('yolo')) return 'Đang phát hiện & theo dõi đối tượng trong cảnh...';
    if (s.includes('visual') || s.includes('scene')) return 'Đang phát hiện phân cảnh & trích xuất slide...';
    if (s.includes('semantic') || s.includes('ocr') || s.includes('florence')) return 'Đang phân tích thị giác & OCR nội dung...';
    if (s.includes('timeline')) return 'Đang đồng bộ trục thời gian bài giảng...';
    if (s.includes('quality')) return 'Đang tối ưu hóa chất lượng slide bài giảng...';
    if (s.includes('storage')) return 'Đang đồng bộ lưu trữ slide keyframes...';
    if (s.includes('intermediate')) return 'Đã sẵn sàng dữ liệu, bắt đầu tóm tắt...';
    if (s.includes('summariz')) return 'Đang tổng hợp nội dung bài học...';
    if (s.includes('text')) return 'Đang tổng hợp nội dung bài học...';
    if (s.includes('rag')) return 'Đang tạo chỉ mục hỏi đáp...';
    if (s.includes('complete')) return 'Xử lý bài giảng hoàn tất!';
    if (s.includes('fail')) return 'Xử lý thất bại';
    return 'Đang phân tích dữ liệu đa phương thức...';
  };

  const startPolling = (videoId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    activeVideoIdRef.current = videoId;
    setProgressPercent(prev => (prev > 0 ? prev : 2));
    
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
              const p = Math.round(Number(job.progress));
              setProgressPercent(prev => Math.max(prev, p));
            }
            if (job.stage || job.progress !== undefined) {
              const nextStep = mapStageToStep(job.stage, job.progress);
              if (job.stage) {
                setCurrentStage(job.stage);
              }
              // Monotonic step progression: a running pipeline step never jumps backwards
              setCurrentStep(prev => Math.max(prev, nextStep));
            }

            const jobStatusUpper = (job.status || "").toUpperCase();
            if (jobStatusUpper === 'COMPLETED' || jobStatusUpper === 'SUCCESS' || jobStatusUpper === 'DONE') {
              if (pollIntervalRef.current !== null) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
                activeVideoIdRef.current = null;
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
              pollIntervalRef.current = null;
              activeVideoIdRef.current = null;
              setIsProcessing(false);
              setErrorMsg(job.errorLog || 'Đã xảy ra lỗi trong quá trình xử lý video.');
              toast.error(job.errorLog || t('upload.toast_err_process'), t('upload.toast_upload_failed'));
            }
          }
        }
      } catch (err) {
        console.error("Error polling job status:", err);
      } finally {
        isPollingRef.current = false;
      }
    }, 2000);
  };

  const handleStartProcessing = async (file: File | null, url: string | null, customLength?: SummaryLength) => {
    if (isUploadingRef.current) return;
    isUploadingRef.current = true;
    setIsProcessing(true);
    setCurrentStep(1);
    setProgressPercent(2);
    setCurrentStage('uploading');
    setErrorMsg('');
    
    const lengthToUse = customLength || selectedSummaryLength;
    try {
      const res = await api.uploadVideo(file, url, sourceLanguage, lengthToUse);
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
      activeVideoIdRef.current = null;
    } finally {
      isUploadingRef.current = false;
    }
  };

  const processSelectedFile = async (file: File) => {
    if (isProcessing) return;
    setValidationErrors([]);
    setErrorMsg('');
    setIsValidatingClient(true);

    try {
      const result = await validateVideoFileClient(file, standards);
      if (!result.valid) {
        setValidationErrors(result.errors);
        setStagedFile(null);
        setClientMetadata(null);
        setStagedPreflightData(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
        toast.error(
          result.errors[0] || "Tệp video không đáp ứng tiêu chuẩn hệ thống.",
          "Tiêu chuẩn Video Không Hợp Lệ"
        );
        return;
      }

      // Valid: save metadata and construct client preflight checklist
      setClientMetadata(result.metadata || null);
      setStagedFile(file);
      setStagedUrl(null);

      if (result.metadata) {
        const clientPreflight = buildClientPreflightData(file, result.metadata, standards);
        setStagedPreflightData(clientPreflight);
      }

      // Open tactile preflight inspector modal
      setPreflightOpen(true);
    } catch (err: any) {
      console.error("Client video validation exception:", err);
      const msg = err.message || "Không thể phân tích metadata tệp video trên trình duyệt.";
      setValidationErrors([msg]);
      toast.error(msg, "Lỗi Tệp Video");
    } finally {
      setIsValidatingClient(false);
    }
  };

  const triggerPreflight = (file: File | null, url: string | null) => {
    if (isProcessing) return;
    setStagedFile(file);
    setStagedUrl(url);
    setPreflightOpen(true);
  };

  const handlePreflightProceed = () => {
    const f = stagedFile;
    const u = stagedUrl;
    setPreflightOpen(false);
    setStagedFile(null);
    setStagedUrl(null);
    setClientMetadata(null);
    setStagedPreflightData(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    handleStartProcessing(f, u, selectedSummaryLength);
  };

  const handlePreflightClose = () => {
    setPreflightOpen(false);
    setStagedFile(null);
    setStagedUrl(null);
    setClientMetadata(null);
    setStagedPreflightData(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleClearStagedFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setStagedFile(null);
    setClientMetadata(null);
    setStagedPreflightData(null);
    setValidationErrors([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleYoutubeSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!youtubeUrl.trim()) return;
    setValidationErrors([]);
    setClientMetadata(null);
    setStagedPreflightData(null);
    triggerPreflight(null, youtubeUrl.trim());
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
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
      processSelectedFile(e.dataTransfer.files[0]);
      return;
    }
    const droppedText = e.dataTransfer.getData('text');
    if (droppedText && (droppedText.includes('youtube.com') || droppedText.includes('youtu.be'))) {
      setYoutubeUrl(droppedText.trim());
      setValidationErrors([]);
      setClientMetadata(null);
      setStagedPreflightData(null);
      toast.info(t('upload.toast_yt_drop'), t('upload.toast_yt_received'));
      triggerPreflight(null, droppedText.trim());
    }
  };

  // Pipeline Steps Configuration matching real AI worker lifecycle
  const pipelineSteps = [
    { id: 1, icon: FileCheck2, title: 'Tải lên & Thẩm định video' },
    { id: 2, icon: Mic, title: 'Xử lý Âm thanh & Người nói' },
    { id: 3, icon: Film, title: 'Phân cảnh & Theo dõi Đối tượng' },
    { id: 4, icon: ImageIcon, title: 'Phân tích Thị giác & OCR Slide' },
    { id: 5, icon: BookOpen, title: 'Tổng hợp Bài học & Trục Thời gian' },
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

            {/* Server Error Message */}
            <AnimatePresence>
              {errorMsg && (
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mb-6 p-4 glass border-l-4 border-l-red-500 rounded-xl flex items-start gap-3 shadow-sm"
                >
                  <AlertCircle className="text-red-500 mt-0.5 shrink-0" size={20} />
                  <span className="font-medium text-slate-700 text-sm">{errorMsg}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Client-Side Validation Error Alert (0ms Instant Pre-upload Rejection) */}
            <AnimatePresence>
              {validationErrors.length > 0 && (
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-2xl shadow-sm text-xs text-rose-900"
                >
                  <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-rose-200/60">
                    <div className="flex items-center gap-2 font-bold text-rose-700">
                      <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
                      <span>Tệp không đáp ứng quy chuẩn hệ thống (Đã chặn trước tải lên):</span>
                    </div>
                    <button 
                      type="button" 
                      onClick={() => setValidationErrors([])}
                      className="p-1 hover:bg-rose-100 rounded text-rose-500 transition-colors"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <ul className="list-disc list-inside space-y-1 font-medium pl-1 text-rose-800">
                    {validationErrors.map((err, idx) => (
                      <li key={idx}>{err}</li>
                    ))}
                  </ul>
                  <p className="mt-2 text-[11px] text-rose-600 italic">
                    Hệ thống đã hủy tiến trình để tiết kiệm 100% băng thông tải lên mạng.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Interactive Drag & Drop Area */}
            <motion.div 
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              className={`relative overflow-hidden cursor-pointer rounded-[2rem] border-2 transition-all duration-300 ${
                dragActive 
                  ? 'border-primary bg-primary/5 shadow-2xl shadow-primary/20 scale-[1.02]' 
                  : stagedFile && clientMetadata
                  ? 'border-emerald-300 bg-emerald-50/30 hover:border-emerald-400'
                  : 'border-slate-200/50 glass hover:border-primary/40 hover:shadow-xl hover:bg-white/40'
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
              
              <div className="flex flex-col items-center justify-center p-8 md:p-12 text-center">
                {isValidatingClient ? (
                  <div className="py-8 flex flex-col items-center gap-3">
                    <Loader2 className="w-12 h-12 text-primary animate-spin" />
                    <p className="text-sm font-semibold text-slate-700">Đang kiểm tra metadata video tại trình duyệt...</p>
                    <span className="text-xs text-slate-500">Đọc container, duration & độ phân giải (0ms upload overhead)</span>
                  </div>
                ) : stagedFile && clientMetadata ? (
                  /* Staged Valid File Preview Card */
                  <div className="w-full max-w-md bg-white rounded-2xl p-5 border border-emerald-200 shadow-sm text-left">
                    <div className="flex items-start justify-between gap-3 mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-emerald-100 text-emerald-600 flex items-center justify-center shrink-0">
                          <FileCheck2 size={24} />
                        </div>
                        <div className="min-w-0">
                          <h4 className="font-bold text-slate-900 text-sm truncate max-w-[220px]" title={stagedFile.name}>
                            {stagedFile.name}
                          </h4>
                          <span className="text-[11px] text-emerald-600 font-semibold flex items-center gap-1">
                            <CheckCircle2 size={12} /> Đã kiểm định hợp lệ tại trình duyệt
                          </span>
                        </div>
                      </div>
                      <button 
                        type="button" 
                        onClick={handleClearStagedFile}
                        className="p-1 text-slate-400 hover:text-rose-500 hover:bg-slate-100 rounded-lg transition-colors"
                        title="Hủy tệp"
                      >
                        <X size={16} />
                      </button>
                    </div>

                    {/* Metadata Badges */}
                    <div className="grid grid-cols-3 gap-2 py-2.5 px-3 bg-slate-50 rounded-xl mb-4 text-xs font-mono">
                      <div>
                        <span className="text-[10px] text-slate-400 block font-sans">Dung lượng</span>
                        <span className="font-bold text-slate-800">{clientMetadata.sizeMb} MB</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-400 block font-sans">Thời lượng</span>
                        <span className="font-bold text-slate-800">
                          {Math.floor(clientMetadata.duration / 60)}m {Math.floor(clientMetadata.duration % 60)}s
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-400 block font-sans">Độ phân giải</span>
                        <span className="px-1.5 py-0.5 bg-emerald-100 text-emerald-800 font-bold rounded text-[11px]">
                          {clientMetadata.width}×{clientMetadata.height}
                        </span>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <button
                        type="button"
                        disabled={isProcessing}
                        onClick={(e) => {
                          e.stopPropagation();
                          setPreflightOpen(true);
                        }}
                        className="btn primary flex-1 py-2 text-xs shadow-md shadow-primary/20 flex items-center justify-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isProcessing ? (
                          <>
                            <Loader2 className="animate-spin" size={14} />
                            <span>Đang xử lý tệp...</span>
                          </>
                        ) : (
                          <>
                            <span>Kiểm tra chi tiết & Bắt đầu</span>
                            <ArrowRight size={14} />
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        disabled={isProcessing}
                        onClick={handleBrowseClick}
                        className="btn bg-slate-100 hover:bg-slate-200 text-slate-700 py-2 px-3 text-xs disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Đổi tệp
                      </button>
                    </div>
                  </div>
                ) : (
                  /* Standard Drag and Drop State with Dynamic Standards Badge */
                  <>
                    <div className="w-16 h-16 md:w-20 md:h-20 bg-gradient-to-br from-primary-light to-white rounded-full flex items-center justify-center mb-4 shadow-inner text-primary">
                      <UploadCloud size={36} />
                    </div>
                    <h3 className="font-heading text-xl md:text-2xl font-bold text-slate-900 mb-2">
                      {t('upload.drag_drop')}
                    </h3>
                    
                    {/* Dynamic Standards Indicators */}
                    <div className="w-full max-w-lg mb-6">
                      <div className="inline-flex flex-wrap items-center justify-center gap-1.5 text-[11px] text-slate-600 font-body">
                        <span className="px-2.5 py-1 rounded-full bg-primary/10 text-primary font-semibold border border-primary/20">
                          Định dạng: {parseAllowedFormats(standards.allowedFormats).map((f) => f.toUpperCase()).join(', ')}
                        </span>
                        <span className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium border border-slate-200/80">
                          Tối đa: {standards.maxFileSize} MB
                        </span>
                        <span className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium border border-slate-200/80">
                          Thời lượng: ≤ {Math.floor(standards.maxDuration / 60)} phút
                        </span>
                        <span className="px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 font-medium border border-slate-200/80">
                          Độ phân giải: {standards.minHeight}p - {standards.maxHeight}p
                        </span>
                      </div>
                    </div>

                    <button 
                      type="button"
                      onClick={handleBrowseClick}
                      disabled={isProcessing}
                      className="btn primary shadow-lg shadow-primary/20 disabled:opacity-70 disabled:cursor-not-allowed text-xs md:text-sm px-6 py-2.5"
                    >
                      {isProcessing ? <Loader2 className="animate-spin mx-auto" size={18} /> : t('upload.browse_files')}
                    </button>
                  </>
                )}
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

            <div className="mt-8">
              <label htmlFor="source-language" className="text-xs font-bold uppercase tracking-wider text-slate-700">
                {t('upload.source_language')}
              </label>
              <select
                id="source-language"
                value={sourceLanguage}
                onChange={(event) => setSourceLanguage(event.target.value as 'auto' | 'vi' | 'en')}
                disabled={isProcessing}
                className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-800"
              >
                <option value="auto">{t('upload.source_language_auto')}</option>
                <option value="vi">{t('upload.source_language_vi')}</option>
                <option value="en">{t('upload.source_language_en')}</option>
              </select>
            </div>

            {/* Summary Depth Selection Cards */}
            <div className="mt-8 mb-4">
              <div className="flex items-center justify-between mb-3">
                <label className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                  <Layers size={14} className="text-primary" />
                  Cấp độ nhận thức & Độ dài tóm tắt mặc định
                </label>
                <span className="text-[11px] text-slate-500 font-medium">
                  Có thể chuyển đổi tức thì trên trang Kết quả
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {/* 1. Short */}
                <button
                  type="button"
                  onClick={() => setSelectedSummaryLength('short')}
                  className={`p-3.5 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
                    selectedSummaryLength === 'short'
                      ? 'border-primary bg-primary/5 ring-1 ring-primary/30 shadow-xs'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-slate-900">
                        <BookOpen size={13} className={selectedSummaryLength === 'short' ? 'text-primary' : 'text-slate-400'} />
                        Ngắn gọn (TL;DR)
                      </span>
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                        Cô đọng
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      Nắm bắt nhanh thông điệp & hành động cốt lõi.
                    </p>
                  </div>
                  {selectedSummaryLength === 'short' && (
                    <div className="mt-2 flex items-center gap-1 text-[10px] font-bold text-primary">
                      <CheckCircle2 size={12} /> Đã chọn
                    </div>
                  )}
                </button>

                {/* 2. Medium */}
                <button
                  type="button"
                  onClick={() => setSelectedSummaryLength('medium')}
                  className={`p-3.5 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
                    selectedSummaryLength === 'medium'
                      ? 'border-primary bg-primary/5 ring-1 ring-primary/30 shadow-xs'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-slate-900">
                        <Layers size={13} className={selectedSummaryLength === 'medium' ? 'text-primary' : 'text-slate-400'} />
                        Tiêu chuẩn (Overview)
                      </span>
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                        Phổ biến
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      Cân đối giữa lý thuyết và các bước thực hành.
                    </p>
                  </div>
                  {selectedSummaryLength === 'medium' && (
                    <div className="mt-2 flex items-center gap-1 text-[10px] font-bold text-primary">
                      <CheckCircle2 size={12} /> Mặc định
                    </div>
                  )}
                </button>

                {/* 3. Detailed */}
                <button
                  type="button"
                  onClick={() => setSelectedSummaryLength('detailed')}
                  className={`p-3.5 rounded-xl border text-left transition-all relative flex flex-col justify-between ${
                    selectedSummaryLength === 'detailed'
                      ? 'border-primary bg-primary/5 ring-1 ring-primary/30 shadow-xs'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-slate-900">
                        <GraduationCap size={13} className={selectedSummaryLength === 'detailed' ? 'text-primary' : 'text-slate-400'} />
                        Chuyên sâu (Guide)
                      </span>
                      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                        Chi tiết
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      Phân tích cặn kẽ từng bước kèm dẫn chứng.
                    </p>
                  </div>
                  {selectedSummaryLength === 'detailed' && (
                    <div className="mt-2 flex items-center gap-1 text-[10px] font-bold text-primary">
                      <CheckCircle2 size={12} /> Đã chọn
                    </div>
                  )}
                </button>
              </div>
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
                  <Layers className="text-primary" size={24} />
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
                        <div className={`text-xs font-mono font-medium ${isActive ? 'text-primary' : isCompleted ? 'text-emerald-500' : 'text-slate-400'}`}>
                          {isActive ? (
                            <span className="flex items-center gap-1.5 font-bold">
                              <span className="inline-block w-2 h-2 rounded-full bg-primary animate-ping" />
                              {getStageText(currentStage)} ({progressPercent}%)
                            </span>
                          ) : isCompleted ? (
                            <span className="font-semibold text-emerald-600">HOÀN TẤT</span>
                          ) : (
                            <span>ĐANG CHỜ</span>
                          )}
                        </div>
                        {isActive && (
                          <div className="w-full bg-slate-200 h-1.5 rounded-full mt-3 overflow-hidden">
                            <motion.div 
                              className="bg-primary h-full" 
                              initial={{ width: 0 }}
                              animate={{ width: `${Math.max(8, progressPercent)}%` }}
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

      {/* Pre-flight Technical Inspector Modal */}
      <VideoPreflightModal
        isOpen={preflightOpen}
        onClose={handlePreflightClose}
        onProceed={handlePreflightProceed}
        file={stagedFile}
        url={stagedUrl}
        preloadedData={stagedPreflightData}
        standards={standards}
        summaryLength={selectedSummaryLength}
        onSummaryLengthChange={setSelectedSummaryLength}
      />
    </div>
  );
};
