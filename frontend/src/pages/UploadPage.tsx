import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { api } from '../services/api';
import { useToast } from '../context/ToastContext';

export const UploadPage: React.FC = () => {
  const toast = useToast();
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
        toast.info("Phát hiện liên kết YouTube từ clipboard. Đang chuẩn bị phân tích...", "Đã nhận liên kết");
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
      case 'queued':
        return 'Đang chờ xử lý trong hàng đợi...';
      case 'download':
        return 'Đang tải xuống và kiểm tra tệp video...';
      case 'audio':
        return 'Đang tách xuất âm thanh và phân tích giọng nói (WhisperX)...';
      case 'speaker':
        return 'Đang nhận dạng và phân vai người nói (ASR)...';
      case 'visual':
        return 'Đang phân tích cảnh quay và trích xuất slide keyframe...';
      case 'semantic':
        return 'Đang quét chữ OCR trên hình ảnh bằng CLIP...';
      case 'timeline':
        return 'Đang đồng bộ dòng thời gian và căn lề bài giảng...';
      case 'text':
        return 'Đang sử dụng LLM để tóm tắt và phân chương bài học...';
      case 'completed':
        return 'Hoàn thành xử lý bài giảng!';
      case 'failed':
        return 'Xử lý thất bại. Vui lòng kiểm tra lại.';
      default:
        return 'Đang xử lý phân tích bài giảng...';
    }
  };

  const startPolling = (videoId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    setProgressPercent(0);
    setCurrentStage('queued');
    
    pollIntervalRef.current = setInterval(async () => {
      // Prevent overlapping updates if already marked as completed
      if (pollIntervalRef.current === null) return;
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
                toast.success('Xử lý video hoàn tất! Đang chuyển hướng...', 'Thành công');
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
              toast.error(job.errorLog || 'Đã xảy ra lỗi trong quá trình xử lý video.', 'Xử lý thất bại');
            } else {
              // Map stage to step
              let step = 1;
              const stage = job.stage;
              if (stage === 'audio' || stage === 'speaker') {
                step = 2;
              } else if (stage === 'visual') {
                step = 3;
              } else if (stage === 'semantic' || stage === 'timeline') {
                step = 4;
              } else if (stage === 'text') {
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
        clearInterval(pollIntervalRef.current);
        setIsProcessing(false);
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
        toast.info("Đã gửi video thành công. Đang bắt đầu phân tích...", "Đã tải lên");
        startPolling(videoId);
      } else {
        throw new Error(res.message || "Tải lên video thất bại.");
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Không thể gửi video lên máy chủ.");
      toast.error(err.message || "Không thể gửi video lên máy chủ.", "Tải lên thất bại");
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

  // Drag and drop handlers
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
      toast.info("Phát hiện liên kết YouTube được thả vào. Đang chuẩn bị phân tích...", "Đã nhận liên kết");
      handleStartProcessing(null, droppedText.trim());
    }
  };

  return (
    <div className="bg-surface text-on-surface font-body-md text-body-md antialiased min-h-screen">
      <main className="flex-1 overflow-y-auto bg-background p-6 md:p-margin-desktop">
        <div className="max-w-container-max mx-auto">
          <header className="mb-10">
            <h1 className="font-headline-xl text-3xl md:text-headline-xl text-deep-navy mb-2 font-bold">
              Initialize Multimodal Analysis
            </h1>
            <p className="text-secondary text-body-lg">
              Tải lên file video bài giảng hoặc nhập liên kết YouTube để hệ thống AI (Audio / Vision / LLM Fusion) bắt đầu xử lý tự động.
            </p>
          </header>

          {errorMsg && (
            <div className="mb-6 p-4 bg-red-50 text-red-700 border border-red-200 rounded-lg flex items-center gap-2">
              <span className="material-symbols-outlined text-red-600">error</span>
              <span className="font-semibold text-sm">{errorMsg}</span>
            </div>
          )}

          {/* Upload Section Bento Grid */}
          <div className="relative">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">
              {/* File Drag & Drop (Span 2) */}
              <div 
                className={`lg:col-span-2 bg-surface border rounded-lg p-8 upload-area-gradient group cursor-pointer transition-all duration-300 relative overflow-hidden ${
                  dragActive ? 'border-vibrant-cyan ring-2 ring-vibrant-cyan/20' : 'border-outline-variant hover:border-vibrant-cyan'
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
                <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-outline-variant/50 rounded-lg group-hover:border-vibrant-cyan/50 transition-colors">
                  <div className="w-16 h-16 bg-primary-fixed text-deep-navy rounded-full flex items-center justify-center mb-4 group-hover:bg-vibrant-cyan group-hover:text-white transition-all">
                    <span className="material-symbols-outlined text-[32px]">upload_file</span>
                  </div>
                  <h3 className="font-headline-md text-xl font-bold text-deep-navy mb-1">Kéo thả file video hoặc liên kết YouTube vào đây</h3>
                  <p className="text-secondary font-body-sm mb-6 text-sm">Hỗ trợ MP4, AVI, MKV (Tối đa 2GB) hoặc thả link YouTube</p>
                  <button 
                    onClick={handleBrowseClick}
                    className="px-6 py-2 bg-deep-navy text-white font-label-md text-label-md rounded hover:bg-primary transition-colors font-semibold"
                  >
                    Browse Files
                  </button>
                </div>
              </div>

              {/* YouTube URL Input */}
              <div className="bg-surface border border-outline-variant rounded-lg p-8 flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 bg-error-container text-on-error-container rounded-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-[24px]">link</span>
                    </div>
                    <h3 className="font-headline-md text-lg font-bold text-deep-navy">Remote Import</h3>
                  </div>
                  <p className="text-secondary font-body-sm text-sm mb-6">
                    Phân tích nội dung trực tiếp từ YouTube bằng cách cung cấp đường dẫn URL công khai.
                  </p>
                  <form onSubmit={handleYoutubeSubmit} className="space-y-4">
                    <div className="relative">
                      <label className="block font-label-sm text-xs font-semibold text-secondary mb-2 uppercase tracking-wider">
                        YouTube URL
                      </label>
                      <div className="relative flex items-center">
                        <input 
                          className="w-full pl-4 pr-10 py-3 bg-background border border-outline-variant rounded-lg focus:ring-1 focus:ring-vibrant-cyan focus:border-vibrant-cyan outline-none transition-all font-mono-data text-sm" 
                          placeholder="https://youtube.com/watch?v=..." 
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
                                toast.success("Đã dán liên kết từ clipboard!", "Thành công");
                              } else if (text) {
                                setYoutubeUrl(text.trim());
                                toast.info("Liên kết đã dán có vẻ không phải là YouTube URL.", "Cảnh báo");
                              } else {
                                toast.error("Bộ nhớ tạm rỗng.", "Không thể dán");
                              }
                            } catch (err) {
                              toast.error("Vui lòng cho phép quyền truy cập clipboard hoặc nhấn Ctrl+V.", "Lỗi truy cập");
                            }
                          }}
                          className="absolute right-3 text-secondary hover:text-vibrant-cyan transition-colors flex items-center justify-center"
                          title="Dán từ Clipboard"
                        >
                          <span className="material-symbols-outlined text-[20px]">content_paste</span>
                        </button>
                      </div>
                    </div>
                    <button 
                      type="submit" 
                      className="w-full py-3 bg-secondary-container text-on-secondary-container font-semibold font-label-md text-label-md rounded hover:bg-slate-200 transition-colors flex items-center justify-center gap-2"
                    >
                      <span className="material-symbols-outlined">cloud_download</span>
                      Fetch Video
                    </button>
                  </form>
                </div>
                <div className="pt-6 border-t border-outline-variant/30 mt-6 lg:mt-0">
                  <div className="flex items-center justify-between text-secondary text-xs">
                    <span className="font-label-sm font-medium">Hạn ngạch API ngày</span>
                    <span className="font-mono-data text-deep-navy font-bold">42 / 100</span>
                  </div>
                  <div className="w-full bg-surface-container-high h-1.5 rounded-full mt-2 overflow-hidden">
                    <div className="bg-deep-navy h-full" style={{ width: '42%' }}></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Progress Overlay (Displayed when isProcessing is true) */}
            {isProcessing && (
              <div className="absolute inset-0 bg-white/95 backdrop-blur-sm rounded-xl flex items-center justify-center p-8 transition-opacity duration-300 z-50 border border-outline-variant shadow-lg">
                <div className="w-full max-w-lg space-y-6">
                  <div className="flex justify-between items-center">
                    <span className="font-headline-md text-lg text-deep-navy font-bold flex items-center gap-2">
                      <span className="material-symbols-outlined text-vibrant-cyan animate-spin">sync</span>
                      Đang phân tích video bài giảng...
                    </span>
                    <span className="font-mono-data text-lg text-vibrant-cyan font-bold">
                      {progressPercent}%
                    </span>
                  </div>
                  <div className="w-full bg-surface-container-high h-3 rounded-full overflow-hidden shadow-inner">
                    <div 
                      className="bg-vibrant-cyan h-full transition-all duration-500 ease-out" 
                      style={{ width: `${progressPercent}%` }}
                    ></div>
                  </div>
                  <p className="text-center text-secondary font-body-sm text-sm italic">
                    Trạng thái hiện tại: <span className="text-deep-navy font-bold not-italic">{getStageText(currentStage)}</span>
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Pipeline Processing (Dynamic State) */}
          {isProcessing && (
            <div className="mb-12 animate-fade-in">
              <h2 className="font-headline-md text-lg font-bold text-deep-navy mb-6 flex items-center gap-2">
                <span className="material-symbols-outlined text-vibrant-cyan">memory</span>
                Real-time Pipeline Status (Celery Tasks)
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                {/* Step 1 */}
                <div className={`bg-surface p-4 rounded-lg flex flex-col items-center text-center border ${
                  currentStep > 1 ? 'border-status-success/30' : 'border-vibrant-cyan active-step-pulse'
                }`}>
                  <span className={`material-symbols-outlined mb-2 ${
                    currentStep > 1 ? 'text-status-success' : 'text-vibrant-cyan'
                  }`} style={currentStep > 1 ? { fontVariationSettings: "'FILL' 1" } : {}}>
                    {currentStep > 1 ? 'check_circle' : 'downloading'}
                  </span>
                  <p className="font-label-sm text-xs font-bold text-deep-navy mb-1">Tải & Kiểm tra video</p>
                  <span className={`text-[10px] font-mono-data font-bold ${
                    currentStep > 1 ? 'text-status-success' : 'text-vibrant-cyan animate-pulse'
                  }`}>
                    {currentStep > 1 ? 'COMPLETE' : 'PROCESSING...'}
                  </span>
                </div>

                {/* Step 2 */}
                <div className={`bg-surface p-4 rounded-lg flex flex-col items-center text-center border ${
                  currentStep > 2 ? 'border-status-success/30' : currentStep === 2 ? 'border-vibrant-cyan active-step-pulse' : 'border-outline-variant/30 opacity-50'
                }`}>
                  <span className={`material-symbols-outlined mb-2 ${
                    currentStep > 2 ? 'text-status-success' : currentStep === 2 ? 'text-vibrant-cyan' : 'text-secondary'
                  }`} style={currentStep > 2 ? { fontVariationSettings: "'FILL' 1" } : {}}>
                    {currentStep > 2 ? 'check_circle' : 'mic'}
                  </span>
                  <p className="font-label-sm text-xs font-bold text-deep-navy mb-1">Xử lý âm thanh (FFmpeg)</p>
                  <span className={`text-[10px] font-mono-data font-bold ${
                    currentStep > 2 ? 'text-status-success' : currentStep === 2 ? 'text-vibrant-cyan animate-pulse' : 'text-secondary'
                  }`}>
                    {currentStep > 2 ? 'COMPLETE' : currentStep === 2 ? 'PROCESSING...' : 'PENDING'}
                  </span>
                </div>

                {/* Step 3 */}
                <div className={`bg-surface p-4 rounded-lg flex flex-col items-center text-center border ${
                  currentStep > 3 ? 'border-status-success/30' : currentStep === 3 ? 'border-vibrant-cyan active-step-pulse' : 'border-outline-variant/30 opacity-50'
                }`}>
                  <span className={`material-symbols-outlined mb-2 ${
                    currentStep > 3 ? 'text-status-success' : currentStep === 3 ? 'text-vibrant-cyan' : 'text-secondary'
                  }`} style={currentStep > 3 ? { fontVariationSettings: "'FILL' 1" } : {}}>
                    {currentStep > 3 ? 'check_circle' : 'transcribe'}
                  </span>
                  <p className="font-label-sm text-xs font-bold text-deep-navy mb-1">Nhận dạng chữ (WhisperX)</p>
                  <span className={`text-[10px] font-mono-data font-bold ${
                    currentStep > 3 ? 'text-status-success' : currentStep === 3 ? 'text-vibrant-cyan animate-pulse' : 'text-secondary'
                  }`}>
                    {currentStep > 3 ? 'COMPLETE' : currentStep === 3 ? 'PROCESSING...' : 'PENDING'}
                  </span>
                </div>

                {/* Step 4 */}
                <div className={`bg-surface p-4 rounded-lg flex flex-col items-center text-center border ${
                  currentStep > 4 ? 'border-status-success/30' : currentStep === 4 ? 'border-vibrant-cyan active-step-pulse' : 'border-outline-variant/30 opacity-50'
                }`}>
                  <span className={`material-symbols-outlined mb-2 ${
                    currentStep > 4 ? 'text-status-success' : currentStep === 4 ? 'text-vibrant-cyan' : 'text-secondary'
                  }`} style={currentStep > 4 ? { fontVariationSettings: "'FILL' 1" } : {}}>
                    {currentStep > 4 ? 'check_circle' : 'image_search'}
                  </span>
                  <p className="font-label-sm text-xs font-bold text-deep-navy mb-1">Keyframes & OCR (CLIP)</p>
                  <span className={`text-[10px] font-mono-data font-bold ${
                    currentStep > 4 ? 'text-status-success' : currentStep === 4 ? 'text-vibrant-cyan animate-pulse' : 'text-secondary'
                  }`}>
                    {currentStep > 4 ? 'COMPLETE' : currentStep === 4 ? 'PROCESSING...' : 'PENDING'}
                  </span>
                </div>

                {/* Step 5 */}
                <div className={`bg-surface p-4 rounded-lg flex flex-col items-center text-center border ${
                  currentStep === 5 ? 'border-vibrant-cyan active-step-pulse' : 'border-outline-variant/30 opacity-50'
                }`}>
                  <span className={`material-symbols-outlined mb-2 ${
                    currentStep === 5 ? 'text-vibrant-cyan' : 'text-secondary'
                  }`}>
                    {currentStep === 5 ? 'auto_awesome' : 'auto_awesome'}
                  </span>
                  <p className="font-label-sm text-xs font-bold text-deep-navy mb-1">LLM Fusion tóm tắt</p>
                  <span className={`text-[10px] font-mono-data font-bold ${
                    currentStep === 5 ? 'text-vibrant-cyan animate-pulse' : 'text-secondary'
                  }`}>
                    {currentStep === 5 ? 'PROCESSING...' : 'PENDING'}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Recent Uploads Table */}
          <section className="animate-fade-in">
            <div className="flex items-center justify-between mb-6">
              <h2 className="font-headline-md text-xl font-bold text-deep-navy">Các video phân tích gần đây</h2>
              <Link to="/history" className="text-vibrant-cyan font-label-md text-label-md hover:underline flex items-center gap-1">
                Xem tất cả <span className="material-symbols-outlined text-sm">arrow_forward</span>
              </Link>
            </div>
            <div className="bg-surface border border-outline-variant rounded-lg overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[600px]">
                  <thead>
                    <tr className="bg-surface-container-low border-b border-outline-variant">
                      <th className="px-6 py-4 font-label-sm text-xs font-bold text-secondary uppercase tracking-wider">Video Source</th>
                      <th className="px-6 py-4 font-label-sm text-xs font-bold text-secondary uppercase tracking-wider">Timestamp</th>
                      <th className="px-6 py-4 font-label-sm text-xs font-bold text-secondary uppercase tracking-wider">Trạng thái</th>
                      <th className="px-6 py-4 font-label-sm text-xs font-bold text-secondary uppercase tracking-wider text-right">Hành động</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-outline-variant/30">
                    {recentVideos.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="px-6 py-8 text-center text-secondary font-body-sm text-sm">
                          Chưa có video nào được phân tích.
                        </td>
                      </tr>
                    ) : (
                      recentVideos.map((video) => {
                        const statusUpper = (video.status || "").toUpperCase();
                        const isDone = statusUpper === 'DONE' || statusUpper === 'COMPLETED' || statusUpper === 'SUCCESS';
                        const isFailed = statusUpper === 'FAILED';
                        return (
                          <tr key={video.videoId} className="hover:bg-surface-container-lowest transition-colors">
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="w-16 h-10 rounded bg-slate-100 overflow-hidden border border-outline-variant shrink-0 flex items-center justify-center">
                                  <span className="material-symbols-outlined text-secondary">video_file</span>
                                </div>
                                <div>
                                  <p className="font-label-md text-sm font-semibold text-deep-navy truncate max-w-xs" title={video.title || video.originalUrl || (video.r2Url ? video.r2Url : (video.filePath && !video.filePath.includes('/stream') ? video.filePath : 'Video bài giảng'))}>
                                    {video.title || (video.originalUrl ? video.originalUrl : (video.r2Url ? (video.r2Url.split('?')[0].split('/').pop() || 'video.mp4') : (video.filePath && !video.filePath.includes('/stream') ? (video.filePath.split('?')[0].split('/').pop() || 'video.mp4') : 'Video bài giảng')))}
                                  </p>
                                  <p className="text-xs text-secondary">Duration: {video.duration ? `${Math.round(video.duration)}s` : 'N/A'}</p>
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4 font-mono-data text-xs text-secondary">
                              {new Date(video.uploadedAt).toLocaleString()}
                            </td>
                            <td className="px-6 py-4">
                              <span className={`px-2 py-1 text-[10px] font-extrabold rounded-full border tracking-wider uppercase ${
                                isDone ? 'bg-green-50 text-status-success border-status-success/20' :
                                isFailed ? 'bg-red-50 text-red-700 border-red-200' :
                                'bg-blue-50 text-vibrant-cyan border-vibrant-cyan/20 animate-pulse'
                              }`} title={video.stage ? `Trạng thái: ${getStageText(video.stage)}` : undefined}>
                                {isDone ? 'Hoàn tất' :
                                 isFailed ? 'Thất bại' :
                                 `Đang xử lý ${video.progress !== undefined && video.progress !== null ? `(${video.progress}%)` : ''}`}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-right">
                              {isDone ? (
                                <Link 
                                  to={`/results?videoId=${video.videoId}`} 
                                  className="text-deep-navy hover:text-vibrant-cyan font-bold text-xs inline-flex items-center gap-1"
                                >
                                  Xem kết quả
                                  <span className="material-symbols-outlined text-sm">arrow_forward</span>
                                </Link>
                              ) : (
                                <span className="text-secondary text-xs italic">
                                  {video.stage ? getStageText(video.stage) : 'Đang xử lý...'}
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};
