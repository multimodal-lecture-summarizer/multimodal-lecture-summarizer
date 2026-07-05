import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { api } from '../services/api';

export const UploadPage: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
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

  const [recentVideos, setRecentVideos] = useState<any[]>([]);

  useEffect(() => {
    api.getVideos(undefined, 5, 0)
      .then((res) => {
        if (res.success && res.data) {
          setRecentVideos(res.data);
        }
      })
      .catch((err) => console.error("Failed to load recent videos:", err));
  }, []);

  const startPolling = (videoId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    
    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await api.getJobStatus(videoId);
        if (res.success && res.data && res.data.length > 0) {
          const job = res.data.find((j: any) => j.jobType === 'SUMMARIZE' || j.job_type === 'SUMMARIZE');
          if (job) {
            if (job.status === 'COMPLETED' || job.status === 'SUCCESS') {
              clearInterval(pollIntervalRef.current);
              setCurrentStep(5);
              setTimeout(() => {
                navigate(`/results?videoId=${videoId}`);
              }, 1000);
            } else if (job.status === 'FAILED') {
              clearInterval(pollIntervalRef.current);
              setIsProcessing(false);
              setErrorMsg(job.errorLog || 'Đã xảy ra lỗi trong quá trình xử lý video.');
            } else {
              // Update step representation based on current state
              setCurrentStep((prev) => (prev < 4 ? prev + 1 : prev));
            }
          }
        }
      } catch (err) {
        console.error("Error polling job status:", err);
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
        startPolling(videoId);
      } else {
        throw new Error(res.message || "Tải lên video thất bại.");
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Không thể gửi video lên máy chủ.");
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
                <h3 className="font-headline-md text-xl font-bold text-deep-navy mb-1">Kéo thả file video bài giảng vào đây</h3>
                <p className="text-secondary font-body-sm mb-6 text-sm">Hỗ trợ MP4, AVI, MKV (Tối đa 2GB)</p>
                <button 
                  onClick={handleBrowseClick}
                  className="px-6 py-2 bg-deep-navy text-white font-label-md text-label-md rounded hover:bg-primary transition-colors font-semibold"
                >
                  Browse Files
                </button>
              </div>

              {/* Progress Overlay (Displayed when isProcessing is true) */}
              {isProcessing && (
                <div className="absolute inset-0 bg-white/95 backdrop-blur-sm flex items-center justify-center p-8 transition-opacity duration-300">
                  <div className="w-full max-w-md">
                    <div className="flex justify-between mb-2">
                      <span className="font-label-md text-label-md text-deep-navy font-bold">
                        Đang phân tích video bài giảng...
                      </span>
                      <span className="font-mono-data text-mono-data text-vibrant-cyan font-bold">
                        {currentStep === 1 ? '15%' : currentStep === 2 ? '40%' : currentStep === 3 ? '65%' : currentStep === 4 ? '85%' : '100%'}
                      </span>
                    </div>
                    <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                      <div 
                        className="bg-vibrant-cyan h-full transition-all duration-500" 
                        style={{ width: `${currentStep * 20}%` }}
                      ></div>
                    </div>
                    <p className="mt-4 text-center text-secondary font-body-sm text-xs italic animate-pulse">
                      Đang xử lý phân tách cảnh và dịch giọng nói bằng AI...
                    </p>
                  </div>
                </div>
              )}
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
                    <input 
                      className="w-full px-4 py-3 bg-background border border-outline-variant rounded-lg focus:ring-1 focus:ring-vibrant-cyan focus:border-vibrant-cyan outline-none transition-all font-mono-data text-sm" 
                      placeholder="https://youtube.com/watch?v=..." 
                      type="url"
                      value={youtubeUrl}
                      onChange={(e) => setYoutubeUrl(e.target.value)}
                    />
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
                      <th className="px-6 py-4 font-label-sm text-xs font-bold text-secondary uppercase tracking-wider">Độ chính xác ASR/OCR</th>
                      <th className="px-6 py-4 font-label-sm text-xs font-bold text-secondary uppercase tracking-wider">Trạng thái</th>
                      <th className="px-6 py-4 font-label-sm text-xs font-bold text-secondary uppercase tracking-wider text-right">Hành động</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-outline-variant/30">
                    {recentVideos.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="px-6 py-8 text-center text-secondary font-body-sm text-sm">
                          Chưa có video nào được phân tích.
                        </td>
                      </tr>
                    ) : (
                      recentVideos.map((video) => (
                        <tr key={video.videoId} className="hover:bg-surface-container-lowest transition-colors">
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-3">
                              <div className="w-16 h-10 rounded bg-slate-100 overflow-hidden border border-outline-variant shrink-0 flex items-center justify-center">
                                <span className="material-symbols-outlined text-secondary">video_file</span>
                              </div>
                              <div>
                                <p className="font-label-md text-sm font-semibold text-deep-navy truncate max-w-xs">
                                  {video.originalUrl ? video.originalUrl : (video.filePath ? video.filePath.split('/').pop() : 'video.mp4')}
                                </p>
                                <p className="text-xs text-secondary">Duration: {video.duration ? `${Math.round(video.duration)}s` : 'N/A'}</p>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 font-mono-data text-xs text-secondary">
                            {new Date(video.uploadedAt).toLocaleString()}
                          </td>
                          <td className="px-6 py-4">
                            <div className="flex items-center gap-2">
                              <span className="font-mono-data text-status-success font-bold text-sm">95%+</span>
                              <div className="w-16 bg-surface-container-high h-1.5 rounded-full overflow-hidden">
                                <div className="bg-status-success h-full" style={{ width: '95%' }}></div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <span className={`px-2 py-1 text-[10px] font-extrabold rounded-full border tracking-wider uppercase ${
                              video.status === 'DONE' ? 'bg-green-50 text-status-success border-status-success/20' :
                              video.status === 'FAILED' ? 'bg-red-50 text-red-700 border-red-200' :
                              'bg-blue-50 text-vibrant-cyan border-vibrant-cyan/20 animate-pulse'
                            }`}>
                              {video.status}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-right">
                            {video.status === 'DONE' ? (
                              <Link 
                                to={`/results?videoId=${video.videoId}`} 
                                className="text-deep-navy hover:text-vibrant-cyan font-bold text-xs inline-flex items-center gap-1"
                              >
                                Xem kết quả
                                <span className="material-symbols-outlined text-sm">arrow_forward</span>
                              </Link>
                            ) : (
                              <span className="text-secondary text-xs">Đang xử lý</span>
                            )}
                          </td>
                        </tr>
                      ))
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
