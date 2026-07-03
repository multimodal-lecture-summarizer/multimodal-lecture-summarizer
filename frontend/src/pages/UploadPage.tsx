import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './UploadPage.css';

export const UploadPage: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const navigate = useNavigate();

  const handleStartProcessing = () => {
    setIsProcessing(true);
    setCurrentStep(1);
  };

  useEffect(() => {
    if (!isProcessing) return;

    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= 5) {
          clearInterval(timer);
          // Redirect to results page when done
          setTimeout(() => {
            navigate('/results');
          }, 1200);
          return 5;
        }
        return prev + 1;
      });
    }, 1200);

    return () => clearInterval(timer);
  }, [isProcessing, navigate]);

  return (
    <div className="upload-page">
      <div className="container">
        {!isProcessing ? (
          <>
            <h1>Phân Tích Video Mới</h1>
            <p className="desc">
              Tải lên video bài giảng hoặc dán đường dẫn YouTube để pipeline AI (Audio/Visual/Fusion) bắt đầu xử lý
            </p>

            <div className="upload-area" onClick={handleStartProcessing}>
              <i className="fa-solid fa-cloud-arrow-up"></i>
              <h3>Kéo thả file video vào đây</h3>
              <p style={{ color: 'var(--text-muted)', marginTop: '10px', fontSize: '0.9rem' }}>
                Hỗ trợ MP4, AVI, MKV (Tối đa 60 phút, kiểm tra bởi Module duyệt hợp lệ)
              </p>
              <button className="btn primary" style={{ marginTop: '20px' }}>Chọn File</button>
            </div>

            <div className="divider"><span>HOẶC URL TỪ YOUTUBE</span></div>

            <div className="url-input-group">
              <input 
                type="text" 
                placeholder="https://www.youtube.com/watch?v=..." 
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
              />
              <button className="btn primary" onClick={handleStartProcessing}>Phân Tích Ngay</button>
            </div>
          </>
        ) : (
          <div className="processing" id="processing-ui" style={{ display: 'block' }}>
            <h3 style={{ marginBottom: '25px', fontFamily: 'Outfit' }}>
              Tiến trình xử lý AI bất đồng bộ (Celery + FastAPI)
            </h3>
            
            <div className={`step ${currentStep > 1 ? 'done' : 'active'}`}>
              <i className={currentStep > 1 ? 'fa-solid fa-check' : 'fa-solid fa-spinner fa-spin'}></i>
              <span>1. Tải và kiểm tra tính hợp lệ video (100%)</span>
            </div>
            
            <div className={`step ${currentStep > 2 ? 'done' : currentStep === 2 ? 'active' : ''}`}>
              <i className={currentStep > 2 ? 'fa-solid fa-check' : currentStep === 2 ? 'fa-solid fa-spinner fa-spin' : 'fa-regular fa-circle'}></i>
              <span>2. Tiền xử lý âm thanh với FFmpeg (Hoàn tất)</span>
            </div>
            
            <div className={`step ${currentStep > 3 ? 'done' : currentStep === 3 ? 'active' : ''}`}>
              <i className={currentStep > 3 ? 'fa-solid fa-check' : currentStep === 3 ? 'fa-solid fa-spinner fa-spin' : 'fa-regular fa-circle'}></i>
              <span>3. Trích xuất văn bản (WhisperX) & word-level timestamps...</span>
            </div>
            
            <div className={`step ${currentStep > 4 ? 'done' : currentStep === 4 ? 'active' : ''}`}>
              <i className={currentStep > 4 ? 'fa-solid fa-check' : currentStep === 4 ? 'fa-solid fa-spinner fa-spin' : 'fa-regular fa-circle'}></i>
              <span>4. Phân tích hình ảnh (PySceneDetect + CLIP + BLIP-2)</span>
            </div>
            
            <div className={`step ${currentStep === 5 ? 'active' : ''}`}>
              <i className={currentStep === 5 ? 'fa-solid fa-spinner fa-spin' : 'fa-regular fa-circle'}></i>
              <span>5. Tổng hợp tóm tắt và phân chương (Fusion LLM)</span>
            </div>
            
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${currentStep * 20}%` }}></div>
            </div>
            
            <p style={{ textAlign: 'center', marginTop: '20px', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              <i className="fa-solid fa-circle-info"></i> Quá trình có thể mất từ 1-3 phút. Websocket đang duy trì kết nối real-time.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
