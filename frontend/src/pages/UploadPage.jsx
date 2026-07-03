/**
 * UploadPage — Trang upload video.
 * Trang: UIs/2_upload.html → frontend/src/pages/UploadPage.jsx
 */
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import UploadArea from '../components/UploadArea';
import ProcessingSteps from '../components/ProcessingSteps';

export default function UploadPage() {
  const [isProcessing, setIsProcessing] = useState(false);
  const navigate = useNavigate();

  const handleUploadStart = () => {
    setIsProcessing(true);
    // Auto redirect to results for demo
    setTimeout(() => navigate('/results'), 3000);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center pt-20">
      {/* Navigation */}
      <div className="fixed top-5 left-5 z-10 flex gap-3">
        <Link to="/" className="bg-white border border-gray-200 text-gray-900 px-4 py-2.5 rounded-lg text-sm no-underline hover:border-primary transition-all shadow-sm">
          <i className="fa-solid fa-house mr-2"></i>Trang chủ
        </Link>
        <Link to="/history" className="bg-white border border-gray-200 text-gray-900 px-4 py-2.5 rounded-lg text-sm no-underline hover:border-primary transition-all shadow-sm">
          <i className="fa-solid fa-clock-rotate-left mr-2"></i>Lịch sử
        </Link>
      </div>

      {isProcessing ? (
        <ProcessingSteps />
      ) : (
        <UploadArea onUploadStart={handleUploadStart} />
      )}
    </div>
  );
}
