/**
 * ResultsPage — Trang kết quả tóm tắt.
 * Trang: UIs/3_results.html → frontend/src/pages/ResultsPage.jsx
 */
import { Link } from 'react-router-dom';
import VideoPlayer from '../components/VideoPlayer';
import SummaryPanel from '../components/SummaryPanel';

export default function ResultsPage() {
  return (
    <div className="grid grid-cols-2 gap-5 p-5 pt-20 h-screen box-border bg-gray-50">
      {/* Top Navigation */}
      <div className="fixed top-5 right-5 z-10 flex gap-4">
        <Link to="/" className="bg-white border border-gray-200 text-gray-900 px-5 py-2.5 rounded-lg text-sm no-underline font-medium hover:bg-gray-50 transition-all shadow-sm">
          <i className="fa-solid fa-house mr-2"></i>Trang chủ
        </Link>
        <Link to="/history" className="bg-white border border-gray-200 text-gray-900 px-5 py-2.5 rounded-lg text-sm no-underline font-medium hover:bg-gray-50 transition-all shadow-sm">
          <i className="fa-solid fa-clock-rotate-left mr-2"></i>Lịch sử
        </Link>
        <button className="bg-white border border-gray-200 text-gray-900 px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-gray-50 transition-all shadow-sm cursor-pointer">
          <i className="fa-solid fa-file-pdf mr-2"></i>Xuất PDF
        </button>
        <button className="bg-white border border-gray-200 text-gray-900 px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-gray-50 transition-all shadow-sm cursor-pointer">
          <i className="fa-solid fa-file-lines mr-2"></i>Xuất TXT
        </button>
        <Link to="/qa" className="bg-primary text-white px-5 py-2.5 rounded-lg text-sm no-underline font-medium shadow-lg shadow-primary/20 hover:bg-primary-hover hover:-translate-y-0.5 transition-all">
          <i className="fa-solid fa-comments mr-2"></i>Hỏi Đáp RAG
        </Link>
      </div>

      <VideoPlayer />
      <SummaryPanel />
    </div>
  );
}
