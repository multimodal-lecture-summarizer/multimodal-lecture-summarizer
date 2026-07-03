/**
 * HistoryPage — Trang lịch sử video đã xử lý.
 * Trang: UIs/6_history.html → frontend/src/pages/HistoryPage.jsx
 */
import { Link } from 'react-router-dom';
import HistoryCard from '../components/HistoryCard';

const mockVideos = [
  { title: 'Bài giảng MIT 6.S191 - Introduction to Deep Learning', duration: '45:10', date: 'Hôm nay, 10:30', status: 'done', thumbnailIcon: 'fa-solid fa-image' },
  { title: 'TED Talk - Tương lai của Trí tuệ Nhân tạo', duration: '12:45', date: 'Hôm qua, 15:20', status: 'processing', statusText: 'Đang trích xuất Audio...', thumbnailIcon: 'fa-brands fa-youtube' },
  { title: 'CS224N - NLP with Deep Learning', duration: '55:30', date: '25/06/2026', status: 'done', thumbnailIcon: 'fa-solid fa-video' },
];

export default function HistoryPage() {
  return (
    <div className="bg-gray-50 min-h-screen pt-20">
      {/* Header */}
      <header className="flex justify-between items-center px-[5%] py-4 bg-white/95 fixed w-full top-0 z-50 border-b border-gray-200 shadow-sm">
        <Link to="/" className="text-xl font-bold text-primary no-underline flex items-center gap-3">
          <i className="fa-solid fa-brain"></i> AI.Summarizer
        </Link>
        <div className="flex gap-4">
          <Link to="/admin" className="text-gray-900 text-sm no-underline hover:text-primary transition-colors flex items-center gap-2">
            <i className="fa-solid fa-shield"></i> Quản trị
          </Link>
          <Link to="/upload" className="bg-primary text-white px-4 py-2 rounded-lg text-sm no-underline font-medium hover:bg-primary-hover transition-all">
            <i className="fa-solid fa-plus mr-2"></i>Upload Video Mới
          </Link>
          <Link to="/auth" className="bg-white border border-gray-200 w-10 h-10 rounded-full flex justify-center items-center text-sm hover:bg-gray-50 no-underline text-gray-900" title="Đăng xuất">
            <i className="fa-solid fa-right-from-bracket"></i>
          </Link>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-5">
        <h1 className="font-outfit text-3xl font-semibold mb-8">Lịch Sử Video Của Bạn</h1>
        {mockVideos.map((video, i) => (
          <HistoryCard key={i} video={video} />
        ))}
      </div>
    </div>
  );
}
