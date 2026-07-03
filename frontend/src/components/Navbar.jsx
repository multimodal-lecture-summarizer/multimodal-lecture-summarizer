/**
 * Navbar — Header navigation chung cho toàn bộ app.
 * Chuyển đổi từ: UIs/1_landing.html (header)
 */
import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <header className="flex justify-between items-center px-[5%] py-5 bg-white/95 backdrop-blur-sm fixed w-full top-0 z-50 border-b border-gray-200">
      <Link to="/" className="text-2xl font-extrabold text-primary no-underline font-outfit">
        <i className="fa-solid fa-brain mr-2"></i>AI.Summarizer
      </Link>
      <nav className="flex gap-8">
        <Link to="/" className="text-gray-500 no-underline font-medium hover:text-primary transition-colors">Trang chủ</Link>
        <Link to="/history" className="text-gray-500 no-underline font-medium hover:text-primary transition-colors">Lịch sử video</Link>
        <Link to="/admin" className="text-gray-500 no-underline font-medium hover:text-primary transition-colors">Quản trị</Link>
        <a href="#" className="text-gray-500 no-underline font-medium hover:text-primary transition-colors">Tài liệu</a>
      </nav>
      <div className="flex gap-4">
        <Link to="/auth" className="bg-white border border-gray-200 text-gray-900 px-5 py-2.5 rounded-lg font-semibold hover:bg-gray-50 transition-all">
          Đăng nhập
        </Link>
        <Link to="/upload" className="bg-primary text-white px-5 py-2.5 rounded-lg font-semibold shadow-lg shadow-primary/20 hover:bg-primary-hover hover:-translate-y-0.5 transition-all">
          Bắt đầu ngay
        </Link>
      </div>
    </header>
  );
}
