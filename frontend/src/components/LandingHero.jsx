/**
 * LandingHero — Hero section component cho trang chủ.
 * Chuyển đổi từ: UIs/1_landing.html (section.hero)
 */
import { Link } from 'react-router-dom';

export default function LandingHero() {
  return (
    <section className="min-h-screen flex flex-col justify-center items-center text-center px-5 pt-24 bg-surface">
      <h1 className="font-outfit text-6xl font-extrabold leading-tight mb-5 max-w-3xl animate-fade-in">
        Tiết Kiệm Hàng Giờ Xem Video Với Trí Tuệ Nhân Tạo
      </h1>
      <p className="text-xl text-gray-500 max-w-xl mb-10 animate-fade-in" style={{ animationDelay: '0.2s' }}>
        Hệ thống trích xuất nội dung, tóm tắt văn bản, phân đoạn chapter và hỗ trợ hỏi đáp trực tiếp (RAG)
        với bất kỳ video học thuật nào bằng các mô hình AI tiên tiến nhất.
      </p>
      <div className="flex gap-5 animate-fade-in" style={{ animationDelay: '0.4s' }}>
        <Link to="/upload" className="bg-primary text-white px-6 py-3 rounded-lg font-semibold shadow-lg shadow-primary/20 hover:bg-primary-hover hover:-translate-y-0.5 transition-all">
          <i className="fa-solid fa-upload mr-2"></i> Xử lý Video Ngay
        </Link>
        <Link to="/results" className="bg-white border border-gray-200 text-gray-900 px-6 py-3 rounded-lg font-semibold hover:bg-gray-50 transition-all">
          <i className="fa-brands fa-youtube mr-2"></i> Xem Demo Kết Quả
        </Link>
      </div>
    </section>
  );
}
