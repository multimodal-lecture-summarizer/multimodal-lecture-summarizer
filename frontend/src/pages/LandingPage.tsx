import React from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';

export const LandingPage: React.FC = () => {
  return (
    <div className="landing-page">
      <section className="hero">
        <h1>Tiết Kiệm Hàng Giờ Xem Video Với Trí Tuệ Nhân Tạo</h1>
        <p>
          Hệ thống trích xuất nội dung, tóm tắt văn bản, phân đoạn chapter và hỗ trợ hỏi đáp trực tiếp (RAG) với bất kỳ video học thuật nào bằng các mô hình AI tiên tiến nhất.
        </p>
        <div className="hero-btns">
          <Link to="/upload" className="btn primary">
            <i className="fa-solid fa-upload"></i> Xử lý Video Ngay
          </Link>
          <Link to="/results" className="btn secondary">
            <i className="fa-brands fa-youtube"></i> Xem Demo Kết Quả
          </Link>
        </div>
      </section>

      <section className="features">
        <h2 className="section-title">Tính Năng Cốt Lõi</h2>
        <div className="grid">
          <div className="card">
            <i className="fa-solid fa-file-lines"></i>
            <h3>Tóm Tắt Tự Động (LLM)</h3>
            <p>Sử dụng GPT-4o / Gemini / Qwen2.5 để trích xuất ý chính, tạo bản tóm tắt ngắn gọn nhưng đầy đủ thông tin từ WhisperX transcript.</p>
          </div>
          <div className="card">
            <i className="fa-solid fa-images"></i>
            <h3>Trích Xuất Keyframe (CLIP)</h3>
            <p>Mô-đun Vision tự động chọn lọc các frame đại diện quan trọng nhất trong video bằng PySceneDetect và nhúng CLIP.</p>
          </div>
          <div className="card">
            <i className="fa-solid fa-comments"></i>
            <h3>Hỏi Đáp Tương Tác (RAG)</h3>
            <p>Lưu trữ nội dung qua ChromaDB. Trò chuyện và đặt câu hỏi, nhận câu trả lời chính xác kèm theo timestamp để xem lại đoạn liên quan.</p>
          </div>
        </div>
      </section>
    </div>
  );
};
