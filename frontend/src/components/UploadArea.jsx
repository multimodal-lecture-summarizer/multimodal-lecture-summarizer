/**
 * UploadArea — Drag & drop upload zone + YouTube URL input.
 * Chuyển đổi từ: UIs/2_upload.html
 */
import { useState } from 'react';

export default function UploadArea({ onUploadStart }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onUploadStart?.(files[0]);
    }
  };

  return (
    <div className="w-full max-w-3xl bg-white border border-gray-200 rounded-2xl shadow-lg p-10 mx-auto">
      <h1 className="font-outfit text-2xl font-semibold text-center mb-2">Phân Tích Video Mới</h1>
      <p className="text-center text-gray-500 mb-8">
        Tải lên video bài giảng hoặc dán đường dẫn YouTube để pipeline AI (Audio/Visual/Fusion) bắt đầu xử lý
      </p>

      {/* Drag & Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => onUploadStart?.()}
        className={`border-2 border-dashed rounded-xl p-16 text-center transition-all cursor-pointer
          ${isDragging ? 'border-primary bg-indigo-50' : 'border-gray-200 bg-gray-50 hover:border-primary hover:bg-indigo-50'}`}
      >
        <i className="fa-solid fa-cloud-arrow-up text-5xl text-primary mb-4 block"></i>
        <h3 className="font-outfit text-lg font-semibold mb-2">Kéo thả file video vào đây</h3>
        <p className="text-gray-500 text-sm mb-5">
          Hỗ trợ MP4, AVI, MKV (Tối đa 60 phút, kiểm tra bởi Module duyệt hợp lệ)
        </p>
        <button className="bg-primary text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-hover transition-all">
          Chọn File
        </button>
      </div>

      {/* Divider */}
      <div className="flex items-center my-8 text-gray-500 text-sm">
        <span className="flex-1 h-px bg-gray-200"></span>
        <span className="px-4">HOẶC URL TỪ YOUTUBE</span>
        <span className="flex-1 h-px bg-gray-200"></span>
      </div>

      {/* YouTube URL Input */}
      <div className="flex gap-3">
        <input
          type="text"
          placeholder="https://www.youtube.com/watch?v=..."
          className="flex-1 px-4 py-3 rounded-lg border border-gray-200 bg-white outline-none focus:border-primary transition-colors"
        />
        <button
          onClick={() => onUploadStart?.()}
          className="bg-primary text-white px-6 py-3 rounded-lg font-semibold hover:bg-primary-hover transition-all whitespace-nowrap"
        >
          Phân Tích Ngay
        </button>
      </div>
    </div>
  );
}
