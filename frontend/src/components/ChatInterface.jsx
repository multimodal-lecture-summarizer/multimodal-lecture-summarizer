/**
 * ChatInterface — Giao diện hỏi đáp RAG với video context.
 * Chuyển đổi từ: UIs/4_qa.html
 */
import { useState } from 'react';

export default function ChatInterface() {
  const [messages] = useState([
    {
      role: 'bot',
      text: 'Chào bạn! Hệ thống RAG đã được nạp toàn bộ Transcript và thông tin Keyframes (BLIP-2) của video. Bạn có muốn hỏi gì về bài giảng này không?',
    },
    {
      role: 'user',
      text: 'Giảng viên định nghĩa thế nào về kiến trúc Transformer? Tại sao nó tốt hơn RNN?',
    },
    {
      role: 'bot',
      text: `Theo bài giảng, kiến trúc Transformer là một mô hình học sâu chủ yếu dựa trên cơ chế <strong>Self-Attention</strong>. Giảng viên giải thích rằng khác với RNN (Recurrent Neural Networks) phải xử lý dữ liệu tuần tự từng từ một, Transformer có khả năng xem xét toàn bộ câu cùng một lúc.<br/><br/>
Điều này mang lại hai lợi ích chính:
<ul class="mt-2 pl-5 list-disc"><li>Khắc phục triệt để vấn đề mất mát thông tin đối với chuỗi văn bản dài.</li><li>Cho phép tính toán song song trên GPU, giúp tốc độ huấn luyện nhanh hơn rất nhiều.</li></ul>`,
      ref: '15:20',
    },
  ]);

  return (
    <div className="flex-1 flex flex-col bg-white">
      {/* Header */}
      <div className="px-8 py-5 border-b border-gray-200 flex justify-between items-center bg-white">
        <h3 className="font-outfit font-medium text-lg">
          <i className="fa-solid fa-robot text-primary mr-3"></i>Trợ lý Video AI
        </h3>
        <button className="bg-white text-gray-900 border border-gray-200 px-4 py-2 rounded-md text-sm hover:bg-gray-50 transition-all">
          <i className="fa-solid fa-trash-can mr-2"></i>Xóa lịch sử
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 p-8 overflow-y-auto flex flex-col gap-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-4 max-w-[85%] ${msg.role === 'user' ? 'self-end flex-row-reverse' : ''}`}>
            <div className={`w-11 h-11 rounded-full flex justify-center items-center text-lg shrink-0
              ${msg.role === 'bot' ? 'bg-primary text-white' : 'bg-gray-200 text-gray-500'}`}>
              <i className={msg.role === 'bot' ? 'fa-solid fa-brain' : 'fa-solid fa-user'}></i>
            </div>
            <div className={`px-6 py-4 rounded-2xl border leading-relaxed text-sm shadow-sm
              ${msg.role === 'user'
                ? 'bg-primary border-primary text-white rounded-tr-sm shadow-primary/20'
                : 'bg-white border-gray-200 text-gray-900 rounded-tl-sm'}`}>
              <div dangerouslySetInnerHTML={{ __html: msg.text }} />
              {msg.ref && (
                <div className="inline-flex items-center gap-2 bg-indigo-50 text-primary px-3 py-1.5 rounded-md font-mono text-sm mt-4 cursor-pointer border border-indigo-200 hover:bg-indigo-100 transition-all">
                  <i className="fa-solid fa-play"></i> Xem đoạn: {msg.ref}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="px-8 py-6 border-t border-gray-200 bg-white">
        <div className="flex bg-white border border-gray-200 rounded-xl p-2 focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/10 transition-all">
          <input
            type="text"
            placeholder="Nhập câu hỏi của bạn để tìm kiếm trong video (Enter để gửi)..."
            className="flex-1 bg-transparent border-none px-4 py-3 outline-none text-base"
          />
          <button className="bg-primary text-white w-14 rounded-lg hover:bg-primary-hover hover:scale-105 transition-all text-xl flex justify-center items-center">
            <i className="fa-solid fa-paper-plane"></i>
          </button>
        </div>
      </div>
    </div>
  );
}
