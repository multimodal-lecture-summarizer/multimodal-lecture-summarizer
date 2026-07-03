/**
 * VideoSidebar — Sidebar với video context cho trang Q&A.
 * Chuyển đổi từ: UIs/4_qa.html (sidebar)
 */
import { Link } from 'react-router-dom';

export default function VideoSidebar() {
  const recentQuestions = [
    'Cơ chế của Transformer?',
    'Hạn chế của mạng RNN là gì?',
    'Giải thích Backpropagation',
  ];

  return (
    <div className="w-96 bg-gray-50 border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="px-5 py-6 border-b border-gray-200 flex items-center gap-4 bg-white">
        <Link to="/results" className="text-gray-500 hover:text-primary">
          <i className="fa-solid fa-arrow-left"></i>
        </Link>
        <h2 className="font-outfit text-lg font-semibold">Ngữ Cảnh Video (RAG)</h2>
      </div>

      <div className="p-5 flex-1 overflow-y-auto">
        {/* Mini Video */}
        <div className="w-full aspect-video bg-black rounded-lg mb-6 flex justify-center items-center relative">
          <i className="fa-solid fa-play text-5xl text-white/50"></i>
        </div>

        {/* Info Card */}
        <div className="bg-white border border-gray-200 p-4 rounded-lg mb-5 shadow-sm space-y-2">
          <p className="text-gray-500 text-sm flex justify-between">Tên file: <span className="text-gray-900 font-medium">MIT_DeepLearning_Lec1.mp4</span></p>
          <p className="text-gray-500 text-sm flex justify-between">Thời lượng: <span className="text-gray-900 font-medium">45:10</span></p>
          <p className="text-gray-500 text-sm flex justify-between">Vector Store: <span className="text-gray-900 font-medium">ChromaDB</span></p>
          <p className="text-gray-500 text-sm flex justify-between">LLM Backend: <span className="text-gray-900 font-medium">Qwen2.5-7B (Local)</span></p>
        </div>

        {/* Recent Questions */}
        <div>
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-4 font-medium">Câu hỏi gần đây</h3>
          {recentQuestions.map((q, i) => (
            <div
              key={i}
              className={`px-3 py-3 rounded-lg cursor-pointer transition-all mb-1 text-sm truncate
                ${i === 0 ? 'bg-indigo-50 text-primary' : 'text-gray-900 hover:bg-indigo-50 hover:text-primary'}`}
            >
              <i className="fa-regular fa-message mr-3"></i>{q}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
