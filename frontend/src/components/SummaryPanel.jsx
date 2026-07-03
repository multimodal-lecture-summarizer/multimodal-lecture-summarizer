/**
 * SummaryPanel — Kết quả tổng hợp: tóm tắt, chapters, keyframes.
 * Chuyển đổi từ: UIs/3_results.html (right column)
 */
export default function SummaryPanel({ summary, chapters = [], keyframes = [] }) {
  const defaultSummary = `Bài giảng cung cấp cái nhìn toàn diện về nền tảng của Deep Learning. Mở đầu bằng việc định nghĩa mạng nơ-ron nhân tạo, giảng viên nhấn mạnh vào cơ chế hoạt động của thuật toán <strong>Backpropagation</strong> trong việc tối ưu hóa trọng số thông qua Gradient Descent. Nửa sau của video mở rộng sang kiến trúc <strong>Transformer</strong> và cơ chế <strong>Self-Attention</strong>, giải thích lý do tại sao nó thay thế RNN trong các bài toán NLP hiện đại.`;

  const defaultChapters = [
    { title: '1. Giới thiệu Deep Learning', ts: '00:00' },
    { title: '2. Thuật toán Backpropagation & Gradient Descent', ts: '05:12' },
    { title: '3. Hạn chế của mạng RNN', ts: '12:30' },
    { title: '4. Kiến trúc Transformer & Self-Attention', ts: '15:20' },
  ];

  const defaultKeyframes = [
    { label: '[Hình ảnh: Sơ đồ]', ts: '02:15' },
    { label: '[Hình ảnh: Đồ thị Loss]', ts: '05:40' },
    { label: '[Hình ảnh: Transformer]', ts: '15:20' },
  ];

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 flex flex-col h-full shadow-sm overflow-y-auto">
      <h2 className="font-outfit text-xl font-semibold flex items-center gap-3 pb-4 border-b border-gray-200 mb-5">
        <i className="fa-solid fa-brain text-primary"></i>
        Kết Quả Tổng Hợp (Multimodal Fusion)
      </h2>

      {/* Summary */}
      <h3 className="font-outfit text-base font-semibold mb-4">
        <i className="fa-solid fa-align-left mr-2"></i>Tóm Tắt Abstractive (GPT/Gemini)
      </h3>
      <div
        className="bg-indigo-50 border-l-4 border-primary px-5 py-4 rounded-r-lg mb-6 leading-7 text-sm"
        dangerouslySetInnerHTML={{ __html: summary || defaultSummary }}
      />

      {/* Chapters */}
      <h3 className="font-outfit text-base font-semibold mb-4">
        <i className="fa-solid fa-list mr-2"></i>Phân Chương Tự Động
      </h3>
      <div className="space-y-2.5 mb-6">
        {(chapters.length > 0 ? chapters : defaultChapters).map((ch, i) => (
          <div
            key={i}
            className="flex justify-between items-center p-4 bg-white border border-gray-200 rounded-lg cursor-pointer hover:border-primary hover:bg-indigo-50 transition-all"
          >
            <span className="font-medium text-sm">{ch.title}</span>
            <span className="text-primary font-mono text-sm">{ch.ts}</span>
          </div>
        ))}
      </div>

      {/* Keyframes */}
      <h3 className="font-outfit text-base font-semibold mb-4">
        <i className="fa-solid fa-images mr-2"></i>Keyframes (PySceneDetect + CLIP)
      </h3>
      <div className="grid grid-cols-3 gap-4">
        {(keyframes.length > 0 ? keyframes : defaultKeyframes).map((kf, i) => (
          <div
            key={i}
            className="rounded-lg overflow-hidden relative border border-gray-200 bg-gray-50 hover:scale-105 hover:border-primary transition-transform cursor-pointer"
          >
            <div className="w-full h-24 bg-gray-200 flex justify-center items-center text-gray-500 text-xs">
              {kf.label}
            </div>
            <div className="absolute bottom-1 right-1 bg-black/70 px-2 py-0.5 rounded text-xs text-white font-mono">
              {kf.ts}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
