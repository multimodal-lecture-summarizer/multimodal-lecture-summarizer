/**
 * VideoPlayer — Video player với transcript word-level timestamps.
 * Chuyển đổi từ: UIs/3_results.html (left column)
 */
export default function VideoPlayer({ videoUrl, transcriptLines = [] }) {
  const defaultLines = [
    { ts: '00:00', text: 'Chào mừng các bạn đến với bài giảng về Trí tuệ Nhân tạo.', active: false },
    { ts: '00:15', text: 'Hôm nay chúng ta sẽ đi sâu vào cấu trúc của mạng nơ-ron nhân tạo.', active: false },
    { ts: '05:12', text: 'Khái niệm quan trọng nhất ở đây là Backpropagation (Lan truyền ngược). Nó giúp điều chỉnh trọng số...', active: true },
    { ts: '05:40', text: 'Hãy nhìn vào đạo hàm của hàm mất mát. Chúng ta tính gradient để tìm điểm cực tiểu.', active: false },
    { ts: '15:20', text: 'Tiếp theo, hãy chuyển sang các kiến trúc hiện đại hơn như mô hình Transformer.', active: false },
  ];

  const lines = transcriptLines.length > 0 ? transcriptLines : defaultLines;

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 flex flex-col h-full shadow-sm overflow-y-auto">
      <h2 className="font-outfit text-xl font-semibold flex items-center gap-3 pb-4 border-b border-gray-200 mb-5">
        <i className="fa-brands fa-youtube text-red-500"></i>
        Bài Giảng Gốc &amp; Transcript (WhisperX)
      </h2>

      {/* Video Container */}
      <div className="w-full aspect-video bg-black rounded-lg mb-5 flex justify-center items-center text-white relative overflow-hidden">
        <i className="fa-solid fa-play text-6xl z-10"></i>
        <div className="absolute bottom-4 left-4 z-10 text-white flex gap-4 items-center w-[calc(100%-2rem)]">
          <i className="fa-solid fa-pause"></i>
          <span className="font-mono text-sm">05:12 / 45:10</span>
          <div className="flex-1 h-1 bg-white/30 rounded">
            <div className="w-[30%] h-full bg-primary rounded"></div>
          </div>
          <i className="fa-solid fa-volume-high"></i>
          <i className="fa-solid fa-expand"></i>
        </div>
      </div>

      {/* Transcript */}
      <div className="flex-1 pt-3">
        <h3 className="text-sm text-gray-500 mb-4 flex justify-between">
          <span>Word-level Timestamps</span>
          <i className="fa-solid fa-magnifying-glass cursor-pointer hover:text-primary"></i>
        </h3>
        <div className="space-y-2">
          {lines.map((line, i) => (
            <div
              key={i}
              className={`flex gap-4 p-3 rounded-lg cursor-pointer border transition-all
                ${line.active
                  ? 'bg-indigo-50 border-indigo-200'
                  : 'bg-white border-transparent hover:bg-indigo-50 hover:border-indigo-200'}`}
            >
              <span className="text-primary font-mono text-sm min-w-[3rem]">{line.ts}</span>
              <span className="text-sm">{line.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
