/**
 * ProcessingSteps — Hiển thị tiến trình xử lý AI pipeline.
 * Chuyển đổi từ: UIs/2_upload.html (processing UI)
 */
export default function ProcessingSteps({ steps = [], progress = 0 }) {
  const defaultSteps = [
    { label: '1. Tải và kiểm tra tính hợp lệ video', status: 'done' },
    { label: '2. Tiền xử lý âm thanh với FFmpeg', status: 'done' },
    { label: '3. Trích xuất văn bản (WhisperX) & word-level timestamps...', status: 'active' },
    { label: '4. Phân tích hình ảnh (PySceneDetect + CLIP + BLIP-2)', status: 'pending' },
    { label: '5. Tổng hợp tóm tắt và phân chương (Fusion LLM)', status: 'pending' },
  ];

  const displaySteps = steps.length > 0 ? steps : defaultSteps;

  const getIcon = (status) => {
    switch (status) {
      case 'done': return <i className="fa-solid fa-check text-emerald-500"></i>;
      case 'active': return <i className="fa-solid fa-spinner fa-spin text-primary"></i>;
      default: return <i className="fa-regular fa-circle text-gray-400"></i>;
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto bg-white border border-gray-200 rounded-2xl shadow-lg p-10">
      <h3 className="font-outfit text-lg font-semibold mb-6">
        Tiến trình xử lý AI bất đồng bộ (Celery + FastAPI)
      </h3>

      <div className="space-y-4">
        {displaySteps.map((step, i) => (
          <div
            key={i}
            className={`flex items-center gap-4 text-base
              ${step.status === 'active' ? 'text-primary font-medium' : ''}
              ${step.status === 'done' ? 'text-emerald-600' : ''}
              ${step.status === 'pending' ? 'text-gray-400' : ''}`}
          >
            <span className="w-6 text-center text-lg">{getIcon(step.status)}</span>
            <span>{step.label}</span>
          </div>
        ))}
      </div>

      {/* Progress Bar */}
      <div className="h-2 bg-gray-200 rounded mt-8 overflow-hidden">
        <div
          className="h-full bg-primary rounded transition-all duration-500"
          style={{ width: `${progress || 45}%`, animation: 'pulse 2s infinite' }}
        ></div>
      </div>

      <p className="text-center mt-5 text-gray-500 text-sm">
        <i className="fa-solid fa-circle-info mr-1"></i>
        Quá trình có thể mất từ 1-3 phút. Websocket đang duy trì kết nối real-time.
      </p>
    </div>
  );
}
