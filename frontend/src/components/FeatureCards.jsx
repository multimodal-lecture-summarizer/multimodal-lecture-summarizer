/**
 * FeatureCards — Grid hiển thị tính năng cốt lõi.
 * Chuyển đổi từ: UIs/1_landing.html (section.features)
 */
export default function FeatureCards() {
  const features = [
    {
      icon: 'fa-solid fa-file-lines',
      title: 'Tóm Tắt Tự Động (LLM)',
      desc: 'Sử dụng GPT-4o / Gemini / Qwen2.5 để trích xuất ý chính, tạo bản tóm tắt ngắn gọn nhưng đầy đủ thông tin từ WhisperX transcript.',
    },
    {
      icon: 'fa-solid fa-images',
      title: 'Trích Xuất Keyframe (CLIP)',
      desc: 'Mô-đun Vision tự động chọn lọc các frame đại diện quan trọng nhất trong video bằng PySceneDetect và nhúng CLIP.',
    },
    {
      icon: 'fa-solid fa-comments',
      title: 'Hỏi Đáp Tương Tác (RAG)',
      desc: 'Lưu trữ nội dung qua ChromaDB. Trò chuyện và đặt câu hỏi, nhận câu trả lời chính xác kèm theo timestamp để xem lại đoạn liên quan.',
    },
  ];

  return (
    <section className="py-24 px-[5%] bg-white">
      <h2 className="font-outfit text-4xl font-semibold text-center mb-16">Tính Năng Cốt Lõi</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {features.map((f, i) => (
          <div
            key={i}
            className="bg-white border border-gray-200 rounded-2xl p-10 text-center transition-transform hover:-translate-y-2.5 hover:border-primary hover:shadow-lg cursor-default"
          >
            <i className={`${f.icon} text-5xl text-primary mb-5 block`}></i>
            <h3 className="font-outfit text-2xl font-semibold mb-4">{f.title}</h3>
            <p className="text-gray-500 leading-relaxed">{f.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
