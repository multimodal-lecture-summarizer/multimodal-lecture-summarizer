import React, { useState } from 'react';

export const DocsPage: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'overview' | 'summary' | 'keyframe' | 'rag'>('overview');

  return (
    <div className="flex h-[calc(100vh-64px)] overflow-hidden bg-background text-on-surface">
      {/* Sidebar for documentation navigation */}
      <aside className="w-64 border-r border-outline-variant bg-surface-container-low flex flex-col shrink-0 p-4 gap-2">
        <div className="px-2 py-4 border-b border-outline-variant/30 mb-2 flex items-center gap-2">
          <span className="material-symbols-outlined text-vibrant-cyan">menu_book</span>
          <span className="font-label-md text-label-md font-bold text-deep-navy">Hướng Dẫn Sử Dụng</span>
        </div>
        <nav className="flex flex-col gap-1 flex-1">
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left text-xs font-semibold ${
              activeSection === 'overview' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
            }`}
            onClick={() => setActiveSection('overview')}
          >
            <span className="material-symbols-outlined text-sm">home_storage</span>
            Tổng quan hệ thống
          </button>
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left text-xs font-semibold ${
              activeSection === 'summary' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
            }`}
            onClick={() => setActiveSection('summary')}
          >
            <span className="material-symbols-outlined text-sm">article</span>
            Tóm tắt &amp; Phân chương
          </button>
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left text-xs font-semibold ${
              activeSection === 'keyframe' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
            }`}
            onClick={() => setActiveSection('keyframe')}
          >
            <span className="material-symbols-outlined text-sm">image</span>
            Trích xuất Keyframe
          </button>
          <button 
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all text-left text-xs font-semibold ${
              activeSection === 'rag' ? 'bg-secondary-container text-primary' : 'text-secondary hover:bg-surface-container-high'
            }`}
            onClick={() => setActiveSection('rag')}
          >
            <span className="material-symbols-outlined text-sm">forum</span>
            Hỏi đáp RAG
          </button>
        </nav>
      </aside>

      {/* Main content display area */}
      <div className="flex-1 overflow-y-auto p-6 md:p-margin-desktop bg-background custom-scrollbar">
        <div className="max-w-3xl mx-auto bg-white border border-outline-variant rounded-xl p-8 shadow-sm">
          
          {/* SECTION 1: OVERVIEW */}
          {activeSection === 'overview' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-2xl font-bold text-deep-navy">Tổng Quan Hệ Thống</h1>
              <p className="text-secondary text-sm leading-relaxed">
                Chào mừng bạn đến với <strong className="text-deep-navy">Multimodal Lecture Summarizer</strong> - Nền tảng phân tích bài giảng thông minh thế hệ mới, hỗ trợ tối đa việc học tập và nghiên cứu video.
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-5 border border-outline-variant/60 rounded-xl bg-background space-y-2">
                  <span className="material-symbols-outlined text-vibrant-cyan text-3xl">magic_button</span>
                  <h3 className="text-xs font-bold text-deep-navy">Trích xuất tức thì</h3>
                  <p className="text-[11px] text-secondary leading-normal">Chỉ cần dán liên kết Youtube hoặc tải lên tệp video, hệ thống tự động bóc tách âm thanh, nhận diện slide bài học.</p>
                </div>
                <div className="p-5 border border-outline-variant/60 rounded-xl bg-background space-y-2">
                  <span className="material-symbols-outlined text-vibrant-cyan text-3xl">history_toggle_off</span>
                  <h3 className="text-xs font-bold text-deep-navy">Tiết kiệm 80% thời gian</h3>
                  <p className="text-[11px] text-secondary leading-normal">Không cần xem hết hàng tiếng video dài, bạn có thể nắm bắt toàn bộ nội dung cốt lõi chỉ sau 2-3 phút đọc bản tóm tắt.</p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-deep-navy pt-2">Quy trình xử lý Video của AI</h2>
              <div className="space-y-4">
                <div className="flex gap-4 items-start">
                  <div className="w-6 h-6 rounded-full bg-deep-navy text-vibrant-cyan flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">1</div>
                  <div>
                    <h4 className="text-xs font-bold text-deep-navy">Tải lên &amp; Khởi tạo</h4>
                    <p className="text-[11px] text-secondary leading-relaxed">Hệ thống nhận video, kiểm tra định dạng và độ dài giới hạn trong DB để tiến hành đưa vào hàng đợi.</p>
                  </div>
                </div>
                <div className="flex gap-4 items-start">
                  <div className="w-6 h-6 rounded-full bg-deep-navy text-vibrant-cyan flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">2</div>
                  <div>
                    <h4 className="text-xs font-bold text-deep-navy">Tách tiếng &amp; Lọc Keyframe</h4>
                    <p className="text-[11px] text-secondary leading-relaxed">Mô hình WhisperX thực hiện dịch giọng nói sang văn bản kèm timestamp, đồng thời mô hình CLIP quét tìm các khoảnh khắc chuyển đổi trang slide.</p>
                  </div>
                </div>
                <div className="flex gap-4 items-start">
                  <div className="w-6 h-6 rounded-full bg-deep-navy text-vibrant-cyan flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">3</div>
                  <div>
                    <h4 className="text-xs font-bold text-deep-navy">Tạo Tóm tắt &amp; Vectors hóa</h4>
                    <p className="text-[11px] text-secondary leading-relaxed">Dùng LLM (GPT-4o/Gemini) viết báo cáo, xuất kết quả chương và đưa toàn bộ dữ liệu vào cơ sở dữ liệu vector ChromaDB.</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SECTION 2: TRANSCRIPTION & SUMMARY */}
          {activeSection === 'summary' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-2xl font-bold text-deep-navy">Tóm Tắt &amp; Chia Phân Đoạn Chương</h1>
              <p className="text-secondary text-sm leading-relaxed">
                Mô-đun ngôn ngữ xử lý toàn bộ bản ghi văn bản từ giọng nói giảng viên để xây dựng tài liệu tổng hợp đầy đủ nhất.
              </p>
              
              <div className="flex gap-4 items-start p-4 bg-background border border-outline-variant/60 rounded-xl">
                <span className="material-symbols-outlined text-vibrant-cyan text-2xl mt-0.5">translate</span>
                <div>
                  <h3 className="text-xs font-bold text-deep-navy mb-1">Nhận diện giọng nói chính xác với WhisperX</h3>
                  <p className="text-[11px] text-secondary leading-relaxed">
                    WhisperX cung cấp khả năng tự sửa lỗi chính tả bằng từ điển ngữ cảnh và căn chỉnh thời gian chuẩn xác đến từng từ (word-level timestamping), giúp bạn bấm vào bất kỳ câu nào trong văn bản dịch để nhảy ngay đến giây đó trong video.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start p-4 bg-background border border-outline-variant/60 rounded-xl">
                <span className="material-symbols-outlined text-vibrant-cyan text-2xl mt-0.5">view_timeline</span>
                <div>
                  <h3 className="text-xs font-bold text-deep-navy mb-1">Tự động phân đoạn bài giảng thành các Chương</h3>
                  <p className="text-[11px] text-secondary leading-relaxed">
                    Bằng thuật toán phân tích ngữ cảnh, mô hình LLM sẽ phát hiện các chủ đề chính và chia video bài học thành từng chương (Chapters) rõ ràng. Mỗi chương sẽ đi kèm thời gian bắt đầu, kết thúc cùng nội dung mô tả ngắn gọn.
                  </p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-deep-navy pt-2">Cách sử dụng Giao diện Tóm tắt:</h2>
              <ul className="list-disc pl-5 text-xs text-secondary space-y-1.5 leading-relaxed">
                <li>Vào trang <strong className="text-deep-navy">Lịch sử video</strong> để xem danh sách video của bạn.</li>
                <li>Chọn video mong muốn để mở trang kết quả phân tích.</li>
                <li>Bên trái giao diện là tab <strong className="text-deep-navy">Bản Tóm Tắt</strong> hiển thị tổng hợp ý chính, và tab <strong className="text-deep-navy">Văn Bản Dịch</strong> giúp bạn đọc lại toàn bộ lời nói giảng viên.</li>
              </ul>
            </div>
          )}

          {/* SECTION 3: KEYFRAME EXTRACTION */}
          {activeSection === 'keyframe' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-2xl font-bold text-deep-navy">Trích Xuất Hình Ảnh Slide (Keyframe)</h1>
              <p className="text-secondary text-sm leading-relaxed">
                Để hỗ trợ việc học trực quan, hệ thống tích hợp giải thuật phân tích hình ảnh giúp bạn không bỏ lỡ nội dung hiển thị trên bảng/màn hình trình chiếu.
              </p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-5 border border-outline-variant/60 rounded-xl bg-background space-y-2">
                  <h3 className="text-xs font-bold text-deep-navy">1. Lọc thay đổi khung cảnh</h3>
                  <p className="text-[11px] text-secondary leading-relaxed">Sử dụng công nghệ dò biên để tìm các thời điểm slide trình chiếu thay đổi hoặc giảng viên viết bảng mới, giảm thiểu việc trùng lặp hình ảnh.</p>
                </div>
                <div className="p-5 border border-outline-variant/60 rounded-xl bg-background space-y-2">
                  <h3 className="text-xs font-bold text-deep-navy">2. Trích xuất đặc trưng với CLIP</h3>
                  <p className="text-[11px] text-secondary leading-relaxed">Mô hình CLIP (Contrastive Language-Image Pre-training) tiến hành ánh xạ hình ảnh bài giảng và văn bản dịch thuật để liên kết slide tương ứng với nội dung giảng viên đang nói.</p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-deep-navy pt-2">Cách xem Slide ảnh trong giao diện:</h2>
              <p className="text-xs text-secondary leading-relaxed">
                Tại trang kết quả video, tab <strong className="text-deep-navy">Hình ảnh Keyframe</strong> sẽ hiển thị toàn bộ slide được chụp lại. Mỗi hình ảnh slide đều đính kèm một mốc thời gian (Ví dụ: <code className="bg-surface-container-high px-1 py-0.5 rounded text-[10px] font-mono-data text-deep-navy">02:45</code>). Bạn chỉ cần nhấp chuột vào hình slide đó để tua trình phát video trực tiếp đến thời điểm slide đó xuất hiện.
              </p>
            </div>
          )}

          {/* SECTION 4: INTERACTIVE Q&A */}
          {activeSection === 'rag' && (
            <div className="space-y-6">
              <h1 className="font-headline-xl text-2xl font-bold text-deep-navy">Hỏi Đáp Tương Tác (RAG)</h1>
              <p className="text-secondary text-sm leading-relaxed">
                Hệ thống không chỉ tóm tắt một chiều mà còn cho phép bạn trò chuyện trực tiếp để tìm kiếm thông tin chuyên sâu trong bài học.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="p-4 border border-outline-variant/60 rounded-xl bg-background text-center space-y-2">
                  <span className="material-symbols-outlined text-vibrant-cyan text-2xl">database</span>
                  <h4 className="text-xs font-bold text-deep-navy">ChromaDB Store</h4>
                  <p className="text-[10px] text-secondary leading-relaxed">Nội dung bài giảng được cắt thành các đoạn nhỏ, chuyển đổi thành chuỗi số và lưu vào CSDL Vector.</p>
                </div>
                <div className="p-4 border border-outline-variant/60 rounded-xl bg-background text-center space-y-2">
                  <span className="material-symbols-outlined text-vibrant-cyan text-2xl">search_check</span>
                  <h4 className="text-xs font-bold text-deep-navy">Similarity Match</h4>
                  <p className="text-[10px] text-secondary leading-relaxed">Khi gửi câu hỏi, hệ thống truy xuất các đoạn văn bản có ý nghĩa gần nhất trong kho dữ liệu bài học.</p>
                </div>
                <div className="p-4 border border-outline-variant/60 rounded-xl bg-background text-center space-y-2">
                  <span className="material-symbols-outlined text-vibrant-cyan text-2xl">smart_toy</span>
                  <h4 className="text-xs font-bold text-deep-navy">GenAI Synthesis</h4>
                  <p className="text-[10px] text-secondary leading-relaxed">LLM nhận câu hỏi kèm ngữ cảnh trích xuất để sinh câu trả lời chính xác nhất, tránh bịa đặt.</p>
                </div>
              </div>

              <h2 className="text-sm font-bold text-deep-navy pt-2">Hướng dẫn hỏi đáp:</h2>
              <ol className="list-decimal pl-5 text-xs text-secondary space-y-1.5 leading-relaxed">
                <li>Trong trang kết quả phân tích, nhấn vào nút <strong className="text-deep-navy">"Hỏi Đáp RAG"</strong> ở góc trên bên phải.</li>
                <li>Nhập các thắc mắc liên quan đến bài giảng (Ví dụ: <em className="italic">"Giảng viên giải thích định lý Bayes như thế nào ở phút thứ 10?"</em>).</li>
                <li>Câu trả lời của AI sẽ chứa các mốc liên kết thời gian. Nhấp vào các mốc này để xem đúng đoạn video giảng bài.</li>
              </ol>
            </div>
          )}



        </div>
      </div>
    </div>
  );
};
