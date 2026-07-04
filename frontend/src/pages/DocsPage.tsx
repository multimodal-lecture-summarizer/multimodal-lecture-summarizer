import React, { useState } from 'react';
import './DocsPage.css';

export const DocsPage: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'overview' | 'summary' | 'keyframe' | 'rag' | 'api'>('overview');

  return (
    <div className="docs-page animate-fade-in">
      {/* Sidebar for documentation navigation */}
      <div className="docs-sidebar">
        <div className="docs-sidebar-title">
          <i className="fa-solid fa-book-open"></i> Hướng Dẫn Sử Dụng
        </div>
        <nav className="docs-nav">
          <button 
            className={`docs-nav-btn ${activeSection === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveSection('overview')}
          >
            <i className="fa-solid fa-house-laptop"></i> Tổng quan hệ thống
          </button>
          <button 
            className={`docs-nav-btn ${activeSection === 'summary' ? 'active' : ''}`}
            onClick={() => setActiveSection('summary')}
          >
            <i className="fa-solid fa-file-invoice"></i> Tóm tắt & Chương
          </button>
          <button 
            className={`docs-nav-btn ${activeSection === 'keyframe' ? 'active' : ''}`}
            onClick={() => setActiveSection('keyframe')}
          >
            <i className="fa-solid fa-image"></i> Trích xuất Keyframe
          </button>
          <button 
            className={`docs-nav-btn ${activeSection === 'rag' ? 'active' : ''}`}
            onClick={() => setActiveSection('rag')}
          >
            <i className="fa-solid fa-comments"></i> Hỏi đáp RAG
          </button>
          <button 
            className={`docs-nav-btn ${activeSection === 'api' ? 'active' : ''}`}
            onClick={() => setActiveSection('api')}
          >
            <i className="fa-solid fa-terminal"></i> Tích hợp API
          </button>
        </nav>
      </div>

      {/* Main content display area */}
      <div className="docs-content-container">
        
        {/* SECTION 1: OVERVIEW */}
        {activeSection === 'overview' && (
          <div className="docs-section">
            <h1>Tổng Quan Hệ Thống</h1>
            <p className="lead">
              Chào mừng bạn đến với <strong>Multimodal Lecture Summarizer</strong> - Nền tảng phân tích bài giảng thông minh thế hệ mới, hỗ trợ tối đa việc học tập và nghiên cứu video.
            </p>
            
            <div className="docs-card-grid">
              <div className="docs-card">
                <i className="fa-solid fa-wand-magic-sparkles text-primary"></i>
                <h3>Trích xuất tức thì</h3>
                <p>Chỉ cần dán liên kết Youtube hoặc tải lên tệp video, hệ thống tự động bóc tách âm thanh, nhận diện slide bài học.</p>
              </div>
              <div className="docs-card">
                <i className="fa-solid fa-clock-rotate-left text-primary"></i>
                <h3>Tiết kiệm 80% thời gian</h3>
                <p>Không cần xem hết hàng tiếng video dài, bạn có thể nắm bắt toàn bộ nội dung cốt lõi chỉ sau 2-3 phút đọc bản tóm tắt.</p>
              </div>
            </div>

            <h2>Quy trình xử lý Video của AI</h2>
            <div className="workflow-steps">
              <div className="step">
                <div className="step-num">1</div>
                <div>
                  <h4>Tải lên & Khởi tạo</h4>
                  <p>Hệ thống nhận video, kiểm tra định dạng và độ dài giới hạn trong DB để tiến hành đưa vào hàng đợi.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">2</div>
                <div>
                  <h4>Tách tiếng & Lọc Keyframe</h4>
                  <p>Mô hình WhisperX thực hiện dịch giọng nói sang văn bản kèm timestamp, đồng thời mô hình CLIP quét tìm các khoảnh khắc chuyển đổi trang slide.</p>
                </div>
              </div>
              <div className="step">
                <div className="step-num">3</div>
                <div>
                  <h4>Tạo Tóm tắt & Vectors hóa</h4>
                  <p>Dùng LLM (GPT-4o/Gemini) viết báo cáo, xuất kết quả chương và đưa toàn bộ dữ liệu vào cơ sở dữ liệu vector ChromaDB.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* SECTION 2: TRANSCRIPTION & SUMMARY */}
        {activeSection === 'summary' && (
          <div className="docs-section">
            <h1>Tóm Tắt & Chia Phân Đoạn Chương</h1>
            <p>
              Mô-đun ngôn ngữ xử lý toàn bộ bản ghi văn bản từ giọng nói giảng viên để xây dựng tài liệu tổng hợp đầy đủ nhất.
            </p>
            
            <div className="feature-block">
              <div className="feature-icon-wrapper">
                <i className="fa-solid fa-spell-check"></i>
              </div>
              <div>
                <h3>Nhận diện giọng nói chính xác với WhisperX</h3>
                <p>
                  WhisperX cung cấp khả năng tự sửa lỗi chính tả bằng từ điển ngữ cảnh và căn chỉnh thời gian chuẩn xác đến từng từ (word-level timestamping), giúp bạn bấm vào bất kỳ câu nào trong văn bản dịch để nhảy ngay đến giây đó trong video.
                </p>
              </div>
            </div>

            <div className="feature-block">
              <div className="feature-icon-wrapper">
                <i className="fa-solid fa-indent"></i>
              </div>
              <div>
                <h3>Tự động phân đoạn bài giảng thành các Chương</h3>
                <p>
                  Bằng thuật toán phân tích ngữ cảnh, mô hình LLM sẽ phát hiện các chủ đề chính và chia video bài học thành từng chương (Chapters) rõ ràng. Mỗi chương sẽ đi kèm thời gian bắt đầu, kết thúc cùng nội dung mô tả ngắn gọn.
                </p>
              </div>
            </div>

            <h2>Cách sử dụng Giao diện Tóm tắt:</h2>
            <ul>
              <li>Vào trang <strong>Lịch sử video</strong> để xem danh sách video của bạn.</li>
              <li>Chọn video mong muốn để mở trang kết quả phân tích.</li>
              <li>Bên trái giao diện là tab <strong>Bản Tóm Tắt</strong> hiển thị tổng hợp ý chính, và tab <strong>Văn Bản Dịch</strong> giúp bạn đọc lại toàn bộ lời nói giảng viên.</li>
            </ul>
          </div>
        )}

        {/* SECTION 3: KEYFRAME EXTRACTION */}
        {activeSection === 'keyframe' && (
          <div className="docs-section">
            <h1>Trích Xuất Hình Ảnh Slide (Keyframe)</h1>
            <p>
              Để hỗ trợ việc học trực quan, hệ thống tích hợp giải thuật phân tích hình ảnh giúp bạn không bỏ lỡ nội dung hiển thị trên bảng/màn hình trình chiếu.
            </p>
            
            <div className="docs-card-grid">
              <div className="docs-card">
                <h3>1. Lọc thay đổi khung cảnh</h3>
                <p>Sử dụng công nghệ dò biên để tìm các thời điểm slide trình chiếu thay đổi hoặc giảng viên viết bảng mới, giảm thiểu việc trùng lặp hình ảnh.</p>
              </div>
              <div className="docs-card">
                <h3>2. Trích xuất đặc trưng với CLIP</h3>
                <p>Mô hình CLIP (Contrastive Language-Image Pre-training) tiến hành ánh xạ hình ảnh bài giảng và văn bản dịch thuật để liên kết slide tương ứng với nội dung giảng viên đang nói.</p>
              </div>
            </div>

            <h2>Cách xem Slide ảnh trong giao diện:</h2>
            <p>
              Tại trang kết quả video, tab <strong>Hình ảnh Keyframe</strong> sẽ hiển thị toàn bộ slide được chụp lại. Mỗi hình ảnh slide đều đính kèm một mốc thời gian (Ví dụ: <code>02:45</code>). Bạn chỉ cần nhấp chuột vào hình slide đó để tua trình phát video trực tiếp đến thời điểm slide đó xuất hiện.
            </p>
          </div>
        )}

        {/* SECTION 4: INTERACTIVE Q&A */}
        {activeSection === 'rag' && (
          <div className="docs-section">
            <h1>Hỏi Đáp Tương Tác (RAG)</h1>
            <p>
              Hệ thống không chỉ tóm tắt một chiều mà còn cho phép bạn trò chuyện trực tiếp để tìm kiếm thông tin chuyên sâu trong bài học.
            </p>

            <div className="rag-workflow">
              <div className="rag-step">
                <i className="fa-solid fa-arrows-down-to-people"></i>
                <h4>Vector Hóa & ChromaDB</h4>
                <p>Nội dung bài giảng được cắt thành các đoạn nhỏ (chunks), chuyển đổi thành chuỗi số và lưu vào CSDL Vector ChromaDB.</p>
              </div>
              <div className="rag-step">
                <i className="fa-solid fa-magnifying-glass-arrow-right"></i>
                <h4>Tìm kiếm tương đồng</h4>
                <p>Khi bạn gửi câu hỏi, hệ thống truy xuất các đoạn văn bản có ý nghĩa gần nhất trong kho dữ liệu bài học.</p>
              </div>
              <div className="rag-step">
                <i className="fa-solid fa-robot"></i>
                <h4>Tạo câu trả lời (LLM)</h4>
                <p>LLM nhận câu hỏi kèm dữ liệu trích xuất để sinh câu trả lời chính xác nhất, cam kết không bịa đặt thông tin.</p>
              </div>
            </div>

            <h2>Hướng dẫn hỏi đáp:</h2>
            <ol>
              <li>Trong trang kết quả phân tích, nhấn vào nút <strong>"Đặt câu hỏi AI (RAG)"</strong> ở góc phải.</li>
              <li>Nhập các thắc mắc liên quan đến bài giảng (Ví dụ: <em>"Giảng viên giải thích định lý Bayes như thế nào ở phút thứ 10?"</em>).</li>
              <li>Câu trả lời của AI sẽ chứa các mốc liên kết thời gian màu xanh. Nhấp vào các mốc này để xem đúng đoạn video giảng bài.</li>
            </ol>
          </div>
        )}

        {/* SECTION 5: API INTEGRATION */}
        {activeSection === 'api' && (
          <div className="docs-section">
            <h1>Tích Hợp API Hệ Thống</h1>
            <p>
              Dành cho các lập trình viên muốn tích hợp lõi xử lý tóm tắt video vào các phần mềm quản lý học tập (LMS) khác.
            </p>
            
            <div className="api-endpoint-box">
              <span className="badge badge-get">GET</span>
              <code>/api/v1/videos</code>
              <p>Lấy danh sách video đã phân tích của người dùng hiện tại.</p>
            </div>

            <div className="api-endpoint-box">
              <span className="badge badge-post">POST</span>
              <code>/api/v1/videos/upload</code>
              <p>Tải tệp video hoặc gửi link Youtube kèm ngôn ngữ xử lý để kích hoạt pipeline AI.</p>
            </div>

            <div className="api-endpoint-box">
              <span className="badge badge-post">POST</span>
              <code>/api/v1/qa/ask</code>
              <p>Gửi câu hỏi tương tác với tài liệu vector bài giảng trong ChromaDB.</p>
            </div>

            <div className="api-endpoint-box">
              <span className="badge badge-get">GET</span>
              <code>/api/v1/videos/standards</code>
              <p>Đọc các quy chuẩn về kích thước, định dạng và giới hạn thời lượng tải lên.</p>
            </div>

            <h2>Tài liệu tự động:</h2>
            <p>
              Bạn có thể truy cập tài liệu Swagger đầy đủ tương tác trực tiếp với Backend tại: <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" className="text-link">http://localhost:8000/docs</a> khi chạy server cục bộ.
            </p>
          </div>
        )}

      </div>
    </div>
  );
};
