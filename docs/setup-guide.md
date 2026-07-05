# Guide to Setup, Run, and Reset the System

Tài liệu này hướng dẫn chi tiết cách chuẩn bị môi trường, chạy các dịch vụ (Backend, Frontend, AI Worker) và cách khởi tạo lại (reset) cơ sở dữ liệu cùng Cloudflare R2 cho dự án **Multimodal Lecture Summarizer**.

---

## 1. Chuẩn bị môi trường & Biến môi trường (.env)

Hệ thống sử dụng các tệp `.env` riêng biệt cho Backend và AI Worker để quản lý cấu hình bảo mật.

### Cấu hình Backend:
1. Sao chép file cấu hình mẫu trong thư mục `backend/`:
   ```powershell
   cd backend
   cp .env.example .env
   ```
2. Mở file `backend/.env` và điền đầy đủ các thông tin:
   * **Database**: `DATABASE_URL` (kết nối PostgreSQL hoặc SQLite dự phòng).
   * **Vector DB**: `CHROMA_HOST` và `CHROMA_PORT` (để kết nối tới ChromaDB RAG).
   * **Cloudflare R2**: `CF_R2_ACCESS_KEY_ID`, `CF_R2_SECRET_ACCESS_KEY`, `CF_R2_BUCKET_NAME`, và `CF_R2_PUBLIC_URL` (dùng để lưu trữ video và keyframe).
   * **AI API Key**: `GROQ_API_KEY` (Sử dụng để chạy mô hình ngôn ngữ lớn tóm tắt và hỏi đáp RAG).

---

## 2. Chuẩn bị và Chạy Backend (FastAPI)

API Backend quản lý các luồng dữ liệu chính, giao tiếp cơ sở dữ liệu và kích hoạt công việc xử lý video.

1. Di chuyển vào thư mục `backend/` và tạo môi trường ảo Python:
   ```powershell
   cd backend
   python -m venv .venv
   ```
2. Kích hoạt và cài đặt các thư viện phụ thuộc:
   ```powershell
   .venv\Scripts\pip.exe install -r requirements.txt
   ```
3. Chạy Uvicorn Web Server ở chế độ phát triển (reload tự động):
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.main:app --reload
   ```
   * *Địa chỉ API*: `http://127.0.0.1:8000`
   * *Tài liệu API Swagger*: `http://127.0.0.1:8000/docs`

> [!NOTE]
> Khi khởi động, Backend sẽ chạy một luồng ngầm tự động quét cơ sở dữ liệu mỗi 10 giây để kiểm tra và đồng bộ trạng thái các công việc Celery từ Redis vào PostgreSQL, giúp lưu kết quả tự động ngay cả khi người dùng đóng tab trình duyệt.

---

## 3. Chuẩn bị và Chạy AI Worker (Celery)

AI Worker thực hiện các công việc nặng về tính toán: Nhận diện giọng nói, tách giọng nói, phân tích hình ảnh/slide, và tóm tắt bằng LLM.

### Chạy bằng CPU (Mặc định):
1. Đứng ở thư mục gốc của dự án, thiết lập `PYTHONPATH` và khởi chạy Celery Worker (chế độ luồng trên Windows):
   ```powershell
   # Tại thư mục gốc (multimodal-lecture-summarizer):
   $env:PYTHONPATH="."
   backend\.venv\Scripts\python.exe -m celery -A ai_workers.core.celery_app worker --loglevel=info -P threads --concurrency=2
   ```

### Chạy bằng GPU CUDA (Tăng tốc gấp 10-20 lần - Khuyên dùng):
Nếu máy tính của bạn có card đồ họa NVIDIA CUDA, bạn có thể tăng tốc đáng kể tiến trình xử lý:
1. Cài đặt phiên bản PyTorch CUDA 12.1 và cập nhật các thư viện bổ trợ:
   ```powershell
   backend\.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
   backend\.venv\Scripts\pip.exe install --upgrade typing-extensions
   ```
2. Khởi chạy lại Celery Worker. Bạn sẽ thấy dòng log xác nhận đã nhận diện GPU thành công:
   ```text
   Device set to use cuda:0
   ```

---

## 4. Chuẩn bị và Chạy Frontend (ReactJS)

Giao diện người dùng cho phép tải lên video bài giảng, theo dõi tiến trình thực tế và hiển thị kết quả trực quan (tóm tắt, chương, slide keyframe, hỏi đáp RAG).

1. Di chuyển vào thư mục `frontend/` và cài đặt các gói NPM:
   ```powershell
   cd frontend
   npm install
   ```
2. Chạy Frontend server ở chế độ Development:
   ```powershell
   npm run dev
   ```
   * *Địa chỉ ứng dụng Web*: `http://localhost:5173` (hoặc cổng được hiển thị trên console).

---

## 5. Khởi tạo lại Hệ thống & Reset Cơ sở dữ liệu

Khi bạn muốn xóa sạch dữ liệu chạy thử để chuẩn bị cho lượt xử lý mới, hệ thống cung cấp một script dọn dẹp toàn diện.

1. Di chuyển vào thư mục `backend/`:
   ```powershell
   cd backend
   ```
2. Chạy tệp script reset an toàn:
   ```powershell
   .venv\Scripts\python.exe reset_r2_and_db.py
   ```

### Tác vụ tự động của script:
* **Relational DB**: Xóa sạch toàn bộ video tải lên, tóm tắt, chương và phân cảnh nhưng **giữ lại tài khoản người dùng mẫu** (admin/user mặc định) để tránh việc phải đăng ký lại tài khoản mới.
* **Cloudflare R2**: Tự động dọn sạch các tệp video và keyframe đã lưu trữ trên đám mây R2.
* **ChromaDB**: Xóa và khởi tạo lại collection vector để chuẩn bị nạp tài liệu RAG mới.
* **Storage cục bộ**: Xóa toàn bộ file cache/temp được tạo ra trong quá trình xử lý video trước đó.
