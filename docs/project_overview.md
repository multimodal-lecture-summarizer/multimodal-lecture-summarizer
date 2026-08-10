# Tổng Quan Dự Án (Project Overview)

Tài liệu này cung cấp cái nhìn tổng thể về kiến trúc, các pipeline và các mô hình AI đang được sử dụng trong hệ thống **Multimodal Lecture Summarizer**.

## 1. Kiến trúc chung
Hệ thống được chia làm hai phần chính hoạt động phối hợp với nhau thông qua Celery và Redis:
- **Backend**: Xây dựng bằng FastAPI, chịu trách nhiệm quản lý API, xác thực, cơ sở dữ liệu quan hệ, và điều phối các job.
- **AI Workers**: Quản lý và thực thi các luồng xử lý (pipeline) đa phương thức (multimodal) bất đồng bộ.

## 2. Các Thành phần và Công nghệ chính

### Backend (`/backend`)
- **Web Framework**: FastAPI
- **Cơ sở dữ liệu quan hệ**: PostgreSQL (thông qua `psycopg2-binary`) với ORM là `SQLAlchemy`. Quản lý migration bằng `alembic`.
- **Cơ sở dữ liệu Vector**: ChromaDB (phục vụ cho RAG Q&A).
- **Task Queue / Broker**: Celery và Redis.
- **Lưu trữ Object Storage**: Cloudflare R2 / AWS S3 (thông qua `boto3`) để lưu trữ video và ảnh keyframe.
- **Xác thực (Authentication)**: JWT thông qua `python-jose` và mã hoá mật khẩu với `passlib`.

### AI Workers (`/ai_workers`)
Chịu trách nhiệm thực thi pipeline trích xuất và phân tích. Pipeline được thực thi bởi Celery (`tasks.process_video`) và bao gồm các chặng sau:

#### Stage 1: Âm thanh & Nhận dạng giọng nói (Audio & ASR)
- **Công nghệ/Mô hình**: `WhisperX`, `faster-whisper`, `torchaudio`.
- **Nhiệm vụ**: Trích xuất âm thanh từ file video và nhận dạng giọng nói (Speech-to-Text), tạo các đoạn transcript ban đầu.

#### Stage 2: Phân biệt người nói (Speaker Diarization)
- **Công nghệ/Mô hình**: `pyannote.audio`.
- **Nhiệm vụ**: Phân tích đoạn âm thanh để phân biệt các giọng nói khác nhau, gán nhãn người nói cho từng đoạn hội thoại.

#### Stage 3: Thị giác máy tính (Visual Processing)
- **Công nghệ/Mô hình**: `PySceneDetect`, `opencv-python`.
- **Nhiệm vụ**: Phân tích các phân cảnh (scene detection) trong video và trích xuất các khung hình chính (keyframes/slides).

#### Stage 4: Phân tích ngữ nghĩa hình ảnh (Semantic Analysis)
- **Công nghệ/Mô hình**:
  - OCR (Nhận dạng chữ trong ảnh): `PaddleOCR`.
  - Image Captioning / Semantic: `Florence-2` hoặc `BLIP-2` (thông qua `transformers` và `Pillow`).
  - Embeddings: CLIP (`sentence-transformers`).
- **Nhiệm vụ**: Phân tích nội dung của các keyframe, loại bỏ các slide trùng lặp (dùng thuật toán K-Means phân cụm embedding) và tạo ra mô tả (caption) cho từng slide đặc trưng.

#### Stage 5: Trục thời gian & Ánh xạ (Timeline Alignment)
- **Nhiệm vụ**: Đồng bộ và liên kết (fusion) dữ liệu văn bản (transcript) với hình ảnh (slide) dựa trên trục thời gian, sau đó chia đoạn (chapter segmentation).

#### Stage 6: Tóm tắt văn bản & Chỉ mục RAG (Summarization & RAG)
- **Công nghệ/Mô hình**:
  - Framework: `LangChain`, `langchain-openai`.
  - LLM Provider: Dùng **Groq API** (như cấu hình trong `tasks.py` để lấy bản tóm tắt tốc độ cao). Các provider khác như OpenAI, Anthropic cũng có thư viện sẵn sàng.
  - Vector DB: `ChromaDB`.
- **Nhiệm vụ**:
  - Gửi dữ liệu đa phương thức (transcript + mô tả slide) qua LLM để sinh bản tóm tắt bài giảng chi tiết.
  - Lưu trữ embedding của các đoạn tóm tắt/transcript vào ChromaDB để phục vụ hệ thống Multimodal RAG (Hỏi đáp dựa trên tài liệu bài giảng).

## 3. Tóm tắt luồng dữ liệu (Data Flow)
1. Video được tải xuống từ R2, YouTube (qua `yt-dlp`), hoặc URL trực tiếp.
2. Job được Celery worker tiếp nhận.
3. Các module âm thanh và hình ảnh chạy độc lập để tạo ra transcripts (có nhãn người nói) và keyframes.
4. Keyframes được đưa qua module phân tích ngữ nghĩa (Semantic) để lấy OCR và mô tả hình ảnh. Keyframes được upload lên Cloudflare R2.
5. Module Fusion (TimelineBuilder) gom toàn bộ dữ liệu lại để ánh xạ các text segment vào đúng slide.
6. Dữ liệu tổng hợp được đưa qua LLM (Groq) để tạo bản tóm tắt, đồng thời index vector vào ChromaDB.
7. Kết quả trả về cho hệ thống Backend để lưu vào cơ sở dữ liệu (PostgreSQL) và hiển thị qua UI.
