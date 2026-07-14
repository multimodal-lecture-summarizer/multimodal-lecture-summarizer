# Video Summarization System

Hệ thống tóm tắt video bài giảng đa phương thức sử dụng AI — kiến trúc microservice với Frontend (ReactJS), Backend API (FastAPI), và AI Workers (Celery).

## 1. Kiến trúc hệ thống

```
VIDEO-SUMMARIZATION-SYSTEM/
├── .github/                        # CI/CD workflows, issue templates
├── frontend/                       # 🟦 ReactJS + Tailwind: Giao diện người dùng
│   ├── src/components/             # UI components (Buttons, Modals...)
│   ├── src/pages/                  # Trang Upload, Kết Quả, Admin, Q&A
│   ├── src/services/               # API client (axios/fetch)
│   └── src/hooks/                  # Custom hooks (WebSocket)
├── backend/                    # 🟦 FastAPI: xử lý HTTP request
│   ├── app/api/                    # Endpoints: /upload, /videos, /stats
│   ├── app/core/                   # Cấu hình hệ thống, JWT security
│   ├── app/db/                     # PostgreSQL (SQLAlchemy), migrations
│   ├── app/schemas/                # Pydantic models (API Contract)
│   └── app/services/               # Logic CRUD database, gửi task vào Celery
├── ai_workers/                     # 🟨 Celery + AI Models: xử lý bất đồng bộ
│   ├── core/                       # Cấu hình chung, kết nối Redis/Celery
│   ├── modules/audio/              # 🧑 NGƯỜI 1: WhisperX, FFmpeg, Noise reduction
│   ├── modules/visual/             # 🧑 NGƯỜI 2: PySceneDetect, CLIP, BLIP-2
│   ├── modules/fusion/             # 🧑 NGƯỜI 3: LangChain, LLM Prompts, ChromaDB RAG
│   └── tasks.py                    # Celery Tasks kết nối 3 modules
├── experiments/                    # Nghiên cứu và ablation studies
│   ├── notebooks/                  # Jupyter Notebooks test mô hình
│   ├── evaluation/                 # Script tính WER, ROUGE-L, BERTScore
│   └── datasets/                   # CHÚ Ý: .gitignore, chỉ sample data
├── docs/                           # Tài liệu dự án
│   ├── api_contracts/              # JSON schema Frontend ↔ Backend
│   ├── architecture/               # Biểu đồ luồng dữ liệu, ERD
│   └── reports/                    # Báo cáo đồ án, file thuyết trình
├── docker-compose.yml              # Triển khai: Frontend, API, Worker, PostgreSQL, Redis
├── .env.example                    # File mẫu biến môi trường
├── .gitignore
└── README.md                       # (file này)
```

## 2. Tính năng cốt lõi

| Tầng | Chức năng | Công nghệ |
|------|-----------|-----------| 
| Audio | Nhận dạng lời nói + timestamp | WhisperX, AssemblyAI |
| Speaker | Xác định ai nói | pyannote.audio |
| Visual | Phát hiện cảnh, trích keyframe | PySceneDetect |
| Semantic | OCR slide, hiểu nội dung ảnh | PaddleOCR, CLIP, GPT-4o vision |
| Timeline | Alignment, chia chương, RAG | ChromaDB, cross-modal matching |
| Text | Tóm tắt có trích dẫn | GPT-4o / Ollama |

## 3. Hướng dẫn Cài đặt & Khởi chạy

### Docker (Khuyến nghị)

```bash
# Copy và cấu hình biến môi trường
cp .env.example .env

# Chạy toàn bộ services
docker-compose up --build
```

Truy cập:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- Redis: localhost:6379
- PostgreSQL: localhost:5432

### Development (Local)

```bash
# Frontend
cd frontend
npm install
npm run dev         # → http://localhost:3000

# Backend API
cd backend_api
pip install -r requirements.txt
uvicorn app.main:app --reload   # → http://localhost:8000

# AI Worker
cd ai_workers
pip install -r requirements.txt
celery -A ai_workers.core.celery_app worker --loglevel=info
```

## 4. Thực nghiệm và Đánh giá hệ thống

### 4.1 Mô tả tập dữ liệu

Quá trình thực nghiệm được tiến hành độc lập trên hai tập dữ liệu khác nhau, phục vụ cho hai bài toán cốt lõi của hệ thống: nhận dạng giọng nói và tóm tắt video.

*   **TEDLIUM**: Đây là một kho dữ liệu âm thanh chuyên dụng được trích xuất từ các bài diễn thuyết thực tế bằng tiếng Anh. Môi trường âm thanh trong tập dữ liệu này rất sát với bài toán thực tế mà đồ án hướng tới: diễn giả nói chuyện trong một hội trường lớn, có tiếng ồn nền, có sự đa dạng về ngữ điệu và tốc độ nói. Việc sử dụng TEDLIUM giúp đo đạc chính xác khả năng bóc tách khoảng lặng và độ chuẩn xác của mô hình giải mã văn bản.
*   **TVSUM**: Tập dữ liệu này chứa năm mươi video ngắn được thu thập từ YouTube, bao trùm mười chủ đề khác nhau như tin tức, phim tài liệu, bài giảng và hướng dẫn thực hành. Điểm giá trị nhất của TVSUM là mỗi video đều đi kèm với nhãn đánh giá mức độ quan trọng ở cấp độ khung hình do con người gán thủ công. Dữ liệu này đóng vai trò làm tiêu chuẩn vàng để hệ thống đối chiếu và đo lường khả năng trích xuất các khung hình mang tính trọng tâm.

*Chi tiết kết quả Phân tích Khám phá Dữ liệu (EDA) có thể tham khảo tại báo cáo: [docs/DATASET_EDA.md](docs/DATASET_EDA.md)*

## 5. Tài liệu dự án

| Tài liệu | Nội dung |
|----------|----------|
| [docs/ROADMAP_EVALUATION.md](docs/ROADMAP_EVALUATION.md) | Đánh giá lộ trình nghiên cứu luận văn & Khảo sát mô hình SOTA |
| [docs/RESEARCH_DIRECTIONS.md](docs/RESEARCH_DIRECTIONS.md) | Thiết kế đề cương nghiên cứu, câu hỏi khoa học (RQs) và kịch bản thử nghiệm |
| [docs/POTENTIAL_RESEARCH_TRENDS.md](docs/POTENTIAL_RESEARCH_TRENDS.md) | Phân tích các hướng phát triển tiên phong (cutting-edge) trong tương lai |
| [docs/HYBRID_MODEL_GUIDE.md](docs/HYBRID_MODEL_GUIDE.md) | Hướng dẫn thiết kế, cài đặt PyTorch và huấn luyện mô hình Hybrid |
| [docs/setup-guide.md](docs/setup-guide.md) | Hướng dẫn cài đặt, chạy local và reset DB |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kiến trúc chi tiết, data flow |
| [docs/STACK_COMPARISON.md](docs/STACK_COMPARISON.md) | So sánh Local GPU vs API vs Hybrid |
| [docs/DATASET_EDA.md](docs/DATASET_EDA.md) | CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG (Phân tích khám phá dữ liệu TEDLIUM & TVSum) |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Khung benchmark trên video thực tế |
| [docs/api_contracts/](docs/api_contracts/) | JSON schema Frontend ↔ Backend |

## 6. Cấu hình Stack

AI Workers hỗ trợ 3 stack:
- **Local GPU** — Full local, cần NVIDIA RTX 4070+
- **Cloud API** — Full cloud (AssemblyAI, GPT-4o)
- **Hybrid** (khuyến nghị) — Local ASR/OCR + API summary

## 7. Trạng thái dự án

**v0.2.0 — Microservice Architecture:** Kiến trúc 3 tầng (Frontend, Backend API, AI Workers) đã sẵn sàng. Các module AI stage đang ở dạng stub — triển khai theo lộ trình.

## 8. License

TBD
