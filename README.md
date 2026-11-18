# Video Summarization System

> 📖 **HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN TỪ A-Z:** Xem chi tiết tại [HUONG_DAN_CHAY_DU_AN.md](file:///d:/Document/Code/Exercise/Workspace/multimodal-lecture-summarizer/HUONG_DAN_CHAY_DU_AN.md) để cài đặt các công cụ bắt buộc (Python 3.10/3.11, Git LFS, FFmpeg, Redis) và khởi chạy không lỗi.

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
│   ├── modules/audio/              # Faster-Whisper, FFmpeg, noise reduction
│   ├── modules/visual/             # 🧑 NGƯỜI 2: PySceneDetect, CLIP, BLIP-2
│   ├── modules/fusion/             # 🧑 NGƯỜI 3: LangChain, LLM Prompts, ChromaDB RAG
│   └── tasks.py                    # Celery Tasks kết nối 3 modules
├── experiments/                    # Nghiên cứu và ablation studies
│   ├── notebooks/                  # Jupyter Notebooks test mô hình
│   ├── evaluation/                 # Script tính WER, ROUGE-L, BERTScore
│   └── datasets/                   # CHÚ Ý: .gitignore, chỉ sample data
├── docs/                           # Tài liệu dự án
│   ├── architecture/               # Biểu đồ luồng dữ liệu, HTML reports
│   └── current-system-architecture.md # Kiến trúc hệ thống mới nhất
├── docker-compose.yml              # Triển khai: Frontend, API, Worker, PostgreSQL, Redis
├── .env.example                    # File mẫu biến môi trường
├── .gitignore
└── README.md                       # (file này)
```

## 2. Tính năng cốt lõi

| Tầng | Chức năng | Công nghệ |
|------|-----------|-----------| 
| Audio | Nhận dạng lời nói + word timestamp | `en`: Qwen3-ASR 0.6B + Qwen3-ForcedAligner (mặc định, lỗi thì lùi về Faster-Whisper). `vi`: Faster-Whisper `small`. `auto`: nhận diện ngôn ngữ trước rồi chọn engine. Đặt `ASR_ENGLISH_BACKEND=whisper` để luôn dùng Faster-Whisper. |
| Speaker | Gán người nói khi có token | pyannote 3.1 chỉ chạy nếu có `HF_TOKEN`. Không có token thì `diarization_status=skipped`, không gán speaker giả. |
| Visual | Phát hiện cảnh, trích keyframe | PySceneDetect `ContentDetector` (mặc định); bộ nhận biết slide khi đặt `SCENE_SEGMENTER=slide` |
| Semantic | OCR slide, hiểu nội dung ảnh | PaddleOCR, CLIP, Florence-2 (CPU); RT-DETR-R18vd + BoT-SORT (bật mặc định, `ENABLE_OBJECT_TRACKING`) |
| Timeline | Alignment, chia chương, RAG | ChromaDB, cross-modal matching |
| Text | Tóm tắt | Đầu ra `en` có GPU: LED, transcript xen OCR (`LED_INCLUDE_OCR`). Trường hợp còn lại: LLM qua OpenRouter hoặc Groq. LLM lỗi: tóm tắt trích xuất dự phòng. |

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
celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1
```

## 4. Thực nghiệm và Đánh giá hệ thống

### 4.1 Mô tả tập dữ liệu

Đánh giá được thực hiện độc lập theo từng thành phần trong workspace `benchmark/` cùng cấp với repository này; giao thức và kết quả nằm ở `benchmark/docs/BENCHMARK.md`. Chưa có lần chạy đầu-cuối được chấm điểm, nên các bộ dữ liệu dưới đây không đánh giá toàn hệ thống.

*   **TED-LIUM Release 3 (ASR)**: 507 đoạn validation và 1.155 đoạn test, bài nói TED tiếng Anh. Chỉ ở mức chẩn đoán vì tập test đã lộ; chưa đại diện cho âm thanh giảng đường.
*   **IIIT-CVID (OCR)**: 15.379 ảnh cắt validation và 1.754 khung test với 40.063 nhãn. Kết luận là `no_selection`.
*   **TIB (tóm tắt văn bản)**: 300 bản ghi validation để chọn và 523 bản ghi test để xác nhận, tham chiếu là abstract của tác giả.
*   **MLPDataset (chuyển slide)**: 333 nguồn; đánh giá chẩn đoán trên 11 video tinh chỉnh và 7 video báo cáo với nhãn phát hành chưa qua duyệt.

TVSum không thuộc benchmark hiện tại; phần mô tả về bộ này trong `docs/DATASET_EDA.md` chỉ còn giá trị lịch sử.

*Phân tích khám phá dữ liệu hiện hành nằm ở `benchmark/docs/DATASET_EDA.md`.*

## 5. Tài liệu dự án

| Tài liệu | Nội dung |
|----------|----------|
| [docs/ROADMAP_EVALUATION.md](docs/ROADMAP_EVALUATION.md) | Đánh giá lộ trình nghiên cứu luận văn & Khảo sát mô hình SOTA |
| [docs/RESEARCH_DIRECTIONS.md](docs/RESEARCH_DIRECTIONS.md) | Thiết kế đề cương nghiên cứu, câu hỏi khoa học (RQs) và kịch bản thử nghiệm |
| [docs/POTENTIAL_RESEARCH_TRENDS.md](docs/POTENTIAL_RESEARCH_TRENDS.md) | Phân tích các hướng phát triển tiên phong (cutting-edge) trong tương lai |
| [docs/HYBRID_MODEL_GUIDE.md](docs/HYBRID_MODEL_GUIDE.md) | Hướng dẫn thiết kế, cài đặt PyTorch và huấn luyện mô hình Hybrid |
| [docs/setup-guide.md](docs/setup-guide.md) | Hướng dẫn cài đặt, chạy local và reset DB |
| [docs/colab-gpu-worker-guide.md](docs/colab-gpu-worker-guide.md) | Docker cho mọi máy, worker GPU trên Colab, R2 + Redis public, định tuyến GPU/CPU |
| [docs/current-system-architecture.md](docs/current-system-architecture.md) | Kiến trúc hệ thống hiện tại mới nhất, quy trình xử lý, checkpoint & fallback |
| [docs/MODEL_SELECTION_README.md](docs/MODEL_SELECTION_README.md) | Báo cáo lựa chọn mô hình Multi-Model dựa trên số liệu thực nghiệm |
| [docs/SPRINT_QUALITY_DEPLOYMENT.md](docs/SPRINT_QUALITY_DEPLOYMENT.md) | Báo cáo Quality Gates & Sprint Stack (S1–S10) |
| [docs/STACK_COMPARISON.md](docs/STACK_COMPARISON.md) | So sánh Local GPU vs API vs Hybrid |
| [docs/DATASET_EDA.md](docs/DATASET_EDA.md) | CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG (TEDLIUM & TVSum, tài liệu lịch sử) |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Khung benchmark trên video thực tế |
| [docs/florence-2-cpu-reproducibility.md](docs/florence-2-cpu-reproducibility.md) | Nguyên nhân Florence-2 sinh text rác trên CPU và giải pháp tái lập Windows/Linux |

## 6. Cấu hình Stack

AI Workers hỗ trợ 3 stack:
- **Local GPU** — Full local, cần NVIDIA RTX 4070+
- **Cloud API** — Full cloud (AssemblyAI, GPT-4o)
- **Hybrid** (khuyến nghị) — Local ASR/OCR + API summary

## 7. Trạng thái dự án

**v0.2.0 — Microservice Architecture:** Kiến trúc 3 tầng (Frontend, Backend API, AI Workers) chạy pipeline AI thực, chi tiết ở [docs/current-system-architecture.md](docs/current-system-architecture.md). Số đo hiện có chỉ ở mức từng thành phần; chưa có lần chạy đầu-cuối được chấm điểm.

## 8. License

TBD
