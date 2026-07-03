# Video Summarization System

Hệ thống tóm tắt video bài giảng đa phương thức sử dụng AI — kiến trúc microservice với Frontend (ReactJS), Backend API (FastAPI), và AI Workers (Celery).

## Kiến trúc

```
VIDEO-SUMMARIZATION-SYSTEM/
├── .github/                        # CI/CD workflows, issue templates
├── frontend/                       # 🟦 ReactJS + Tailwind: Giao diện người dùng
│   ├── src/components/             # UI components (Buttons, Modals...)
│   ├── src/pages/                  # Trang Upload, Kết Quả, Admin, Q&A
│   ├── src/services/               # API client (axios/fetch)
│   └── src/hooks/                  # Custom hooks (WebSocket)
├── backend_api/                    # 🟦 FastAPI: xử lý HTTP request
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

## Tính năng

| Tầng | Chức năng | Công nghệ |
|------|-----------|-----------| 
| Audio | Nhận dạng lời nói + timestamp | WhisperX, AssemblyAI |
| Speaker | Xác định ai nói | pyannote.audio |
| Visual | Phát hiện cảnh, trích keyframe | PySceneDetect |
| Semantic | OCR slide, hiểu nội dung ảnh | PaddleOCR, CLIP, GPT-4o vision |
| Timeline | Alignment, chia chương, RAG | ChromaDB, cross-modal matching |
| Text | Tóm tắt có trích dẫn | GPT-4o / Ollama |

## Cài đặt & Chạy

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

## Tài liệu

| Tài liệu | Nội dung |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kiến trúc chi tiết, data flow |
| [docs/STACK_COMPARISON.md](docs/STACK_COMPARISON.md) | So sánh Local GPU vs API vs Hybrid |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | Khung benchmark trên video thực tế |
| [docs/api_contracts/](docs/api_contracts/) | JSON schema Frontend ↔ Backend |

## Cấu hình stack

AI Workers hỗ trợ 3 stack:
- **Local GPU** — Full local, cần NVIDIA RTX 4070+
- **Cloud API** — Full cloud (AssemblyAI, GPT-4o)
- **Hybrid** (khuyến nghị) — Local ASR/OCR + API summary

## Trạng thái dự án

**v0.2.0 — Microservice Architecture:** Kiến trúc 3 tầng (Frontend, Backend API, AI Workers) đã sẵn sàng. Các module AI stage đang ở dạng stub — triển khai theo lộ trình.

## License

TBD
