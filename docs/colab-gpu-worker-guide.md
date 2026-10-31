# Chạy pipeline AI trên GPU Colab và Docker

Website (frontend + backend) chạy ở một máy. Pipeline AI (`ai_workers.process_video`) chạy ở worker GPU: Colab, hoặc bất kỳ máy NVIDIA nào có Docker. Khi không có worker GPU nào online, backend gửi job cho worker CPU local.

## 1. Cách định tuyến job

```
Website ──upload──► Backend ──send_task──► Redis (public) ──► queue "gpu"    ──► Colab / Docker GPU worker
                                                          └─► queue "celery" ──► worker CPU local (+ RAG indexing)
```

- Backend gọi `choose_video_queue` (`backend/app/services/worker_routing.py`) trước mỗi job: hỏi Redis xem có worker nào nghe queue `gpu` không (timeout 1s).
- Có worker GPU **và** video là URL (R2/YouTube) → queue `gpu`. Ngược lại → queue `celery`.
- Video lưu ở mock storage (đường dẫn trên ổ máy backend) luôn chạy CPU, vì worker từ xa không đọc được.
- `build_rag_index` / `reindex_video` ghi thẳng vào DB nên luôn ở queue `celery`, cần một worker local (hoặc worker có quyền tới DB) nghe queue này.
- Job đã vào queue `gpu` thì ở đó tới khi worker GPU nhận. Nếu Colab tắt giữa chừng, task được giao lại sau `visibility_timeout` của Redis (mặc định 1 giờ) khi một worker `gpu` quay lại.

## 2. Chuẩn bị một lần

### 2.1 Cloudflare R2 (bắt buộc cho worker từ xa)

Worker GPU tải video từ R2 và upload keyframe lên R2 để website hiển thị.

1. Cloudflare Dashboard → R2 → **Create bucket** (ví dụ `lecture-summarizer-assets`).
2. Bucket → Settings → **Public access**: bật `r2.dev` subdomain (hoặc gắn custom domain). Ghi lại URL `https://pub-xxxx.r2.dev`.
3. R2 → **Manage R2 API Tokens** → Create token, quyền *Object Read & Write*, giới hạn vào bucket trên. Ghi lại Access Key ID, Secret Access Key, Account ID.
4. Điền vào `backend/.env`:

```
CF_R2_ACCOUNT_ID=...
CF_R2_ACCESS_KEY_ID=...
CF_R2_SECRET_ACCESS_KEY=...
CF_R2_BUCKET_NAME=lecture-summarizer-assets
CF_R2_PUBLIC_URL=https://pub-xxxx.r2.dev
```

Kiểm tra: khởi động backend, log phải báo kết nối R2 thành công (`R2Service.verify_connection`). Upload một video, `file_path` trong DB phải là URL `https://pub-...`, không phải `/static/mock_r2/...`.

### 2.2 Redis public

Colab chỉ kết nối ra ngoài, nên Redis phải truy cập được từ Internet và có mật khẩu. Ưu tiên TLS (`rediss://`).

- Upstash: chọn gói **fixed-price**. Gói free/pay-as-you-go bị Celery poll hết quota request.
- Redis Cloud: gói free 30 MB đủ cho broker; chỉ có database 0, nên dùng cùng một URL cho broker và result backend.

Với URL `rediss://`, Celery cần tham số `ssl_cert_reqs`:

```
rediss://default:<password>@<host>:<port>/0?ssl_cert_reqs=CERT_REQUIRED
```

Điền cùng một URL vào:
- `backend/.env`: `CELERY_BROKER_URL` và `CELERY_RESULT_BACKEND` (khi chạy backend không dùng Docker).
- `./.env` (thư mục gốc): `SHARED_REDIS_URL` (khi chạy bằng Docker Compose).
- Colab Secrets: `SHARED_REDIS_URL`.

## 3. Chạy bằng Docker

Image dùng chung cho mọi máy: `ai_workers/Dockerfile` (Python 3.11, torch 2.5.1 CUDA 12, Faster-Whisper với cuDNN 9). Máy có GPU chỉ cần driver NVIDIA + NVIDIA Container Toolkit (Windows: Docker Desktop với WSL2).

| Mục đích | Lệnh |
|---|---|
| Cả hệ thống, worker CPU | `docker compose up --build` |
| Cả hệ thống, worker dùng GPU máy này | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build` |
| Chỉ worker GPU, website ở máy khác | `docker compose -f docker-compose.worker.yml up --build` |

- Frontend: http://localhost:3000, API: http://localhost:8000/docs.
- `VITE_API_BASE_URL` được build vào frontend. Nếu người dùng truy cập từ máy khác, đặt `VITE_API_BASE_URL=http://<ip-may-chu>:8000/api/v1` trong `./.env` rồi build lại.
- Không đặt `SHARED_REDIS_URL` thì mọi service dùng container `redis` nội bộ (không cho Colab tham gia).
- Model (Whisper, CLIP, YOLO, HF) cache ở `./cache`, chỉ tải lần đầu.

## 4. Chạy worker trên Colab

Colab không chạy được Docker, nên notebook cài cùng các file requirements với image Docker vào một venv Python 3.11.

1. Push code lên GitHub (notebook clone nhánh `feat/update-project`; sửa `BRANCH` nếu khác).
2. Mở `ai_workers/colab_gpu_worker.ipynb` trên Colab, chọn runtime GPU.
3. Thêm Colab Secrets theo bảng trong notebook (`SHARED_REDIS_URL`, `CF_R2_*`, `GROQ_API_KEY`, tùy chọn `OPENROUTER_API_KEY`, `HF_TOKEN`, `GITHUB_TOKEN`).
4. Chạy lần lượt các cell. Cell cuối chạy worker và in log; khi thấy `colab@... ready.` là website bắt đầu gửi job sang Colab.
5. Dừng cell cuối (hoặc runtime hết hạn) → job mới tự quay về worker CPU.

Cấu hình trên Colab: `FLORENCE_DEVICE=cuda` (caption nhanh hơn, nhưng token không giống hệt bản CPU, xem `docs/florence-2-cpu-reproducibility.md`), `SEMANTIC_CLIP_DEVICE=cuda`, Faster-Whisper float16 trên GPU. PaddleOCR vẫn chạy CPU vì gói `paddlepaddle` GPU cần cuDNN 8, xung đột với torch; OCR chỉ chạy trên vài keyframe nên ảnh hưởng nhỏ.

## 5. Giới hạn cần biết

- Colab Pro: session tối đa 12 giờ, có idle timeout, GPU không được bảo đảm. Cần mở lại notebook mỗi session; cài đặt mất vài phút.
- Điều khoản Colab cấm chạy distributed worker trên runtime free không còn compute unit. Gói Pro có compute unit thì được.
- YouTube thường chặn tải từ IP datacenter (Colab). Video YouTube có thể lỗi trên Colab; upload file (đi qua R2) ổn định hơn.
- Checkpoint (`storage/checkpoints/`) nằm trên đĩa Colab, mất khi session kết thúc.
- Worker Colab chỉ nghe queue `gpu`; vẫn cần worker local nghe `celery` cho RAG indexing và job CPU.
