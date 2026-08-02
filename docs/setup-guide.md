# Hướng dẫn chạy dự án trên Windows và Linux

Tài liệu này hướng dẫn cài mới, cập nhật dependency và chạy **Multimodal Lecture Summarizer**
trên Windows, Linux hoặc Docker. Thực hiện lệnh từ thư mục gốc có `run.bat`, `run.sh`,
`backend/`, `ai_workers/` và `frontend/`.

## 1. Yêu cầu chung

| Thành phần | Phiên bản hoặc yêu cầu |
|---|---|
| Git và Git LFS | Bắt buộc để tải checkpoint Florence-2 |
| Python | 3.10 hoặc 3.11 |
| Node.js và npm | Node.js LTS |
| Redis | Cổng `6379` nếu chạy local |
| PostgreSQL | Theo `DATABASE_URL` nếu không dùng SQLite |
| Docker | Chỉ cần khi chạy bằng container |
| NVIDIA driver/Container Toolkit | Chỉ cần khi chủ động chạy GPU |

> [!IMPORTANT]
> Florence-2 mặc định chạy CPU/FP32/eager trên mọi máy. Đây là chế độ chuẩn để giữ kết quả ổn định
> giữa Windows và Linux. Python 3.12 trở lên chưa nằm trong runtime đã xác minh.

## 2. Chuẩn bị chung

### 2.1. Tải đầy đủ model

Sau khi clone hoặc pull repository:

```bash
git lfs install
git lfs pull
```

Nếu `model.safetensors` chỉ là Git LFS pointer, AI Worker sẽ từ chối khởi động.

### 2.2. Tạo file môi trường

**Windows PowerShell:**

```powershell
Copy-Item backend\.env.example backend\.env
```

**Linux Bash:**

```bash
cp backend/.env.example backend/.env
```

Cấu hình `backend/.env` theo môi trường thực tế:

- `DATABASE_URL`
- `CELERY_BROKER_URL` và `CELERY_RESULT_BACKEND`
- Thông tin Cloudflare R2 nếu lưu video trên R2
- API key của LLM nếu chạy tóm tắt hoặc hỏi đáp

Khi chạy Docker, tạo thêm file `.env` ở thư mục gốc từ `.env.example`.

## 3. Chạy trên Windows

### 3.1. Dùng launcher PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_win.ps1
```

Trong menu:

1. Chọn **5 - Install / Update Dependencies**.
2. Chờ cài xong Python và frontend dependencies.
3. Chọn **1 - Local Dev Mode**.
4. Giữ các cửa sổ Backend, Celery Worker và Frontend đang mở.

Windows luôn chạy lại cả hai file requirements:

```powershell
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt
backend\.venv\Scripts\pip.exe install -r ai_workers\requirements.txt
```

Do `ai_workers/requirements.txt` khóa `transformers==4.57.6`, pip sẽ tự nâng hoặc hạ
Transformers về đúng phiên bản nếu quá trình cài thành công.

### 3.2. Dùng launcher CMD

```powershell
.\run.bat
```

Chọn **5 - Install / Update Dependencies**, sau đó chọn **1 - Local Dev Mode**.

### 3.3. Cài và chạy thủ công

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install --upgrade pip
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
backend\.venv\Scripts\python.exe -m pip install -r ai_workers\requirements.txt
npm --prefix frontend install
```

Mở ba PowerShell riêng.

**Backend:**

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

**AI Worker, chạy từ thư mục gốc:**

```powershell
$env:PYTHONPATH="."
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
backend\.venv\Scripts\python.exe -m celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1
```

**Frontend:**

```powershell
Set-Location frontend
npm run dev
```

## 4. Chạy trên Linux

### 4.1. Dùng launcher

```bash
chmod +x run.sh
./run.sh
```

Chọn **4 - Install / Update Dependencies** khi cài lần đầu, sau đó chọn
**1 - Local Dev Mode**.

Trên Linux desktop có giao diện đồ họa, launcher tự mở ba terminal riêng:

- `MLS_Backend_API`
- `MLS_Celery_Worker`
- `MLS_Frontend`

Launcher hỗ trợ `gnome-terminal` và `x-terminal-emulator`. Khi chạy qua SSH, trên server
headless hoặc không tìm thấy terminal tương thích, ba dịch vụ chạy nền trong terminal hiện tại và
log được gộp lại. Có thể dùng các lệnh thủ công tại mục 4.3 nếu muốn tự quản lý từng terminal.

Tùy chọn **4 - Install / Update Dependencies** luôn chạy lại hai file requirements và
`npm install`, kể cả khi `backend/.venv` và `frontend/node_modules` đã tồn tại. Có thể gọi
trực tiếp cùng chức năng bằng:

```bash
./run.sh --install
```

### 4.2. Cài mới hoặc cập nhật dependency

```bash
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/python -m pip install -r backend/requirements.txt
backend/.venv/bin/python -m pip install -r ai_workers/requirements.txt
npm --prefix frontend install
```

Nếu `backend/.venv` đã tồn tại, bỏ qua lệnh tạo venv nhưng vẫn chạy lại hai lệnh
`pip install -r`. Nếu cần, thay `python3` bằng `python3.10` hoặc `python3.11`.

### 4.3. Chạy thủ công

Mở ba terminal riêng.

**Backend:**

```bash
cd backend
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

**AI Worker, chạy từ thư mục gốc:**

```bash
export PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
backend/.venv/bin/python -m celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1
```

**Frontend:**

```bash
cd frontend
npm run dev
```

## 5. Kiểm tra sau khi cài

**Windows:**

```powershell
backend\.venv\Scripts\python.exe -m pip show transformers
backend\.venv\Scripts\python.exe -m pip check
backend\.venv\Scripts\python.exe -m unittest ai_workers.tests.test_florence_runtime -v
```

**Linux:**

```bash
backend/.venv/bin/python -m pip show transformers
backend/.venv/bin/python -m pip check
backend/.venv/bin/python -m unittest ai_workers.tests.test_florence_runtime -v
```

Transformers phải có đúng phiên bản:

```text
Version: 4.57.6
```

Worker hợp lệ phải in:

```text
[Startup] Florence-2 runtime verified: cpu/float32/eager, all asset SHA-256 checks OK.
```

Nếu không có dòng này, chưa gửi video vào hàng đợi.

## 6. Chạy bằng Docker

### CPU mặc định

```bash
docker compose up --build
```

Không cần NVIDIA GPU; Florence-2 dùng `FLORENCE_DEVICE=cpu`.

### Cho phép container truy cập GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Overlay cấp GPU cho các stage tăng tốc. Florence-2 vẫn dùng CPU nếu không đặt rõ
`FLORENCE_DEVICE=cuda`. CUDA dùng FP32 và deterministic settings nhưng không được cam kết giống
CPU từng token trên mọi GPU/driver.

## 7. Địa chỉ dịch vụ

| Dịch vụ | Địa chỉ |
|---|---|
| Frontend | URL do Vite in ra, thường là `http://localhost:5173` |
| Backend API | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Redis | `localhost:6379` |
| PostgreSQL | Theo `DATABASE_URL` |

## 8. Dừng dịch vụ

- Windows PowerShell launcher: chọn **2 - Stop All Local Services**.
- Linux: nhấn `Ctrl+C` tại terminal chạy `run.sh`.
- Docker: chạy `docker compose down`.

## 9. Reset dữ liệu thử nghiệm

> [!CAUTION]
> Reset có thể xóa video, kết quả xử lý, cache và dữ liệu R2 theo cấu hình hiện tại.

**Windows:**

```powershell
Set-Location backend
.\.venv\Scripts\python.exe reset_r2_and_db.py
```

**Linux:**

```bash
cd backend
.venv/bin/python reset_r2_and_db.py
```

## 10. Xử lý lỗi thường gặp

### Transformers sai phiên bản

Cài lại requirements rồi khởi động lại Celery.

**Windows:**

```powershell
backend\.venv\Scripts\python.exe -m pip install -r ai_workers\requirements.txt
```

**Linux:**

```bash
backend/.venv/bin/python -m pip install -r ai_workers/requirements.txt
```

Nếu pip báo conflict, không dùng `--no-deps` để bỏ qua. Xử lý conflict rồi chạy lại `pip check`.

### Thiếu hoặc sai Florence-2 asset

```bash
git lfs pull
```

Nếu vẫn sai SHA-256, khôi phục đúng model checkout thay vì sửa checksum thủ công.

### Worker không kết nối Redis

Kiểm tra Redis và `CELERY_BROKER_URL`. Giá trị local mặc định:

```text
redis://localhost:6379/0
```

### Đã cập nhật package nhưng worker vẫn dùng code cũ

Dừng hoàn toàn Celery Worker rồi khởi động lại. Worker đang chạy không tự nạp lại package Python.

## 11. Tài liệu liên quan

- `docs/florence-2-cpu-reproducibility.md`
- `docs/ARCHITECTURE.md`
- `docs/BENCHMARK.md`
