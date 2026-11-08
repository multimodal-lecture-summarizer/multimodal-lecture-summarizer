# HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN MULTIMODAL LECTURE SUMMARIZER TỪ A ĐẾN Z

> Tài liệu này được biên soạn đầy đủ và chi tiết nhất dành cho bất kỳ ai mới pull mã nguồn về máy. Chỉ cần làm theo từng bước dưới đây, bạn sẽ khởi chạy được toàn bộ hệ thống (Frontend, Backend API, AI Worker Pipeline) một cách trơn tru mà không gặp phải bất kỳ trở ngại nào.

---

## 1. TỔNG QUAN KIẾN TRÚC DỰ ÁN

Hệ thống gồm 3 thành phần chính hoạt động đồng bộ:
1. **Frontend (React + Vite + Tailwind/Vanilla CSS):** Giao diện người dùng xem video, timeline, đối tượng phân tích, tóm tắt bài giảng và hỏi đáp chatbot. (Mặc định: `http://localhost:5173`).
2. **Backend API (FastAPI + SQLAlchemy + Alembic):** Quản lý xác thực, API upload, quản lý job, lịch sử và database. (Mặc định: `http://localhost:8000`).
3. **AI Worker (Celery + PyTorch + WhisperX + Florence-2 + YOLOv8 + PaddleOCR + LLM):** Tiến trình xử lý nền các tác vụ thị giác máy tính, bóc tách âm thanh, nhận diện slide, trích xuất thực thể và tóm tắt đa phương thức.
4. **Dịch vụ hỗ trợ:** **Redis** (Message Broker & Result Backend cho Celery) và **PostgreSQL** (hoặc tự động fallback **SQLite** nếu không có Postgres).

---

## 2. CÁC PHẦN MỀM & MÔI TRƯỜNG BẮT BUỘC TRÊN MÁY BẠN (PREREQUISITES)

Trước khi bắt đầu, máy tính của bạn **bắt buộc** phải tự cài đặt các công cụ sau:

### ⚠️ 2.1. Python 3.10 hoặc Python 3.11 (QUAN TRỌNG NHẤT)
- **Yêu cầu:** Bắt buộc dùng **Python 3.10** hoặc **3.11**.
- **Cảnh báo đỏ:** **TUYỆT ĐỐI KHÔNG DÙNG Python 3.12 hoặc 3.13**. Mã nguồn Florence-2 và PaddlePaddle sẽ kích hoạt runtime check và từ chối khởi động nếu phát hiện Python >= 3.12.
- Kiểm tra phiên bản máy bạn:
  ```bash
  python --version
  ```

### ⚠️ 2.2. Git và Git LFS (Large File Storage)
- Dự án sử dụng mô hình AI thị giác Florence-2 nặng ~463 MB lưu trữ trực tiếp trong repo thông qua Git LFS.
- Nếu không có Git LFS, file weights chỉ là file text rỗng (~130 bytes) và AI Worker sẽ báo lỗi SHA-256 ngay khi khởi động.
- Cài Git LFS:
  - **Windows:** Đã tích hợp sẵn khi cài [Git for Windows](https://git-scm.com/).
  - **Linux (Ubuntu/Debian):** `sudo apt install git-lfs`
  - **macOS:** `brew install git-lfs`

### ⚠️ 2.3. FFmpeg (Bắt buộc trong System PATH)
- Module bóc tách âm thanh từ video (`ai_workers/modules/audio_v2/transcriber.py`) gọi trực tiếp lệnh `ffmpeg` của hệ điều hành.
- **Cách cài đặt:**
  - **Windows (PowerShell với quyền Admin):**
    ```powershell
    winget install Gyan.FFmpeg
    # hoặc dùng Chocolatey:
    choco install ffmpeg
    ```
    *(Sau khi cài, mở cửa sổ terminal mới và gõ `ffmpeg -version` để chắc chắn lệnh đã nhận trong PATH).*
  - **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update && sudo apt install -y ffmpeg
    ```
  - **macOS:**
    ```bash
    brew install ffmpeg
    ```

### 2.4. Node.js và npm
- Cần **Node.js LTS (v18.x hoặc v20.x trở lên)** để chạy Frontend.
- Kiểm tra: `node -v` và `npm -v`.

### 2.5. Redis Server (Cổng 6379)
- Celery bắt buộc cần Redis để lắng nghe hàng đợi tác vụ.
- **Cách chạy Redis đơn giản nhất bằng Docker (Khuyên dùng):**
  ```bash
  docker run -d --name mls-redis -p 6379:6379 redis:alpine
  ```
- **Nếu không dùng Docker:**
  - **Windows:** Tải bản Redis Windows (Memurai hoặc [Redis-x64 MSI](https://github.com/tporadowski/redis/releases)) và start service.
  - **Linux:** `sudo apt install -y redis-server && sudo systemctl start redis`

### 2.6. Card đồ họa NVIDIA GPU (Không bắt buộc)
- Dự án **mặc định chạy CPU** rất ổn định, bất kể máy không có GPU vẫn chạy tốt.
- Nếu máy có GPU NVIDIA (VRAM >= 4GB) và muốn tăng tốc, bạn chỉ cần cài **NVIDIA Driver** + **CUDA Toolkit** (11.8 hoặc 12.x) và đặt `FLORENCE_DEVICE=cuda` trong file cấu hình.

---

## 3. CẤU HÌNH FILE MÔI TRƯỜNG (`.env`)

Người chia sẻ dự án sẽ gửi cho bạn file chứa các API Key và credentials bảo mật. Bạn cần đặt file này đúng vị trí:

📁 **Vị trí đặt file:** `backend/.env`

*(Nếu bạn chạy bằng Docker Compose toàn bộ, hãy copy thêm một bản thành `.env` ngay tại thư mục gốc của dự án).*

### Chi tiết các biến cấu hình cần lưu ý trong `backend/.env`:

| Nhóm biến | Biến cấu hình | Ý nghĩa & Lưu ý cho máy cá nhân |
|---|---|---|
| **Hệ thống** | `APP_ENV="development"` | Để chế độ phát triển để kích hoạt auto-reload & self-healing SQLite. |
| **Cơ sở dữ liệu** | `DATABASE_URL` | Chuỗi kết nối PostgreSQL (ví dụ Neon Cloud). **Lưu ý:** Nếu bạn không điền hoặc rớt mạng, hệ thống **tự động fallback sang SQLite (`backend/local.db`)** mà không làm crash app. |
| **Hàng đợi** | `CELERY_BROKER_URL` | `redis://localhost:6379/0` (Trỏ về Redis đang chạy ở cổng 6379). |
| | `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` |
| **Lưu trữ đám mây** | `CF_R2_*` | Thông tin tài khoản Cloudflare R2 để upload/stream video. (Được cấp sẵn trong file env). |
| **LLM Tóm tắt** | `OPENROUTER_API_KEY` | Key OpenRouter để gọi mô hình Qwen 2.5 / DeepSeek tóm tắt bài giảng. |
| | `GROQ_API_KEY` | Key dự phòng cực nhanh để hỗ trợ tóm tắt / trích xuất. |
| **AI Hardware** | `FLORENCE_DEVICE` | Đặt `cpu` nếu máy không có card NVIDIA rời. Đặt `cuda` nếu máy có card NVIDIA mạnh. |

---

## 4. QUY TRÌNH CÀI ĐẶT TỪNG BƯỚC (STEP-BY-STEP)

### BƯỚC 1: Clone mã nguồn và kéo weights mô hình (Git LFS)
Mở Terminal / PowerShell tại thư mục bạn muốn chứa dự án:
```bash
git clone <URL_CUA_REPO>
cd multimodal-lecture-summarizer

# Kích hoạt và kéo đầy đủ file weights mô hình Florence-2 (~463MB):
git lfs install
git lfs pull
```

🔍 **Cách kiểm tra file weights đã tải chuẩn chưa:**
- Xem file: `ai_workers/modules/visual_v2/florence2_vendor/model.safetensors`.
- Nếu dung lượng file **khoảng ~463 MB (463,221,266 bytes)**: ✅ **Đạt chuẩn**.
- Nếu dung lượng chỉ có **130 bytes**: ❌ Bạn chưa chạy `git lfs pull` (Hãy chạy lệnh trên lại ngay).

---

### BƯỚC 2: Tạo Virtual Environment và cài đặt Python dependencies
Bạn nên dùng chung 1 Virtual Environment đặt tại `backend/venv` cho cả Backend và AI Worker:

#### Trên Windows (PowerShell):
```powershell
# Tạo môi trường ảo với Python 3.10 hoặc 3.11
python -m venv backend\venv

# Nâng cấp pip
backend\venv\Scripts\python.exe -m pip install --upgrade pip

# Cài đặt thư viện Backend
backend\venv\Scripts\pip.exe install -r backend\requirements.txt

# Cài đặt thư viện AI Pipeline (WhisperX, PyTorch, Florence, OpenCV...)
backend\venv\Scripts\pip.exe install -r ai_workers\requirements.txt
```

> **Mẹo cài PyTorch hỗ trợ GPU NVIDIA (Windows):**
> Nếu muốn tận dụng GPU với CUDA 12.x, chạy lệnh sau:
> ```powershell
> backend\venv\Scripts\pip.exe install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
> ```

#### Trên Linux / macOS:
```bash
python3 -m venv backend/venv
backend/venv/bin/pip install --upgrade pip
backend/venv/bin/pip install -r backend/requirements.txt
backend/venv/bin/pip install -r ai_workers/requirements.txt
```

---

### BƯỚC 3: Cài đặt thư viện Frontend
```bash
cd frontend
npm install
cd ..
```

---

### BƯỚC 4: Khởi động Redis Server
Đảm bảo Redis đang hoạt động ở cổng `6379`:
```bash
# Bằng Docker (nhanh nhất):
docker run -d --name mls-redis -p 6379:6379 redis:alpine

# Kiểm tra Redis:
docker ps
```

---

## 5. HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG

Bạn có 2 cách chạy tùy sở thích:

### CÁCH 1: Khởi chạy 1-Click tự động (Khuyên dùng)

Dự án đã chuẩn bị sẵn launcher tự động phát hiện môi trường và mở đủ các cửa sổ:

- **Trên Windows (PowerShell):**
  ```powershell
  Set-ExecutionPolicy -Scope Process Bypass
  .\run_local.ps1
  ```
  *(Menu hiện ra, bạn chỉ cần bấm phím `1` - Local Dev Mode).*
  Hệ thống sẽ tự động bật 3 cửa sổ: Backend API, Celery Worker và Frontend Vite.

- **Trên Windows (CMD):**
  ```cmd
  run.bat
  ```

- **Trên Linux / macOS:**
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

---

### CÁCH 2: Khởi chạy thủ công từng dịch vụ (Dễ xem log / debug)

Mở 3 tab Terminal / PowerShell riêng biệt:

#### 🔹 Terminal 1: Backend API (FastAPI)
- **Windows:**
  ```powershell
  cd backend
  $env:PYTHONPATH=".."
  .\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
  ```
- **Linux / macOS:**
  ```bash
  cd backend
  export PYTHONPATH=".."
  ./venv/bin/python -m uvicorn app.main:app --reload --port 8000
  ```
*Backend chạy thành công khi thấy dòng: `Uvicorn running on http://127.0.0.1:8000`*

---

#### 🔹 Terminal 2: AI Celery Worker (Xử lý tác vụ AI)
- **Windows (Bắt buộc chạy từ thư mục gốc của dự án):**
  ```powershell
  $env:PYTHONPATH="."
  $env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
  .\backend\venv\Scripts\python.exe -m celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1
  ```
- **Linux / macOS:**
  ```bash
  export PYTHONPATH="."
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  ./backend/venv/bin/celery -A ai_workers.core.celery_app worker --loglevel=info --concurrency=1
  ```
*Worker sẵn sàng khi thấy in ra danh sách task: `ai_workers.tasks.process_video`, v.v.*

---

#### 🔹 Terminal 3: Frontend (React Vite)
```bash
cd frontend
npm run dev
```
*Frontend chạy tại: `http://localhost:5173`*

---

## 6. ĐỊA CHỈ TRUY CẬP & TÀI KHOẢN MẪU

Khi cả 3 dịch vụ đã chạy:
- **Giao diện Web:** [http://localhost:5173](http://localhost:5173)
- **Tài liệu API Swagger:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Tài khoản mặc định hệ thống tự khởi tạo (Seed Account):
Nếu bạn dùng cơ sở dữ liệu mới, hệ thống tự động tạo sẵn 2 tài khoản sau:
1. **Quản trị viên (Admin):**
   - **Email:** `hungphitran.22@gmail.com`
   - **Mật khẩu:** `AdminPass123@`
2. **Người dùng tiêu chuẩn (User):**
   - **Email:** `nguyen.van.a@gmail.com`
   - **Mật khẩu:** `UserPass123@`

---

## 7. TỔNG HỢP CÁC LỖI THƯỜNG GẶP & CÁCH XỬ LÝ (TROUBLESHOOTING)

### 🔴 Lỗi 1: `FlorenceRuntimeError: Asset 'model.safetensors' failed SHA-256 check`
- **Nguyên nhân:** File weights mô hình bị thiếu do clone bình thường mà không có Git LFS. File hiện tại chỉ là con trỏ text ~130 bytes.
- **Cách khắc phục:**
  ```bash
  git lfs install
  git lfs pull
  ```
  Kiểm tra xem file `ai_workers/modules/visual_v2/florence2_vendor/model.safetensors` đã đạt dung lượng ~463 MB chưa.

---

### 🔴 Lỗi 2: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
- **Nguyên nhân:** Máy tính của bạn chưa cài FFmpeg hoặc chưa thêm vào biến môi trường PATH.
- **Cách khắc phục:**
  - Cài FFmpeg theo mục **2.3** ở trên.
  - Sau khi cài xong, nhớ **tắt hết tất cả các terminal và mở lại** để hệ điều hành nhận diện lệnh `ffmpeg`.

---

### 🔴 Lỗi 3: `Consumer: Cannot connect to redis://localhost:6379/0: Error 10061 connecting to localhost:6379`
- **Nguyên nhân:** Redis server chưa được bật.
- **Cách khắc phục:** Chạy container Redis:
  ```bash
  docker run -d --name mls-redis -p 6379:6379 redis:alpine
  ```

---

### 🔴 Lỗi 4: `FlorenceRuntimeError: Python version (3, 12) is not supported`
- **Nguyên nhân:** Bạn đang chạy môi trường Python 3.12 hoặc mới hơn.
- **Cách khắc phục:** Cài đặt Python 3.10 hoặc 3.11, sau đó xóa thư mục `backend/venv` cũ đi và tạo lại venv bằng Python 3.10/3.11.

---

### 🔴 Lỗi 5: `CUDA out of memory` hoặc GPU báo tràn bộ nhớ
- **Nguyên nhân:** Card đồ họa không đủ VRAM hoặc mô hình chiếm nhiều bộ nhớ.
- **Cách khắc phục:** Chuyển sang chế độ CPU bằng cách sửa trong `backend/.env`:
  ```ini
  FLORENCE_DEVICE=cpu
  ```
  Sau đó khởi động lại Celery Worker.

---

### 🔴 Lỗi 6: Lỗi khi cài thư viện `lapx` (ByteTrack) trên Windows
- **Nguyên nhân:** `lapx` yêu cầu C++ Build Tools nếu không tìm thấy prebuilt binary wheel.
- **Cách khắc phục:**
  - Hãy chắc chắn bạn đang dùng đúng **Python 3.10 hoặc 3.11** 64-bit (phiên bản này luôn có sẵn prebuilt wheel trên PyPI).
  - Hoặc cài đặt gói [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

---

### 🔴 Lỗi 7: Backend báo kết nối PostgreSQL thất bại
- **Yên tâm:** Trong chế độ `APP_ENV="development"`, nếu không kết nối được PostgreSQL ở Neon, Backend sẽ **tự động fallback sang file SQLite cục bộ (`backend/local.db`)** và tự động khởi tạo bảng. Bạn vẫn có thể trải nghiệm toàn bộ tính năng bình thường.

---

Chúc bạn khởi chạy dự án thành công! Nếu gặp bất kỳ vấn đề gì phát sinh, hãy kiểm tra tab log của Celery Worker hoặc terminal Backend để xem thông báo chi tiết.
