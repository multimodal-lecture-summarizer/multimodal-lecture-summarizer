# Hướng Dẫn Sử Dụng Backend - Hệ Thống Tóm Tắt Video Ngắn AI

Tài liệu này hướng dẫn cách cài đặt, chạy thử và giải thích các thiết kế kiến trúc chính của Backend FastAPI được phát triển cho đề tài: **"Xây dựng hệ thống tóm tắt video ngắn dựa trên các kỹ thuật Trí Tuệ Nhân Tạo"**.

---

## 1. Cấu Trúc Thư Mục

Backend được thiết kế theo mô hình **Modular & Clean Architecture**, chia rõ ràng trách nhiệm của từng file/thư mục:

```text
backend/
├── app/
│   ├── core/
│   │   ├── config.py         # Đọc và quản lý cấu hình (pydantic-settings)
│   │   ├── constants.py      # Định nghĩa Enums, Error Codes và các hằng số hệ thống
│   │   ├── database.py       # Khởi tạo SQLAlchemy engine, session maker
│   │   └── exceptions.py     # Các custom exceptions (Auth, NotFound, Validation...)
│   ├── middleware/
│   │   ├── case_converter.py # Xử lý chuyển đổi camelCase <-> snake_case cho Query và models
│   │   └── exception_handler.py # Bắt lỗi toàn cục, format output thành BaseDTO chuẩn hóa
│   ├── models/               # SQLAlchemy Models ánh xạ với PostgreSQL tables
│   │   ├── __init__.py       # Export tập trung các models
│   │   ├── user.py           # Bảng users
│   │   ├── video.py          # Bảng videos và video_standards
│   │   ├── job.py            # Bảng jobs (quản lý tác vụ chạy nền)
│   │   ├── summary.py        # Bảng summaries (kết quả tóm tắt, chapter, keyframes)
│   │   ├── qa.py             # Bảng qa_logs (lịch sử hỏi đáp)
│   │   └── stats.py          # Bảng system_stats (thống kê tổng hợp)
│   ├── schemas/              # Pydantic Schemas (DTOs)
│   │   ├── __init__.py
│   │   ├── base.py           # BaseDTO (success, data, error, code, message, metadata)
│   │   ├── user.py           # DTO cho Authentication
│   │   ├── video.py          # DTO cho Video
│   │   ├── job.py            # DTO cho Jobs
│   │   ├── summary.py        # DTO cho Tóm tắt (Chapters, Keyframes)
│   │   ├── qa.py             # DTO cho RAG Q&A
│   │   └── stats.py          # DTO cho Báo cáo Admin
│   ├── services/             # Integrations với các dịch vụ AI / Storage ngoài
│   │   ├── r2.py             # Cloudflare R2 (Boto3 Client) có mock dự phòng
│   │   ├── groq.py           # Groq API Client có mock dự phòng
│   │   └── chromadb.py       # ChromaDB Vector DB Client có mock dự phòng
│   ├── api/
│   │   ├── deps.py           # Dependency injection (Auth check, Get DB...)
│   │   ├── v1/               # Các API Endpoints
│   │   │   ├── auth.py       # Đăng ký, đăng nhập, JWT, OAuth2
│   │   │   ├── videos.py     # Upload video, duyệt theo standards
│   │   │   ├── jobs.py       # Truy vấn trạng thái jobs
│   │   │   ├── summaries.py  # Lấy tóm tắt, export TXT/SRT/PDF
│   │   │   ├── qa.py         # Hỏi đáp RAG
│   │   │   └── stats.py      # Báo cáo thống kê Admin
│   │   └── router.py         # Định tuyến Router v1
│   └── main.py               # Entry point chính của ứng dụng
├── requirements.txt          # Các thư viện phụ thuộc
└── .env.example              # Biến môi trường mẫu
```

---

## 2. Các Cơ Chế Kỹ Thuật Đặc Biệt

### A. Tự động Chuyển Đổi CamelCase <-> snake_case
* **Vấn đề**: Backend Python sử dụng chuẩn `snake_case` (ví dụ: `original_url`), trong khi Frontend ReactJS giao tiếp qua chuẩn `camelCase` (ví dụ: `originalUrl`).
* **Giải pháp**:
  1. **Request/Response Body**: Sử dụng `CamelModel` làm lớp cơ sở cho tất cả DTOs. Lớp này kế thừa `BaseModel` của Pydantic v2 và cấu hình:
     ```python
     model_config = ConfigDict(
         alias_generator=to_camel,
         populate_by_name=True,
         from_attributes=True
     )
     ```
     Điều này giúp Pydantic tự động serialize thuộc tính `snake_case` thành key `camelCase` khi trả về response, và tự động parse key `camelCase` từ client gửi lên thành thuộc tính `snake_case`.
  2. **Query Parameters**: Tạo custom `CamelCaseAPIRoute` kế thừa từ `APIRoute` để bắt các query parameters dạng `camelCase` từ client (ví dụ `?maxDuration=3600`) và tự động chuyển đổi thành `snake_case` trước khi truyền vào hàm xử lý của router.

### B. BaseDTO Response & Error Handler
* Mọi API response đều được bọc trong class `BaseDTO` chứa cấu trúc chuẩn:
  ```json
  {
    "success": true,
    "data": { ... },
    "error": null,
    "code": 200,
    "message": "Success",
    "metadata": null
  }
  ```
* Hệ thống có bộ **Exception Handler** toàn cục để bắt mọi lỗi:
  * **Lỗi Nghiệp vụ (`AppException`)**: Trả về `success: false` kèm mã lỗi chi tiết.
  * **Lỗi Validate Dữ liệu (`RequestValidationError`)**: Trích xuất chi tiết trường bị lỗi, đổi tên trường bị lỗi từ `snake_case` sang `camelCase` để frontend dễ hiển thị lỗi cho người dùng.
  * **Lỗi Hệ thống không xác định (`Exception`)**: Bắt lỗi và in log stacktrace ra console. Nếu đang ở chế độ `DEBUG=True`, trả về đầy đủ stacktrace và loại lỗi về client để phục vụ debug nhanh, tránh tình trạng hiển thị lỗi thô sơ "Internal Server Error" không rõ nguyên nhân.

### C. Dữ liệu Gán Trực tiếp (Direct Mapping)
* Pydantic v2 hỗ trợ `from_attributes=True`. Vì thế, dữ liệu từ SQLAlchemy Models có thể được mapping trực tiếp sang DTO bằng phương thức:
  ```python
  UserDTO.model_validate(db_user)
  ```
  Không cần phải gán thủ công từng trường (`email=db_user.email`, `role=db_user.role`...), giúp tăng tốc độ phát triển và giảm thiểu sai sót.

### D. Cơ Chế Mock Dự Phòng (External API Fallback)
* Nếu bạn chưa cấu hình Postgres, ChromaDB, Cloudflare R2 hay Groq API, hệ thống sẽ **tự động chuyển sang chế độ Mock** tại chỗ:
  * **PostgreSQL**: Nếu không cấu hình, FastAPI sẽ dùng SQLite memory DB hoặc log lỗi chi tiết (bạn có thể thay URI kết nối Postgres trong `.env` để kiểm tra).
  * **Cloudflare R2**: Tự động lưu file video/keyframes vào thư mục `storage/mock_r2_bucket` của backend và sinh URL tĩnh để frontend có thể hiển thị ảnh keyframe bình thường.
  * **ChromaDB**: Nếu không connect được, backend dùng một dictionary in-memory hoạt động như một vector database thu nhỏ để lưu chunk transcript và tìm kiếm keyword phục vụ tính năng RAG.
  * **Groq API**: Nếu không cung cấp key, service sẽ tự động sinh tóm tắt hoặc câu trả lời Q&A giả lập chất lượng cao dựa theo từ khóa trong câu hỏi của người dùng.

---

## 3. Hướng Dẫn Cài Đặt & Chạy Thử

### Bước 1: Tạo môi trường ảo & cài đặt thư viện
Chạy các lệnh sau trong thư mục `backend/`:
```bash
# Tạo môi trường ảo python
python -m venv .venv

# Kích hoạt môi trường ảo (Windows)
.venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux/macOS)
source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Cấu hình biến môi trường
1. Sao chép file `.env.example` thành `.env`:
   ```bash
   cp .env.example .env
   ```
2. Cập nhật các thông số kết nối PostgreSQL hoặc giữ nguyên mặc định nếu bạn muốn test chế độ debug với database local của mình.

Chạy lệnh uvicorn (giới hạn watch directory để tránh vòng lặp reload khi tạo/sửa đổi file static trong thư mục `storage`):
```bash
uvicorn app.main:app --reload --reload-dir app
```

Server sẽ khởi chạy tại: `http://127.0.0.1:8000`.

---

## 4. Các Endpoints API Phục Vụ Đồ Án

Sau khi chạy server, bạn hãy mở trình duyệt truy cập `http://127.0.0.1:8000/docs` để xem tài liệu Swagger trực quan và đầy đủ:

1. **Authentication (`/api/v1/auth`)**:
   * `POST /register`: Đăng ký người dùng mới.
   * `POST /login`: Đăng nhập lấy JWT Access Token (JSON request).
   * `POST /login/oauth2`: Đăng nhập dạng OAuth2 (Dùng cho Swagger UI authenticate).
   * `GET /me`: Lấy thông tin profile hiện tại.
2. **Videos (`/api/v1/videos`)**:
   * `POST /upload`: Kéo thả file video hoặc nhập URL YouTube. Tự động kiểm duyệt video dựa trên bảng cấu hình standards.
   * `GET /`: Lấy danh sách video của user.
   * `GET /standards`: Lấy cấu hình kiểm duyệt video.
   * `PUT /standards`: Cập nhật cấu hình video standards (Chỉ Admin).
   * `GET /{video_id}`: Lấy thông tin trạng thái video.
3. **Jobs (`/api/v1/jobs`)**:
   * `GET /video/{video_id}`: Lấy tiến trình xử lý AI của video.
   * `GET /{job_id}`: Lấy chi tiết tác vụ xử lý nền.
4. **Summaries (`/api/v1/summaries`)**:
   * `GET /video/{video_id}`: Trả về văn bản tóm tắt, chapters (timestamp), gallery keyframes (mô tả ngữ nghĩa).
   * `GET /video/{video_id}/export`: Tải file tóm tắt dưới dạng TXT, SRT (subtitles), hoặc PDF (report).
5. **Interactive Q&A (`/api/v1/qa`)**:
   * `POST /video/{video_id}`: Cho phép đặt câu hỏi về nội dung bài giảng. Hệ thống tự động truy vấn ChromaDB lấy 3 đoạn transcript gần nhất, đẩy vào Groq sinh câu trả lời RAG và log lịch sử.
6. **Admin Dashboard (`/api/v1/stats`)**:
   * `GET /dashboard`: Thống kê người dùng theo thời gian, tỷ lệ job thành công/thất bại, thời gian xử lý và chi phí API (chỉ Admin).
