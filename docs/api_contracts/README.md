# API Contracts

Định nghĩa JSON schema cho giao tiếp Frontend ↔ Backend.

## Endpoints

### POST /api/upload
Upload video file hoặc YouTube URL.

**Request:** `multipart/form-data`
- `file` (optional): Video file (MP4, AVI, MKV)
- `youtube_url` (optional): YouTube URL
- `config_stack`: "hybrid" | "local_gpu" | "api"

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued"
}
```

### GET /api/videos/{video_id}/status
Kiểm tra trạng thái xử lý.

**Response:**
```json
{
  "video_id": "uuid-string",
  "status": "processing",
  "progress": 45,
  "current_stage": "audio"
}
```

### GET /api/videos/{video_id}/results
Lấy kết quả đầy đủ.

**Response:**
```json
{
  "video_id": "uuid-string",
  "transcript": [...],
  "summary": "...",
  "chapters": [...],
  "keyframes": [...]
}
```

### POST /api/videos/{video_id}/qa
Hỏi đáp RAG.

**Request:**
```json
{
  "question": "Transformer hoạt động thế nào?"
}
```

**Response:**
```json
{
  "answer": "...",
  "references": [{"timestamp": "15:20", "text": "..."}],
  "question": "..."
}
```

### WebSocket /ws/videos/{video_id}/progress
Real-time progress updates.

**Message format:**
```json
{
  "stage": "audio",
  "progress": 45,
  "message": "Đang trích xuất WhisperX..."
}
```
