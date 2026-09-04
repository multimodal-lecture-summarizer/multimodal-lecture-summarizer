---
phase: 1
title: "Model Architecture & Fusion Refactoring"
status: completed
priority: P1
dependencies: []
---

# Phase 1: Model Architecture & Fusion Refactoring

## 1. Overview

Tái cấu trúc 3 mô hình hạt nhân trong `benchmarks/models/` (`chaptering.py`, `summarization.py`, `retrieval_qa.py`) bằng cách chuyển hóa trực tiếp các thuật toán tối ưu hóa đã được kiểm chứng từ `ai_workers` và `backend`. 

Giai đoạn này loại bỏ triệt để các nguyên nhân kỹ thuật khiến $C_5$ bị phạt WindowDiff nặng nề ($0.3217$), đưa $C_5$ trở thành mô hình có WindowDiff tốt nhất toàn bảng ($\le 0.0950$), chuẩn hóa cơ chế trích dẫn chứng cứ ngữ nghĩa cho $S_3\text{+ev}$, và đưa thuật toán RRF chuẩn mực vào hệ thống RAG.

---

## 2. Requirements

### Functional Requirements:
1. **$C_5$ Decoupled Cross-Attention [Red Team Finding 1]:**
   - Tách biệt luồng xử lý: Loại bỏ hoàn toàn phép cộng trung bình thô `torch.stack().mean()` tại `chaptering.py:381`.
   - Sử dụng đặc trưng thị giác (Visual Stream từ DINOv2 ViT-S/14, 384d) làm **Query Anchor**, trong khi Transcript, OCR và Acoustic đóng vai trò **Keys & Values** qua `nn.MultiheadAttention`.
2. **Acoustic Dimension Correction [Red Team Finding 2]:**
   - Sửa tham số khởi tạo `d_ac: int = 32` tại `chaptering.py:314` (thay cho giá trị mặc định cũ `64`) để khớp hoàn toàn với tensor thực tế `[T, 32]` trong `benchmarks/data/cached_features/*.pt`.
3. **Visual-Anchor Snapping & Dynamic Thresholding [Red Team Finding 4]:**
   - **Cơ chế trích xuất mốc chuyển slide từ `visual_features`:** Trong ngữ cảnh dữ liệu cached không có mốc thời gian slide thô, hệ thống tính khoảng cách cosine giữa các vector đặc trưng thị giác liên tiếp:
     $$\Delta v_t = 1 - \frac{\mathbf{v}_t \cdot \mathbf{v}_{t-1}}{\|\mathbf{v}_t\|_2 \|\mathbf{v}_{t-1}\|_2}, \quad t \in [1, T-1]$$
     Điểm chuyển cảnh thị giác $\mathcal{S}_{\text{visual}}$ được xác định tại các thời điểm $t$ có $\Delta v_t > \tau_v$ với ngưỡng thích ứng $\tau_v = \max\big(0.20, \mu(\Delta v) + 0.5\sigma(\Delta v)\big)$, hoặc tại các đỉnh cực đại cục bộ từ đầu dò thị giác $C_3$.
   - **Quy tắc bắt dính (Snapping):** Kế thừa từ [`ai_workers/modules/fusion/timeline.py:196-218`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/ai_workers/modules/fusion/timeline.py#L196-L218): Nếu ranh giới dự đoán $t_i$ nằm trong bán kính $\delta = 45\text{s}$ so với một điểm $s_j \in \mathcal{S}_{\text{visual}}$, bắt dính ranh giới: $b_i^* = \arg\min_{s_j} |t_i - s_j|$.
   - **Silent Intro Guard:** Tích hợp kiểm tra từ `backend/app/api/v1/qa.py:118-120`: Tuyệt đối không snap ranh giới chương 1 về $0.0\text{s}$ nếu câu thoại đầu tiên bắt đầu sau $0.0\text{s}$ ($t_{\text{first\_sent}} > 0.0$), tránh nuốt phần slide intro/nhạc dạo im lặng vào nội dung bài giảng.
   - Thay thế ngưỡng tĩnh $0.40$ bằng ngưỡng thích ứng trung bình trượt $\tau = \mu_{\text{sim}} - 1.0 \cdot \sigma_{\text{sim}}$.
   - Giảm hệ số phạt mẫu dương `pos_weight` trong BCEWithLogitsLoss từ $15.0 \to 4.0$ để chấm dứt hiện tượng sinh ranh giới giả (spurious over-segmentation).
4. **Semantic Evidence Grounding & Tensor Fallback ($S_3\text{+ev}$ / $S_4$) [Red Team Finding 5]:**
   - Thay thế heuristic thô "lấy 5 từ đầu của OCR" bằng phép đo độ tương đồng ngữ nghĩa Cosine sử dụng SBERT (`all-MiniLM-L6-v2`) giữa từng mệnh đề tóm tắt và văn bản OCR slide.
   - **Cơ chế Tensor Fallback trên dữ liệu cached:** Trong trường hợp tệp `.pt` chỉ chứa `ocr_features` dạng tensor `[T, 384]` (đã được nhúng sẵn từ văn bản slide/cụm từ nổi bật), hệ thống cho phép tính cosine similarity trực tiếp giữa embedding câu tóm tắt $e(c) \in \mathbb{R}^{384}$ và vector đặc trưng `ocr_features[k]`:
     $$\text{Score}(c, k) = \frac{e(c) \cdot \mathbf{f}_{\text{ocr}, k}}{\|e(c)\|_2 \|\mathbf{f}_{\text{ocr}, k}\|_2}$$
     Một mệnh đề được coi là có chứng cứ xác thực (grounded) khi $\max_k \text{Score}(c, k) \ge \tau_{\text{ev}} = 0.45$.
   - Xử lý chia khối batch ($\le 32$ câu) khi tính toán cosine similarity và gọi `torch.cuda.empty_cache()` định kỳ để tránh tràn VRAM trên GPU Tesla T4 (16GB).
   - Tích hợp hàm chấm điểm đa yếu tố từ [`ai_workers/modules/fusion/quality_postprocess.py:58-75`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/ai_workers/modules/fusion/quality_postprocess.py#L58-L75), giới hạn độ dài OCR $\le 120$ ký tự để slide mục lục/tài liệu tham khảo không lấn át slide công thức/khái niệm.
5. **Reciprocal Rank Fusion (RRF) cho RAG:**
   - Cập nhật [`benchmarks/models/retrieval_qa.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/benchmarks/models/retrieval_qa.py): Chuyển đổi công thức cộng tuyến tính chưa chuẩn hóa ($0.60\cos + 0.40\text{BM25}$) sang thuật toán xếp hạng chuẩn:
     $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{BM25}}(d)}$$
6. **Lưu trữ C5 Checkpoint phục vụ RQ3:**
   - Xuất trọng số tối ưu của mô hình $C_5$ tốt nhất sau quá trình huấn luyện/đánh giá vào tệp `checkpoints/c5_real.pt` để pipeline RQ3 (`run_rq3_benchmark.py`) nạp trực tiếp mà không gặp lỗi `FileNotFoundError`.

### Non-Functional Requirements:
- Giữ nguyên ngân sách phần cứng $D\text{-}T08$: Hoạt động hoàn toàn trong $\le 16\text{GB}$ VRAM trên NVIDIA Tesla T4.
- Đảm bảo 100% tương thích ngược với dữ liệu thực tại `benchmarks/data/cached_features/*.pt`.

---

## 3. Mathematical & Architectural Formalization

### 3.1. $C_5$ Cross-Attention with Visual Anchor Snapping

```
[Visual DINOv2]    ──► Proj (384->256) ──► Query (Q_v) ┐
[Transcript Text]  ──► Proj (384->256) ─┐               ├─► Multi-Head Cross-Attn
[Slide OCR]        ──► Proj (384->256) ─┼─► Keys/Values ┘   (4 heads, 256 hidden)
[Acoustic Whisper] ──► Proj (32->256)  ─┘                         │
                                                                  ▼
                                                       Temporal Transformer Block
                                                                  │
                                                                  ▼
                                                       Boundary Probability Head
                                                                  │
                                                                  ▼
                                                       Dynamic Moving-Avg NMS
                                                                  │
                                                                  ▼
                                                       Visual-Anchor Snapping (±45s)
                                                       (Tích hợp Silent Intro Guard)
                                                                  │
                                                                  ▼
                                                       Final Segmented Chapters
```

Toán học của phân đoạn:
1. Xác suất ranh giới thô: $p_t = \sigma(W_b \cdot h_t + b_b)$.
2. Lọc đỉnh cục bộ thích ứng: $\hat{B} = \{t \mid p_t > \mu(p) + k\sigma(p) \land p_t = \max_{i \in [t-w, t+w]} p_i\}$.
3. Bắt dính thị giác:
   $$b_i^* = \begin{cases} s_j & \text{nếu } \exists s_j \in \mathcal{S}_{\text{visual}} \text{ sao cho } |t_i - s_j| \le \delta \land (t_i > 0 \lor t_{\text{first\_sent}} = 0) \\ t_i & \text{ngược lại} \end{cases}$$

### 3.2. Evidence Grounding Alignment

Với mỗi câu $c$ trong bản tóm tắt:
- Khi có văn bản OCR thô:
  $$\text{Score}(c, \text{slide}_k) = \cos\big(e(c), e(\text{ocr}_k)\big) \cdot \left(\frac{\min(\text{len}(\text{ocr}_k), 120)}{120}\right) \cdot \mathbb{I}(\text{time}(c) \cap \text{time}(\text{slide}_k) \ne \emptyset)$$
- Khi chạy trên tensor đặc trưng đã trích xuất:
  $$\text{Score}(c, k) = \frac{e(c) \cdot \mathbf{f}_{\text{ocr}, k}}{\|e(c)\|_2 \|\mathbf{f}_{\text{ocr}, k}\|_2}$$

---

## 4. Related Code Files

- **Modify:** [`benchmarks/models/chaptering.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/benchmarks/models/chaptering.py)
- **Modify:** [`benchmarks/models/summarization.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/benchmarks/models/summarization.py)
- **Modify:** [`benchmarks/models/retrieval_qa.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/benchmarks/models/retrieval_qa.py)
- **Export Checkpoint:** `checkpoints/c5_real.pt`
- **Verify:** [`tests/test_validation_gate.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/tests/test_validation_gate.py)

---

## 5. Implementation Steps

1. **Refactor $C_5$ trong `benchmarks/models/chaptering.py`:**
   - Thay thế khối ghép đặc trưng phẳng bằng `DecoupledCrossAttentionBlock` (Visual làm Query, các modality khác làm Keys/Values).
   - Thêm hàm `extract_visual_transitions(visual_features, timestamps, threshold)` dựa trên cosine distance của khung hình liên tiếp.
   - Bổ sung hàm `apply_visual_snapping(predicted_boundaries, visual_transitions, window_sec=45.0, first_sentence_time=0.0)`.
   - Cập nhật tham số khởi tạo $d_{\text{ac}} = 32$ và hạ `pos_weight = 4.0`.
2. **Refactor Evidence Grounding trong `benchmarks/models/summarization.py`:**
   - Thay thế việc cắt 5 từ OCR bằng việc nhúng vector `MiniLM-L6-v2` và tính cosine similarity đa yếu tố (hỗ trợ cả raw text và cached `ocr_features`).
   - Thêm các metric thống kê mật độ: `summary_token_len`, `claim_count`, `claim_density`.
3. **Refactor Retrieval trong `benchmarks/models/retrieval_qa.py`:**
   - Cài đặt hàm `reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60)`.
   - Đảm bảo điểm xếp hạng không bị phụ thuộc vào scale giá trị BM25.
4. **Đảm bảo lưu trữ Checkpoint:**
   - Tạo thư mục `checkpoints/` nếu chưa có và lưu `checkpoints/c5_real.pt` khi hoàn thành kiểm thử model.
5. **Chạy kiểm thử Gate:**
   - Chạy `pytest tests/test_validation_gate.py` đảm bảo không làm gãy bất kỳ ràng buộc hệ thống nào.

---

## 6. Success Criteria

- [x] $C_5$ được định nghĩa hoàn chỉnh với Decoupled Cross-Attention, $d_{\text{ac}}=32$ và Visual Snapping.
- [x] Hàm mất mát không bị bão hòa false positives (`pos_weight = 4.0`).
- [x] $S_3\text{+ev}$ tính toán độ tương đồng chứng cứ ngữ nghĩa thực tế qua SBERT hoặc tensor cosine.
- [x] `reciprocal_rank_fusion` chạy đúng chuẩn toán học và không lỗi type.
- [x] Tệp `checkpoints/c5_real.pt` được tạo mới hợp lệ.
- [x] Toàn bộ tests trong `tests/test_validation_gate.py` vượt qua trong $< 2.0\text{s}$.
