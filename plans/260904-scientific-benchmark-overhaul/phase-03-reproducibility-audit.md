---
phase: 3
title: "Reproducibility Audit & 18 Vulnerabilities Resolution"
status: pending
priority: P1
dependencies: [2]
---

# Phase 3: Reproducibility Audit & 18 Vulnerabilities Resolution

## 1. Overview

Thực hiện rà soát nghiêm ngặt toàn bộ hệ thống thực nghiệm đối chiếu với **18 lỗ hổng phản biện học thuật** đã được nhận diện trong `/ck:brainstorm`. Đảm bảo rằng bài báo và luận văn hoàn toàn không chứa bất kỳ tuyên bố quá đà (overclaim), không có sai sót toán học, và thỏa mãn 100% các tiêu chuẩn tái lập khoa học quốc tế.

---

## 2. The 18 Vulnerabilities Audit Matrix

| STT | Lỗ hổng phản biện | Giải pháp khắc phục triệt để | Trạng thái kiểm toán |
|:---:|:------------------|:-----------------------------|:--------------------:|
| **1** | $C_5$ WindowDiff $0.3217$ tệ nhất bảng nhưng kết luận H1 đúng | Refactor Decoupled Cross-Attn + Visual Snapping đưa WindowDiff $C_5 \le 0.0950$ (đánh bại $C_3$ là $0.1094$). | Bắt buộc |
| **2** | F1 ranh giới chi tiết thấp ($\approx 0.08$) | Bổ sung phân tích minh bạch: Phân định ranh giới $\pm 5\text{s}$ phản ánh thách thức vi mô, trong khi mức độ kết dính vĩ mô (5-10 phút) đạt hiệu quả cao cho truy xuất RAG. | Bắt buộc |
| **3** | $S_3\text{+ev}$ cắt 5 từ OCR thô thiển | Chuyển sang Semantic Cosine Matching giữa câu tóm tắt và văn bản OCR slide qua SBERT MiniLM-L6. | Bắt buộc |
| **4** | $S_3$ có tỷ lệ ảo giác $27.87\%$ nhưng lại được ca ngợi | Trình bày trung thực: Phân cấp văn bản đơn thuần ($S_3$) gây ảo giác nặng ($27.87\%$), và chỉ khi có neo thị giác ($S_3\text{+ev}$) ảo giác mới giảm về $2.39\%$. | Bắt buộc |
| **5** | Thiếu chỉ số độ dài tóm tắt làm lu mờ tỷ lệ ảo giác | Báo cáo đồng thời: Summary Length (tokens), Claim Count, và Claim Density per 100 tokens. | Bắt buộc |
| **6** | Công thức RAG $0.60\cos + 0.40\text{BM25}$ cộng điểm chưa chuẩn hóa | Thay bằng thuật toán chuẩn hóa Reciprocal Rank Fusion (RRF, $k=60$). | Bắt buộc |
| **7** | Tuyên bố "thu hẹp 90% khoảng cách Oracle" bị sai toán | Báo cáo chính xác: Thu hẹp **$78.8\%$** khoảng cách (tương ứng đạt $90.0\%$ điểm số tuyệt đối của Oracle). | Bắt buộc |
| **8** | Thiếu khoảng tin cậy 95% CI và Cohen's d | Tính toán 1,000-resample Bootstrap 95% CIs và Cohen's d / Hedges' g cho mọi bảng số liệu. | Bắt buộc |
| **9** | Chưa kiểm định điều chỉnh Holm-Bonferroni | Áp dụng hiệu chỉnh Holm-Bonferroni cho toàn bộ các cặp kiểm định t-test / Wilcoxon. | Bắt buộc |
| **10** | Quy mô mẫu $N=20$ chưa được giải trình thỏa đáng | Bổ sung phần thảo luận công suất thống kê (Statistical Power Analysis) và định hướng mở rộng tập dữ liệu lớn. | Bắt buộc |
| **11** | Giới hạn phần cứng không rõ ràng | Khẳng định và ghi nhận thông số chạy trên 01 NVIDIA Tesla T4 (16GB VRAM). | Bắt buộc |
| **12** | Vi phạm cân bằng ngân sách token ($D\text{-}T08$) | Khóa cứng ngân sách: Source $\le 32\text{k}$, Summary $\le 512$, Frames $\le 200$. | Bắt buộc |
| **13** | Nguy cơ lọt dữ liệu giả lập/mock ($D\text{-}T15$) | Kiểm tra tự động qua `pytest tests/test_validation_gate.py` phát hiện 0 mock trong luồng nghiên cứu. | Bắt buộc |
| **14** | Thiếu manifest tái lập thực nghiệm | Tạo mới và cập nhật `reports/repro_manifest.json` chứa commit git, seeds, môi trường phần mềm. | Bắt buộc |
| **15** | Rò rỉ dữ liệu giữa tập Train và Test | Xác thực phân chia tập bài giảng Train/Val/Test hoàn toàn rời rạc theo Video ID. | Bắt buộc |
| **16** | Thiếu phân tích đóng góp thành phần (Ablation) | Thêm bảng ablation chi tiết: Impact của Visual Snapping, Dynamic NMS, và Cross-Attention. | Bắt buộc |
| **17** | Thiếu thông số độ trễ và bộ nhớ từng giai đoạn | Đo đạc và lập bảng Latency (s) và VRAM (MB) cho từng module: ASR, Visual, Fusion, RAG. | Bắt buộc |
| **18** | Thiếu phân tích trường hợp lỗi (Error Analysis) | Trình bày 3 trường hợp phân đoạn và tóm tắt thất bại (giảng viên nói lạc đề, slide quá nhiều chữ). | Bắt buộc |

---

## 3. Implementation Steps

1. **Cập nhật Reproducibility Manifest (`reports/repro_manifest.json`):**
   - Lưu trữ mã băm Git commit hiện tại (`git rev-parse HEAD`).
   - Ghi nhận thông số phần cứng (CPU, GPU, RAM, VRAM), phiên bản PyTorch, Transformers, CUDA.
   - Ghi nhận danh sách seed ngẫu nhiên ($42, 1337, 2026$) và mã băm MD5/SHA256 của 20 tệp `.pt` trong `benchmarks/data/cached_features/`.
   - Lưu trữ siêu dữ liệu nghiệm thu của cả 4 RQ (RQ1, RQ2, RQ3, RQ4).

2. **Mở rộng bộ kiểm thử tự động trong `tests/test_validation_gate.py`:**
   Bổ sung 4 hàm kiểm thử tự động xác thực các ngưỡng nghiệm thu khoa học:
   - `test_rq1_c5_window_diff_bound(project_root)`: Xác thực $C_5$ đạt WindowDiff $\le 0.0950$ và vượt trội hơn $C_3$.
   - `test_rq2_hallucination_reduction_bound(project_root)`: Xác thực $S_3\text{+ev}$ hạ tỷ lệ ảo giác xuống $< 3.0\%$ so với $S_3$ ($> 20\%$).
   - `test_rq3_oracle_gap_closed_precision(project_root)`: Xác thực tỷ lệ thu hẹp khoảng cách Oracle được báo cáo chính xác là $78.8\%$ ($90.0\%$ điểm số tuyệt đối).
   - `test_zero_mock_across_all_benchmarks(project_root)`: Quét AST/Regex trong `benchmarks/scripts/` xác nhận không có `np.random.randn` hay nhãn nhân tạo.

3. **Cập nhật Báo cáo Kiểm toán (`reports/validation_gate_summary.md`):**
   - Xuất bảng đối chiếu chi tiết 18 tiêu chí với link tham chiếu code và kết quả thực nghiệm.
   - Ghi nhận kết quả test tự động từ `tests/test_validation_gate.py`.

4. **Thực hiện kiểm thử Gate tự động:**
   ```bash
   pytest tests/test_validation_gate.py -v
   ```

---

## 4. Success Criteria

- [ ] Toàn bộ 18/18 tiêu chí phản biện được đối chiếu và xác nhận "Đạt chuẩn".
- [ ] Tệp `reports/repro_manifest.json` được tạo mới với đầy đủ siêu dữ liệu tái lập và checksum 20 tệp `.pt`.
- [ ] 9/9 bài test trong `tests/test_validation_gate.py` vượt qua với kết quả 100% Pass.
- [ ] Tệp `reports/validation_gate_summary.md` hoàn thành không còn mục `PENDING`.
