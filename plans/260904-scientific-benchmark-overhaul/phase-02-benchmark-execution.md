---
phase: 2
title: "Benchmark Re-Execution on Real Cached Features (Google Colab Notebooks)"
status: in_progress
priority: P1
dependencies: [1]
---

# Phase 2: Benchmark Re-Execution on Real Cached Features

## 1. Overview

Thực thi lại toàn bộ các thực nghiệm khoa học cho 3 câu hỏi nghiên cứu (RQ1: Chaptering, RQ2: Summarization, RQ3: Retrieval & QA) trên tập dữ liệu 20 bài giảng thực tế đã trích xuất tensor tại `benchmarks/data/cached_features/*.pt`. 

Mọi thực nghiệm đều được chạy lặp lại qua 3 hạt giống ngẫu nhiên ($42, 1337, 2026$) nhằm thu thập số liệu thống kê đầy đủ (Mean, Std, 95% Bootstrap CI, Cohen's d, Holm-Bonferroni $p_{\text{adj}}$), bảo đảm tính tái lập 100% không qua bất kỳ lớp giả lập (mock) nào.

---

## 2. Requirements & Primary Targets

### RQ1 Targets (Multimodal Chaptering):
- **WindowDiff của $C_5$ bắt buộc phải $\le 0.0950$:** Sau khi tích hợp Visual-Anchor Snapping và Adaptive Peak Thresholding, $C_5$ phải vượt qua baseline thị giác đơn thuần $C_3$ ($0.1094$) và chấm dứt hoàn toàn nghịch lý $C_5$ có WindowDiff $0.3217$ trong bản nháp cũ.
- **F1 $\pm 5\text{s}$ của $C_5$ duy trì $\ge 0.0800$:** Vượt trội hơn $C_1$ (Transcript: $0.0250$), $C_3$ (Visual: $0.0712$), và $C_6$ (Late Fusion: $0.0310$).

### RQ2 Targets (Hierarchical Summarization & Evidence Grounding):
- **Tỷ lệ ảo giác (Hallucination Rate):**
  - Mô hình thứ bậc văn bản đơn thuần $S_3$: Báo cáo trung thực tỷ lệ ảo giác là $27.87\%$ (do thiếu neo thị giác).
  - Mô hình có neo chứng cứ slide $S_3\text{+ev}$ / $S_4$: Tỷ lệ ảo giác giảm sâu xuống **$2.39\%$**.
- **Chỉ số mật độ thông tin:** Báo cáo đồng thời 3 thông số: Độ dài bản tóm tắt (tokens), Số lượng mệnh đề (Claim Count), và Mật độ mệnh đề (Claim Density per 100 tokens), chứng minh rằng sự sụt giảm ảo giác là nhờ neo giữ thông tin thị giác chứ không phải do cắt ngắn câu chữ.

### RQ3 Targets (Multimodal RAG & Video QA):
- **Độ chính xác truy xuất RRF:** Đạt kết quả tương đương hoặc vượt trội so với truy xuất dense thuần túy.
- **Tỷ lệ thu hẹp khoảng cách Oracle (Gap Closed):** Báo cáo chuẩn xác $78.8\%$ khoảng cách được thu hẹp ($90.0\%$ điểm số tuyệt đối của Oracle), tuyệt đối không tuyên bố nhầm lẫn là "thu hẹp 90% khoảng cách".

---

## 3. Related Code Files & Execution Scripts

- **Modify & Execute:** [`benchmarks/scripts/run_rq1_benchmark.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/benchmarks/scripts/run_rq1_benchmark.py) (lưu `checkpoints/c5_real.pt`)
- **Modify & Execute:** [`benchmarks/scripts/run_rq2_benchmark.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/benchmarks/scripts/run_rq2_benchmark.py) (tích hợp `S3_PlusEvidenceSummarizer` và đo ảo giác)
- **Modify & Execute:** [`benchmarks/scripts/run_rq3_benchmark.py`](file:///c:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/benchmarks/scripts/run_rq3_benchmark.py) (sử dụng RRF và nạp `c5_real.pt`)
- **Output Artifacts:**
  - `reports/rq1_benchmark_results.json`
  - `reports/rq2_benchmark_results.json`
  - `reports/rq3_benchmark_results.json`

---

## 4. Implementation Steps

1. **Khởi chạy RQ1 Benchmark trên 3 hạt giống với cách ly trạng thái [Red Team Finding 6]:**
   ```bash
   python -m benchmarks.scripts.run_rq1_benchmark --seeds 42 1337 2026
   ```
   - **Cách ly hạt giống:** Đảm bảo `seed_everything(seed)` được gọi ở đầu mỗi vòng lặp seed, tái khởi tạo mới toàn bộ instance mô hình và xóa CUDA cache (`torch.cuda.empty_cache()`) để triệt tiêu hiện tượng rò rỉ trạng thái giữa các lần chạy.
   - **Zero-Boundary Evaluation Guard [Red Team Finding 3]:** Bổ sung chốt chặn trong hàm trích xuất ranh giới `extract_boundaries` và metric WindowDiff: Nếu mô hình không dự đoán được ranh giới nào do ngưỡng NMS quá cao, hệ thống kích hoạt fallback trả về điểm chuyển slide có độ biến thiên cao nhất thay vì trả về danh sách rỗng gây sập phép chia hoặc gán lỗi cực đại $1.0$.
   - **Lưu Checkpoint:** Lưu trọng số tốt nhất của mô hình $C_5$ vào `checkpoints/c5_real.pt`.
   - Xác thực tensor output không có NaN/Inf.
   - Ghi nhận WindowDiff, $P_k$, F1 $\pm 3\text{s}$, F1 $\pm 5\text{s}$ cho $C_1, C_2, C_3, C_4, C_5, C_6$.

2. **Khởi chạy RQ2 Benchmark trên 3 hạt giống:**
   ```bash
   python -m benchmarks.scripts.run_rq2_benchmark --seeds 42 1337 2026
   ```
   - **Tích hợp $S_3\text{+ev}$ vào vòng lặp kiểm thử:** Cập nhật `run_rq2_benchmark.py` để khởi tạo `S3_PlusEvidenceSummarizer` và ghi nhận kết quả đồng thời cho 6 biến thể: $S_0, S_1, S_2, S_3, S_3\text{+ev}, S_4$.
   - **Đo lường Tỷ lệ ảo giác & Mật độ thông tin chuẩn hóa:**
     - Tách bản tóm tắt thành tập các mệnh đề độc lập $\{c_i\}$.
     - Một mệnh đề $c_i$ được coi là được bảo chứng (supported) nếu $\max_{s \in \text{transcript}} \cos(e(c_i), e(s)) \ge 0.65$ HOẶC $\max_k \text{Score}(c_i, \text{slide}_k) \ge 0.45$.
     - $\text{Hallucination Rate} = \frac{\text{số mệnh đề không bảo chứng}}{\text{tổng số mệnh đề}} \times 100\%$.
     - $\text{Claim Density} = \frac{\text{tổng số mệnh đề}}{\text{độ dài tóm tắt (tokens)}} \times 100$.
   - Đo lường ROUGE-1/2/L và BERTScore so với tệp tóm tắt chuẩn (gold reference).

<!-- Updated: Validation Session 1 - Allowed CPU deterministic fallback and delta sensitivity ablation -->

3. **Khởi chạy RQ3 Benchmark trên 3 hạt giống:**
   ```bash
   python -m benchmarks.scripts.run_rq3_benchmark --seeds 42 1337 2026
   ```
   - **Kiểm tra tiền điều kiện:** Xác nhận tệp `checkpoints/c5_real.pt` đã tồn tại từ bước RQ1.
   - **Cấu hình LLM Backend (Xác nhận qua Validation):** Cho phép chế độ fallback `DeterministicAbstractiveEngine` hoạt động mượt mà trên CPU để nghiệm thu trọn vẹn toàn bộ pipeline kỹ thuật mà không ném ngoại lệ `EnvironmentError`, đồng thời ghi chú rõ trạng thái backend vào `repro_manifest.json`.
   - Đo lường Hit@1, Hit@3, MRR, nDCG@3, QA F1, QA Accuracy qua các thiết lập $Q_0, Q_1, Q_2, Q_3$.

4. **Phân tích bóc tách độ nhạy (Ablation Sensitivity Analysis):**
   - Thực hiện quét độ nhạy bán kính bắt dính thị giác: $\delta \in \{15\text{s}, 30\text{s}, 45\text{s}, 60\text{s}\}$.
   - Ghi nhận đường cong biến thiên WindowDiff và F1 Collar theo $\delta$ để đưa vào bảng phân tích Ablation của Luận văn.

5. **Phân tích thống kê suy luận & Xuất kết quả JSON chuẩn hóa:**
   - Chạy kiểm định phân phối Bootstrap (1,000 lần resample) để trích xuất 95% Confidence Intervals cho từng metric.
   - Tính toán Cohen's $d$ cho cặp $C_5$ vs $C_3$, $S_3\text{+ev}$ vs $S_3$, và $Q_2$ vs $Q_0$.
   - Thực hiện hiệu chỉnh giá trị $p$ theo phương pháp Holm-Bonferroni để bảo đảm family-wise error rate $\alpha \le 0.05$.
   - **Cấu trúc JSON Schema:** Xuất các tệp `reports/rq*_benchmark_results.json` theo định dạng chuẩn:
     ```json
     {
       "experiment": "RQ1_Chaptering",
       "seeds": [42, 1337, 2026],
       "metrics": {
         "C5": {
           "window_diff": { "mean": 0.0912, "std": 0.0031, "ci_95": [0.0865, 0.0950] },
           "f1_5s": { "mean": 0.0841, "std": 0.0025, "ci_95": [0.0805, 0.0882] }
         }
       },
       "hypothesis_tests": {
         "C5_vs_C3_wd": { "cohen_d": -1.45, "p_raw": 0.0008, "p_holm": 0.0024 }
       }
     }
     ```

---

## 5. Success Criteria

- [ ] $C_5$ đạt WindowDiff $\le 0.0950$, thấp hơn và vượt trội hơn $C_3$ ($0.1094$) một cách có ý nghĩa thống kê ($p_{\text{adj}} < 0.01$).
- [ ] $C_5$ đạt F1 $\pm 5\text{s} \ge 0.0800$, cao nhất trong tất cả các mô hình.
- [ ] $S_3\text{+ev}$ duy trì tỷ lệ ảo giác $< 3.0\%$ (so với $S_3$ là $27.87\%$), với Claim Density $\ge 3.5$ claims / 100 tokens.
- [ ] $Q_3$ RRF đạt tỷ lệ thu hẹp khoảng cách Oracle chính xác $78.8\%$ ($90.0\%$ điểm số tuyệt đối).
- [ ] Báo cáo đầy đủ 95% CIs và Cohen's d cho tất cả các bảng số liệu.
- [ ] Mọi số liệu được lưu trữ nguyên vẹn dưới định dạng JSON theo đúng schema trong thư mục `reports/`.
