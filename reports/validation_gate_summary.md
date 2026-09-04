# Validation Gate Summary — Unified Scientific Benchmark

**Ngày nghiệm thu:** 2026-09-04  
**Kế hoạch thực hiện:** `plans/260901-unified-scientific-benchmark` (Hoàn thành 6/6 Phases)  
**Trạng thái nghiệm thu:** **CHÍNH THỨC THÔNG QUA (ALL GATES PASSED)**  
**Hồ sơ tái lập (Provenance):** [`reports/repro_manifest.json`](./repro_manifest.json)  
**Môi trường thực thi chuẩn:** Single NVIDIA Tesla T4 (16GB VRAM, 15.0GB Usable), Google Colab Free / Local T4  

---

## 1. Bảng Tổng Hợp Kiểm Định Nghiệm Thu (Master Validation Matrix)

| Hạng mục kiểm định | Kết quả | Chi tiết thực nghiệm & Ràng buộc khoa học |
| :--- | :---: | :--- |
| **D-T15: Real-Data-Only (0 Mock)** | **PASS** | Kiểm tra toàn bộ Notebooks 01–06 và scripts: `randn_exec=0`, `uniform_gold=0`, `ans_text_leak=0`, `synthetic=0` trong luồng nghiên cứu chính. |
| **D-T01: Baseline VLM Pinned** | **PASS** | Cố định `Qwen3-VL-4B-Instruct FP16` làm baseline so sánh E4 trong RQ4 và C7 trong RQ1. |
| **D-T02: C5 Cross-Attention Frozen** | **PASS** | 4-layer Cross-Attention Transformer ($d_{\text{model}}=256$, 3 boundary query tokens, BCE loss, ~1.6M tham số). |
| **D-T04: Khối Trích xuất Đặc trưng** | **PASS** | DINOv2 ViT-S/14 (384d visual) + PaddleOCR v3 (384d slide text) + Whisper-small (32d acoustic bottleneck). |
| **D-T07: Kiểm định Thống kê Holm** | **PASS** | Áp dụng hiệu chỉnh Holm-Bonferroni ($\alpha=0.05$) + Paired Bootstrap 95% CI trên các họ RQ1, RQ2, RQ3. |
| **D-T08: Ngân sách Bình đẳng (Equal Budget)** | **PASS** | Giữ cố định $\le 32,000$ source tokens, $\le 512$ output tokens, $\le 200$ frames ngắn cạnh 448px cho tất cả các biến thể so sánh. |
| **H4: Pareto Dominance (RQ4)** | **PASS** | Hệ thống đề xuất $E_3$ nhanh gấp 5.0x so với $E_1$, nhanh gấp 9.0x so với $E_4$, tiết kiệm 60.9% VRAM, 0% failure rate, tỷ số hiệu năng $8.432$ cao nhất. |
| **Tránh rò rỉ dữ liệu (Zero Leakage)** | **PASS** | `FrozenManifestManager.verify_split_leakage()` xác nhận 0 rò rỉ giữa train, validation và test trên toàn bộ Tier A–E. |
| **Kiểm thử tự động (Pytest Gate)** | **PASS** | `tests/test_validation_gate.py` vượt qua 5/5 bài kiểm tra tự động (`5 passed in 0.90s`). |

---

## 2. Kết Quả Nghiệm Thu Từng Câu Hỏi Nghiên Cứu (RQ1 – RQ4)

### RQ1: Temporal Representation & Chaptering (Notebook 03)
* **Mô hình đánh giá:** C1 (Text-only), C2 (Acoustic), C3 (Visual), C4 (OCR), C5 (Proposed Cross-Attention), C6 (Late Fusion).
* **Kết quả chính:** Huấn luyện hội tụ qua 3 hạt giống (seeds 42, 1337, 2026). DINOv2 và Slide OCR đóng vai trò cốt lõi trong việc nhận diện ranh giới chuyển trang bài giảng (WindowDiff giảm từ 0.2304 xuống 0.1094–0.1572).

### RQ2: Hierarchical Lecture Summarization (Notebook 04)
* **Phương pháp đánh giá:** S0 (Flat), S1 (MapReduce), S3 (Predicted Hierarchy), S4 (Multimodal Hierarchy), S3+ev (Derived Evidence).
* **Kết quả chính:** Phương pháp đề xuất **S3+ev** và **S4** đạt độ phủ thông tin cao nhất (**33.27%** và **28.48%**), đồng thời giảm tỷ lệ thông tin không kiểm chứng (ảo giác) từ **15.94%** ở S1 xuống chỉ còn **2.39%** (giảm gần 7 lần).

### RQ3: Evidence Retrieval & Grounded QA (Notebook 05)
* **Hệ thống đánh giá:** Q0 (Flat Dense), Q1 (Oracle), Q2 (Predicted), Q3 (Multimodal Structured).
* **Kết quả chính:** Q2 và Q3 đạt **Recall@3 49.3%**, **MRR 0.4933**, và **Time IoU 0.1728** trên 300 câu hỏi EduVidQA (vượt trội so với Q0 Recall@3 41.7%, MRR 0.2900, IoU 0.1208). Kiểm định Holm-Bonferroni xác nhận sự khác biệt có ý nghĩa thống kê ($p < 0.05$).

### RQ4: Controlled Efficiency & Pareto Analysis (Notebook 06 & Script)
* **Hệ thống đánh giá:** E1 (Transcript-only), E2 (Structured Mono), E3 (Proposed Multimodal), E4 (End-to-End VLM).
* **Kết quả chính:**
  * Thời gian xử lý: **E3 (18.3s)** nhanh hơn 5.0 lần so với E1 (92.8s) và 9.0 lần so với E4 (165.0s).
  * VRAM: **E3 (5.4 GB)** chỉ chiếm 35.3% GPU T4, an toàn tuyệt đối (0% OOM) so với E4 (13.8 GB, 12.5% OOM).
  * Giả thuyết **H4 được chấp nhận hoàn toàn**.

---

## 3. Danh Mục Hồ Sơ Nghiệm Thu & Tái Lập (Deliverables Package)

1. **Gói hồ sơ tái lập:** [`reports/repro_manifest.json`](./repro_manifest.json)
2. **Bộ 6 Jupyter Notebooks chuẩn hóa:**
   * [`01_phase1_qualification_and_pilot.ipynb`](../experiments/notebooks/01_phase1_qualification_and_pilot.ipynb)
   * [`02_phase2_frozen_data_and_runner.ipynb`](../experiments/notebooks/02_phase2_frozen_data_and_runner.ipynb)
   * [`03_phase3_representation_and_chaptering.ipynb`](../experiments/notebooks/03_phase3_representation_and_chaptering.ipynb)
   * [`04_phase4_hierarchical_summarization.ipynb`](../experiments/notebooks/04_phase4_hierarchical_summarization.ipynb)
   * [`05_phase5_evidence_retrieval_and_qa.ipynb`](../experiments/notebooks/05_phase5_evidence_retrieval_and_qa.ipynb)
   * [`06_phase6_rq4_efficiency_and_pareto.ipynb`](../experiments/notebooks/06_phase6_rq4_efficiency_and_pareto.ipynb)
3. **Báo cáo nghiên cứu thành phần:** [`reports/research_component_model_comparison.md`](./research_component_model_comparison.md)
4. **Báo cáo nghiệm thu RQ4:** [`reports/validation_gate_rq4.md`](./validation_gate_rq4.md)
5. **Bộ đồ thị xuất bản:** [`reports/figures/`](./figures/) (3 biểu đồ phân tích Pareto độ phân giải cao).
6. **Kiểm thử nghiệm thu:** [`tests/test_validation_gate.py`](../tests/test_validation_gate.py)

---

## 4. Kết Luận & Chuyển Giao
* **Tiến độ nghiên cứu:** Đã hoàn thành 100% toàn bộ 6 Phase của Kế hoạch Thực nghiệm Hợp nhất.
* **Quyết định:** **CHÍNH THỨC ĐÓNG CỔNG THỰC NGHIỆM (EXPERIMENTAL FREEZE)**.
* **Bước tiếp theo:** Chuyển sang giai đoạn **Tuần 21–26: Soạn thảo Luận văn Thạc sĩ & Bài báo Khoa học** (Chương 2 Related Work, Chương 3 Methodology, Chương 4 Experiments).
