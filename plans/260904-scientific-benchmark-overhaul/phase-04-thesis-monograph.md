---
phase: 4
title: "Full Master Thesis Monograph (7 Chapters)"
status: pending
priority: P1
dependencies: [3]
---

# Phase 4: Full Master Thesis Monograph (7 Chapters)

## 1. Overview

Soạn thảo toàn diện bộ bản thảo Luận văn Thạc sĩ Khoa học (Master Thesis Monograph) chuẩn mực quốc tế bằng tiếng Anh trong thư mục `docs/thesis/`. Tài liệu này tổng hợp toàn bộ nền tảng lý thuyết, phương pháp luận kiến trúc, kết quả thực nghiệm thực tế sau khi khắc phục 18 điểm phản biện, và phân tích sâu sắc về bài toán hiểu video bài giảng đa phương thức.

Toàn bộ bản thảo được tổ chức thành 7 chương độc lập cùng 01 tệp tổng hợp duy nhất `MASTER_RESEARCH_MANUSCRIPT_EN.md`.

---

### Chapter Breakdown & Word Budgets

| Chương | Tệp Markdown | Mục tiêu dung lượng | Nội dung trọng tâm |
|:------:|:-------------|:-------------------:|:-------------------|
| **Ch 1** | `docs/thesis/01_introduction.md` | 1,500 - 2,000 từ | Bối cảnh bài giảng video, bất đối xứng thời gian, 3 câu hỏi RQ1-RQ3, 4 đóng góp chính (C1-C4), ràng buộc $D\text{-}T08$ (Tesla T4, $\le 32\text{k}$ tokens). |
| **Ch 2** | `docs/thesis/02_related_work.md` | 2,500 - 3,000 từ | Khảo cứu chuyên sâu: ASR (WhisperX), Visual Segmentation (PySceneDetect, DINOv2), Evidence Grounding, Video QA, RAG & Reciprocal Rank Fusion. |
| **Ch 3** | `docs/thesis/03_methodology.md` | 3,000 - 3,500 từ | Kiến trúc toán học chi tiết: $C_5$ Decoupled Cross-Attention (Visual làm Query), Snapping $\delta=45\text{s}$, Dynamic NMS, $S_3\text{+ev}$ Evidence Scoring, RRF RAG. |
| **Ch 4** | `docs/thesis/04_experimental_setup.md` | 2,000 - 2,500 từ | 20 bài giảng thực tế (EduVidQA), hệ thống độ đo (Collar F1, WindowDiff, Pk, ROUGE, Claim Density), quy trình thống kê Bootstrap 95% CI & Holm-Bonferroni. |
<!-- Updated: Validation Session 1 - Added delta sensitivity ablation table to Chapter 5 -->

| **Ch 5** | `docs/thesis/05_empirical_results.md` | 3,500 - 4,000 từ | Bảng số liệu thực nghiệm đầy đủ (RQ1: $C_5 \le 0.0950$, RQ2: Ảo giác $2.39\%$, RQ3: Gap closed $78.8\%$), Ablation studies (bao gồm khảo sát độ nhạy $\delta \in \{15, 30, 45, 60\}\text{s}$), Latency & VRAM profiling. |
| **Ch 6** | `docs/thesis/06_discussion_limitations.md` | 2,000 - 2,500 từ | Phân tích dung hòa F1 vi mô vs liên kết vĩ mô, 3 Failure Cases điển hình, giới hạn $N=20$ & phân tích công suất thống kê. |
| **Ch 7** | `docs/thesis/07_conclusion_future_work.md` | 1,200 - 1,500 từ | Đánh giá giả thuyết H1-H3, tác động thực tiễn EdTech, mở rộng quy mô lớn và streaming. |
| **Tổng** | `docs/thesis/MASTER_RESEARCH_MANUSCRIPT_EN.md` | **15,000 - 20,000 từ** | Bản thảo hợp nhất hoàn chỉnh, đồng nhất thuật ngữ và ký hiệu toán học. |

### Bảng chuẩn hóa ký hiệu toán học (Mathematical Notation Standard)

Toàn bộ 7 chương bắt buộc phải tuân thủ ký hiệu thống nhất:
- $\mathbf{X}_t = [\mathbf{v}_t, \mathbf{t}_t, \mathbf{o}_t, \mathbf{a}_t]$: Bộ vector đa phương thức tại timestamp $t$.
- $\mathbf{Q}_v = \mathbf{W}_q \mathbf{v}_t$: Query anchor thị giác trích xuất từ DINOv2 ($384 \to 256$).
- $\mathbf{K}, \mathbf{V}$: Khóa và giá trị chiếu từ Transcript, OCR và Acoustic.
- $\delta = 45\text{s}$: Bán kính cửa sổ bắt dính Visual Snapping.
- $\tau = \mu(p) - 1.0\sigma(p)$: Ngưỡng động trung bình trượt phân định ranh giới.
- $\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$: Điểm xếp hạng kết hợp truy xuất.

---

## 3. Implementation Steps

1. Tạo thư mục `docs/thesis/` nếu chưa tồn tại.
2. Xây dựng kịch bản kiểm tra đồng bộ dữ liệu `benchmarks/scripts/sync_thesis_data.py`:
   - Đọc trực tiếp từ `reports/rq*_benchmark_results.json`.
   - Tự động thay thế các placeholder giá trị số trong bản thảo để loại bỏ 100% nguy cơ gõ sai số liệu bằng tay.
3. Soạn thảo tuần tự từng chương từ Chương 1 đến Chương 7 bằng tiếng Anh học thuật chuẩn mực cao (Academic English).
4. Biên dịch và tổng hợp tệp `docs/thesis/MASTER_RESEARCH_MANUSCRIPT_EN.md`.
5. Rà soát tính nhất quán nội tại: Ký hiệu toán học, thuật ngữ mô hình ($C_1 - C_6$, $S_0 - S_4$, $Q_0 - Q_3$), và trích dẫn bảng số liệu.

---

## 4. Success Criteria

- [ ] Toàn bộ 7 tệp chương trong `docs/thesis/` được hoàn thành đầy đủ nội dung (không có placeholder `TODO`).
- [ ] Dung lượng toàn bộ bản thảo đạt chuẩn luận văn thạc sĩ ($\ge 15,000$ từ).
- [ ] Tệp tổng hợp `MASTER_RESEARCH_MANUSCRIPT_EN.md` hoàn thiện với cấu trúc luận văn thạc sĩ chuyên nghiệp.
- [ ] Mọi số liệu thực nghiệm khớp chính xác 100% với các tệp JSON trong `reports/`.
- [ ] Không còn bất kỳ phát biểu gây hiểu lầm hay sai sót toán học nào từ 18 điểm phản biện.
