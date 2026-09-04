---
phase: 5
title: "Publication-Grade LaTeX Package & Build System"
status: pending
priority: P1
dependencies: [4]
---

# Phase 5: Publication-Grade LaTeX Package & Build System

## 1. Overview

Xây dựng bộ mã nguồn LaTeX xuất bản hoàn chỉnh trong thư mục `latex/`, đạt chuẩn trình bày của các hội nghị hàng đầu (IEEE / ACM / NeurIPS format) hoặc quy chuẩn luận văn thạc sĩ quốc tế. 

Gói tài liệu bao gồm: Tệp gốc `main.tex`, các phân đoạn mô-đun hóa trong `sections/`, cơ sở dữ liệu trích dẫn `references.bib` sạch 100%, hệ thống bảng biểu `booktabs` chuyên nghiệp, sơ đồ kiến trúc vector/TikZ, kịch bản biên dịch tự động `build.bat`, và tệp nén lưu trữ `latex.zip` sẵn sàng tải lên Overleaf.

---

## 2. Directory Structure & Deliverables

```
latex/
├── main.tex                       # Tệp điều phối trung tâm
├── references.bib                 # Toàn bộ danh mục tài liệu tham khảo BibTeX
├── build.bat                      # Kịch bản biên dịch tự động cho Windows (pdflatex + bibtex)
├── build.sh                       # Kịch bản biên dịch cho Linux/macOS
├── README.md                      # Hướng dẫn biên dịch và cài đặt gói CTAN
├── sections/                      # Nội dung các chương học thuật
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_experimental_setup.tex
│   ├── 05_results.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
├── figures/                       # Sơ đồ kiến trúc vector và biểu đồ
│   ├── system_architecture.pdf
│   ├── cross_attention_snapping.pdf
│   └── ablation_pareto.pdf
└── tables/                        # Bảng biểu chuyên nghiệp (booktabs)
    ├── table_rq1_chaptering.tex
    ├── table_rq2_summarization.tex
    ├── table_rq3_retrieval.tex
    └── table_ablation.tex
```

---

<!-- Updated: Validation Session 1 - Pinned document class to 2-column IEEEtran format -->

## 3. Implementation Specifications

1. **`main.tex`:**
   - **Định dạng tài liệu:** Chuẩn hóa sang khuôn dạng bài báo 2 cột chuẩn IEEE (`\documentclass[10pt,journal,compsoc]{IEEEtran}`).
   - Sử dụng gói lệnh chuẩn: `amsmath`, `amssymb`, `booktabs`, `graphicx`, `microtype`, `hyperref`, `cleveref`, `tikz`, `cite`.
   - Cấu hình siêu dữ liệu bài báo rõ ràng: Title, Abstract, Keywords, Author Affiliation.
2. **`references.bib` & Kiểm toán trích dẫn tự động (`benchmarks/scripts/check_bib_citations.py`):**
   - Chứa đầy đủ các bài báo kinh điển và cập nhật: Whisper (Radford et al.), WhisperX (Bain et al.), DINOv2 (Oquab et al.), Florence-2 (Xiao et al.), SBERT (Reimers & Gurevych), Reciprocal Rank Fusion (Cormack et al.), Video Summarization surveys, RAG frameworks.
   - Kịch bản kiểm toán tự động quét toàn bộ `\cite{}` trong `sections/*.tex` đối chiếu với `references.bib`, bảo đảm 0 trích dẫn gãy (`cite undefined`) và 0 khóa trích dẫn rỗng.
3. **Kịch bản tự động xuất bảng biểu (`benchmarks/scripts/export_latex_tables.py`):**
   - Tự động phân giải số liệu từ `reports/rq*_benchmark_results.json` và kết xuất ra các tệp `.tex` trong `tables/`.
   - 100% sử dụng chuẩn `booktabs` (`\toprule`, `\midrule`, `\bottomrule`), không có đường kẻ dọc (`|`).
   - Mọi số liệu đều bao gồm $\pm$ độ lệch chuẩn, khoảng tin cậy 95% CI và ký hiệu ý nghĩa thống kê ($^*p < 0.05, ^{**}p < 0.01$).
4. **Kịch bản biên dịch (`build.bat` & `build.sh`):**
   - Chạy tuần tự 4 bước: `pdflatex main` $\to$ `bibtex main` $\to$ `pdflatex main` $\to$ `pdflatex main` để giải quyết triệt để các tham chiếu chéo (cross-references) và mục lục.
   - Hỗ trợ kiểm tra cú pháp độc lập và báo cáo lỗi nếu môi trường thiếu trình biên dịch TeX cục bộ.
5. **Đóng gói phân phối (`latex.zip`):**
   - Nén toàn bộ thư mục `latex/` thành tệp `latex.zip` tại thư mục gốc của dự án để người dùng có thể tải về và mở ngay trên Overleaf chỉ với một cú nhấp chuột.

---

## 4. Success Criteria

- [ ] Cây thư mục `latex/` được khởi tạo đầy đủ với tất cả các tệp thành phần.
- [ ] Kịch bản `benchmarks/scripts/export_latex_tables.py` chạy thành công, tạo 4 tệp bảng biểu chính xác 100% với báo cáo JSON.
- [ ] Kịch bản kiểm toán trích dẫn `check_bib_citations.py` báo cáo 0 cảnh báo tham chiếu gãy (`ref/cite undefined`).
- [ ] Tệp `build.bat` và `README.md` được kiểm thử hoạt động ổn định.
- [ ] Tệp `latex.zip` được tạo mới thành công và chứa đầy đủ dữ liệu biên dịch.
