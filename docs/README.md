# Tài liệu Dự án — Multimodal Lecture Summarizer

Thư mục này chứa tất cả các tài liệu kỹ thuật, kiến trúc, nghiên cứu, hướng dẫn thiết lập và đánh giá của hệ thống **Tóm tắt bài giảng đa phương thức**.

## Danh mục tài liệu

| File Tài liệu | Mô tả chi tiết |
| :--- | :--- |
| 🏗️ **[current-system-architecture.md](./current-system-architecture.md)** | **Kiến trúc hệ thống hiện tại mới nhất** — Topo dịch vụ, quy trình 9 stage tuần tự, checkpoint persistence, extractive fallback, decoupled async RAG, cơ sở dữ liệu và ràng buộc tài nguyên phần cứng. |
| 🛠️ **[setup-guide.md](./setup-guide.md)** | Hướng dẫn cài đặt chi tiết môi trường phát triển (Frontend, Backend API, AI Workers), công cụ khởi chạy PowerShell/Bash và Docker. |
| 📝 **[project_overview.md](./project_overview.md)** | Tổng quan dự án, công nghệ cốt lõi và tóm tắt luồng dữ liệu đa phương thức. |
| 🔬 **[florence-2-cpu-reproducibility.md](./florence-2-cpu-reproducibility.md)** | Phân tích chuyên sâu về hiện tượng sinh text rác và giải pháp chuẩn hóa suy luận CPU/FP32 deterministic cho mô hình Florence-2. |
| 📊 **[MODEL_SELECTION_README.md](./MODEL_SELECTION_README.md)** | Báo cáo lựa chọn mô hình Multi-Model (WhisperX, Florence-2 vs BLIP, PaddleOCR) dựa trên số liệu thực nghiệm đo đạc thực tế. |
| ✅ **[SPRINT_QUALITY_DEPLOYMENT.md](./SPRINT_QUALITY_DEPLOYMENT.md)** | Báo cáo triển khai Quality Gates & Sprint Stack (S1–S10): làm mịn chương mục, làm giàu OCR, lọc keyframe có căn cứ ngữ cảnh. |
| ⚖️ **[STACK_COMPARISON.md](./STACK_COMPARISON.md)** | Phân tích và so sánh chi phí, latency, yêu cầu phần cứng giữa các stack triển khai: Local GPU vs Cloud API vs Hybrid. |
| 🧪 **[BENCHMARK.md](./BENCHMARK.md)** | Khung benchmark và quy trình đo lường hiệu năng hệ thống (WER, CER, F1, ROUGE, BERTScore) trên video thực tế. |
| 📋 **[EVAL_TABLES.md](./EVAL_TABLES.md)** | Khung đánh giá bảng luận văn (pipeline AI không gồm RAG/Q&A): metrics, runner và script tổng hợp. |
| 📊 **[DATASET_EDA.md](./DATASET_EDA.md)** | Báo cáo phân tích khám phá dữ liệu (EDA) chi tiết cho hai tập dữ liệu chuẩn TED-LIUM và TVSum. |
| 📖 **[ROADMAP_EVALUATION.md](./ROADMAP_EVALUATION.md)** | Đánh giá lộ trình triển khai đề tài luận văn, phân tích tính khả thi và khảo sát chi tiết các mô hình SOTA từ tạp chí/hội nghị lớn. |
| 🧪 **[RESEARCH_DIRECTIONS.md](./RESEARCH_DIRECTIONS.md)** | Đề cương thiết kế nghiên cứu chuyên sâu, các câu hỏi nghiên cứu cốt lõi (RQ1–RQ4) và kịch bản thực nghiệm suy hao (Ablation Studies). |
| 🚀 **[POTENTIAL_RESEARCH_TRENDS.md](./POTENTIAL_RESEARCH_TRENDS.md)** | Phân tích 6 xu hướng nghiên cứu tiên phong trong xử lý video bài giảng đa phương thức. |
| 🤖 **[HYBRID_MODEL_GUIDE.md](./HYBRID_MODEL_GUIDE.md)** | Hướng dẫn chi tiết thiết kế mạng Multimodal Scene Encoder dùng PyTorch, cơ chế Cross-modal Attention và Contrastive Learning. |

---
*Lưu ý: Mọi đóng góp mới về kiến trúc hay lộ trình nghiên cứu cần cập nhật trực tiếp vào file tài liệu tương ứng.*
