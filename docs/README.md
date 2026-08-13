# Tài liệu Dự án — Multimodal Lecture Summarizer

Thư mục này chứa tất cả các tài liệu kỹ thuật, nghiên cứu, hướng dẫn thiết lập và đánh giá của hệ thống **Tóm tắt bài giảng đa phương thức**.

## Danh mục tài liệu

| File Tài liệu | Mô tả chi tiết |
| :--- | :--- |
| 📖 **[ROADMAP_EVALUATION.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/ROADMAP_EVALUATION.md)** | Đánh giá lộ trình triển khai đề tài luận văn, phân tích tính khả thi từng giai đoạn, khảo sát chi tiết ưu/nhược điểm các mô hình SOTA từ các tạp chí lớn. |
| 🧪 **[RESEARCH_DIRECTIONS.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/RESEARCH_DIRECTIONS.md)** | Đề cương thiết kế nghiên cứu chuyên sâu, các câu hỏi nghiên cứu cốt lõi (RQs), kịch bản thực nghiệm (Ablation Studies) và hệ thống metrics khoa học. |
| 🚀 **[POTENTIAL_RESEARCH_TRENDS.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/POTENTIAL_RESEARCH_TRENDS.md)** | Phân tích 6 hướng nghiên cứu tiên phong (cutting-edge) trong xử lý video đa phương thức để mở rộng quy mô đề tài nghiên cứu. |
| 🤖 **[HYBRID_MODEL_GUIDE.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/HYBRID_MODEL_GUIDE.md)** | Hướng dẫn chi tiết thiết kế mạng Multimodal Scene Encoder dùng PyTorch, cơ chế Cross-modal Attention và quy trình huấn luyện tự giám sát (Contrastive Learning). |
| 🏗️ **[ARCHITECTURE.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/ARCHITECTURE.md)** | Mô tả kiến trúc chi tiết hệ thống, luồng dữ liệu giữa các AI Workers, cấu trúc dữ liệu lưu trữ (Data Models). |
| ⚖️ **[STACK_COMPARISON.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/STACK_COMPARISON.md)** | So sánh chi phí, hiệu năng, yêu cầu phần cứng giữa các stack chạy: Local GPU vs Cloud API vs Hybrid. |
| 📊 **[DATASET_EDA.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/DATASET_EDA.md)** | **CHƯƠNG 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ HỆ THỐNG** — Báo cáo phân tích khám phá dữ liệu (EDA) chi tiết cho hai tập dữ liệu TED-LIUM và TVSum. |
| 🧪 **[BENCHMARK.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/BENCHMARK.md)** | Quy trình và các chỉ số đo lường hiệu năng của hệ thống (WER, ROUGE, F1, BERTScore) trên các video bài giảng thực tế. |
| 📋 **[EVAL_TABLES.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/EVAL_TABLES.md)** | **TTTN/DATN — bảng pipeline (không RAG):** metric, runner, GT templates và script điền bảng đánh giá. |
| 🛠️ **[setup-guide.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/setup-guide.md)** | Hướng dẫn cài đặt chi tiết môi trường phát triển (Frontend, Backend API, AI Workers) và Docker. |
| ✅ **[SPRINT_QUALITY_DEPLOYMENT.md](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/SPRINT_QUALITY_DEPLOYMENT.md)** | Báo cáo Quality Gates & Sprint 1–10: vấn đề giải quyết, thuật toán, kết quả offline/GPU, và hướng triển khai vào `tasks.py`. |
| 🔌 **[api_contracts/](file:///c:/Users/admin/multimodal-lecture-summarizer/docs/api_contracts/)** | Định nghĩa chi tiết các JSON contract trao đổi API giữa Frontend và Backend. |

---
*Lưu ý: Mọi đóng góp mới về kiến trúc hay lộ trình nghiên cứu cần cập nhật trực tiếp vào file tài liệu tương ứng.*
