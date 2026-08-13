# Khung đánh giá bảng luận văn (pipeline, không gồm RAG/Q&A)

Tài liệu triển khai phần **“Các bảng đánh giá cần bổ sung cho pipeline AI”** (TTTN/DATN).

## Đã triển khai

| Thành phần | Đường dẫn |
|---|---|
| Metric thuần (WER/CER/RTF, VAD/scene/chapter P/R/F1, OCR, caption, ROUGE-L…) | `experiments/evaluation/metrics.py` |
| Schema + normalize GT | `experiments/evaluation/schemas.py` |
| Stage runners | `experiments/evaluation/runners.py` |
| Render Markdown bảng pipeline (không RAG) | `experiments/evaluation/report.py` |
| CLI tổng hợp | `experiments/scripts/run_eval_tables.py` |
| Unit tests | `experiments/scripts/test_eval_metrics.py` |
| Dataset routing TED-LIUM / TVSum | `experiments/evaluation/datasets.py` |
| Manifest + GT mẫu | `benchmarks/manifest_eval.csv`, `benchmarks/references/` |

## Không nằm trong phạm vi đánh giá (theo đề bài)

- CPU/GPU/RAM hardware profiling
- UI / authentication / admin / deployment

## Thứ tự ưu tiên

1. ASR → 2. Scene/Keyframe → 3. OCR/Caption → 4. Timeline/Chapter → 5. Summary  
(+ Ablation multimodal, bảng tổng hợp stage). Không đánh giá RAG/Q&A.

## Cách chạy

```bash
# Unit tests (không cần GPU)
python -m pytest experiments/scripts/test_eval_metrics.py -q

# Sinh khung TBD
python experiments/scripts/run_eval_tables.py --dry-report

# Mặc định: TED-LIUM (ASR/VAD/timeline/chapter/OCR/caption/summary/ablation) + TVSum (scene/keyframe)
python experiments/scripts/run_eval_tables.py --out-dir outputs/eval_tables_real --asr-limit 12 --tvsum-limit 4 --asr-models base.en,small.en

# Chỉ ASR trên TED-LIUM
python experiments/scripts/run_eval_tables.py --asr-only --asr-models base.en,small.en --asr-limit 12

# Manifest thủ công (không auto TED/TVSum)
python experiments/scripts/run_eval_tables.py --no-auto-datasets --manifest benchmarks/manifest_eval.csv
```

Output: `outputs/eval_tables_*/EVAL_TABLES.md` (+ `.json`).

## Annotate dữ liệu thật

Làm theo `benchmarks/references/README.md`. Mỗi bảng trong báo cáo nên có **một câu kết luận ngắn** sau bảng (script tự sinh khi có số liệu).
