# Khung đánh giá bảng luận văn (pipeline, không gồm RAG/Q&A)

Tài liệu triển khai phần **“Các bảng đánh giá cần bổ sung cho pipeline AI”** (TTTN/DATN).

## Dataset

| Tên | Nội dung |
|-----|----------|
| **TED** | Một dataset thống nhất: clip ASR + transcript TED-LIUM + video talk cùng `talk_id` (không tách TED-LIUM / TED video). |
| **TVSum** | Scene / keyframe (GT importance). |

## 5 stage ưu tiên (metric)

1. **ASR** — WER, CER, RTF (+ so sánh multi-model vs production `base.en`)
2. **Scene / Keyframe** — Precision, Recall, F1
3. **OCR / Caption** — CER, human score (proxy 1–5), hallucination rate
4. **Timeline / Chapter** — MAE, boundary P/R/F1
5. **Summary** — ROUGE-L, BERTScore, factuality, coverage

## Thành phần code

| Thành phần | Đường dẫn |
|---|---|
| Metrics | `experiments/evaluation/metrics.py` |
| So sánh model | `experiments/evaluation/aggregate.py` |
| TED unified | `experiments/evaluation/datasets.py` (`TED_DATASET`, `pick_ted_unified_talks`) |
| Runners / model compare | `experiments/evaluation/runners.py`, `model_compare.py` |
| Báo cáo Markdown | `experiments/evaluation/report.py` |
| CLI | `experiments/scripts/run_eval_tables.py` |

## Cách chạy

```bash
# Unit tests (không cần GPU)
python experiments/scripts/test_eval_metrics.py

# Demo nhanh (CPU nếu thiếu cuDNN)
$env:CUDA_VISIBLE_DEVICES="-1"
python experiments/scripts/run_eval_tables.py \
  --out-dir outputs/eval_tables_demo \
  --ted-limit 2 --asr-limit 4 \
  --asr-models tiny.en,base.en,small.en \
  --production-asr base.en \
  --skip-ocr

# So sánh candidate vs production (Bảng 10) — mặc định bật
python experiments/scripts/run_eval_tables.py \
  --out-dir outputs/eval_model_compare \
  --ted-limit 1 --tvsum-limit 1 --model-compare

# Tắt so sánh (nhanh hơn)
python experiments/scripts/run_eval_tables.py --no-model-compare

# BERTScore (chậm hơn)
python experiments/scripts/run_eval_tables.py --bertscore
```

Output: `outputs/eval_tables_*/EVAL_TABLES.md` (+ `.json`, gồm `model_comparison`).

## Production vs candidates (Bảng 10)

| Stage | Production | Candidates |
|-------|------------|------------|
| ASR | `base.en` | `tiny.en`, `small.en`, `medium.en` |
| Scene | PySceneDetect **27** | threshold 20, 35 |
| Keyframe | CLIP agglomerative | keep-all, temporal-dedup |
| Caption | Florence-2 | placeholder, OCR-grounded |
| OCR | PaddleOCR | EasyOCR, Tesseract |

## Production defaults

| Stage | Production |
|-------|------------|
| ASR | Faster-Whisper `base.en` |
| Scene | PySceneDetect threshold 27 |
| Keyframe | CLIP ViT-B/32 agglomerative |
| OCR | PaddleOCR |
| Caption | Florence-2-base |
| Summary | Extractive TF-IDF (proxy) hoặc LLM khi có API key |
