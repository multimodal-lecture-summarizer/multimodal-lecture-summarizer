# Báo cáo lựa chọn mô hình Multi-Model (chỉ số liệu đo được)

Tài liệu tổng hợp kết quả **so sánh model vs model** cho pipeline **Multimodal Lecture Summarizer**.

**Nguyên tắc:**
- Chỉ đưa metric có file nguồn và script sinh ra.
- Stage **TBD** hoặc chưa A/B → ghi rõ, không suy đoán từ benchmark ngoài.
- **Không** gộp so sánh thuật toán (TF-IDF timeline/chapter, keep-all/temporal-dedup, ablation modality, ngưỡng PySceneDetect) vào phần biện minh lựa chọn model — các số đó nằm ở [§5 Baseline thuật toán](#5-baseline-thuật-toán-không-phải-so-sánh-model).

---

## 1. Nguồn dữ liệu (dataset)

| Dataset | Đường dẫn local | Loader | Stage dùng |
|---------|-----------------|--------|------------|
| **TED-LIUM** | `D:\datasets\TEDLIUM` (`audio/`, `videos/`, `metadata.csv`) | `experiments/evaluation/datasets.py` | ASR, OCR, Caption, Summary |
| **TVSum** | `D:\datasets\tvsum` hoặc fallback `experiments/datasets/tvsum_extracted/` | cùng file trên | Scene, Keyframe |

Mô tả EDA (150 clip TED, 50 video TVSum): [`docs/DATASET_EDA.md`](DATASET_EDA.md).

**Lưu ý proxy GT:**
- **Scene (TVSum):** pseudo boundary từ importance score drops — `tvsum_scene_boundaries()`.
- **Keyframe (TVSum):** cửa sổ frame quan trọng — `tvsum_important_windows()` (ngưỡng = mean + 0.4).
- **Summary (TED):** reference = câu đầu mỗi cửa sổ 40s transcript; hypothesis eval = **extractive TF-IDF** (proxy, **không phải** LLM Qwen production).

---

## 2. File kết quả (output)

| File | Script sinh ra | Phạm vi |
|------|----------------|---------|
| [`outputs/model_choice/ASR_COMPARE.md`](../outputs/model_choice/ASR_COMPARE.md) | script ASR bakeoff | **Model vs model:** base / small / large-v3 (12 clip) |
| [`outputs/model_choice/CAPTION_BAKEOFF.md`](../outputs/model_choice/CAPTION_BAKEOFF.md) | caption bakeoff | **Model vs model:** Florence-2 vs BLIP |
| [`outputs/model_choice/MODEL_CHOICE.md`](../outputs/model_choice/MODEL_CHOICE.md) | `run_model_choice_eval.py` | Tổng hợp + một số đối chứng chiến lược (§5) |
| [`outputs/eval_tables_real/EVAL_TABLES.md`](../outputs/eval_tables_real/EVAL_TABLES.md) | `run_eval_tables.py` | Baseline đầy đủ pipeline (12 clip ASR, 4 TVSum) |
| [`outputs/eval_tables_real/EVAL_TABLES.json`](../outputs/eval_tables_real/EVAL_TABLES.json) | cùng script | JSON gốc |
| [`outputs/eval_model_compare/EVAL_TABLES.md`](../outputs/eval_model_compare/EVAL_TABLES.md) | `run_eval_tables.py --model-compare` | Subset nhanh (2 clip ASR + tiny.en) |

Khung eval và lệnh CLI: [`docs/EVAL_TABLES.md`](EVAL_TABLES.md).

Registry production: `experiments/evaluation/model_compare.py` → `PRODUCTION`.  
Pipeline inference: `ai_workers/tasks.py` → `process_video`.

---

## 3. Công thức metric (code)

| Metric | Định nghĩa | File |
|--------|------------|------|
| **WER** | Word Error Rate | `experiments/evaluation/metrics.py` → `wer()` |
| **CER** | Character Error Rate | `metrics.py` → `cer()` |
| **RTF** | `wall_time_sec / audio_duration_sec` (< 1 = nhanh hơn realtime) | `metrics.py` → `rtf()` |
| **CLIPScore** | Độ khớp ảnh↔caption (CLIP ViT-B/32) | caption bakeoff |
| **Caption content_ok** | Heuristic hallucination/grounding | `metrics.py` → `caption_hallucination_flags()` |
| **Scene / Keyframe P/R/F1** | Boundary hoặc importance-window match | `runners.py` |

---

## 4. So sánh model vs model

Phần này dùng cho luận văn khi biện minh **tại sao chọn model A thay vì model B**. Mỗi bảng là A/B giữa các **weights/pretrained model khác nhau**, cùng pipeline và dataset.

### 4.1 ASR — Faster-Whisper

**Production:** `faster-whisper-base.en` (`ai_workers/modules/audio_v2/transcriber.py`)

**Nguồn:** [`outputs/model_choice/ASR_COMPARE.md`](../outputs/model_choice/ASR_COMPARE.md) — 12 clip TED-LIUM, CPU int8, audio qua DeepFilterNet giống production.

| Model | Params | WER (%) | CER (%) | RTF | Chậm hơn base.en |
|-------|--------|---------|---------|-----|------------------|
| **base.en (production)** | 74M | **14.67** | 4.93 | **0.190** | 1.0× |
| small.en | 244M | **10.80** | 2.93 | 0.429 | 2.3× |
| large-v3 | 1550M | 14.92 | 5.00 | 3.857 | **20.4×** |

**Kết luận model:**
- **`large-v3` loại:** WER không tốt hơn `base.en` (14.92 vs 14.67) nhưng chậm ~20×, ~3 GB load — không phù hợp worker CPU/RAM hạn chế. Config `WHISPERX_MODEL=large-v3` trong `config.py` **không dùng** cho inference chính.
- **`small.en` chính xác hơn** (~−3.9 điểm WER, −2.0 CER) nhưng chậm 2.3× (RTF 0.43, vẫn < 1).
- **Giữ `base.en`:** WER ~15% chấp nhận được, nhanh nhất (~150 MB), phù hợp pipeline lecture ưu tiên latency.

Run phụ (`eval_tables_real`, mean 12 clip): WER base **13.5%** vs small **16.1%** — dao động theo clip; không đảo ngược kết luận về `large-v3` và RTF.

---

### 4.2 Caption — Florence-2 vs BLIP

**Production:** `microsoft/Florence-2-base` (`ai_workers/modules/visual_v2/florence_runtime.py`)

**Nguồn:** [`outputs/model_choice/CAPTION_BAKEOFF.md`](../outputs/model_choice/CAPTION_BAKEOFF.md) — 4 keyframe TED (`Blaise_Agueray_Arcas`), GPU, metric **CLIPScore** (cao hơn = caption khớp ảnh hơn).

| Model | CLIPScore ↑ | s/frame | Generic rate |
|-------|-------------|---------|--------------|
| placeholder *(không phải model)* | 17.43 | 0.00 | 1.00 |
| **Florence-2 (production)** | **30.17** | 2.24 | 0.00 |
| BLIP-base | 27.93 | 1.13 | 0.00 |

**Kết luận model:** Florence-2 **> BLIP** (+2.24 CLIPScore) trên frame bài giảng; BLIP nhanh hơn (~2×) nhưng mô tả kém hơn. Giữ Florence vì đã vendored, prompt `<CAPTION>`, và chất lượng caption tốt hơn BLIP trên tập đo.

> Run `MODEL_CHOICE.md` §3: Florence **skipped** khi RAM < 6 GB — pipeline fallback placeholder/OCR-grounded; không dùng số đó để so model.

---

### 4.3 OCR — PaddleOCR vs candidate

**Production:** PaddleOCR (`compare_ocr_engines()` default)

| Model | CER | Trạng thái |
|-------|-----|------------|
| **PaddleOCR (production)** | TBD | Chưa chạy A/B |
| EasyOCR | TBD | Candidate trong `model_compare.py` |
| Tesseract | TBD | Candidate trong `model_compare.py` |

**Script:** `experiments/evaluation/model_compare.py` → `compare_ocr_engines()` — cần annotate slide TED.

---

### 4.4 Summary LLM — Qwen vs candidate

**Production:** Qwen 2.5 7B Instruct qua OpenRouter (`OPENROUTER_MODEL` trong `config.py`)

| Model | ROUGE-L | Trạng thái |
|-------|---------|------------|
| **Qwen 2.5 7B (production)** | TBD | Chưa eval LLM thật |
| gpt-4o-mini / Llama / Groq | TBD | Chưa chạy |

Eval hiện tại (`eval_summary_pair`) dùng **extractive TF-IDF** — chỉ đo baseline thuật toán, không so LLM (xem §5.6).

---

### 4.5 Stage chỉ có một model (chưa A/B model khác)

| Stage | Model production | Model thay thế đã thử? |
|-------|------------------|------------------------|
| Scene detection | PySceneDetect ContentDetector | ❌ Chưa (TransNetV2, …) |
| Keyframe embedding | CLIP ViT-B/32 | ❌ Chưa (ViT-L/14, SigLIP, …) |
| VAD | Built-in Faster-Whisper | ❌ Chưa (Silero, pyannote, …) |

Chỉ ghi **metric tuyệt đối** ở §5; không dùng để biện minh lựa chọn giữa nhiều model.

---

### 4.6 Tóm tắt lựa chọn model

| Stage | Production | Candidate tốt nhất (metric) | Lý do giữ production |
|-------|------------|----------------------------|----------------------|
| ASR | `base.en` | `small.en` (WER thấp hơn) | Nhanh 2.3×; `large-v3` không cải thiện WER |
| Caption | Florence-2 | BLIP (CLIPScore thấp hơn) | +2.24 CLIPScore, đã tích hợp |
| OCR | PaddleOCR | — | Chưa có A/B |
| LLM Summary | Qwen 2.5 7B | — | Chưa eval ROUGE |

---

## 5. Baseline thuật toán (không phải so sánh model)

Các số dưới đây đo **chất lượng stage** hoặc **ablation chiến lược**, không phải chọn giữa hai pretrained model. **Không đưa vào mục biện minh model** trong luận văn.

### 5.1 Scene — PySceneDetect (metric tuyệt đối)

| Con số | Giá trị | Nguồn |
|--------|---------|-------|
| F1 mean (4 video TVSum) | **0.464** | `eval_tables_real/EVAL_TABLES.md` Bảng 3 |
| Threshold production | **27.0** | `config.py` → `SCENE_THRESHOLD` |

So ngưỡng 20/27/35 là **tuning hyperparameter**, không phải so model — xem `compare_scene_thresholds()` khi cần.

### 5.2 Keyframe — chiến lược lọc (không phải so model)

So keep-all / temporal-dedup / CLIP agglomerative (`MODEL_CHOICE.md` §2) là **chiến lược post-processing** trên cùng embedding CLIP ViT-B/32, không phải so CLIP vs model visual khác.

| Chiến lược | F1 (EE-bNr36nyA) | Nén | Ghi chú |
|------------|------------------|-----|---------|
| keep_all | 0.667 | 1.0 | F1 cao nhất |
| CLIP agglomerative | 0.600 | 0.50 | Production — trade-off nén |

### 5.3 VAD (built-in Whisper)

F1 mean **~0.404** — `eval_tables_real` Bảng 2. Một phương pháp, chưa so model VAD khác.

### 5.4 Timeline (TF-IDF + temporal)

Accuracy **1.000**, MAE **0.000 s** (2 talk) — `timeline.py`. Một thuật toán fusion.

### 5.5 Chapter (TF-IDF topic shift)

Boundary F1 mean **~0.18** — `timeline.py`. Một thuật toán.

### 5.6 Summary — ablation modality (TF-IDF proxy)

| Modality | ROUGE-L (Blaise / Barry) | Nguồn |
|----------|--------------------------|-------|
| Audio + Visual | 0.298 / 0.389 | `eval_tables_real` Bảng 9–10 |
| Audio only | 0.304 / 0.391 | Ablation |
| Visual only | 0.045 / 0.010 | OCR+caption < 5% đóng góp |

Đây là **ablation đầu vào**, không so Qwen vs Llama vs GPT.

---

## 6. Pipeline production (tham chiếu code)

```
Video
 → Faster-Whisper base.en          [§4.1 — model compare]
 → PySceneDetect thr=27            [§5.1 — baseline]
 → CLIP ViT-B/32 agglomerative     [§5.2 — chiến lược trên 1 model]
 → PaddleOCR                       [§4.3 — TBD]
 → Florence-2-base / OCR fallback  [§4.2 — model compare]
 → TF-IDF timeline + chapter       [§5.4, §5.5]
 → Qwen 2.5 7B (OpenRouter)        [§4.4 — TBD]
 → ChromaDB RAG (async)            [chưa eval]
```

Config: `ai_workers/core/config.py`, `backend/.env`.

---

## 7. Cách tái tạo số liệu

```powershell
cd c:\Users\admin\multimodal-lecture-summarizer

# Unit test metric (không GPU)
python experiments/scripts/test_eval_metrics.py

# Baseline pipeline đầy đủ
python experiments/scripts/run_eval_tables.py `
  --out-dir outputs/eval_tables_real `
  --ted-limit 2 --tvsum-limit 4 --asr-limit 12 `
  --asr-models base.en,small.en

# Bakeoff model (ASR + caption) → outputs/model_choice/
python experiments/scripts/run_model_choice_eval.py

# Subset nhanh ASR multi-size
python experiments/scripts/run_eval_tables.py `
  --out-dir outputs/eval_model_compare `
  --ted-limit 1 --tvsum-limit 1 --asr-limit 2 `
  --asr-models tiny.en,base.en,small.en --model-compare
```

**Yêu cầu:** TED-LIUM và TVSum tại `D:\datasets\...`. Thiếu cuDNN → ASR fallback CPU.

---

## 8. Thí nghiệm model còn thiếu (trước luận văn)

| # | So sánh model | Script / hàm | Mở khóa |
|---|---------------|--------------|---------|
| 1 | PaddleOCR vs EasyOCR vs Tesseract | `compare_ocr_engines()` | CER slide TED |
| 2 | Qwen vs gpt-4o-mini / Llama | mở rộng `eval_summary_pair()` + API key | ROUGE-L LLM thật |
| 3 | CLIP ViT-B/32 vs ViT-L/14 | mở rộng `eval_clip_filter_video()` | Keyframe model A/B |
| 4 | PySceneDetect vs TransNetV2 | stage mới | Scene model A/B |
| 5 | tiny.en / medium.en trên 12 clip | `run_eval_tables.py --asr-models ...` | Hoàn thiện thang ASR |

*Không liệt kê ở đây:* scene threshold 20/27/35, keep-all vs temporal — thuộc §5 (thuật toán/hyperparameter).

---

## 9. Tài liệu liên quan

| Tài liệu | Nội dung |
|----------|----------|
| [`docs/EVAL_TABLES.md`](EVAL_TABLES.md) | Khung bảng luận văn, lệnh CLI |
| [`docs/DATASET_EDA.md`](DATASET_EDA.md) | EDA TED-LIUM + TVSum |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | Kiến trúc pipeline |
| [`experiments/evaluation/model_compare.py`](../experiments/evaluation/model_compare.py) | Production registry + compare helpers |

---

*Cập nhật: 2026-08-14 — §4 chỉ gồm so sánh model; §5 tách baseline thuật toán. Số liệu từ output liệt kê mục 2.*
