# Báo cáo lựa chọn mô hình Multi-Model (chỉ số liệu đo được)

Tài liệu tổng hợp kết quả so sánh mô hình cho pipeline **Multimodal Lecture Summarizer**.  
**Nguyên tắc:** chỉ đưa metric có file nguồn và script sinh ra; stage **TBD** hoặc chưa A/B → **bỏ qua**, không suy đoán từ benchmark ngoài.

---

## 1. Nguồn dữ liệu (dataset)

| Dataset | Đường dẫn local | Loader | Stage dùng |
|---------|-----------------|--------|------------|
| **TED-LIUM** | `D:\datasets\TEDLIUM` (`audio/`, `videos/`, `metadata.csv`) | `experiments/evaluation/datasets.py` | ASR, VAD, Timeline, Chapter, Summary |
| **TVSum** | `D:\datasets\tvsum` hoặc fallback `experiments/datasets/tvsum_extracted/` | cùng file trên | Scene, Keyframe |

Mô tả EDA (150 clip TED, 50 video TVSum): [`docs/DATASET_EDA.md`](DATASET_EDA.md).

**Lưu ý proxy GT:**
- **Scene (TVSum):** pseudo boundary từ importance score drops — `tvsum_scene_boundaries()` trong `datasets.py`.
- **Keyframe (TVSum):** cửa sổ frame quan trọng — `tvsum_important_windows()` (ngưỡng = mean + 0.4).
- **Summary (TED):** reference = câu đầu mỗi cửa sổ 40s transcript; hypothesis = **extractive TF-IDF** (proxy, **không phải** LLM Qwen production).

---

## 2. File kết quả (output)

| File | Script sinh ra | Mô tả |
|------|----------------|-------|
| [`outputs/eval_tables_real/EVAL_TABLES.md`](../outputs/eval_tables_real/EVAL_TABLES.md) | `experiments/scripts/run_eval_tables.py` | Run đầy đủ nhất: 12 clip ASR, 4 TVSum, 2 TED talk |
| [`outputs/eval_tables_real/EVAL_TABLES.json`](../outputs/eval_tables_real/EVAL_TABLES.json) | cùng script | JSON gốc (ASR rows, scene, keyframe, …) |
| [`outputs/model_choice/MODEL_CHOICE.md`](../outputs/model_choice/MODEL_CHOICE.md) | `experiments/scripts/run_model_choice_eval.py` | ASR aggregate + CLIP 3 chiến lược + Florence bakeoff |
| [`outputs/model_choice/MODEL_CHOICE.json`](../outputs/model_choice/MODEL_CHOICE.json) | cùng script | JSON gốc clip/caption |
| [`outputs/eval_model_compare/EVAL_TABLES.md`](../outputs/eval_model_compare/EVAL_TABLES.md) | `run_eval_tables.py --model-compare` | Subset nhỏ: 2 clip ASR (+ tiny.en), 1 TVSum |

Khung eval và lệnh chạy: [`docs/EVAL_TABLES.md`](EVAL_TABLES.md).

---

## 3. Công thức metric (code)

| Metric | Định nghĩa | File |
|--------|------------|------|
| **WER** | Word Error Rate (jiwer hoặc Levenshtein/token) | `experiments/evaluation/metrics.py` → `wer()` |
| **CER** | Character Error Rate | `metrics.py` → `cer()` |
| **RTF** | `wall_time_sec / audio_duration_sec` (< 1 = nhanh hơn realtime) | `metrics.py` → `rtf()` |
| **Scene P/R/F1** | Boundary match ±2s giữa pred cuts vs pseudo GT | `runners.py` → `eval_scene_boundaries()` |
| **Keyframe P/R/F1** | Hit trong TVSum importance windows ±0.5s | `runners.py` → `eval_clip_filter_video()` → `_score()` |
| **Nén keyframe** | `n_after / n_before` (scene timestamps) | `eval_clip_filter_video()` |
| **Timeline Acc/MAE** | Utterance ↔ scene alignment | `runners.py` → `eval_timeline_alignment()` |
| **Chapter F1** | Boundary P/R/F1 chapter (± tolerance) | `runners.py` → `eval_chapters()` |
| **ROUGE-L** | ROUGE-L F1 extractive summary | `metrics.py` → `rouge_l_f1()` |
| **Caption content_ok** | Heuristic hallucination/grounding | `metrics.py` → `caption_hallucination_flags()` |

Model production được đăng ký tại: `experiments/evaluation/model_compare.py` → `PRODUCTION`.

Pipeline inference thật: `ai_workers/tasks.py` → `process_video`.

---

## 4. Bảng tra nguồn từng con số

### 4.1 ASR — Faster-Whisper

| Con số | Giá trị | Nguồn trực tiếp | Cách tính |
|--------|---------|-----------------|-----------|
| WER base.en (12 clip) | **13.5%** (0.135) | `MODEL_CHOICE.md` §1 ← `EVAL_TABLES.json` key `asr` | Mean WER 12 row `base.en` trong JSON |
| WER small.en (12 clip) | **16.1%** (0.161) | cùng nguồn | Mean WER 12 row `small.en` |
| CER base.en | **4.6%** | `MODEL_CHOICE.md` | Mean CER từ JSON |
| RTF base.en | **0.226** | `MODEL_CHOICE.md` | Mean RTF từ JSON |
| RTF small.en | **0.635** | `MODEL_CHOICE.md` | Mean RTF; ratio ≈ 2.8× chậm hơn base |
| WER base/small/tiny (2 clip) | 19.3% / 12.5% / 14.6% | `eval_model_compare/EVAL_TABLES.md` Bảng 1 | Subset nhỏ, **không** dùng làm kết luận chính |
| RTF (2 clip) | 0.96 / 2.24 / 1.58 | cùng file | Subset 2 clip |

**Script ASR:** `runners.py` → `eval_asr_file()` gọi `AudioTranscriber` (`ai_workers/modules/audio_v2/transcriber.py`, default `base.en`).

**Aggregate ASR:** `run_model_choice_eval.py` → `load_asr_compare()` đọc `outputs/eval_tables_real/EVAL_TABLES.json`.

---

### 4.2 Scene — PySceneDetect (chỉ absolute, chưa so sánh threshold)

| Con số | Giá trị | Nguồn | Ghi chú |
|--------|---------|-------|---------|
| F1 mean (4 video) | **0.464** | `eval_tables_real/EVAL_TABLES.md` Bảng 3 | EE-bNr36nyA 0.326, iVt07TCkFM0 0.721, 91IHQYk1IQM 0.440, -esJrBWj2d8 0.368 |
| F1 (1 video) | **0.326** | `eval_model_compare/EVAL_TABLES.md` Bảng 2 | P=1.0, R=0.194, 7 pred / 36 GT |
| Threshold | **27.0** | `ai_workers/core/config.py` → `SCENE_THRESHOLD` | Chưa có bảng so sánh 20/35 trong output hiện tại |

**Detector:** `ai_workers/modules/visual_v2/scene_detector.py` (PySceneDetect ContentDetector, `frame_skip=4`).

---

### 4.3 Keyframe — CLIP vs keep-all vs temporal-dedup

| Con số | Giá trị | Nguồn | Ghi chú |
|--------|---------|-------|---------|
| CLIP F1 EE-bNr36nyA | P=0.75, R=0.50, F1=**0.60**, nén=0.50 | `MODEL_CHOICE.md` §2 / `MODEL_CHOICE.json` | 8 scene → 4 sau CLIP |
| keep_all F1 cùng video | F1=**0.667**, nén=1.0 | cùng nguồn | |
| CLIP F1 iVt07TCkFM0 | F1=**0.556**, nén=0.40 | cùng nguồn | 25 → 10 scene |
| keep_all F1 cùng video | F1=**0.611**, nén=1.0 | cùng nguồn | |
| F1 mean CLIP (4 video) | **~0.482** | `eval_tables_real/EVAL_TABLES.md` Bảng 4 | Run rộng hơn model_choice |

**CLIP model:** `openai/clip-vit-base-patch32` — `ai_workers/modules/visual_v2/semantic.py` → `filter_scenes_clip()`.

**Eval:** `runners.py` → `eval_clip_filter_video()`; bakeoff: `run_model_choice_eval.py`.

---

### 4.4 VAD (baseline, không so sánh model)

| Con số | Giá trị | Nguồn |
|--------|---------|-------|
| F1 mean | **~0.404** | `eval_tables_real/EVAL_TABLES.md` Bảng 2 |
| Barry: P/R | 0.865 / 0.333 | cùng file |
| Blaise: P/R | 0.379 / 0.289 | cùng file |

Built-in VAD Faster-Whisper (`transcriber.py`, `min_silence_duration_ms=2000`).

---

### 4.5 Timeline (baseline, không so sánh model)

| Con số | Giá trị | Nguồn |
|--------|---------|-------|
| Accuracy | **1.000** (2 talk) | `eval_tables_real/EVAL_TABLES.md` Bảng 7 |
| MAE | **0.000 s** | cùng file |

Thuật toán: `ai_workers/modules/fusion/timeline.py` (70% temporal + 30% TF-IDF).

---

### 4.6 Chapter (baseline, không so sánh model)

| Con số | Giá trị | Nguồn |
|--------|---------|-------|
| Boundary F1 Blaise | **0.364** | `eval_tables_real/EVAL_TABLES.md` Bảng 8 |
| Boundary F1 Barry | **0.000** | cùng file |
| F1 mean | **~0.18** | mean 2 talk |

Thuật toán: TF-IDF topic shift — `timeline.py`.

---

### 4.7 Summary — ablation modality (không so sánh LLM)

| Con số | Giá trị | Nguồn | Ghi chú |
|--------|---------|-------|---------|
| ROUGE-L Audio+Visual Blaise | **0.298** | `eval_tables_real` Bảng 9 | Extractive TF-IDF proxy |
| ROUGE-L Barry | **0.389** | cùng file | |
| Audio only | 0.304 / 0.391 | Bảng 10 ablation | |
| Visual only | 0.045 / 0.010 | Bảng 10 ablation | OCR+caption đóng góp <5% |

**Không có số liệu** Qwen 2.5 7B vs Llama vs GPT — production LLM: `ai_workers/modules/fusion/summarizer.py`.

---

### 4.8 Stage bỏ qua (TBD / chưa A/B)

| Stage | Trạng thái | Nguồn ghi nhận |
|-------|------------|----------------|
| **OCR** | CER = TBD | `eval_tables_real` Bảng 5; `eval_model_compare` Bảng 4 |
| **Caption Florence vs alt.** | Florence **skipped** (RAM < 6 GB) | `MODEL_CHOICE.md` §3; `florence_runtime.py` gate |
| **Scene threshold 20/35** | Chưa có output | Candidate trong `model_compare.py` → `compare_scene_thresholds()` |
| **LLM model compare** | Chưa chạy | Summary eval = TF-IDF proxy |
| **Speaker diarization** | Stub | `ai_workers/modules/audio_v2/speaker.py` |
| **RAG embedding** | Không eval | `backend/app/services/chromadb.py` |

---

## 5. Kết luận lựa chọn (chỉ từ số liệu trên)

### 5.1 ASR — **giữ `faster-whisper-base.en`**

| Hạng | Model | Căn cứ |
|:----:|-------|--------|
| **1** | base.en | WER 0.135 < 0.161 (small), RTF 0.226 < 0.635 — nguồn: 12 clip `MODEL_CHOICE` |
| 2 | tiny.en | Chỉ có 2-clip eval |
| 3 | small.en | WER cao hơn base trên 12 clip; RTF ×2.8 |

### 5.2 Keyframe — **giữ CLIP ViT-B/32 agglomerative**

| Hạng | Chiến lược | Căn cứ |
|:----:|------------|--------|
| **1** | CLIP agglomerative | Nén 40–50%; F1 giảm ~0.05–0.07 vs keep_all — `MODEL_CHOICE.json` |
| 2 | keep_all | F1 cao nhất, nén = 1.0 |
| 3 | temporal_dedup | F1 ≤ keep_all trên 2 video đo |

> Chọn CLIP vì **trade-off F1 / nén đã đo**, không vì SOTA visual.

### 5.3 Các stage khác

| Stage | Kết luận eval | Biện minh lựa chọn model? |
|-------|---------------|:--:|
| Scene PySceneDetect-27 | F1 ~0.33–0.46 (TVSum proxy) | ❌ Chưa so candidate |
| Timeline TF-IDF | Acc 1.0, MAE 0s | ❌ Một thuật toán |
| Chapter TF-IDF | F1 ~0.18 — yếu | ❌ Một thuật toán |
| VAD | F1 ~0.40 — cần cải thiện | ❌ Một phương pháp |
| Summary | Audio >> Visual (ablation) | ❌ Không so LLM |
| OCR / Caption / LLM | TBD hoặc skipped | ❌ |

---

## 6. Pipeline production (tham chiếu code)

```
Video
 → Faster-Whisper base.en          [WER/RTF: §4.1]
 → PySceneDetect thr=27            [F1: §4.2]
 → CLIP ViT-B/32 agglomerative     [F1/nén: §4.3]
 → PaddleOCR (vi)                  [chưa có CER — §4.8]
 → Florence-2-base hoặc OCR-grounded fallback  [§4.8]
 → TF-IDF timeline + chapter       [§4.5, §4.6]
 → Qwen 2.5 7B (OpenRouter)        [chưa eval ROUGE — §4.7]
 → ChromaDB RAG (async)            [chưa eval]
```

Config: `ai_workers/core/config.py`, `backend/.env`.

---

## 7. Cách tái tạo số liệu

```powershell
# Từ thư mục gốc repo
cd c:\Users\admin\multimodal-lecture-summarizer

# Unit test metric (không GPU)
python experiments/scripts/test_eval_metrics.py

# Eval đầy đủ → outputs/eval_tables_real/
python experiments/scripts/run_eval_tables.py `
  --out-dir outputs/eval_tables_real `
  --ted-limit 2 --tvsum-limit 4 --asr-limit 12 `
  --asr-models base.en,small.en `
  --model-compare

# Bakeoff 3 model production → outputs/model_choice/
python experiments/scripts/run_model_choice_eval.py

# Subset nhanh + tiny.en → outputs/eval_model_compare/
python experiments/scripts/run_eval_tables.py `
  --out-dir outputs/eval_model_compare `
  --ted-limit 1 --tvsum-limit 1 --asr-limit 2 `
  --asr-models tiny.en,base.en,small.en --model-compare
```

**Yêu cầu:** TED-LIUM và TVSum tại `D:\datasets\...` (xem `datasets.py`). Nếu thiếu cuDNN 8, ASR fallback CPU — log mẫu: `outputs/eval_full_run.log`.

---

## 8. Thí nghiệm cần chạy trước khi bổ sung luận văn

| # | Thí nghiệm | Script / hàm | Mở khóa |
|---|-----------|--------------|---------|
| 1 | Annotate slide + OCR CER | `compare_ocr_engines()` | PaddleOCR vs EasyOCR |
| 2 | Scene threshold 20/27/35 | `compare_scene_thresholds()` | Biện minh thr=27 |
| 3 | Florence trên RAM ≥ 6 GB | `run_model_choice_eval.py` / `run_florence_eval.py` | Caption A/B |
| 4 | LLM thật ROUGE trên TED | mở rộng `eval_summary_pair()` + API key | Qwen vs fallback |
| 5 | tiny.en trên 12 clip | `run_eval_tables.py --asr-models tiny.en,base.en,small.en` | Hoàn thiện ASR ranking |

---

## 9. Tài liệu liên quan

| Tài liệu | Nội dung |
|----------|----------|
| [`docs/EVAL_TABLES.md`](EVAL_TABLES.md) | Khung bảng luận văn, lệnh CLI |
| [`docs/DATASET_EDA.md`](DATASET_EDA.md) | EDA TED-LIUM + TVSum |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | Kiến trúc pipeline |
| [`experiments/evaluation/model_compare.py`](../experiments/evaluation/model_compare.py) | Production registry + compare helpers |

---

*Cập nhật: 2026-08-14 — số liệu lấy từ các file output liệt kê mục 2; không dùng benchmark bên ngoài repo.*
