# LED as primary summarizer for English output

Status: implemented, reviewed and tested 2026-11-03; not committed, never run on a GPU. Worker tests pass per file (14 files, 129 passed; florence_runtime has 5 env failures from torch/numpy/Pillow pin drift, unrelated). Review fix: `sentence_prefix` drops an unterminated tail.

## Requirements (user-confirmed 2026-11-03)
- LED (benchmark profile `led-ngram3`) replaces the LLM as the main summarizer. Scope is English lectures.
- OCR: PaddleOCR language `en` (config `PADDLEOCR_LANG`). Object detection/tracking off by default (`ENABLE_OBJECT_TRACKING=false`).
- Earlier decisions kept: Qwen3-ASR for English, Whisper for vi.

## Design
- Active when `output_language == "en"` and a CUDA GPU exists. `vi` output cannot be produced by LED (no translation step in scope) -> LLM path as before.
- Benchmarked profile reproduced: `allenai/led-large-16384-arxiv` (revision pinned), fp16, eager attention, encoder 4096 tokens padded to multiple of 1024, global attention on first token, beams 4, `no_repeat_ngram_size=3`, map/reduce notes 256 tokens, final 768 tokens, reduce groups joined by blank line, plain transcript input (no timestamps, no OCR).
- Delivery cap by sentence prefix: short 120 / medium 240 / detailed 350 words (benchmark S/M/L views + 350 cap).
- Title, key takeaways and chapter summaries stay extractive (existing `ExtractiveSummarizer.generate_fallback`); only `summary` is LED. Result: `summary_method="led"`, `fallback_used=False`.
- Any LED failure (no GPU, OOM, empty/implausible output) -> existing LLM path -> extractive fallback.
- Backend switch: `SUMMARIZER_ENGLISH_BACKEND` = `led` (default) | `llm`.

## Touchpoints
- new `ai_workers/modules/summarization/led_summarizer.py`
- `ai_workers/modules/summarization/summarizer.py` (early hook in `summarize`)
- `ai_workers/core/config.py`
- tests: `ai_workers/tests/test_led_summarizer.py`
- docs: `docs/current-system-architecture.md` Stage 7

## Acceptance
- LED path used only for `en` output + CUDA + backend `led`; otherwise unchanged behavior.
- Map-reduce covers every source token; single pass when transcript fits.
- Output word count <= tier cap, ends on a sentence boundary.
- LED failure falls through to LLM.
- Result keys match LLM/extractive schema (video_title, summary, key_takeaways, chapters, model_used, summary_variants, ...).
- Existing worker tests pass per file.

## Known limits (from benchmark; not fixed here)
- LED: 15% outputs without terminal EOS, 27/519 repetitive, 0.76% failed cells; less faithful than BART by MiniCheck (over-flags; not a hallucination rate).
- LED is below the app LLM on ROUGE-L (0.1438 vs 0.1595). Chosen as a product decision, not by the data.
- Long-lecture map-reduce was benchmarked only up to ~11.7k words; real long lectures untested.
- Chapters/takeaways/title are extractive; OCR is not used by LED.
- Never run on a real GPU in this session; weights not downloaded.
