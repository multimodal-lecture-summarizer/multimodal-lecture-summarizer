---
phase: 3
title: RateLimit-Free Engine
status: completed
priority: P1
dependencies:
  - phase-02
---

# Phase 3: RateLimit-Free Engine

## Overview
Làm `04_phase4_hierarchical_summarization.ipynb` + `05` retrieval QA hết phụ thuộc Gemini free-tier (429 abort) bằng engine stack offline-first đã implement ở `4013290`: `Qwen2.5-1.5B-Instruct` singleton (3.5 GB FP16) làm default, `DeterministicAbstractiveEngine` v2 SBERT-centrality làm guaranteed fallback, `GeminiLLMEngine` chỉ opt-in với `retry 2× (2s/4s) + fallback` never `""`. Thêm chunked `outputs/phase4_cache` + `outputs/phase5_cache` resume để survive Colab 12h kill. Code đã xong — phase này là tích hợp, hardening cache/resume, và wiring vào notebooks đã fix ở P2.

## Requirements
- Functional: `get_llm_engine(preference="auto"|"hf"|"deterministic"|"gemini"|"hf-bart")` factory single source; `auto`→HF nếu khả dụng else deterministic; `hf`→raise nếu HF fail; `gemini`→retry 2× exponential rồi fallback HF/deterministic với warning never raise 429; `04`/`05` + `run_rq2/rq3_benchmark.py` share 1 `llm` instance (no per-chapter `get_llm_engine()`); `04` 100 vids `<30 min` T4, 0 Gemini khi `auto`; `outputs/phase4_cache/{sha1(prompt)}.json` hit `>80%` on resume; `05` 5,252 QA offline no `google.generativeai` import.
- Non-functional: `pip install <2 min`, model load `<60s` `HF_HOME=/root/.cache/huggingface`, CPU→skip HF→deterministic directly, VRAM `<12 GB`, `D-T08` `32k/512` strict (fail not scale), `notebook diff <100 LOC`.

## Architecture
```
llm_engine.py (4013290 patched):
  BaseLLMEngine.generate(prompt, max_tokens, temperature)  # contract
  ├─ HuggingFaceLLMEngine(model_id=Qwen/Qwen2.5-1.5B-Instruct | facebook/bart-large-cnn)
  │    device auto cuda/cpu, dtype fp16 on cuda, HF_HOME env, chat vs seq2seq branch
  │    singleton _HF_CACHE so S0/S1/S3/S4 share 1 model (no reload per chapter)
  ├─ DeterministicAbstractiveEngine v2
  │    sbert_available? SBERT-centroid else TF-IDF keyword scoring → **Chapter N [ts]**: bullets
  │    respects max_tokens truncation (max_words=int(max_tokens/1.3))
  └─ GeminiLLMEngine + RateLimitBackoff (2 retries 2s/4s, final fallback, never "")
  get_llm_engine(preference):
    deterministic→Det; hf→try HF except→Det; gemini→try Gemini except→_auto_engine warn; auto→try HF except→Det

04 cell 4 (patched):
  LLM_PREFERENCE = os.getenv("LLM_PREFERENCE","auto")  # auto|hf|deterministic|gemini|hf-bart
  llm = get_llm_engine(preference=LLM_PREFERENCE)
  print(f"[LLM Engine] Using {llm.__class__.__name__} (preference={LLM_PREFERENCE})")
  summarizers = {S0(...,llm_engine=llm), S1(...,llm_engine=llm), S3(...,llm_engine=llm), S4(...,llm_engine=llm)}

04 hardened loop:
  for lecture in manifest_v2 (100):
    for chapter in C5 predicted_boundaries:
      cache_key=sha1(chapter_text+variant_id)  # must include boundaries hash
      if cache_key in outputs/phase4_cache/*.json: load else llm.generate→write cache
  Batch 4 chapters/call, torch.cuda.empty_cache() per 10 lectures

05 hardened:
  full EduVidQA 5252 QA → DenseEmbedder all-MiniLM-L6-v2 SBERT + BM25 hybrid → top_k=3 (600 tok) → llm.generate
  per-question cache outputs/phase5_cache/{q_hash}.json
```

Selection matrix (covers `preference × key × HF × device`):

| pref | key | HF ok | device | chosen |
|------|-----|-------|--------|--------|
| auto | any | yes | cuda | Qwen |
| auto | any | any | cpu | Deterministic (warn `HF skipped on CPU`) |
| hf | any | yes | any | Qwen |
| hf | any | no | any | Deterministic |
| gemini | yes | any | any | Gemini (backoff→fallback) |
| gemini | no | any | any | fallback HF/Det warn |
| deterministic | any | any | any | Deterministic |

## Related Code Files
- Modify: `benchmarks/models/llm_engine.py` (extend HF/Det/factory/backoff — đã ở `4013290`, phase này thêm `phase4_cache` helper nếu cần), `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb` (cell 4 LLM_PREFERENCE + cell 6 `llm_engine=llm` + cache resume cell + batch), `benchmarks/models/summarization.py` (type hint tightening nếu cần, no logic), `benchmarks/models/retrieval_qa.py` (ensure `DenseEmbedder` singleton)
- Create (optional): `benchmarks/models/hf_engine.py` chỉ nếu `llm_engine.py >250 LOC` (YAGNI)
- Delete: (none)
- Reference: `benchmarks/models/summarization.py:41-328 SummarizerConfig S0-S4`, `benchmarks/models/llm_engine.py:22-117`, `requirements.freeze.txt`/`lock.txt` (transformers 4.57.6)

## Implementation Steps
1. **Audit existing code (4013290):** `Read llm_engine.py` confirm `HuggingFaceLLMEngine` (Qwen/BART branch, `device_map auto`, `torch.float16` on cuda, `HF_HOME`, singleton `_HF_CACHE`), `Deterministic v2` (SBERT-centroid path), `get_llm_engine(preference)` + `RateLimitBackoff` `retry 2×` already present. Diff vs phase spec — patch only missing pieces (e.g. `HF skipped on CPU` log, `never ""` guarantee).
2. **Patch `llm_engine.py` (if needed):**
   ```python
   def get_llm_engine(preference="auto"):
       pref = os.getenv("LLM_PREFERENCE", preference)
       if pref == "deterministic": return DeterministicAbstractiveEngine()
       if pref == "hf":
           try: return HuggingFaceLLMEngine()
           except Exception as e: warnings.warn(f"HF failed {e}, fallback"); return DeterministicAbstractiveEngine()
       if pref == "gemini":
           try: return GeminiLLMEngine()  # inside generate: 2 retries 2s/4s, final fallback not ""
           except Exception as e: print(f"Gemini failed {e}, fallback"); return _auto_engine()
       # auto
       if not torch.cuda.is_available(): warnings.warn("HF skipped on CPU"); return DeterministicAbstractiveEngine()
       try: return HuggingFaceLLMEngine()
       except Exception: return DeterministicAbstractiveEngine()
   ```
   Add `cache_key = sha1(chapter_text+variant_id+boundaries_hash)` helper if absent (cache invalidation on C5 change).
3. **Patch `04` cell 4 & 6:** Insert 4-line config (`LLM_PREFERENCE` + `get_llm_engine` + `print`) before `test_files` block; wire `summarizers = {S0(..., llm_engine=llm), S1(...,llm_engine=llm), S3(...), S4(...)}`. Preserve `USE_CACHED_FALLBACK` / `vtssum_clone_target` logic from `4ab8963` (rebase, additive not replace).
4. **Add cache resume to `04` cell 7:** `cache_dir=PROJECT_ROOT/"outputs/phase4_cache"; cache_dir.mkdir(parents=True,exist_ok=True)`; before `llm.generate` check `cache_file=cache_dir/f"{sha1(prompt+variant_id)}.json"` load else generate→write. Batch 4 prompts, `torch.cuda.empty_cache()` per 10 lectures, flush per lecture.
5. **Harden `05` cache:** Same pattern `outputs/phase5_cache/{question_hash}.json`; remove `MAX_QUESTIONS_PER_LECTURE` cap, load full `q_and_a.json` (≥5000), ensure `DenseEmbedder` singleton (SBERT `all-MiniLM-L6-v2`) offline; assert no `google.generativeai` import when `auto`.
6. **Budget & shared-instance checks:** Assert `cfg.max_source_tokens==32000` + `res.token_usage["output_tokens"]<=512` for S0/S1/S3/S4; assert `summarizers["S0"].llm is summarizers["S3"].llm`; `On CPU: get_llm_engine("auto").__class__==DeterministicAbstractiveEngine`.
7. **Verification (static + T4 dry-run):** `grep` 0 per-chapter `get_llm_engine()` calls; `python -m json.tool` + `ast.parse` patched cells pass; T4 Run A `LLM_PREFERENCE=auto` (no key) → log `HuggingFaceLLMEngine` or `Deterministic` (allowed fallback), zero `429`; Run B `deterministic` → bullet summaries `**Chapter` prefix, ROUGE non-zero, `token_usage ≤D-T08`; optional Run C `gemini`.

## Success Criteria
- [ ] `llm_engine.py` có `preference` param, HF singleton, deterministic v2 SBERT-centrality, Gemini backoff 2× never `""`, `auto` trên CPU→deterministic. **Hybrid cache ready.**
- [ ] `04` cell 4 4-line `LLM_PREFERENCE` config + cell 6 pass `llm_engine=llm` cho S0/S1/S3/S4 (share instance), batch + `outputs/phase4_cache` resume present, diff `<100 LOC`, JSON valid.
- [ ] `04` T4 (no key) 0 `GeminiLLMEngine` instantiations, 25 vids `<10 min` hoặc 100 `<30 min` (300 vids ~60' with 6 shards, resume second run `>80%` hit <5'), 0 `llm.generate` for cached chapters.
- [ ] `05` full EduVidQA `≥1000` QA items offline (target 5,252), no `429`, `D-T08` holds, `test_summarization.py 7/7` + `test_retrieval_qa.py 6/6` pass.
- [ ] `git diff --stat` chỉ `llm_engine.py` + `04` notebook (+ minimal `summarization.py` if needed).

## Colab Free Execution (Approach A, 2026-09-01 — keep 5 notebooks, >60' with checkpoint, Hybrid)

- **04 notebook** (16c) **20–30'** for 100 vids / **~60'** for 300 vids (6 shards). **Hybrid:** `hybrid_load()` tries `DRIVE_CACHE/outputs/phase4_cache` → fallback `Qwen2.5-1.5B` generation and saves `sha1(prompt+boundaries)` JSON. Second run <5' (cache hit).
- **Sharding:** Loop `for shard in 0..5: lectures[shard::6]` or `shard 0/6` flag; each shard ~17 vids (100) / 50 vids (300). Cell per shard with `print(f"Shard {s}/6 done — resume --shard {s+1}")` and `torch.cuda.empty_cache()` per 10 lectures.
- **05 retrieval** stays inside same notebooks but per-question cache `outputs/phase5_cache/{hash}.json` makes 5k QA <25' (offline SBERT+BM25, no Gemini). Full 300 `>60'` chấp nhận với checkpoint per-question.
- **>60' policy:** User approved `>60' with checkpoint` — 04/05 will intentionally run >60' for 300 and rely on resume after 12h kill. No split to 7 notebooks needed.

## Risk Assessment
- **R HF download timeout:** Run B deterministic proves offline path; A được phép fallback deterministic và vẫn pass.
- **R VRAM OOM Qwen FP16:** catch `CUDA OOM` → fallback BART/Deterministic per remaining lectures + log, not crash.
- **R Cache invalidation nếu C5 boundaries đổi (new manifest v2):** `cache_key` phải include `predicted_boundaries` hash; otherwise stale summaries — spec đã ghi.
- **R Patch overwrites `USE_CACHED_FALLBACK` (`4ab8963`):** Depends on P2; apply sau rebase; verify `possible_test_dirs` + `vtssum_clone_target` intact.
- **R ROUGE thấp hơn Gemini:** Document trong markdown `ROUGE offline thấp hơn Gemini là expected; invariant là hierarchy gain S3-S1, S4-S3, không phải absolute ROUGE`.
- **R deterministic quality:** v2 phải ra bullet format `**Chapter N [ts]**`; nếu không, ROUGE/coverage mismatch.
