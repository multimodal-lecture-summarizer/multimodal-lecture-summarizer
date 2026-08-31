---
phase: 3
title: Implementation & Notebook Patch
status: completed
priority: P1
dependencies:
  - phase-02
---

# Phase 3: Implementation & Notebook Patch

## Overview
Implement the factory and patch `04` so it runs rate-limit-free on Colab T4 with a single config cell, preserving D-T08/D-T07 and re-using the real-data fix from `260831-fix-notebooks-3-5`.

## Requirements
- Functional: `04` cell 4 gains `LLM_PREFERENCE` + `get_llm_engine` injection; S0/S1/S3/S4 all use the same engine instance; no per-chapter `get_llm_engine()` calls (prevents 20× model loads).
- Non-functional: Notebook diff <100 LOC, no new dependencies beyond `transformers` already in `requirements`; edits are JSON cell patches, not rewrites.

## Architecture
```
04 cell 4 (patched):
  LLM_PREFERENCE = "auto"  # user edits this 1 line to switch
  llm = get_llm_engine(preference=LLM_PREFERENCE)  # singleton, logs backend
  # ... then all summarizers share `llm`:
  summarizers = {
    "S0": S0_FlatSummarizer(cfg_s0, llm_engine=llm),
    "S1": S1_FixedChunkMapReduceSummarizer(cfg_s1, llm_engine=llm),
    "S3": S3_PredictedHierarchySummarizer(cfg_s3, llm_engine=llm),
    "S4": S4_MultimodalHierarchySummarizer(cfg_s4, llm_engine=llm),
  }

llm_engine.py (patched):
  class HuggingFaceLLMEngine:  # improved, handles Qwen vs BART
  class DeterministicAbstractiveEngine:  # v2 with SBERT-centrality
  def get_llm_engine(preference):  # auto/hf/deterministic/gemini with fallback
```

## Related Code Files
- Modify: `benchmarks/models/llm_engine.py`
- Modify: `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb`
- Modify: `benchmarks/models/summarization.py` (only if injection type hint needs tightening; otherwise no change)
- Delete: (none)

## Implementation Steps
1. **Patch `llm_engine.py`:**
   - Update `HuggingFaceLLMEngine.__init__(model_id, device)` to auto-detect device, use `torch.float16` on cuda, `HF_HOME` from env, and branch `generate` for `bart` (seq2seq) vs `qwen` (chat). Add singleton cache `_HF_CACHE`.
   - Upgrade `DeterministicAbstractiveEngine.generate` to try SBERT centrality when `sentence_transformers` available (re-use `DenseEmbedder`), else keep keyword scoring. Keep `max_tokens` truncation.
   - Rewrite `get_llm_engine(preference="auto")`:
     ```python
     def get_llm_engine(preference="auto"):
         pref = os.getenv("LLM_PREFERENCE", preference)
         if pref == "deterministic": return DeterministicAbstractiveEngine()
         if pref == "hf": 
             try: return HuggingFaceLLMEngine()
             except: return DeterministicAbstractiveEngine()
         if pref == "gemini":
             try: return GeminiLLMEngine()
             except Exception as e: print(f"Gemini failed {e}, fallback"); return _auto_engine()
         # auto
         try: return HuggingFaceLLMEngine()
         except: return DeterministicAbstractiveEngine()
     ```
   - Add `RateLimitBackoff` to `GeminiLLMEngine.generate`: 2 retries with `time.sleep(2**attempt)` on 429/ResourceExhausted; on final fail return fallback call, not `""`.

2. **Patch `04` notebook cell 4 (after `possible_test_dirs` block, before `test_files`):**
   ```python
   # LLM backend selection (rate-limit-free, D-T08 compliant)
   LLM_PREFERENCE = os.getenv("LLM_PREFERENCE", "auto")  # auto | hf | deterministic | gemini | hf-bart
   from benchmarks.models.llm_engine import get_llm_engine
   llm = get_llm_engine(preference=LLM_PREFERENCE)
   print(f"[LLM Engine] Using {llm.__class__.__name__} (preference={LLM_PREFERENCE})")
   ```
   Then cell 6 (summarizer instantiation) must pass `llm_engine=llm` to all 4 constructors. Currently `summarization.py:43` already supports `llm_engine` param — just wire it.

3. **Preserve fix-notebooks patch:** Ensure `USE_CACHED_FALLBACK` / `vtssum_clone_target` logic from `4ab8963` remains intact; this plan's LLM patch is additive, not replacing. Rebase if conflict.

4. **Do not edit `requirements`:** `transformers`, `sentence-transformers`, `torch` already present. If `Qwen2.5-1.5B` needs `accelerate`, add `accelerate>=0.28` only if not present (check `requirements.lock.txt`).

5. **Strip outputs:** After edit, clear all `execution_count`/`outputs` (already done for `4ab8963`) so push does not carry Colab run artifacts.

## Success Criteria
- [ ] `llm_engine.py` has `preference` param, HF singleton, deterministic v2, and Gemini backoff+fallback.
- [ ] `04` cell 4 has 4-line LLM config, cell 6 passes `llm_engine=llm` to S0/S1/S3/S4.
- [ ] `git diff --stat` shows only `llm_engine.py` + `04` notebook (and minimal `summarization.py` if needed).
- [ ] Notebook JSON is valid (`python -m json.tool` passes) and `ast.parse` of patched cell succeeds.

## Risk Assessment
- **Risk:** HF model load fails on Colab due to `transformers` version mismatch. **Mitigation:** Pin `transformers==4.57.6` already; test import in Phase 4 dry-run before push.
- **Risk:** Patch overwrites the `USE_CACHED_FALLBACK` logic from `260831-fix-notebooks-3-5`. **Mitigation:** This plan explicitly depends on that plan; apply this patch **after** rebasing onto `4ab8963`.
- **Risk:** Deterministic fallback produces short/bullet summaries that lower ROUGE vs Gemini, causing user to think RQ2 is worse. **Mitigation:** Document in notebook markdown: "ROUGE with offline engine is expected lower than Gemini; RQ2 hierarchy gain (S3-S1, S4-S3) is the invariant, not absolute ROUGE."

<!-- Updated: Validation Session 2 - Gemini 429 now retry 2× (2s/4s) + fallback HF/deterministic, never return "" -->
**Validation Update (2026-08-31):** `GeminiLLMEngine.generate` must retry 2× on 429 then fallback to HF/deterministic with warning (never `""`). `get_llm_engine(preference="gemini")` on key missing must also fallback with warning.

