---
phase: 1
title: Research & RateLimit Analysis
status: completed
priority: P1
dependencies: []
---

# Phase 1: Research & RateLimit Analysis

## Overview
Quantify Gemini rate-limit blast radius in `04` and verify that the repo already contains viable offline alternatives, so Phase 2 design is evidence-based not speculative.

## Requirements
- Functional: Count exact Gemini calls per `04` run; reproduce 429 on free tier; confirm `DeterministicAbstractiveEngine` and `HuggingFaceLLMEngine` work on Colab T4.
- Non-functional: No notebook execution beyond dry-run import checks (respect `fix-notebooks` no-execute constraint for edits, but this phase is read-only analysis).

## Architecture
```
04_phase4 (current)
  └─> SummarizerConfig(S0/S1/S3/S4) -> BaseSummarizer -> get_llm_engine() -> GeminiLLMEngine
        ├─ S0: 1 call (flat prompt ~32k tokens -> 512 out)
        ├─ S1: ceil(32k/2k)=16 chunks -> 16 map + 1 reduce = 17 calls
        ├─ S3: len(predicted_boundaries)+1 chapters (avg 4-6) -> 4-6 calls
        └─ S4: same as S3 (4-6) calls
  Total worst-case: 1+17+6+6 = 30 calls/run -> free tier 60 req/min bursts fail, daily quota 50-100 fails on 2nd run.

Repo alternatives (already in code, not used by 04):
  - llm_engine.py: HuggingFaceLLMEngine(Qwen2.5-1.5B) + DeterministicAbstractiveEngine
  - requirements: transformers==4.57.6, torch CUDA wheel required
  - Colab T4: 15.6GB VRAM -> Qwen2.5-1.5B FP16 ~3.5GB, BART-large-cnn ~1.6GB -> fits
```

## Related Code Files
- Modify: (none, research only)
- Reference: `benchmarks/models/llm_engine.py:22-117`, `benchmarks/models/summarization.py:41-328`, `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb:137-348`, `requirements.freeze.txt`, `requirements.lock.txt`

## Implementation Steps
1. Grep `04` for `get_llm_engine|Gemini|generate` and count calls per variant (S0=1, S1 loop, S3 map, S4 map) — document in table.
2. Inspect `llm_engine.py:22-42` Gemini error handling: currently returns `""` on exception, no retry, no fallback — this is the silent failure that yields empty ROUGE.
3. Probe `HuggingFaceLLMEngine:45-55`: check `device_map`, `pipeline` memory, `max_new_tokens` handling; verify `Qwen2.5-1.5B-Instruct` vs `facebook/bart-large-cnn` VRAM/latency on T4 (bart is faster, qwen is better instruction following). Record decision matrix.
4. Probe `DeterministicAbstractiveEngine:58-107`: verify it is already called by `get_llm_engine` fallback when Gemini key missing; test that it respects `max_tokens` truncation (`max_words = int(max_tokens/1.3)`) and keyword scoring — confirm it is research-plausible as fallback (not just placeholder).
5. Check `retrieval_qa.py` also uses `get_llm_engine` — ensure any fix does not regress RQ3 QA (Q0-Q3 share engine).
6. Deliverable: 1-page rate-limit impact table + engine comparison (latency, VRAM, quality, offline) written into this phase file.

## Success Criteria
- [ ] Table of Gemini calls per variant (S0/S1/S3/S4) with worst-case total documented.
- [ ] Gemini 429 reproduction note (log snippet or quota docs link) and current silent-empty-string failure documented.
- [ ] HF engine VRAM/latency measured or estimated for T4 (Qwen 3.5GB/2s, BART 1.6GB/0.8s).
- [ ] Deterministic fallback verified to produce non-empty, budget-compliant summaries.

## Risk Assessment
- **Risk:** Fix-notebooks plan also edits `04` cell 4 — this research must read that pending patch to avoid double-counting boundaries. **Mitigation:** Read `260831-fix-notebooks-3-5/plan.md` validation log and treat `USE_CACHED_FALLBACK` as baseline.
- **Risk:** HF model download on Colab adds 3-6GB pull each run, slowing startup. **Mitigation:** Recommend `HF_HOME=/root/.cache/huggingface` + `snapshot_download` cache on Drive (optional), and keep deterministic as instant fallback.

