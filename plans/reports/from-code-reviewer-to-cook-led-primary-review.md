# LED primary summarizer review

Status: DONE_WITH_CONCERNS

## Verification run
- py_compile on led_summarizer.py, summarizer.py, config.py, test_led_summarizer.py: OK. ruff is not installed, so no lint run.
- test_led_summarizer.py: 15 passed. test_extractive_fallback.py: 14 passed. test_llm_resilience.py: 14 passed. test_reliability_chaos.py and test_checkpoint_persistence.py were still running when this was written.

## Acceptance criteria
- LED only for en + CUDA + backend led: met. `led_enabled` checks backend and language; CUDA is checked via `torch.cuda.is_available()` inside try. vi, backend `llm` and no-GPU all return None.
- Map-reduce covers every source token: met. Windows are contiguous slices of PAYLOAD_CAP, reduce groups pack every note, and the loop raises if the note count does not shrink. Tests assert this.
- Word cap by tier ending at a sentence boundary: PARTIAL. See C1 and C2.
- Any LED failure falls through to LLM: met. Load, generate, close, extractive-structure failure and an empty summary all return None. A `status == "failed"` result also returns None.
- Result keys match the LLM schema: met, and the result is a superset. It has video_title, summary, key_takeaways, chapters, model_used, fallback_used, summary_method, summary_length, word_count, reading_time_sec and summary_variants. Extra keys: status, executive_summary, structure_method, llm_error=None.

## Concerns
- C1 (medium): a truncated tail passes the cap check without a sentence boundary. The benchmark says 15% of LED outputs have no terminal EOS, and `sentence_prefix` treats a trailing fragment as a sentence. Verified: `sentence_prefix("One two three. Four five six seven eight", 20)` returns the whole string, ending mid-sentence. Fix: when the text does not end in `.!?`, drop the last fragment if at least one complete sentence exists. Add a test for this.
- C2 (low): the "first sentence too long" branch cuts mid-sentence by design. It is acceptable, but the plan claim "ends on a sentence boundary" is not absolute.
- C3 (low): the regex splits on abbreviations. `sentence_prefix("Dr. Smith said hi. Next one here.", 3)` returns just "Dr.". This is rare, but a lecture summary can hit it.
- C4 (low): `dt.datetime.utcnow()` is deprecated. It matches the existing code, so no change is needed.
- C5 (info): the tests mock LedModel, so LedModel.generate has never run on a GPU. The padding, global attention and `dtype=` kwarg are untested. `dtype=` is only valid on recent transformers; older versions need `torch_dtype=`. If the installed version rejects it, the failure is caught and falls back to the LLM silently, so LED would never run. Check the pinned transformers version in requirements.
- C6 (info): the test suite relies on an autouse conftest fixture that forces the backend to `llm`, so the default `led` is never exercised by other tests. This is intentional and documented in the fixture.

## Side effects and regressions
- No regression on vi or on the backend=llm path. The hook is an early return that only fires when `build_led_result` returns non-None. Existing callers of `summarize()` (tasks.py, the `summarize` wrapper at ~L1462) get the same signature and return type.
- tasks.py reads `fallback_used` and `summary_method` with `.get`. LED returns `fallback_used=False`, so it is treated like a successful LLM run. The new `summary_method="led"` flows through to the stored result. Check whether the frontend, API or DB enums restrict `summary_method` to `llm` or `extractive_fallback`. A grep of Python outside tests found no validation.
- Behaviour change for English jobs on GPU hosts: the default is now `led`. The LLM no longer produces the title, takeaways or chapters from OCR and slides for those jobs. They come from the extractive path, and OCR is unused in the summary. That is documented and was decided by the user.
- Every English GPU job loads the LED model fresh and unloads it, which adds latency. If other GPU models are resident (ASR, Florence), OOM falls back to the LLM, so there is no crash.
- The LED path does not need an API key. English jobs without a Groq/OpenRouter key now succeed, whereas before they hit the "no API key" placeholder. This is a benign improvement.
- The architecture doc's Stage 7 paragraph is updated, but the mermaid node (line 139) still says "Stage 7: LLM Summarization". Minor doc drift.
- No public contract breaks. New config keys have defaults, and the existing schema is preserved.
