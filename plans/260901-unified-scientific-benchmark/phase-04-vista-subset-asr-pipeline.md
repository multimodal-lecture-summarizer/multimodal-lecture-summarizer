---
phase: 4
title: VISTA Subset ASR Pipeline
status: completed
priority: P1
dependencies:
  - phase-01
---

# Phase 4: VISTA Subset ASR Pipeline

## Overview
Build VISTA primary cho RQ2 ở quy mô scoped `n=300` (không phải 18,599 / 1.93 TB), với self-ASR `openai/whisper-small` (D-T04) như `D-T05` đã xác định (`VISTA has no transcript field`). Unlock Nominal Path 1 (`D-T12` approved 2026-08-31) mà không thổi Drive quota hay Colab 12h. Hardening validate 2026-09-01 khóa `VISTA bắt buộc` (no auto TIB fallback) — fail-loud nếu gate không đạt.

## Requirements
- Functional: `benchmarks/scripts/vista_subset_asr.py` fetch `dongqi-me/VISTA` 300 random (form-gated, approved), download raw video via `huggingface_hub.snapshot_download(resume_download=True)` tới `HF_HOME` local SSD (model) + Drive temp (raw video per D-T09 split), chạy `whisper-small` (`faster-whisper`/`ctranslate2`) `language=en vad_filter=True`, align với paper abstract → `probes/cache/vista_subset/*.json` (`id, title, transcript_sentences≥20, timestamps, segmentation, summarization_data`). D-T14 audit 20 samples (`source_support 0-2, coverage 0-2, style, action keep/flag/exclude`) frozen trước S0-S4 gen. `tib-bench` 80 luôn chạy như external validation (không phải primary fallback).
- Non-functional: Idempotent `--resume`, chunked per 50 videos, 300 ASR `<12h` T4 (~2 min/vid →10h) fits 1 session với resume; resume không duplicate JSON; `WER` audit `<30%` before full run gate (fail-loud, không silent fallback).

## Architecture
```
HF dongqi-me/VISTA 18,599 (form-gated, CC BY 4.0) → random 300 (seed 42)
  → snapshot_download(allows *.mp4, resume) → local SSD / Drive temp (--shard 0..5 ×50)
  → whisper-small (openai/whisper-small, D-T04) language=en vad_filter=True
  → transcript_sentences + timestamps (sec)
  ├─→ align paper abstract → summarization_data
  └─→ probes/cache/vista_subset/{id}.json  {id,title,segmentation(transcript),summarization(abstract)}
        │
        ├─ D-T14 audit 20 random: source_support/coverage/style/action → exclusion list frozen
        └─ S3/S4 (P3 engine) consumes transcript_sentences with D-T08 32k/512 (P5/P6)
             │
             └─ nếu WER>30% hoặc attrition>50% → FAIL-LOUD (user decision 2026-09-01), không auto fallback TIB
                TIB 80 external vẫn chạy ở P6 nhưng không promote thành primary
```

Scale note: full 18,599 ≈1.93 TB + 25 days ASR; scoped 300 ≈32 GB +10h, fits Drive 30 GB + Colab resume.

## Related Code Files
- Create: `benchmarks/scripts/vista_subset_asr.py`, `probes/cache/vista_subset/*.json` (300), `tests/test_vista_asr.py` (post-hoc)
- Modify: `benchmarks/core/feature_store.py` (reuse ASR path if needed, add `--shard`/`--resume` flags), `requirements.freeze.txt` (ensure `faster-whisper`/`ctranslate2`)
- Delete: (none)
- Reference: `plans/260830-1917-scientific-benchmark/01-dataset-manifest.md §2 VISTA`, `03-colab-runbook.md §4 HF cache`, `04-rq-mapping.md RQ2`, `decisions-log.md D-T04/D-T05/D-T12/D-T14`, `reports/brainstorm-260831-ratelimit-data-expansion.md` (C-light scoped)

## Implementation Steps
1. **Pre-check gates:** Confirm HF `dongqi-me/VISTA` access still `approved` (user đã approved 2026-08-31); `hf auth login` check. `load_dataset("dongqi-me/VISTA")` inspect splits `train_part1/2, validation, test` + schema (video_path is preprocessed artifact, not transcript). Record `dataset_revision` vào `frozen_decisions.json`.
2. **Implement `vista_subset_asr.py`:**
   ```python
   # argparse: --limit 300 --shard 0/6 --resume --output probes/cache/vista_subset
   # 1) load 300 random IDs (seed 42) from VISTA test/validation
   # 2) for each shard 50: snapshot_download(repo_id="dongqi-me/VISTA", allow_patterns=["*.mp4"], local_dir=f"{DRIVE}/vista_raw/shard{shard}", resume_download=True)
   # 3) for each mp4: faster-whisper WhisperModel("small", device="cuda" if available else "cpu") transcribe(language="en", vad_filter=True) → segments with start/end/text
   # 4) build transcript_sentences=[s.text for s in segments], timestamps=[s.start ...], segmentation=segments
   # 5) load paper abstract as summarization_data (from HF record "abstract" field)
   # 6) write probes/cache/vista_subset/{id}.json {id,title,transcript_sentences,timestamps,segmentation,summarization_data, provenance:{asr_model:"whisper-small", whisper_version, timestamp}}
   # 7) idempotent: if json exists and --resume, skip (check mtime/hash)
   ```
   Use `huggingface_hub` + `faster-whisper` already in `requirements`; fallback `openai-whisper` if missing `ctranslate2`.
3. **WER audit gate:** Trên 5 videos có gold transcript (nếu VISTA subset nào có) chạy `jiwer` WER; assert mean `<0.3` hoặc at least `transcript_sentences` non-empty + reasonable length (`≥20 sents` avg). Nếu WER>30% hoặc attrition>50% → `raise SystemExit("VISTA gate failed: WER>30%/attrition>50% — manual decision required, no auto TIB fallback")` + ghi `reports/vista_gate_failure.md`. Post-hoc test `test_asr_wer_audit_below_30` verifies.
4. **D-T14 audit (20 samples):** Random 20 `vista_subset/*.json`, apply rubric `source_support/coverage/style/action` (from `01-dataset-manifest.md §5`). Nếu `flag>30%` hoặc `exclude>10%` → freeze exclusion list `probes/cache/vista_subset/exclusion_list.json` trước khi S0-S4 generation. Report stats.
5. **Shard + resume verification:** Run `--shard 0 --limit 5` twice, assert no duplicate JSONs, `len(json)==5`, each has `transcript_sentences≥20` and `summarization_data.abstract` non-empty.
6. **Wire to P3 engine:** Confirm P3 `get_llm_engine(auto)` + `S3/S4` can consume `vista_subset` transcripts under `32k/512` (truncate if needed, strict). Cache key includes `vista_subset` revision hash.

## Success Criteria
- [ ] `vista_subset_asr.py` tồn tại, `--shard`/`--resume`/`--limit` work,  `--dry-run --limit 5` idempotent no duplicates. **6 shards integrated.**
- [ ] `probes/cache/vista_subset/` có `300` JSONs, mỗi có `transcript_sentences≥20`, `summarization_data` với abstract, `timestamps` aligned. **~1.7h/shard ×6 = ~10h total.**
- [ ] WER audit gate chạy và `<30%` hoặc non-empty guard pass; attrition gate documented fail-loud (không auto fallback).
- [ ] D-T14 audit 20 samples done, `exclusion_list.json` frozen trước S0-S4 (nếu có).
- [ ] `test_vista_asr.py` post-hoc: `test_vista_subset_has_300_with_transcript`, `test_asr_wer_audit_below_30`, `test_vista_idempotent_resume` pass (hoặc manual verify nếu TDD bỏ).
- [ ] `tib-bench` 80 external validation vẫn reachable (không bị gộp/nhầm với VISTA primary).

## Colab Free Execution (Approach A, 2026-09-01 — keep 5 notebooks, >60' with checkpoint, Hybrid)

- **Budget:** 300 VISTA `~32GB` raw + **~10h** ASR (2'/video whisper-small T4). **Colab Free 12h kill → cần resume.** 6 shards ×50 videos = **~1.7h/shard** — mỗi shard là 1 cell trong `04` hoặc 1 lần `!python -m benchmarks.scripts.vista_subset_asr --shard X --resume`. Sau mỗi shard `rm` raw mp4 để không vượt Drive 30GB.
- **Hybrid (per user choice):** `vista_subset_asr.py` tries `probes/cache/vista_subset/*.json` first (Drive cache). Nếu miss, `snapshot_download(..., resume_download=True)` → `faster-whisper transcribe(vad_filter=True)` → save JSON. Second run <1' (cache hit).
- **Notebook integration:** `04_phase4_hierarchical_summarization.ipynb` cell đầu checks `len(list((PROJECT_ROOT/"probes/cache/vista_subset").glob("*.json")))` — nếu `≥300` skip download, else prints `Run !python -m benchmarks.scripts.vista_subset_asr --shard 0` (6 lần) hoặc auto-loop shards với `try/except` và `time.sleep`.
- **User approved `>60'`:** Phase 4 intentionally runs `~10h` across shards; full 300 not demo 20 — thesis power requires it. No split to separate notebook needed (keep 5).

## Risk Assessment
- **R form-gated revoked:** VISTA requires HF access request (approved 2026-08-31) — nếu revoked, phase fail-loud per user mandatory decision; cần supervisor explicit switch to TIB primary + update `D-T12`. Mitigation: check access trước implement.
- **R 1.93 TB nếu không scope:** Scoped 300 (~32 GB) + shard streaming tránh quota blowup; mỗi shard 50 xử lý xong `rm` raw mp4 sau khi ASR xong.
- **R self-ASR 25 days full:** Scoped 300 =10h, fits 1 Colab session `resume per 50`; `vad_filter=True` giảm hallucination.
- **R HF snapshot_download corrupt:** Use `resume_download=True` + verify `local_dir` hash; Drive temp có thể corrupt — keep model weights ở `HF_HOME` SSD only.
- **R transcript ngắn/noise:** Whisper-small trên conference audio (EN) thường WER 15-25%; nếu >30% cần note limitation trong P6 human eval, không silent.
