# Research Report: Dataset Qualification, Compact VLM & Human Eval for Scientific Benchmark

**Research conducted:** 2026-08-30
**Scope:** M1 dataset/compute gate for plan `260830-1917-scientific-benchmark`
**Sources consulted:** ~14 (HF dataset cards, ACL/arXiv papers, GitHub repos, VLM deployment guides)

## Executive Summary

- **VISTA is viable but gated and heavy.** License CC BY 4.0, official 80/10/10 splits, but dataset access is **restricted (form gating)** and media is **1.93 TB of raw video** (`dongqi-me/VISTA`). **No transcript field exists** — the plan's assumption that "provided transcripts/video features" can substitute raw video is **wrong** and blocks fallback scenario 3 for VISTA without running own ASR.
- **TIB is the lowest-friction external set.** Per-record CC licenses, direct MP4 `video_url`, official 7282/910/911 splits, transcript + timestamped segments + keyframe timecodes. Caveat: `keyframes` are **time codes only, not images**; abstract varies 0–6.18k chars. A sibling set `gigant/tib-bench-mm-*` adds real slide images + SigLIP logits for the multimodal runs.
- **EduVidQA is MIT, ungated, GitHub-hosted** (`data/train|dev|test`). 5,252 QA pairs (270 real/99 vids + 4,982 synthetic/197 vids). Media = YouTube URLs → yt-dlp risk for visual; transcripts from official sources make text-only RQ3 safe.
- **T4 (16 GB, Turing) cannot run Qwen3-VL-8B at FP16 (~17 GB).** Recommended E4 baseline: **Qwen3-VL-8B-Instruct AWQ/INT8 (~6–10 GB)** or **Qwen3-VL-4B-Instruct FP16 (~10 GB)**; SmolVLM2-2.2B (~5 GB) as the far-edge row. Turing has **no native FP8**; avoid FP8 claims.
- **Frame budget: 150–200 scene-level keyframes @ ≤256 tokens/frame fits a 32K source budget** for a 1 h lecture (~8–16K transcript tokens). Budget excludes KV-cache growth; must be measured, not assumed.
- **Second annotator** exists in tooling (Label Studio / Argilla / custom blinded HTML). Use **weighted Cohen's κ** for ordinal 0–5 dimensions; plain κ for pairwise preference; report CI.

## Research Methodology

- Sources consulted: 14 (5 web-search batches; HF cards for VISTA/TIB/tib-bench; ACL Anthology 2025.acl-long.310 & 2025.emnlp-main.1760; arXiv 2502.08279, 2504.05299, 2504.10479, 2509.14769, 2511.21631; GitHub dongqi-me/VISTA, giganttheo/tib-dataset, sourjyadip/eduvidqa-emnlp25; VLM VRAM deploy guides)
- Date range: 2023-09 (TIB CBMI'23) → 2026-08 (Qwen3-VL processor issue)
- Key search terms: `VISTA huggingface license schema splits`, `TIB gigant license schema keyframes`, `EduVidQA license schema`, `compact VLM T4 16GB Qwen3-VL frame budget`, `SmolVLM InternVL3 small VLM`, `human evaluation summarization kappa blind`
- Gemini CLI: unavailable (no `~/.Codex/.ck.json`) → WebSearch only.

## Key Findings

### 1. VISTA — license/schema/split verified

| Aspect | Finding |
|---|---|
| License | **CC BY 4.0** (HF card) + **restricted access form** (non-commercial/academic only, IRB-if-applicable clause) |
| Scale | 18,599 videos; **1.93 TB total file size** |
| Splits | **Official 80/10/10**: `train_part1` (8,000) + `train_part2` (remaining) / `validation` / `test` |
| Schema | `id`, `title`, `authors`, `abstract`, `video_file`, `video_path`, `paper_url`, `venue` |
| Transcript? | **NO.** No ASR/transcript field on the hub card. `video_path` is a preprocessed video feature path. |
| Target | `abstract` = paper abstract = ground-truth summary |
| Paper | ACL 2025 Long, 10.18653/v1/2025.acl-long.310; splits proportional by venue for domain balance |

**Gaps for the plan:**
- `01-dataset-manifest.md` L35 "can use provided transcripts/video features if raw video access is limited" — **no transcripts shipped**. Fallback 3 (text-only VISTA) requires self-ASR on 2 TB.
- 1.93 TB exceeds Colab/Kaggle disk; storage plan must be explicit before adoption.
- Gate approval latency unknown → cannot be assumed instant; probe must include access-approval time.

### 2. TIB — license/schema/split verified

| Aspect | Finding |
|---|---|
| License | **Per-record** CC variants (24 values: CC BY 2.0/3.0/4.0, CC BY-NC, etc., incl. `\n`-wrapped strings to clean). Dataset hub terms apply otherwise. |
| Scale | 9,103 records; 502 MB parquet only (media external) |
| Splits | **Official 7,282 train / 910 valid / 911 test** |
| Schema | `doi,title,url,video_url,license,subject,genre,release_year,author,contributors,abstract,transcript,transcript_segments,keyframes,language` |
| Transcript | Provided (auto **whisper-small**); `transcript_segments` = timestamped segments; length up to ~159k chars |
| Keyframes | **Time codes only** — NOT images. Visual content must be decoded from `video_url` |
| Media | `video_url` = direct MP4 at `tib.flowcenter.de/mfc/medialink/...` (tokenized); genre-filterable (9 genres incl. Lecture) |
| Abstract | Author-provided; lengths 0–6.18k; filtered at curation to >30 chars |

**Multimodal upgrade path:** `gigant/tib-bench-mm-*` + `tib-bench-mm-filtering-part1-2` add **`slides` (real images) + `siglip-logits`** per record → directly useful for S4/Q3 without decoding video. Worth evaluating as S3/S4 evidence source.

### 3. EduVidQA — license/schema/split verified

| Aspect | Finding |
|---|---|
| License | **MIT** (GitHub repo header) |
| Access | Ungated; hosted on GitHub (`sourjyadip/eduvidqa-emnlp25/data`); no HF mirror found |
| Splits | **Official train/dev/test** in repo |
| Scale | **5,252 QA pairs / 296 videos**: 270 real (99 videos, expert-verified, ~68 min avg) + 4,982 synthetic (197 videos, ~22 min avg) |
| Schema | QA pairs with question, expert answer, **timestamps**, course, YouTube video, transcript source |
| Media | YouTube URLs (CS courses); visual experiments require yt-dlp |
| Paper | EMNLP 2025 Main; arXiv:2509.24120 |

**QA pairs are grounded in timestamps and transcripts** → Q0–Q3 text branches and evidence-localization metric run on safe artifacts. Visual variant depends on YouTube availability (page-level risk, R3).

### 4. Media-availability mechanics

| Dataset | Media host | Gating | Download model | Dominant risk |
|---|---|---|---|---|
| VISTA | HF hub video files | **Form-gated** | Direct HF download (1.93 TB) | Gate latency; disk:bandwidth; 2 TB transfer |
| TIB | TIB flowcenter MP4 (tokenized URL) | None | Direct HTTP | Link expiry; ~40 min avg/decode cost |
| EduVidQA | YouTube | None | yt-dlp | Takedown/geo; rate-limit |

**Recommendation for the 20-media probe:** publish per-dataset probe matrix (status, duration, ffprobe checksum, decode pass/fail) before any model run; VISTA probe only after gate approval, and start with the smallest `test` subset items to bound cost.

### 5. Reference-quality audit (100-references)

Compute-guided process (data must be downloaded first):

- **VISTA (50):** source = paper abstract. Check: abstract readability (flesch/token count), whether abstract mentions methods/results vs venue boilerplate, abstract-venue duplicates. Report % "clean summary-like" abstracts.
- **TIB (50):** source = author abstract vs transcript. Check: abstract length distribution, title-only/minimal abstracts, conference-banner abstracts (curation already filtered same-abstract dupes — verify per-split), genre == Lecture/Conference filtering effect.
- **Per-file record:** `id, source_support (0–2), coverage (0–2), style (summary-like/boilerplate/mixed), action (keep/flag/exclude)`.
- **Freeze** the 100-IDs + audit sheet + exclusion rule BEFORE any system output (R4/R8).

### 6. Compact VLM selection for T4

T4 constraints: 16 GB GDDR6, Turing (no native FP8/BF16 tensor cores; FP16 tensor cores present). Weights-only FP16 ≈ 2 GB/B params + ViT overhead.

| Candidate | Params | FP16 | INT8/AWQ | Fit T4 16GB | Notes |
|---|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** (E4) | 8B | ~17 GB ✗ | **~6–10 GB ✓** | INT4/INT8 | DocVQA 96.1; best accuracy; needs quant |
| **Qwen3-VL-4B-Instruct** (E4 alt) | 4.4B | ~10 GB ✓ | ~5 GB ✓ | FP16 or INT8 | cleaner eval, no quant confound; DocVQA 91 |
| **Qwen3-VL-2B** | 2B | ~5 GB | ~2–3 GB | ✓ | edge reference only |
| SmolVLM2-2.2B (extreme) | 2.2B | ~5 GB | – | ✓ | Video-MME 52.1; ~0.6–1.7 ex/s A100-class but far lighter |
| InternVL3-2B | 2B | ~4–6 GB | – | ✓ | strong doc/OCR (DocVQA 88.3) |

**Decision (E4):** run **Qwen3-VL-4B-Instruct FP16** as the primary compact-VLM row (no quantization confound, fits with headroom) and optionally **Qwen3-VL-8B-Instruct AWQ** as the quality row if VRAM budget verifies in the 10-video pilot. Freeze ONE checkpoint + revision before Q0–Q4 (R12).

### 7. Token/frame budget

Qwen3-VL video compression: 32× spatial + 2× temporal → ~196 tokens/224²-frame before pair-merge. Standard caps: ≤768 tokens/frame (VideoMMMU), ≤640 (others); recommended total_pixels < 24576×32×32 (≈ 25 M px over all frames).

Budget math for 1 h lecture, 32K source-token cap:

| Setup | Tokens |
|---|---|
| Transcript (~7–10k words) | ~10–16K |
| Frames: 150 scene keyframes @ 256 tok | ~38K ✗ too high |
| Frames: 100 @ 128–200 tok (448 px short edge) | ~13–20K ✓ |
| OCR text (slide text) | ~2–5K |
| **Total target** | **≤ 30–32K** |

**Rule:** sample **scene/keyframe-level**, ~1 frame / 20–40 s, cap 150–200 frames, resize short edge 448–512 px. Total visual tokens must leave transcript + OCR room. Lock `frames`, `source_tokens`, `output_tokens` (512) per run record and keep equal across S0–S4 / Q0–Q3 (R5).

### 8. Compute measurement → matrix lock (RQ4)

Per-item record (from `03-colab-runbook.md` `run_variant`): wall time, `torch.cuda.max_memory_allocated/reserved`, throughput (items/h), tokens in/out, failure rows, storage. T4 expectations: KV-cache dominates long-context memory → record per-item context length too. Extrapolate pilot 10-videos × 2–3× for full run before freezing matrix (R9). No FP8 measurement on T4 — report FP16/INT8 only.

### 9. Second annotator & human-evaluation tooling

- **Metrics:** weighted Cohen's κ for ordinal 0–5 (factual support, coverage, coherence, concision); unweighted κ for pairwise preference; report kappa + 95% CI (bootstrap), not just point estimate. Landis–Koch scale for interpretation (0.61–0.80 substantial).
- **Workflow:** calibration batch → blinded, randomized order → 50+videos × 2 raters → adjudication for ties/disagreement → frozen hashes before runs.
- **Tooling options:**
  - Custom static HTML review pages (auto-randomize A/B side, no method names) + CSV/JSON output → most control, matches plan's "blinded method labels" precisely.
  - Label Studio (open-source; supports pairwise/rating tasks; audit trails).
  - Argilla (Python-native, LLM-eval oriented).
  - `annotation-eval`-style CLI (Cohen/Fleiss/Krippendorff from tidy CSV) for agreement reports.
- **Second annotator practical source:** an independent CS/MSc rater (domain exppertise matters for factuality judgments); same rubric, separate sessions, no communication during annotation.

## Comparative Analysis

| Criteria | VISTA | TIB | EduVidQA |
|---|---|---|---|
| License friction | Gated form + CC BY 4.0 | Per-record CC (mixed NC variants) | MIT, zero friction |
| Input ready | Raw video only | Transcript ✓ + keyframe t.c. + direct MP4 ✓ | Transcript ✓ (YouTube for visual) |
| Target quality | Abstract (high quality, summary-like) | Author abstract (noisy, 0–6.18k) | Expert answers (high quality) |
| Multimodal | Video only (self-ASR needed) | Segments + keyframes + (tib-bench slides via sibling) | Video + transcript |
| Split integrity | Official, venue-proportioned | Official | Official (in-repo, verify) |
| Fallback role | Primary RQ2 | External RQ2; primary if VISTA fails | Primary RQ3 |

## Implementation Recommendations

1. **Submit VISTA access form now** (longest lead-time risk); on approval, probe ≥20 smallest test items and record download bandwidth + decode cost before committing to 1.93 TB.
2. **Adopt TIB as the safe external.** Filter genre∈{Lecture,Conference}, clean license strings, verify transcript_segments ↔ keyframe alignment, freeze test-subset IDs. Evaluate `gigant/tib-bench-mm-*` slide-image field for S4/Q3 multimodal evidence.
3. **EduVidQA:** download repo, verify train/dev/test + timestamp schema, extract YouTube IDs, yt-dlp the 296 videos for visual; keep text-only branch as guaranteed fallback.
4. **Freeze E4 = Qwen3-VL-4B-Instruct FP16** (+ optional 8B-AWQ row) after 10-video pilot confirms VRAM.
5. **Lock budgets:** source ≤32K tokens, output 512, frames ≤150–200 scene keyframes @ 128–256 tok/frame, 448 px short edge. Record per-item context length.
6. **100-ref audit day-1 after download**, with frozen ID list and report template.
7. **Annotator #2 contract before M1 gate:** calibration batch, weighted-κ target report, adjudication rule, tooling chosen.

## Common Pitfalls

- Assuming VISTA ships transcripts → forces unplanned ASR on 2 TB. Verify post-access; otherwise restrict fallback claim.
- FP16 8B VLM on T4 → OOM at 32K context (17 GB weights alone). Quantize or downgrade to 4B; record failure (R10).
- `keyframes` in TIB are timestamps, not images → streaming video decode or tib-bench slides needed for real visual evidence.
- Token-blind frame sampling → S4/Q3 silently exceed budget vs S1/Q0 (R5).
- Unweighted kappa on ordinal 0–5 → under-states near-miss agreement; use weighted.

## Resources & References

### Official
- VISTA paper: https://aclanthology.org/2025.acl-long.310/ · Dataset (gated): https://huggingface.co/datasets/dongqi-me/VISTA · GitHub: https://github.com/dongqi-me/VISTA
- TIB paper: https://doi.org/10.1145/3617233.3617238 · Dataset: https://huggingface.co/datasets/gigant/tib · Blog: https://giganttheo.github.io/tib · Multimodal extension collection: https://huggingface.co/collections/gigant/summarization-of-multimodal-presentations
- EduVidQA paper: https://aclanthology.org/2025.emnlp-main.1760/ · arXiv: https://arxiv.org/abs/2509.24120 · GitHub: https://github.com/sourjyadip/eduvidqa-emnlp25
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL · TR: https://arxiv.org/abs/2511.21631
- SmolVLM: https://arxiv.org/abs/2504.05299 · SmolVLM2: https://huggingface.co/blog/smolvlm2
- InternVL3: https://arxiv.org/abs/2504.10479

### Tools
- Label Studio (pairwise/rating): https://labelstud.io · Argilla: https://argilla.io
- Frame-sampling benchmark (small VLMs): https://arxiv.org/abs/2509.14769
- IAA toolkit (Cohen/Fleiss/Krippendorff): https://github.com/BobbY-24/human-annotation-eval-toolkit

## Appendices

### A. Glossary
- **ASR** automatic speech recognition; **VAE/ViT** vision encoder; **MoE** mixture-of-experts; **IAA** inter-annotator agreement; **Collared F1** boundary tolerance metric (±3 s); **Pk/WD** segmentation metrics; **KV cache** key-value cache (dominant long-context memory); **siglip-logits** SigLIP visual alignment scores.

### B. Version/compat notes
- TIB curated with `whisper-small` transcripts (2023) — older ASR than current pipeline; treat transcript as noise source for RQ2 external.
- Qwen3-VL eval caps: 2,048 frames max, 1–2 fps sampling; long-video accuracy 100% ≤30 min at 256K ctx — irrelevant at 32K budget, cited only for context.
- T4 (Turing) has no FP8/native-BF16 tensor cores → INT8/INT4/FP16 only.

### C. Unresolved questions
1. VISTA: do transcripts exist in downloadable artifacts beyond the card fields? (Only resolvable post gate-approval.)
2. VISTA: is `video_path` a downloadable feature vector or a local-processed artifact path?
3. TIB: do flowcenter medialink tokens expire (TTL) — affects batch download pause/resume design?
4. `gigant/tib-bench-mm-*`: exact item overlap with base `gigant/tib` splits?
5. EduVidQA repo `dair-iitd/EduVidQA` vs `sourjyadip/eduvidqa-emnlp25`: which is canonical for train/dev/test + timestamps?
6. Actual T4 32K-context peak VRAM for Qwen3-VL-4B FP16 (must be measured, not estimated).
7. YouTube status of the 296 EduVidQA video IDs as of pilot date.
8. VISTA abstract duplicates across venues/IDs (leakage audit outcome).

### D. Next steps (locked)
1. Run VISTA access-form + monitor approval.
2. Download TIB + EduVidQA artifacts; run 20-media probe + 100-ref audit.
3. 10-video end-to-end T4 pilot; measure VRAM/latency; select/freeze E4 checkpoint.
4. Lock token/frame budget + RQ4 matrix from measured numbers.
5. Stand up two-rater human-eval package with calibration batch.