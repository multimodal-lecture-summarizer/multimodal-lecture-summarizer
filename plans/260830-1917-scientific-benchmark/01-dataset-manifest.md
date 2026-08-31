# Dataset Manifest and Qualification Plan

**Status:** Candidate datasets verified at source level; local schema/media/license gates still required.  
**Last patched:** 2026-08-31 — removed false VISTA transcript claim; added tib-bench eval target; added three-tier fallback; added audit rubric.

## 1. Core dataset matrix

| Tier | Dataset | Scale | Inputs | Targets | Research role | Main caveat |
|------|---------|-------|--------|---------|---------------|-------------|
| A | **YTSeg** | 19,299 videos / 6,533 h | Transcripts, audio; source video download script | Creator chapter timestamps/titles | RQ1 chaptering | English, multi-domain, creator labels, media attrition |
| B | **VISTA** | 18,599 AI conference presentations | Raw video (1.93 TB, form-gated); **no transcript field** | Paper abstract as summary | RQ2 primary abstractive summarization | Form-gated; must download raw video and self-run ASR; not text-only fallback |
| C | **TIB** | 9,103 long presentations/lectures | Transcript (Whisper-small), timestamped segments, keyframe timecodes, direct MP4 URL | Author-provided abstract | RQ2 external multimodal summarization; primary if VISTA fails | Abstracts are noisy (0–6.18k chars); keyframes are time codes only (not images); `tib-bench` adds slide PNGs |
| D | **EduVidQA** | 5,252 QA pairs / 296 CS videos | Transcript, timestamp-linked context | Expert + synthetic answers | RQ3 external QA | YouTube media for visual variants; text-only branch is safe |
| E | **VT-SSum** | 9,616 videos / 125K pairs | Spoken transcripts, slide-derived segmentation | Weakly supervised extractive labels | RQ1/RQ2 auxiliary diagnostic | Old, extractive, weak labels; not primary abstractive evidence |
| F | Custom evidence subset | 60–100 items on ≥ 20 videos | Transcript, OCR, keyframe | Answer + evidence time range | RQ3 visual evidence only | Requires two annotators and freeze before experiments |

## 2. Dataset decisions

### YTSeg — primary chaptering

- Official source: https://huggingface.co/datasets/retkowski/ytseg
- Paper: https://aclanthology.org/2024.eacl-long.25/
- Current protocol update: https://aclanthology.org/2026.acl-long.396/
- License: CC BY-NC-SA 4.0.
- Use full available official test split for text/audio evaluation.
- Create a deterministic 50–100-video lecture/science subset for visual/OCR experiments after media pilot.
- Do not call creator chapters expert gold.
- Do not use chapter timestamps as shot-boundary ground truth.

### VISTA — primary abstractive summarization

- Paper: https://aclanthology.org/2025.acl-long.310/
- Dataset: https://huggingface.co/datasets/dongqi-me/VISTA
- Scale: 18,599 recorded AI conference presentations paired with paper abstracts.
- License: CC BY 4.0 + restricted access form (non-commercial/academic; IRB clause may apply).
- **VISTA does not ship transcripts.** The `video_path` field is a preprocessed artifact path, not a downloadable transcript or feature vector. Text-only VISTA evaluation requires self-running ASR on raw video. This is out of scope unless media is fully downloaded and an ASR pipeline is budgeted. _(See also `decisions-log.md` D-T05.)_
- Total media size: 1.93 TB. Storage plan must be confirmed before adoption.
- Use official train/dev/test splits when available.
- Before adoption, verify dataset license approval, downloadable fields, split integrity and media access.

**Acceptance gate:**

1. Submit access form; record approval date in `decisions-log.md` (✅ **Approved 2026-08-31**).
2. Load schema and verify official splits (`train_part1`, `train_part2`, `validation`, `test`).
3. Probe 20 media items and record failures / preprocessed feature availability.
4. Audit test references using the calibrated rubric in §5.
5. Check duplicate paper/video IDs across splits.
6. Pass if required inputs/targets are available and reference noise is reportable without invalidating the comparison.

**Status:** Nominal path active (VISTA Primary + TIB External Validation).

### TIB — external summarization validation (primary if VISTA fails)

- Dataset: https://huggingface.co/datasets/gigant/tib
- Paper: https://doi.org/10.1145/3617233.3617238
- Scale: 9,103 long multimodal presentation/lecture records.
- Official splits: train 7,282 / valid 910 / test 911.
- Fields: `doi, title, url, video_url, license, subject, genre, release_year, author, contributors, abstract, transcript, transcript_segments, keyframes, language`.
- Transcripts: provided (Whisper-small, timestamped segments). Treat as noisy ASR; report quality.
- Keyframes: **time codes only, not images**. Visual evidence requires either streaming video decode from `video_url` or using `gigant/tib-bench` slide images.

**Evaluation target (frozen):** `gigant/tib-bench` test subset — 80 records, zero leakage confirmed by probe 2026-08-30 (train 674 / valid 68 / test 80 / outside-splits 0). Multimodal evidence = `slides` column (PIL RGB PNG 512×288, ~30–35 per talk). No video decoding needed for slide-based visual evidence. _(See `reports/260830-1917-tib-audio-visual-probe.md` and `decisions-log.md` D-T06.)_

**S2 oracle diagnostic on TIB:** Attempt to construct oracle chapter inputs using `keyframes.timestamp` boundaries (Week-14 deliverable). If ≥ 80% of test videos have ≥ 3 segments, run S2 and report. Otherwise, move S2 to appendix. _(See `decisions-log.md` D-T11.)_

- Per-record CC licenses must be verified; restrict to records with redistribution-compatible licenses for public manifest release.
- Use a predeclared genre-filtered lecture/presentation subset.
- Treat abstracts as noisy author references; report style/length/support audit using the rubric in §5.

### VT-SSum — auxiliary only

- Paper: https://arxiv.org/abs/2106.05606
- Repository: https://github.com/Dod-o/VT-SSum
- Splits: 7,692 train / 962 dev / 962 test videos.
- Provides transcript segmentation and extractive sentence labels derived using slide content as weak supervision.
- Valid for extractive/segmentation diagnostics and optional pretraining.
- Not valid as the sole benchmark for abstractive summary factuality.

### EduVidQA — primary external QA

- Paper: https://aclanthology.org/2025.emnlp-main.1760/
- Repository: https://github.com/sourjyadip/eduvidqa-emnlp25
- Scale: 5,252 QA pairs from 296 computer-science lecture videos (270 real/99 videos + 4,982 synthetic/197 videos).
- License: MIT.
- Report real-world and synthetic subsets separately.
- Do not tune on official test.
- Text-only branch (transcript + timestamps) is safe. Visual branch depends on YouTube availability; yt-dlp needed; treat as a risk (R3).

## 3. Summarization evaluation sets

### Automatic evaluation set

- Primary: full feasible VISTA official test split.
- External: `gigant/tib-bench` test subset (80 records, predeclared).
- Auxiliary: VT-SSum official test for extractive sentence selection.
- If compute is limited, select a deterministic stratified subset before any model output.

Stratify by duration, transcript length, topic and reference length. Publish IDs and selection script.

### Human quality audit & automated judge evaluation

- Pilot audit: 20–50 videos audited directly by the researcher using the standardized rubric in §5.
- Scale evaluation: Standardized LLM-as-a-Judge protocol (G-Eval criteria: factual support, key-point coverage, coherence, 1–5 scale) + objective metrics (Salient QA F1, AlignScore, ROUGE-L, BERTScore).
- Source: VISTA primary test (if accessible) or TIB `tib-bench` test subset if VISTA fails.
- Dimensions on 0–5 scale: factual support, key-point coverage, coherence, concision.
- Pairwise preference between fixed-chunk (S1) and predicted-hierarchy (S3) summaries.
- Calibration round on pilot subset to verify correlation between human audit scores and LLM-judge scores.

### QA-based summary evaluation

For each evaluated source:

1. Build source-grounded salient questions without seeing system outputs (or leverage pre-labeled QA pairs).
2. Test whether the summary supports correct answers.
3. Verify every summary claim against transcript/OCR/keyframe evidence.
4. Report coverage and unsupported-claim rate.

ROUGE-L and BERTScore are secondary because lexical/semantic overlap alone does not establish factuality or coverage.

## 4. Custom evidence subset

Leverage pre-labeled multimodal evidence from EduVidQA and TIB-bench slides to minimize manual annotation overhead:

```json
{
  "item_id": "F001",
  "video_id": "V001",
  "question": "...",
  "answer": "...",
  "evidence": {
    "time_range_sec": [120.0, 148.0],
    "transcript_ids": [31, 32],
    "ocr_text": "...",
    "keyframe_id": "V001_K004"
  }
}
```

Protocol:

- Evidence extracted from pre-labeled ground-truth datasets (EduVidQA timestamps / TIB-bench slide-transcript alignments).
- Blind quality check on sample items before freezing.
- Freeze file hash before Q0–Q3 runs.
- Report by video and question type.

## 5. Reference quality audit rubric

Per-record audit schema (applies to both VISTA and TIB samples):

| Field | Values | Notes |
|-------|--------|-------|
| `id` | string | Dataset item ID |
| `source_support` | 0 = not supported / 1 = partial / 2 = well-supported | Does the abstract/reference reflect the source content? |
| `coverage` | 0 = missing key points / 1 = partial / 2 = covers main topics | How well does the reference cover the source? |
| `style` | `summary-like` / `boilerplate` / `mixed` | Is the reference a genuine summary or a template/venue text? |
| `action` | `keep` / `flag` / `exclude` | Final decision for inclusion in evaluation |

**Protocol:** Researcher audits the 20–50 record calibration/pilot set using this rubric. Exclusion list is frozen before any system output is generated (R8). Report exclusion rate and predominant failure mode.

## 6. Global qualification gates

Every dataset must have:

- official URL, paper, version/date and license record;
- schema snapshot and checksums;
- official or deterministic split;
- media availability report;
- duplicate/leakage audit;
- target-quality audit (using rubric in §5 for summarization datasets);
- frozen exclusion/missing-item policy;
- statistical unit definition;
- release policy for IDs/features/media.

## 7. Three-tier summarization fallback

1. **VISTA primary + TIB external** — nominal path.
2. **TIB primary** — if VISTA fails media, license, or schema gate; exclude VISTA with documented reason; narrow abstractive claim to scientific presentations.
3. **No abstractive summarization** — if both VISTA and TIB fail; narrow to (a) RQ1 on YTSeg, (b) RQ3 on EduVidQA + custom evidence; submit as a **short paper**; do not claim abstractive lecture summarization.

This fallback is a pre-registered scope cut. No post-hoc narrative replacement.

## 8. Excluded from the core plan

- TVSum: importance labels do not match semantic chaptering or textual lecture summarization.
- SlideVQA: slide-deck VQA, gated, unnecessary after EduVidQA/custom evidence tier.
- TED-LIUM: ASR component benchmark does not test the central thesis.
- IIIT-CVID: optional OCR implementation test, not a core research dataset.
- ViCocktail: Vietnamese AVSR, not lecture summarization.
- Self-generated eight-video silver summaries: insufficient as primary evidence.

## 9. Data ethics and PII handling

- Custom evidence subset (Tier F): only videos with explicit redistribution rights or fair-use academic-research exception; no student PII in published manifests; face-blur on any thumbnails published.
- EduVidQA: YouTube CS-course content; QA pairs are grounded in transcripts; no student PII in questions or answers.
- TIB: academic talks from TIB AV-Portal; per-record licenses; do not publish media; release IDs and derived features only after verifying per-record CC variant allows it.
- VISTA: conference recordings; CC BY 4.0 + access restrictions; release only IDs and non-restricted derived features after legal review.
- All human annotators must consent to using their annotations in published results.
