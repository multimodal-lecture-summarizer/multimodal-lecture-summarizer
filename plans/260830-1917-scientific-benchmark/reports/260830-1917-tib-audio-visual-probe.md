# TIB Media & Multimodal Assets Probe (2026-08-30)

Status: COMPLETED. Verifies feasibility of the M1 dataset/compute gate for the scientific
benchmark on real TIB media + the multimodal benchmark assets.

## 1. TIB official splits — loaded (source of record)

- Dataset: `gigant/tib` on Hugging Face (mirror of TIB AV-Portal).
- Loaded via `datasets.load_dataset(..., trust_remote_code=True)`.
- Splits frozen per repo: **train 7282 / valid 910 / test 911** (total 9103).
- 15 columns; verified inner schemas:
  - `transcript_segments` = Whisper JSON verbatim per segment: `id, seek, start, end,
    text, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob`
    (example record: 177 segments). → enables transcript-quality scoring.
  - `keyframes` = slide-boundary dict: `frames` (start/end frame indices), `slide`
    (PNG filename like `mobile_calls_0_623.png`), `timestamp` (seconds,
    slide onset→offset). Keyframes are time references, **not images** in the base repo.
- Caveats: `license` strings contain embedded newlines (needs whitespace cleaning);
  `genre` column is messy/newline-joined; `subject` is a coarse category.
- Manifest of every record persisted:
  `C:\Users\hung\AppData\Local\Temp\opencode\tib_manifest.json` (doi, title, video_url,
  genre, subject, license, release_year, abstract_len, transcript_len, segments,
  keyframes_n) for all three splits.

### Genre focus (benchmark targets lectures)
Test split (911): 792 Conference/Talk, 91 Lecture, 18 Webinar/Tutorial
(+ 7 multi-genre entries containing "Lecture" ⇒ 98 records containing Lecture).
Probe selection below prioritized pure-Lecture test records.

## 2. Multimodal benchmark assets — inspected

HF repos under `gigant/*` (VLM-paper artifacts):
- `gigant/tib-bench` — **822 records** with embedded **real slide images**
  (`slides` column: PIL RGB PNG @ **512×288**, ~30–35 per talk; schema columns = tib
  columns + `slides`). 2.4 GB parquet. NOTE: repo currently exposes **13 parquet shards**
  (`data/part_*`), not the 51-shard layout the hub UI implied.
- `gigant/tib-bench-mm-test` — identical schema, 500 rows (subset overlapping bench).
- `gigant/tib_slides` — **484,843** slide PNGs (~132 GB), the full slide store.
- Others: `tib-bench-text`, `tib-bench-vlm/vl`, `tib-bench-mm-part2..15`,
  `tib-bench-mm-filtering*` (visual-filtering set).

### Split-containment check (important for eval integrity)
All 822 `tib-bench` DOIs fall inside official TIB splits with **zero leakage**:
train 674 / valid 68 / **test 80** / outside-splits 0.
→ The 80-test-record mm-bench subset is the natural all-in-one evaluation set.

## 3. MP4 media probe — 20 URLs

Selection: 12 pure-Lecture (test) + 8 Conference/Talk (valid). Genre filter prioritized
Lecture per benchmark design.

### HEAD probe (20/20)
- **HTTP 200 on all 20** `tib.flowcenter.de/mfc/medialink/<token>/<slug>.mp4`.
- Content-Type `video/mp4`, sizes **107 MB – 493 MB** (mean ≈ **310 MB**).
- No 403/404/redirect wall; direct anonymous GET works.

### Decode probe — partial download, 12 MB each (20/20)
- Range requests honored; `moov` atom **at start** ⇒ streamable/faststart ⇒ partial
  downloads are fully probe-able.
- All H.264 video + AAC audio; 1024×576 (768×576) @ 25 fps, ~700 kbps video.
- Durations 17 – 79 min (probe durations 1059 s – 4721 s), consistent with talk lengths.

### Full-download test — 3 smallest DOIs
- `10.5446/47268` 107.5 MB in 10.0 s (10.7 MB/s), dur 1038.5 s
- `10.5446/49241` 111.4 MB in  9.6 s (11.6 MB/s), dur 1059.8 s
- `10.5446/38186` 109.3 MB in  9.6 s (11.4 MB/s), dur 1059.5 s
- ffprobe-complete, durations unambiguous ⇒ URLs are long-lived, byte-correct, and
  **resumable** (accept-ranges).

### Slide image sample (metric-visualization sanity)
5 `tib-bench` records × 8 slides saved (`...\Temp\opencode\media\slides\`) — PNGs decode
fine at 512×288.

Artifacts: `tib_head_20.json`, `tib_partial_20.json`, `tib_full_download.json`,
`tib_bench_dois.json` under `C:\Users\hung\AppData\Local\Temp\opencode\`.

## 4. Implications for the benchmark pipeline

1. **Slides: no video decoding needed.** `tib-bench` already ships the slide PNGs and
   keyframe↔slide mapping. Use `slides` + `keyframes.timestamp` for visual evidence and
   metric visualization (don't transcode the source MP4).
2. **Video = low-res, directly streamable** (1024×576 @ ~700 kbps). A ~40-min talk is
   ~300 MB; decoding at 1 fps with `-ss`/`-vf fps=1` via ffmpeg is feasible locally.
   Frame decoding budget stays within the 128–256 tok/frame plan (Qwen3-VL 4B FP16).
3. **Eval set recommendation (update to plan):** evaluate on the 80 test-DOI `tib-bench`
   subset (Lecture-filtered: of those 80, check pure-Lecture count during harness build)
   + the remaining test-split Lecture records for breadth. This yields ~full-coverage
   multimodal evaluation with images from bench + video/audio from TIB.
4. **Download strategy:** range-verified, resumable, ~10–12 MB/s single-stream;
   1000+ test videos ≈ 300 GB ⇒ use streaming decode or a subset; the 80-bench subset
   ≈ 24 GB (download once, keep slides from bench only).
5. **Risks resolved vs. deferred:** flowcenter token now confirmed live & stable;
   residual risk = long-term URL expiry (store media once); VISTA gating (1.93 TB)
   unchanged/out-of-scope for this gate.

## 5. Environment notes
- Windows local, Python 3.11.1; `datasets 2.19.2`, `huggingface_hub 0.36.2`,
  `pyarrow`, `requests`; ffmpeg/ffprobe 7.x via chocolatey.
- `hf://` fs note: parquet column-projection reads via
  `fsspec.filesystem('hf')` + repo-path prefix `datasets/<org>/<repo>/...`;
  `datasets` streaming iterates slowly for image-bearing rows.