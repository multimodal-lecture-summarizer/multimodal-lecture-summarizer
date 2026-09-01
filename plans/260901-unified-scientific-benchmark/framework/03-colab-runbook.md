# T4 Experiment Runbook

GPU type, quota, RAM, disk and session duration are dynamic. Record actual resources for every run.

## 1. Execution order

1. Validate dataset schema/license/splits.
2. Build frozen manifests.
3. Run 10-item end-to-end pilot.
4. Measure storage, GPU-hour, latency, VRAM and failures.
5. Freeze models, prompts, budgets and metrics.
6. Precompute immutable features.
7. Run RQ1 → RQ2 → RQ3.
8. Reuse outputs for RQ4; do not duplicate generation.

## 2. Runtime policy

- One heavy model loaded at a time.
- T4 local comparisons run on the same GPU class. Kaggle T4×2 is not one pooled 32GB GPU without explicit parallelism.
- API experiments are a separate track; never merge into local Pareto tables.
- Checkpoint after each item.
- Keep timeout/failure rows.
- Do not use keep-alive workarounds.
- Reserve 2–3× pilot runtime for full execution.
- No FP8 measurements on Turing T4 (no native FP8 tensor cores); report FP16/INT8 only.

## 3. Environment

Record the following and commit to `provenance.json` per run:

```text
python version
CUDA/driver version
GPU name and memory
package lock hash
model IDs and revisions (E4/C7 commit hash from decisions-log.md D-T01)
dataset revisions and manifest hashes
git commit
random seed
```

Prefer a pinned lockfile over repeatedly installing latest packages.

## 4. Hugging Face cache — corrected policy

**Model weights → Colab local SSD only:**

```bash
# Use the local SSD for model weights; Drive is for feature stores only
export HF_HOME=/root/.cache/huggingface
mkdir -p $HF_HOME
```

**Finished feature stores → Google Drive:**

```bash
# Mount Drive and use it only for precomputed features, predictions, and manifests
# Never symlink model weight cache to Drive
FEATURE_STORE="/content/drive/MyDrive/feature_stores"
mkdir -p $FEATURE_STORE
```

**Rationale:** Drive sync can corrupt partial model weight writes. Re-downloading a 10 GB model on session start is faster than debugging a corrupted cache. Finished feature stores are small, deterministic, and safe to sync.

**Integrity check before each run:**

```python
import hashlib, json, pathlib

def verify_manifest(manifest_path: str) -> bool:
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    for item in manifest["items"]:
        p = pathlib.Path(item["local_path"])
        if not p.exists():
            print(f"MISSING: {item['item_id']}")
            return False
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != item["checksum"]:
            print(f"CHECKSUM MISMATCH: {item['item_id']}")
            return False
    return True
```

Run this before every full experiment; fail loudly if any checksum mismatches.

## 5. Dataset pilot

For each candidate dataset:

```json
{
  "dataset": "VISTA",
  "revision": "...",
  "item_id": "...",
  "split": "test",
  "media_status": "available",
  "failure_reason": null,
  "duration_sec": 0,
  "transcript_tokens": 0,
  "target_tokens": 0,
  "checksum": "...",
  "license": "..."
}
```

Pilot requirements:

- 20 media probes per dataset;
- 10 complete pipeline runs;
- 100-reference summary-quality audit across VISTA/TIB (using rubric in `01-dataset-manifest.md` §5, both annotators, 20-record calibration first);
- duplicate/leakage check;
- measured download/decode/storage cost;
- no replacement after model results are seen.

**VISTA note:** no transcript field; text-only branch requires self-ASR on raw video. Probe must test raw media download bandwidth and decode cost explicitly.

## 6. Feature cache

Precompute once; freeze extractor revisions in `decisions-log.md` D-T04 before starting:

```text
features/{dataset_revision}/{video_id}/
├── transcript.json          # source transcript (Whisper-small for TIB; self-ASR for VISTA)
├── acoustic.npy             # pause/energy/pitch embedding
├── visual.npy               # DINOv2 ViT-S/14 frame embeddings
├── ocr.json                 # PaddleOCR v3 ch_PP-OCRv4, conf >= 0.6
└── provenance.json          # checksum, extractor revision, sampling rate, precision, timestamp alignment
```

Feature cache changes create a new dataset revision. Never overwrite existing features; create a new version directory.

**E4/C7 cache sharing:** C7 (RQ1) and E4 (RQ4) use the same `Qwen3-VL-4B-Instruct` FP16 checkpoint. Store output under a shared run key; do not run the model twice for the same items. _(See `decisions-log.md` D-T03.)_

## 7. Memory-safe model lifecycle

```python
import gc
import time
import torch

def run_variant(load_model, items, infer):
    model = load_model()
    outputs = []

    for item in items:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        try:
            prediction = infer(model, item)
            status, error = "ok", None
        except torch.cuda.OutOfMemoryError as exc:
            prediction = None
            status, error = "oom", repr(exc)
        except TimeoutError as exc:
            prediction = None
            status, error = "timeout", repr(exc)
        except Exception as exc:
            prediction = None
            status, error = "failed", repr(exc)

        outputs.append({
            "item_id": item["id"],
            "prediction": prediction,
            "status": status,
            "error": error,
            "latency_sec": time.perf_counter() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "context_length": item.get("context_length", 0),
        })
        save_checkpoint(outputs)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs
```

OOM and timeout are **explicit result rows**; they stay in the denominator. Never silently retry an OOM — it may indicate a budget violation.

## 8. Token/frame budget enforcement

Before every S0–S4 / Q0–Q3 run, assert equal budgets:

```python
BUDGET = {
    "source_tokens": 32000,
    "output_tokens": 512,
    "max_frames": 200,
    "frame_resolution_px": 448,
}

def assert_budget(variant_config: dict) -> None:
    for k, v in BUDGET.items():
        assert variant_config[k] == v, (
            f"Budget mismatch for {variant_config['variant_id']}: "
            f"{k} = {variant_config[k]}, expected {v}"
        )
```

If S3/S4 cannot fit within `BUDGET["source_tokens"]`, the run is a **failure**. Reduce the hierarchical method (shorter chapter summaries, scene selection) until it fits; record the reduction as an ablation. Do not report a "separate scaling curve". _(See `decisions-log.md` D-T08.)_

## 9. Summarization run protocol

For S0–S4:

1. Load the same frozen evaluation items.
2. Assert equal source/output/frame budgets (§8).
3. Use the same LLM revision, system prompt, output limit and decoding.
4. Save summary, hierarchy/chunks, evidence citations, token usage and latency.
5. Cache model outputs before scoring.
6. Score automatic metrics from cached outputs.
7. Randomize/blind method labels for human review.

Do not regenerate outputs when changing metrics.

## 10. Human evaluation package

Each review item contains:

- source transcript/evidence view;
- anonymized summaries in randomized order (method names replaced with letters);
- factual support, coverage, coherence and concision rubrics (0–5; see `01-dataset-manifest.md` §3);
- pairwise preference (S1 vs S3, S3 vs S4);
- rationale field;
- adjudication status.

**Pre-annotation checklist:**
- [ ] IRB/ethics approval received.
- [ ] Both annotators have completed calibration batch (20 items).
- [ ] Weighted Cohen's κ on calibration batch ≥ 0.50; if not, refine rubric and recalibrate.
- [ ] Method names are not visible to annotators.
- [ ] Annotation order is randomized independently per annotator.

Run a calibration batch before the final-n evaluation. Report inter-rater agreement (weighted κ for ordinal; plain κ for pairwise). Adjudicate disagreements before computing metrics.

## 11. Retrieval/QA protocol

For Q0–Q3:

- same embedder and revision;
- same top-k;
- same maximum context tokens;
- same generator and decoding;
- same frozen questions;
- save retrieved evidence before generation;
- score retrieval separately from answer generation.

## 12. Statistics and multiple-comparison correction

After collecting per-video raw metrics:

```python
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import numpy as np

def holm_bonferroni_family(deltas: dict[str, np.ndarray]) -> dict:
    """
    deltas: {comparison_label: array of per-video metric differences}
    Returns raw p-values, Holm-corrected p-values, and Cohen's d.
    """
    labels = list(deltas.keys())
    raw_pvals = []
    cohens_d = []
    for label in labels:
        d = deltas[label]
        n = len(d)
        t_stat, p = ttest_rel(d, np.zeros(n))
        raw_pvals.append(p)
        # Cohen's d with Hedges' g correction for small n
        mean_d = np.mean(d)
        sd_d = np.std(d, ddof=1)
        d_val = mean_d / sd_d if sd_d > 0 else 0.0
        correction = 1 - (3 / (4 * n - 5)) if n > 2 else 1.0  # Hedges' g
        cohens_d.append(d_val * correction)

    reject, corrected_pvals, _, _ = multipletests(raw_pvals, method="holm")
    return {
        label: {
            "raw_p": raw_pvals[i],
            "corrected_p": corrected_pvals[i],
            "reject_h0": reject[i],
            "cohens_d": cohens_d[i],
        }
        for i, label in enumerate(labels)
    }
```

Apply within each RQ family:
- RQ1: 4 deltas (C2−C1, C3−C1, C4−C1, C5−C6).
- RQ2: 4 S-pairs (S1−S0, S3−S1, S4−S3, S2−S1 where available).
- RQ3: 3 Q-pairs (Q1−Q0, Q2−Q0, Q3−Q2).

Report both raw and corrected p-values. A negative result after correction is still a valid result. _(See `decisions-log.md` D-T07 and `04-rq-mapping.md`.)_

## 13. Failure policy

- OOM, timeout and model failure remain explicit result rows; they stay in the denominator.
- Retry transient infrastructure failure once with the same config.
- Do not retry a poor model output.
- Config changes create a new run ID.
- Never remove failed items from only one compared system.
- Budget violation (unequal tokens/frames) is a **failure**, not a separate result.

## 14. Aggregation

Aggregate by video before dataset mean:

1. per-item raw metric;
2. per-video aggregate;
3. per-seed aggregate;
4. paired video-level effect;
5. bootstrap 95% CI;
6. Holm-corrected p-value and Cohen's d;
7. failure rate and resource metrics.

No result is final without raw-output provenance.
