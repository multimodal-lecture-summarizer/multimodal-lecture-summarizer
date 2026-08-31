"""Probe YTSeg (retkowski/ytseg) for the 6-month research plan.

Steps:
  1. Try to load the dataset card and split information via datasets.
  2. Read first 2 records of the official test split (streaming) to confirm schema.
  3. Inspect `chapter_titles` / `chapter_timestamps` field shape.
  4. Sample 5 random video IDs and check yt-dlp availability (head probe).

Output: probes/output/ytseg_summary.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "ytseg_summary.json"


def main() -> int:
    started = time.perf_counter()
    summary = {"loaded": False}

    print("[1/4] Loading YTSeg dataset info ...", flush=True)
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        ds_info = api.dataset_info("retkowski/ytseg")
        summary["gating"] = "gated" if getattr(ds_info, "gated", False) else "open"
        summary["license"] = getattr(ds_info, "license", "unknown")
        summary["siblings_count"] = len(getattr(ds_info, "siblings", []) or [])
    except Exception as exc:
        summary["info_error"] = repr(exc)

    print("[2/4] Streaming 2 records from test split ...", flush=True)
    sample = None
    try:
        from datasets import load_dataset
        ds = load_dataset("retkowski/ytseg", split="test", streaming=True, trust_remote_code=True)
        for i, r in enumerate(ds):
            if i == 0:
                sample = r
            if i >= 1:
                break
        summary["streaming_ok"] = True
        if sample is not None:
            summary["sample_keys"] = list(sample.keys())
            summary["sample_video_id"] = sample.get("video_id") or sample.get("id") or sample.get("yt_id")
            for k in list(sample.keys()):
                v = sample[k]
                if isinstance(v, (str, int, float)):
                    summary[f"sample_{k}"] = str(v)[:200]
                elif isinstance(v, list):
                    summary[f"sample_{k}_len"] = len(v)
                    if v and isinstance(v[0], dict):
                        summary[f"sample_{k}_first_item_keys"] = list(v[0].keys())
    except Exception as exc:
        summary["streaming_error"] = repr(exc)
        print(f"  streaming failed: {exc!r}", file=sys.stderr)

    print("[3/4] Full dataset load (test only, no media) ...", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("retkowski/ytseg", split="test", trust_remote_code=True)
        summary["test_size"] = len(ds)
    except Exception as exc:
        summary["full_load_error"] = repr(exc)

    print("[4/4] yt-dlp HEAD probe on 3 sample video ids ...", flush=True)
    probes = []
    candidates = []
    if sample is not None:
        for k in ("video_id", "id", "yt_id"):
            if k in sample and sample[k]:
                candidates.append(sample[k])
                break
    for url_k in ("url", "video_url", "youtube_url"):
        if sample is not None and url_k in sample and sample[url_k]:
            candidates.append(sample[url_k])
            break
    candidates = candidates[:3]
    for c in candidates:
        try:
            r = subprocess.run(
                ["yt-dlp", "-J", "--no-warnings", "--no-playlist", "--skip-download", f"https://www.youtube.com/watch?v={c}" if len(c) == 11 else c],
                capture_output=True, text=True, timeout=60,
            )
            ok = r.returncode == 0
            probes.append({"id": c, "ok": ok, "stdout_first_200": (r.stdout or "")[:200], "stderr_first_200": (r.stderr or "")[:200]})
        except Exception as exc:
            probes.append({"id": c, "ok": False, "error": repr(exc)})
    summary["yt_probe"] = probes

    summary["elapsed_sec"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"  wrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
