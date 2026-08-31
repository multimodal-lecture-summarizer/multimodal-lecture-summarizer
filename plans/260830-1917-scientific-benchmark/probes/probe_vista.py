"""Probe VISTA (dongqi-me/VISTA) for the 6-month research plan.

VISTA is form-gated. This script:
  1. Tries to load the dataset card without auth.
  2. If denied, fetches metadata about the access form.
  3. Attempts to load the dataset with no token (will fail gracefully).
  4. Reports whether any parquet files are visible in the repo tree.
  5. Summarizes gating status for the project owner.

Output: probes/output/vista_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "vista_summary.json"


def main() -> int:
    started = time.perf_counter()
    print("[1/4] Probing dongqi-me/VISTA via huggingface_hub ...", flush=True)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not available", file=sys.stderr)
        return 1

    api = HfApi()
    info = {
        "loaded": False,
        "gating": "unknown",
        "license": "unknown",
        "downloads_30d": None,
        "last_modified": None,
        "siblings_count": None,
        "files": [],
        "error": None,
    }

    # Get dataset card metadata (no auth required for metadata)
    try:
        ds_info = api.dataset_info("dongqi-me/VISTA")
        info["license"] = getattr(ds_info, "license", "unknown") or "unknown"
        info["last_modified"] = str(getattr(ds_info, "lastModified", None))
        info["gating"] = "gated" if getattr(ds_info, "gated", False) else "open"
        info["siblings_count"] = len(getattr(ds_info, "siblings", []) or [])
        siblings = getattr(ds_info, "siblings", []) or []
        info["files"] = [s.rfilename for s in siblings[:30]]
        info["downloads_30d"] = getattr(ds_info, "downloads", None) or getattr(ds_info, "downloadsAllTime", None)
    except Exception as exc:
        info["error"] = repr(exc)
        print(f"  failed to read dataset metadata: {exc!r}", file=sys.stderr)

    # Try to read the README
    print("[2/4] Reading README ...", flush=True)
    readme_path = OUT_DIR / "vista_readme.md"
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id="dongqi-me/VISTA", filename="README.md", repo_type="dataset", cache_dir=None)
        Path(p).replace(readme_path)
        info["readme_chars"] = readme_path.stat().st_size
    except Exception as exc:
        info["readme_error"] = repr(exc)
        print(f"  failed to fetch README: {exc!r}", file=sys.stderr)

    # Try to list parquet files (without downloading) to gauge structure
    print("[3/4] Listing parquet/tree ...", flush=True)
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files("dongqi-me/VISTA", repo_type="dataset")
        info["all_files_sample"] = files[:30]
        info["all_files_count"] = len(files)
        parquet_files = [f for f in files if f.endswith(".parquet")]
        info["parquet_count"] = len(parquet_files)
        info["parquet_sample"] = parquet_files[:5]
        json_files = [f for f in files if f.endswith(".json") or f.endswith(".jsonl")]
        info["json_count"] = len(json_files)
        info["json_sample"] = json_files[:5]
    except Exception as exc:
        info["tree_error"] = repr(exc)

    # Try to load via `datasets` library (will likely fail without gate approval)
    print("[4/4] Trying datasets.load_dataset (no token) ...", flush=True)
    try:
        from datasets import load_dataset
        # Try a streaming load of just the test split, only first 2 rows
        ds = load_dataset("dongqi-me/VISTA", split="test", streaming=True, trust_remote_code=True)
        rows = []
        for i, r in enumerate(ds):
            rows.append(r)
            if i >= 1:
                break
        info["load_streaming_ok"] = True
        info["load_streaming_first_keys"] = list(rows[0].keys()) if rows else []
        info["load_streaming_first_id"] = rows[0].get("id") if rows else None
    except Exception as exc:
        info["load_streaming_ok"] = False
        info["load_streaming_error"] = repr(exc)
        print(f"  streaming load failed (expected if gated): {exc!r}", flush=True)

    info["elapsed_sec"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print(f"  wrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
