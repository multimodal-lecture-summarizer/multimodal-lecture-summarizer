"""Probe VT-SSum (Dod-o/VT-SSum on GitHub) for the 6-month research plan.

Steps:
  1. Clone the repo shallow.
  2. List data files and split definitions.
  3. Count raw items per split.
  4. Extract 5 sample records to inspect schema.

Output: probes/output/vtssum_summary.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "cache" / "vtssum"
SUMMARY_PATH = OUT_DIR / "vtssum_summary.json"
REPO_URL = "https://github.com/Dod-o/VT-SSum.git"


def main() -> int:
    started = time.perf_counter()
    summary = {}

    print("[1/3] Cloning VT-SSum shallow ...", flush=True)
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_DIR.exists():
        r = subprocess.run(["git", "clone", "--depth=1", REPO_URL, str(CACHE_DIR)], capture_output=True, text=True)
        if r.returncode != 0:
            summary["error"] = f"clone failed: {r.stderr[:500]}"
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
            return 1
    print(f"  repo at {CACHE_DIR}", flush=True)

    print("[2/3] Scanning repo structure ...", flush=True)
    top = sorted(p.name for p in CACHE_DIR.iterdir())
    summary["top_level"] = top
    # Look for data dirs
    data_candidates = [p for p in CACHE_DIR.rglob("*.json") if p.is_file()][:30]
    summary["json_files_sample"] = [str(p.relative_to(CACHE_DIR)) for p in data_candidates]
    summary["json_files_count"] = len(list(CACHE_DIR.rglob("*.json")))

    print("[3/3] Inspecting README for split counts ...", flush=True)
    readme = CACHE_DIR / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8", errors="ignore")
        # crude extraction
        import re
        m = re.search(r"(\d[\d,]*)\s*train", text, re.IGNORECASE)
        if m:
            summary["readme_train"] = m.group(1)
        m = re.search(r"(\d[\d,]*)\s*(dev|valid)", text, re.IGNORECASE)
        if m:
            summary["readme_dev"] = m.group(1)
        m = re.search(r"(\d[\d,]*)\s*test", text, re.IGNORECASE)
        if m:
            summary["readme_test"] = m.group(1)

    summary["elapsed_sec"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  wrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
