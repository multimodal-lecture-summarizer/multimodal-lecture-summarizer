"""Probe EduVidQA (sourjyadip/eduvidqa-emnlp25) for the 6-month research plan.

Steps:
  1. Clone (or shallow clone) the GitHub repo into probes/cache/eduvidqa.
  2. Verify the train/dev/test split files exist and parse.
  3. Count total QA pairs, count by real/synthetic split.
  4. Extract YouTube video IDs and check 5-min yt-dlp availability probe.
  5. Report overall availability rate from a 30-video sample.

Output: probes/output/eduvidqa_summary.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "cache" / "eduvidqa"
SUMMARY_PATH = OUT_DIR / "eduvidqa_summary.json"
REPO_URL = "https://github.com/sourjyadip/eduvidqa-emnlp25.git"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main() -> int:
    started = time.perf_counter()
    summary = {
        "loaded": False,
        "error": None,
    }

    print("[1/4] Cloning EduVidQA repo (shallow) ...", flush=True)
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_DIR.exists():
        r = run(["git", "clone", "--depth=1", REPO_URL, str(CACHE_DIR)])
        if r.returncode != 0:
            summary["error"] = f"clone failed: {r.stderr[:500]}"
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
            print(f"  clone FAILED: {r.stderr[:200]}", file=sys.stderr)
            return 1
    else:
        print("  already cloned, skipping", flush=True)

    print("[2/4] Listing data directory ...", flush=True)
    data_dir = CACHE_DIR / "data"
    if not data_dir.exists():
        # Try alt paths
        for p in [CACHE_DIR / "EduVidQA", CACHE_DIR / "dataset"]:
            if p.exists():
                data_dir = p
                break
    if not data_dir.exists():
        # Show top-level
        summary["top_level"] = sorted(p.name for p in CACHE_DIR.iterdir())
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        print(f"  no data dir; top-level: {summary['top_level']}", flush=True)
        return 0

    summary["data_dir"] = str(data_dir.relative_to(CACHE_DIR))
    files = sorted(p.name for p in data_dir.rglob("*") if p.is_file())
    summary["data_files"] = files[:50]
    summary["data_files_count"] = len(files)

    print("[3/4] Parsing QA files ...", flush=True)
    qa_count = 0
    real_count = 0
    synth_count = 0
    sample_rows = []
    video_ids = set()
    import csv
    for csv_file in data_dir.rglob("*.csv"):
        try:
            with open(csv_file, encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for i, row in enumerate(rdr):
                    qa_count += 1
                    fname = csv_file.name.lower()
                    if "real" in fname:
                        real_count += 1
                    elif "synth" in fname:
                        synth_count += 1
                    # Extract video id
                    for k in ("video_id", "video", "yt_id", "youtube_id"):
                        if k in row and row[k]:
                            video_ids.add(row[k])
                            break
                    if len(sample_rows) < 5:
                        sample_rows.append({k: row.get(k) for k in list(row.keys())[:6]})
        except Exception as exc:
            print(f"  parse error on {csv_file.name}: {exc!r}", file=sys.stderr)

    summary["qa_total"] = qa_count
    summary["qa_real"] = real_count
    summary["qa_synthetic"] = synth_count
    summary["unique_video_ids"] = len(video_ids)
    summary["sample_rows"] = sample_rows
    summary["video_id_samples"] = list(video_ids)[:5]

    print(f"  QA pairs: {qa_count} (real={real_count}, synth={synth_count}), videos: {len(video_ids)}", flush=True)

    # 4. yt-dlp availability probe on 10 video ids
    print("[4/4] yt-dlp availability probe (10 videos) ...", flush=True)
    yt_probe = []
    if video_ids:
        sample = list(video_ids)[:10]
        for vid in sample:
            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                r = run(["yt-dlp", "-J", "--no-warnings", "--no-playlist", url], timeout=60)
                ok = r.returncode == 0 and "title" in (r.stdout or "")
                yt_probe.append({"video_id": vid, "ok": ok, "snippet": (r.stdout or "")[:120]})
            except Exception as exc:
                yt_probe.append({"video_id": vid, "ok": False, "error": repr(exc)})
        summary["yt_probe"] = yt_probe
        summary["yt_available"] = sum(1 for x in yt_probe if x.get("ok"))
        summary["yt_total_probed"] = len(yt_probe)
    else:
        # Try to extract from sample rows
        urls = []
        for row in sample_rows:
            for k, v in row.items():
                if v and ("youtube.com" in str(v) or "youtu.be" in str(v)):
                    urls.append(str(v))
        summary["extracted_youtube_urls"] = urls[:5]

    summary["elapsed_sec"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  wrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
