"""Probe EduVidQA with the correct schema (id/vid_id)."""
import csv
import json
import re
import subprocess
import time
from pathlib import Path

OUT = Path(r"C:\Users\hung\Documents\GitHub\multimodal-lecture-summarizer\multimodal-lecture-summarizer\plans\260830-1917-scientific-benchmark\probes\output\eduvidqa_v2_summary.json")
DATA = Path(r"C:\Users\hung\Documents\GitHub\multimodal-lecture-summarizer\multimodal-lecture-summarizer\plans\260830-1917-scientific-benchmark\probes\cache\eduvidqa\data")


def main():
    started = time.perf_counter()
    summary = {"files": []}

    video_ids = set()
    sample_rows = []
    for f in DATA.glob("*.csv"):
        with open(f, encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            rows = list(rdr)
        ids = set()
        for row in rows:
            if "id" in row and row["id"]:
                ids.add(row["id"])
            if "vid_id" in row and row["vid_id"]:
                ids.add(row["vid_id"])
        video_ids.update(ids)
        sample_rows.append({
            "file": f.name,
            "rows": len(rows),
            "cols": rdr.fieldnames,
            "unique_ids": len(ids),
            "sample_row": {k: str(row.get(k))[:200] for k, row in zip(rdr.fieldnames, [rows[0]]) if rows} if rows else None,
        })
    summary["files"] = sample_rows
    summary["qa_total"] = sum(s["rows"] for s in sample_rows)
    summary["unique_video_ids"] = len(video_ids)
    summary["video_ids_sample"] = sorted(video_ids)[:20]

    # yt-dlp HEAD probe
    print(f"[yt-dlp] Probing 15 video IDs ...", flush=True)
    probes = []
    for vid in sorted(video_ids)[:15]:
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            r = subprocess.run(
                ["yt-dlp", "-J", "--no-warnings", "--no-playlist", "--skip-download", url],
                capture_output=True, text=True, timeout=45,
            )
            ok = r.returncode == 0 and '"title"' in (r.stdout or "")
            err_excerpt = (r.stderr or "")[:200]
            probes.append({"id": vid, "ok": ok, "stderr_excerpt": err_excerpt})
        except subprocess.TimeoutExpired:
            probes.append({"id": vid, "ok": False, "error": "timeout"})
        except Exception as e:
            probes.append({"id": vid, "ok": False, "error": repr(e)[:200]})
    summary["yt_probe_count"] = len(probes)
    summary["yt_available"] = sum(1 for p in probes if p.get("ok"))
    summary["yt_probe"] = probes

    # Compute total estimated: scale up to all video ids
    if probes:
        rate = summary["yt_available"] / len(probes)
        summary["estimated_total_available"] = round(rate * len(video_ids))

    summary["elapsed_sec"] = time.perf_counter() - started
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
