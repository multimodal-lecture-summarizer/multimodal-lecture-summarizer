"""Quick YTSeg text-only probe: extract video IDs, sample chapter schema, yt-dlp probe."""
import json
import subprocess
import time
from pathlib import Path

OUT = Path(r"C:\Users\hung\Documents\GitHub\multimodal-lecture-summarizer\multimodal-lecture-summarizer\plans\260830-1917-scientific-benchmark\probes\output\ytseg_text_only.json")


def main():
    started = time.perf_counter()
    summary = {}

    from datasets import load_dataset, get_dataset_split_names

    print("[1] List splits in 'text' config", flush=True)
    try:
        names = get_dataset_split_names("retkowski/ytseg", "text")
        summary["text_splits"] = names
    except Exception as e:
        summary["text_splits_error"] = repr(e)[:300]
        OUT.write_text(json.dumps(summary, indent=2))
        return

    print("[2] Get 10 records from text/test streaming", flush=True)
    try:
        ds = load_dataset("retkowski/ytseg", "text", split="test", streaming=True, trust_remote_code=True)
        rows = []
        for i, r in enumerate(ds):
            rows.append(r)
            if i >= 9:
                break
        summary["sample_count"] = len(rows)
        if rows:
            summary["sample_keys"] = list(rows[0].keys())
            summary["sample_row_0"] = {k: str(rows[0].get(k))[:300] for k in rows[0].keys()}
            if len(rows) > 1:
                summary["sample_row_1"] = {k: str(rows[1].get(k))[:300] for k in rows[1].keys()}
            # Extract video IDs
            video_ids = []
            for r in rows:
                for k in ("video_id", "id", "yt_id"):
                    if k in r and r[k]:
                        video_ids.append(r[k])
                        break
            summary["video_ids_sample"] = video_ids
    except Exception as e:
        summary["text_streaming_error"] = repr(e)[:500]

    print("[3] yt-dlp probe on 5 video IDs", flush=True)
    probes = []
    for vid in summary.get("video_ids_sample", [])[:5]:
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            r = subprocess.run(
                ["yt-dlp", "-J", "--no-warnings", "--no-playlist", "--skip-download", url],
                capture_output=True, text=True, timeout=45,
            )
            ok = r.returncode == 0 and '"title"' in (r.stdout or "")
            err = (r.stderr or "")[:200]
            probes.append({"id": vid, "ok": ok, "stderr_excerpt": err})
        except subprocess.TimeoutExpired:
            probes.append({"id": vid, "ok": False, "error": "timeout"})
        except Exception as e:
            probes.append({"id": vid, "ok": False, "error": repr(e)[:200]})
    summary["yt_probe"] = probes
    summary["yt_available"] = sum(1 for p in probes if p.get("ok"))

    summary["elapsed_sec"] = time.perf_counter() - started
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
