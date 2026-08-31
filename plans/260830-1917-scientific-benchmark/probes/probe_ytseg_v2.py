"""Probe YTSeg with the correct config (audio/text/titles)."""
import json
import subprocess
import time
from pathlib import Path

OUT_DIR = Path(r"C:\Users\hung\Documents\GitHub\multimodal-lecture-summarizer\multimodal-lecture-summarizer\plans\260830-1917-scientific-benchmark\probes\output")
OUT = OUT_DIR / "ytseg_v2_summary.json"


def main():
    started = time.perf_counter()
    summary = {}

    from datasets import load_dataset, get_dataset_split_names
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info("retkowski/ytseg")
    summary["license"] = getattr(info, "license", "unknown")
    summary["gating"] = "gated" if getattr(info, "gated", False) else "open"
    summary["siblings_count"] = len(getattr(info, "siblings", []) or [])

    print("[1] List splits per config ...", flush=True)
    for cfg in ["audio", "text", "titles"]:
        try:
            names = get_dataset_split_names("retkowski/ytseg", cfg)
            summary[f"config_{cfg}_splits"] = names
        except Exception as e:
            summary[f"config_{cfg}_error"] = repr(e)[:300]

    print("[2] Load 5 rows from 'text' config test split ...", flush=True)
    try:
        ds = load_dataset("retkowski/ytseg", "text", split="test", streaming=True, trust_remote_code=True)
        sample_keys = None
        first_row = None
        video_ids = []
        for i, r in enumerate(ds):
            if i == 0:
                first_row = dict(r)
                sample_keys = list(r.keys())
            for k in ("video_id", "id", "yt_id", "videoID"):
                if k in r and r[k]:
                    video_ids.append(r[k])
                    break
            if i >= 4:
                break
        summary["text_test_first_keys"] = sample_keys
        summary["text_test_first_row"] = {k: str(first_row.get(k))[:200] for k in sample_keys} if first_row else None
        summary["text_test_video_ids_sample"] = video_ids
    except Exception as e:
        summary["text_test_error"] = repr(e)[:500]

    print("[3] Load 5 rows from 'audio' config ...", flush=True)
    try:
        ds = load_dataset("retkowski/ytseg", "audio", split="test", streaming=True, trust_remote_code=True)
        sample_keys = None
        first_row = None
        for i, r in enumerate(ds):
            if i == 0:
                first_row = dict(r)
                sample_keys = list(r.keys())
            if i >= 4:
                break
        summary["audio_test_first_keys"] = sample_keys
        summary["audio_test_first_row"] = {k: str(first_row.get(k))[:200] for k in sample_keys} if first_row else None
    except Exception as e:
        summary["audio_test_error"] = repr(e)[:500]

    print("[4] yt-dlp HEAD probe on 5 video IDs from text config ...", flush=True)
    probes = []
    for vid in summary.get("text_test_video_ids_sample", [])[:5]:
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            r = subprocess.run(
                ["yt-dlp", "-J", "--no-warnings", "--no-playlist", "--skip-download", url],
                capture_output=True, text=True, timeout=60,
            )
            ok = r.returncode == 0 and '"title"' in (r.stdout or "")
            probes.append({"id": vid, "ok": ok, "stderr_first_300": (r.stderr or "")[:300]})
        except Exception as e:
            probes.append({"id": vid, "ok": False, "error": repr(e)})
    summary["yt_probe"] = probes
    summary["yt_available"] = sum(1 for p in probes if p.get("ok"))

    summary["elapsed_sec"] = time.perf_counter() - started
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
