"""Re-verify TIB media URLs are still alive (small HEAD probe on 10 lecture test records)."""
import json
import re
import subprocess
import time
from pathlib import Path

OUT = Path(r"C:\Users\hung\Documents\GitHub\multimodal-lecture-summarizer\multimodal-lecture-summarizer\plans\260830-1917-scientific-benchmark\probes\output\tib_media_health.json")
IDS = Path(r"C:\Users\hung\Documents\GitHub\multimodal-lecture-summarizer\multimodal-lecture-summarizer\plans\260830-1917-scientific-benchmark\probes\output\tib_lecture_test_ids.json")


def main():
    started = time.perf_counter()
    summary = {}

    if not IDS.exists():
        summary["error"] = "tib_lecture_test_ids.json not found; run probe_tib.py first"
        OUT.write_text(json.dumps(summary, indent=2))
        return

    from datasets import load_dataset
    print("[1] Loading TIB to get fresh video_url field ...", flush=True)
    ds = load_dataset("gigant/tib", split="test", trust_remote_code=True)

    records = json.loads(IDS.read_text(encoding="utf-8"))
    test_dois = {r["doi"]: r for r in records if r.get("doi")}

    probes = []
    for i, rec in enumerate(ds):
        if len(probes) >= 10:
            break
        doi = rec.get("doi")
        if not doi or doi not in test_dois:
            continue
        url = rec.get("video_url")
        if not url:
            continue
        try:
            r = subprocess.run(
                ["curl", "-sI", "-L", "--max-time", "30", url],
                capture_output=True, text=True, timeout=40,
            )
            ok = r.returncode == 0 and "200" in r.stdout
            ct = ""
            for line in r.stdout.splitlines():
                if line.lower().startswith("content-type"):
                    ct = line.split(":", 1)[1].strip()
                    break
            size = ""
            for line in r.stdout.splitlines():
                if line.lower().startswith("content-length"):
                    size = line.split(":", 1)[1].strip()
                    break
            probes.append({
                "doi": doi,
                "url": url,
                "ok": ok,
                "status_line": r.stdout.splitlines()[0] if r.stdout else "",
                "content_type": ct,
                "content_length": size,
            })
        except Exception as e:
            probes.append({"doi": doi, "ok": False, "error": repr(e)[:200]})

    summary["probe_count"] = len(probes)
    summary["available"] = sum(1 for p in probes if p.get("ok"))
    summary["probes"] = probes

    summary["elapsed_sec"] = time.perf_counter() - started
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {OUT}", flush=True)
    print(f"available: {summary['available']} / {summary['probe_count']}", flush=True)


if __name__ == "__main__":
    main()
