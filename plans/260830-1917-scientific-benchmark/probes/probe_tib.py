"""Probe TIB (gigant/tib) for the 6-month research plan.

This script:
  1. Loads the official TIB splits (train/valid/test).
  2. Counts records containing "Lecture" in `genre`.
  3. Inspects license string cleanliness.
  4. Checks abstract quality (length distribution, % empty/boilerplate).
  5. Verifies transcript presence and length distribution.
  6. Inspects `tib-bench` (multimodal extension) split containment via STREAMING.
  7. Writes a per-record manifest and a JSON summary (always, even on partial failure).
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "tib_summary.json"
LECTURE_IDS_PATH = OUT_DIR / "tib_lecture_test_ids.json"
USABLE_PATH = OUT_DIR / "tib_lecture_usable.json"

summary: dict = {}


def clean_license(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


def is_summary_like(abstract: str, title: str) -> tuple[bool, str]:
    if not abstract:
        return False, "empty"
    n = len(abstract)
    if n < 60:
        return False, f"too-short({n})"
    if n > 4000:
        return False, f"too-long({n})"
    lower = abstract.lower()
    if "this work is licensed under" in lower:
        return False, "license-banner"
    if "all rights reserved" in lower and n < 300:
        return False, "rights-reserved"
    if "©" in abstract and n < 200:
        return False, "copyright-only"
    if title and abstract.strip().lower() == title.strip().lower():
        return False, "title-only"
    return True, "ok"


def main() -> int:
    started = time.perf_counter()
    summary["elapsed_sec_so_far"] = 0

    try:
        from datasets import load_dataset

        print("[1/5] Loading gigant/tib ...", flush=True)
        ds = load_dataset("gigant/tib", trust_remote_code=True)
        splits = {name: len(ds[name]) for name in ds.keys()}
        summary["splits"] = splits
        print(f"  splits: {splits}", flush=True)

        print("[2/5] Counting Lecture genre in test split ...", flush=True)
        test = ds["test"]
        lecture_records = []
        for i, rec in enumerate(test):
            genre_raw = rec.get("genre", "") or ""
            genres_norm = re.sub(r"\s+", " ", str(genre_raw)).lower()
            if "lecture" in genres_norm:
                lecture_records.append({"index": i, "doi": rec.get("doi"), "title": rec.get("title"), "genre": genre_raw})
        summary["lecture_test_total"] = len(lecture_records)
        print(f"  pure-or-mixed Lecture records in test: {len(lecture_records)}", flush=True)

        print("[3/5] Inspecting license cleanliness & abstract quality ...", flush=True)
        cleaned_licenses = {}
        usable = []
        excluded_reasons = {}
        for rec in lecture_records:
            idx = rec["index"]
            full = test[idx]
            license_raw = full.get("license", "") or ""
            license_clean = clean_license(license_raw)
            cleaned_licenses[license_clean] = cleaned_licenses.get(license_clean, 0) + 1
            abstract = full.get("abstract", "") or ""
            title = full.get("title", "") or ""
            ok, reason = is_summary_like(abstract, title)
            transcript = full.get("transcript", "") or ""
            if not ok:
                excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
            if ok and len(transcript) > 200:
                usable.append({
                    "doi": rec["doi"],
                    "title": rec["title"],
                    "license": license_clean,
                    "abstract_len": len(abstract),
                    "transcript_len": len(transcript),
                    "has_keyframes": bool(full.get("keyframes")),
                    "video_url_present": bool(full.get("video_url")),
                    "language": full.get("language"),
                })
        summary["lecture_test_usable"] = len(usable)
        summary["excluded_reasons"] = excluded_reasons
        summary["license_distribution_top10"] = dict(sorted(cleaned_licenses.items(), key=lambda x: -x[1])[:10])

        # Also count Lecture in train and valid
        lecture_train = 0
        lecture_valid = 0
        for i, rec in enumerate(ds["train"]):
            genre_raw = rec.get("genre", "") or ""
            if "lecture" in re.sub(r"\s+", " ", str(genre_raw)).lower():
                lecture_train += 1
        for i, rec in enumerate(ds["valid"]):
            genre_raw = rec.get("genre", "") or ""
            if "lecture" in re.sub(r"\s+", " ", str(genre_raw)).lower():
                lecture_valid += 1
        summary["lecture_train"] = lecture_train
        summary["lecture_valid"] = lecture_valid
        print(f"  usable lecture test records: {len(usable)} / {len(lecture_records)}", flush=True)
        print(f"  lecture in train/valid: {lecture_train}/{lecture_valid}", flush=True)

        # Save partial
        LECTURE_IDS_PATH.write_text(json.dumps(lecture_records, indent=2, ensure_ascii=False))
        USABLE_PATH.write_text(json.dumps(usable, indent=2, ensure_ascii=False))

    except Exception as exc:
        summary["load_error"] = repr(exc)
        print(f"  TIB load FAILED: {exc!r}", file=sys.stderr)

    print("[4/5] Checking gigant/tib-bench split containment (streaming) ...", flush=True)
    try:
        from datasets import load_dataset
        # STREAMING so we don't load images
        bench = load_dataset("gigant/tib-bench", split="test", streaming=True, trust_remote_code=True)
        test_dois = set()
        for rec in LECTURE_IDS_PATH.read_text(encoding="utf-8") if LECTURE_IDS_PATH.exists() else "[]":
            pass
        # Recompute test_dois from saved file
        if LECTURE_IDS_PATH.exists():
            test_dois = {r["doi"] for r in json.loads(LECTURE_IDS_PATH.read_text(encoding="utf-8")) if r.get("doi")}
        bench_in_test = []
        bench_lecture_in_test = []
        n_seen = 0
        for rec in bench:
            n_seen += 1
            doi = rec.get("doi")
            if doi and doi in test_dois:
                bench_in_test.append(doi)
                bench_lecture_in_test.append(doi)
            if n_seen >= 2000:  # safety
                break
        summary["tib_bench_seen"] = n_seen
        summary["tib_bench_in_test"] = len(bench_in_test)
        summary["tib_bench_lecture_in_test"] = len(bench_lecture_in_test)
        print(f"  scanned {n_seen} bench records; in-test: {len(bench_in_test)}; in-lecture-test: {len(bench_lecture_in_test)}", flush=True)
    except Exception as exc:
        summary["bench_error"] = repr(exc)
        print(f"  bench check FAILED: {exc!r}", file=sys.stderr)

    print("[5/5] Writing final summary ...", flush=True)
    summary["elapsed_sec"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  wrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
