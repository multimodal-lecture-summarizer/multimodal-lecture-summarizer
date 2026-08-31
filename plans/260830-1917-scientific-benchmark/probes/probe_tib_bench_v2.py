"""Verify: does gigant/tib-bench have a 'split' column to filter test, or
are all 822 records exposed as one 'train' split that needs DOI-based filtering?
"""
import json
import time
from pathlib import Path

OUT = Path(r"C:\Users\hung\Documents\GitHub\multimodal-lecture-summarizer\multimodal-lecture-summarizer\plans\260830-1917-scientific-benchmark\probes\output\tib_bench_v2_summary.json")


def main():
    started = time.perf_counter()
    summary = {}

    from datasets import load_dataset
    print("Loading gigant/tib-bench (streaming) ...", flush=True)
    try:
        bench = load_dataset("gigant/tib-bench", split="train", streaming=True, trust_remote_code=True)
        first_keys = None
        n = 0
        first_row = None
        for r in bench:
            if n == 0:
                first_keys = list(r.keys())
                first_row = dict(r)
            n += 1
            if n >= 5:
                break
        summary["first_keys"] = first_keys
        summary["first_row"] = {k: str(first_row.get(k))[:200] for k in first_keys} if first_row else None
        summary["streaming_ok"] = True
    except Exception as e:
        summary["streaming_error"] = repr(e)[:500]
        OUT.write_text(json.dumps(summary, indent=2))
        return

    # Check all 822 records (no streaming limit)
    print("Loading full tib-bench and checking for split/dataset_split column ...", flush=True)
    try:
        bench = load_dataset("gigant/tib-bench", split="train", trust_remote_code=True)
        summary["total_records"] = len(bench)
        # Check if there's a 'split' column
        keys = bench.column_names
        summary["all_columns"] = keys
        if "split" in keys:
            from collections import Counter
            summary["split_distribution"] = dict(Counter(bench["split"]))
        if "dataset_split" in keys:
            from collections import Counter
            summary["dataset_split_distribution"] = dict(Counter(bench["dataset_split"]))
        # Check what fields are available for filtering
        for k in keys[:20]:
            sample = bench[k][0] if len(bench) > 0 else None
            summary[f"sample_{k}"] = str(sample)[:200] if not isinstance(sample, (list, dict)) else f"<{type(sample).__name__}>"
    except Exception as e:
        summary["full_load_error"] = repr(e)[:500]

    # Also check the sibling files on the hub
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info("gigant/tib-bench")
        siblings = [s.rfilename for s in (info.siblings or [])]
        summary["siblings_count"] = len(siblings)
        summary["parquet_files"] = [s for s in siblings if s.endswith(".parquet")]
        summary["json_files"] = [s for s in siblings if s.endswith(".json")]
        summary["all_files_sample"] = siblings[:20]
    except Exception as e:
        summary["hub_error"] = repr(e)[:500]

    summary["elapsed_sec"] = time.perf_counter() - started
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
