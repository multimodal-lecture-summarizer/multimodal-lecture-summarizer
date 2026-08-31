"""Quick check: which splits does gigant/tib-bench actually expose?"""
import json
from datasets import load_dataset, get_dataset_split_names

OUT = "C:/Users/hung/Documents/GitHub/multimodal-lecture-summarizer/multimodal-lecture-summarizer/plans/260830-1917-scientific-benchmark/probes/output/tib_bench_splits.json"

# Check several candidate repo names
candidates = [
    "gigant/tib-bench",
    "gigant/tib-bench-mm-test",
    "gigant/tib-bench-mm-part1",
    "gigant/tib_slides",
]

out = {}
for name in candidates:
    try:
        names = get_dataset_split_names(name)
        out[name] = {"splits": names, "ok": True}
    except Exception as e:
        out[name] = {"error": repr(e)[:300], "ok": False}

# Try to load and count rows in test split for each
for name in candidates:
    if out[name].get("ok") and "test" in out[name]["splits"]:
        try:
            ds = load_dataset(name, split="test", streaming=True, trust_remote_code=True)
            n = 0
            for r in ds:
                n += 1
                if n >= 5:
                    break
            out[name]["test_streaming_first_5"] = n
        except Exception as e:
            out[name]["test_streaming_error"] = repr(e)[:300]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
