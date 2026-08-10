import os
import sys
import time
from datasets import load_dataset

# Force UTF-8 stdout/stderr for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def main():
    # Set HF_HOME to D:/datasets/hf_cache to utilize 400+ GB disk space on D: drive
    cache_dir = "D:/datasets/hf_cache"
    os.environ["HF_HOME"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("TED-LIUM Dataset Downloader & Verifier")
    print(f"Cache Location: {cache_dir}")
    print("=" * 60)

    splits = ["validation", "test", "train"]
    dataset_summary = {}

    for split in splits:
        print(f"\n[+] Processing split: '{split}'...")
        start_time = time.time()
        try:
            ds = load_dataset(
                "distil-whisper/tedlium",
                "default",
                revision="refs/convert/parquet",
                split=split,
                cache_dir=cache_dir
            )
            elapsed = time.time() - start_time
            num_examples = len(ds)
            print(f"  [OK] Successfully downloaded/loaded '{split}' split!")
            print(f"  [OK] Examples count: {num_examples:,}")
            print(f"  [OK] Time elapsed: {elapsed:.2f} seconds")
            dataset_summary[split] = num_examples
        except Exception as e:
            print(f"  [ERROR] Error downloading split '{split}': {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All TED-LIUM splits (train, validation, test) successfully downloaded and verified!")
    print("Summary:")
    for split, count in dataset_summary.items():
        print(f"  * {split}: {count:,} samples")
    print("=" * 60)

if __name__ == "__main__":
    main()
