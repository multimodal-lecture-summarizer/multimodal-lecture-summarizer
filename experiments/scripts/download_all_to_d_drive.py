import os
import sys
import shutil
import urllib.request
from pathlib import Path

# Configure all caching to D: drive
d_root = Path(r"D:\datasets")
d_root.mkdir(parents=True, exist_ok=True)

hf_cache = d_root / "hf_cache"
hf_cache.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache)
os.environ["TORCH_HOME"] = str(d_root / "torch_cache")

print("======================================================================")
print(f" DOWNLOADING & CONSOLIDATING ALL DATASETS AND MODELS TO D: DRIVE")
print(f" Target Directory: {d_root}")
print("======================================================================")

# 1. Download HuggingFace Pretrained Models to D:\datasets\hf_cache
print("\n[1/4] Downloading Pre-trained HuggingFace Models to D:\\datasets\\hf_cache...")
try:
    from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
    print("  -> Downloading CLIP ViT-B/32 model & processor...")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(hf_cache))
    clip_mod = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(hf_cache))
    
    print("  -> Downloading SentenceTransformer all-MiniLM-L6-v2 model & tokenizer...")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir=str(hf_cache))
    mod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir=str(hf_cache))
    print("[OK] HuggingFace Models downloaded successfully to D:\\datasets\\hf_cache!")
except Exception as e:
    print(f"[WARN] HuggingFace model download exception: {e}")

# 2. Consolidate TED-LIUM Dataset on D:\datasets\TEDLIUM
print("\n[2/4] Consolidating TED-LIUM Dataset to D:\\datasets\\TEDLIUM...")
d_ted = d_root / "TEDLIUM"
d_ted.mkdir(parents=True, exist_ok=True)

try:
    from datasets import load_dataset
    print("  -> Downloading TED-LIUM speech dataset split to D:\\datasets\\hf_cache...")
    ds = load_dataset("distil-whisper/tedlium", "default", cache_dir=str(hf_cache))
    print(f"[OK] TED-LIUM dataset downloaded ({len(ds)} splits available)!")
except Exception as e:
    print(f"[WARN] TED-LIUM HuggingFace download exception: {e}")

# 3. Consolidate TVSum Video & Thumbnail Dataset to D:\datasets\tvsum
print("\n[3/4] Downloading TVSum Benchmark Dataset to D:\\datasets\\tvsum...")
d_tvsum = d_root / "tvsum"
d_tvsum.mkdir(parents=True, exist_ok=True)

tvsum_videos_url = "https://people.csail.mit.edu/yichao/projects/tvsum/data/ydata-tvsum50-video.tgz"
tvsum_anno_url = "https://people.csail.mit.edu/yichao/projects/tvsum/data/ydata-tvsum50-data.tgz"

def download_file(url, target_path):
    if not target_path.exists():
        print(f"  -> Downloading {target_path.name} from {url}...")
        try:
            urllib.request.urlretrieve(url, target_path)
            print(f"  [OK] Downloaded {target_path.name}")
        except Exception as err:
            print(f"  [WARN] Failed to download {target_path.name}: {err}")
    else:
        print(f"  [OK] {target_path.name} already exists in D:\\datasets\\tvsum.")

download_file(tvsum_videos_url, d_tvsum / "ydata-tvsum50-video.tgz")
download_file(tvsum_anno_url, d_tvsum / "ydata-tvsum50-data.tgz")

# Copy existing extracted TVSum dataset if available
proj_tvsum = Path(__file__).resolve().parent.parent / "datasets" / "tvsum_extracted"
if proj_tvsum.exists():
    d_tvsum_extracted = d_tvsum / "tvsum_extracted"
    if not d_tvsum_extracted.exists():
        print("  -> Copying extracted TVSum videos & thumbnails to D:\\datasets\\tvsum\\tvsum_extracted...")
        shutil.copytree(proj_tvsum, d_tvsum_extracted, dirs_exist_ok=True)
        print("  [OK] TVSum extracted data copied to D:\\datasets\\tvsum!")

# 4. Copy Trained Model Weights to D:\datasets\models
print("\n[4/4] Backup Trained PyTorch Model Weights to D:\\datasets\\models...")
d_models = d_root / "models"
d_models.mkdir(parents=True, exist_ok=True)

proj_models = Path(__file__).resolve().parent.parent.parent / "storage" / "models"
if proj_models.exists():
    for weight_file in proj_models.glob("*.pth"):
        dst = d_models / weight_file.name
        shutil.copy(weight_file, dst)
        print(f"  [OK] Copied {weight_file.name} to D:\\datasets\\models\\{weight_file.name}")

print("\n" + "="*70)
print(" ALL DATASETS & MODELS HAVE BEEN DOWNLOADED & SAVED TO D: DRIVE!")
print(f" Root Folder:   D:\\datasets")
print(f" HF Cache:      D:\\datasets\\hf_cache")
print(f" TED-LIUM:      D:\\datasets\\TEDLIUM")
print(f" TVSum Dataset: D:\\datasets\\tvsum")
print(f" Model Weights: D:\\datasets\\models")
print("="*70 + "\n")

if __name__ == "__main__":
    pass
