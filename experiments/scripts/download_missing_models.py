"""Download missing local models into project ./cache."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Prefer classic HF downloads; Xet CAS DNS often fails on this machine.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Point HuggingFace cache into project so pipeline finds local weights.
os.environ["HF_HOME"] = str(CACHE / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(CACHE / "torch_hub")


def mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def ensure_clip_from_local_copy() -> None:
    """CLIP already exists under cache/clip-vit-base-patch32; also ensure HF hub snapshot."""
    local = CACHE / "clip-vit-base-patch32" / "pytorch_model.bin"
    print(f"[CLIP] local pytorch_model.bin = {mb(local):.1f} MB | exists={local.exists()}")
    if not local.exists():
        from huggingface_hub import snapshot_download

        print("[CLIP] downloading openai/clip-vit-base-patch32 ...")
        snapshot_download(
            repo_id="openai/clip-vit-base-patch32",
            local_dir=str(CACHE / "clip-vit-base-patch32"),
            local_dir_use_symlinks=False,
        )
    # Also materialize in HF hub cache used by from_pretrained("openai/...")
    from huggingface_hub import snapshot_download

    print("[CLIP] ensuring HF hub cache snapshot ...")
    snapshot_download(repo_id="openai/clip-vit-base-patch32")
    print(f"[CLIP] done. folder={mb(CACHE / 'clip-vit-base-patch32'):.1f} MB")


def download_blip() -> None:
    target = CACHE / "blip-image-captioning-base"
    weight = target / "pytorch_model.bin"
    hub_weight = None
    hub_root = CACHE / "huggingface" / "hub" / "models--Salesforce--blip-image-captioning-base"
    if hub_root.exists():
        cand = list(hub_root.rglob("pytorch_model.bin"))
        if cand:
            hub_weight = cand[0]

    print(f"[BLIP] local dir weight={mb(weight):.1f} MB | hub weight={(mb(hub_weight) if hub_weight else 0):.1f} MB")
    from huggingface_hub import snapshot_download

    print("[BLIP] downloading Salesforce/blip-image-captioning-base ...")
    snapshot_download(
        repo_id="Salesforce/blip-image-captioning-base",
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    snapshot_download(repo_id="Salesforce/blip-image-captioning-base")
    print(f"[BLIP] done. folder={mb(target):.1f} MB")


def download_faster_whisper(model_size: str) -> None:
    """Download CT2 Faster-Whisper model into cache/faster-whisper-{size}."""
    out_dir = CACHE / f"faster-whisper-{model_size}"
    model_bin = out_dir / "model.bin"
    if model_bin.exists() and mb(model_bin) > 50:
        print(f"[Whisper] {model_size} already present ({mb(model_bin):.1f} MB) — skip")
        return

    print(f"[Whisper] downloading {model_size} into {out_dir} ...")
    from huggingface_hub import snapshot_download

    repo_id = f"Systran/faster-whisper-{model_size}"
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[Whisper] {model_size} done. model.bin={mb(model_bin):.1f} MB")


def check_florence() -> None:
    florence = ROOT / "ai_workers" / "modules" / "visual_v2" / "florence2_vendor" / "model.safetensors"
    print(f"[Florence-2] vendor model.safetensors = {mb(florence):.1f} MB | exists={florence.exists()}")
    if not florence.exists() or mb(florence) < 100:
        from huggingface_hub import snapshot_download

        dest = CACHE / "Florence-2-base"
        print("[Florence-2] downloading microsoft/Florence-2-base ...")
        snapshot_download(
            repo_id="microsoft/Florence-2-base",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        # Copy weights into vendor path expected by SemanticAnalyzer
        vendor = ROOT / "ai_workers" / "modules" / "visual_v2" / "florence2_vendor"
        src = dest / "model.safetensors"
        if src.exists():
            shutil.copy2(src, vendor / "model.safetensors")
            print(f"[Florence-2] copied weights -> {vendor / 'model.safetensors'}")
    else:
        print("[Florence-2] already complete — skip download")


def warmup_paddleocr() -> None:
    print("[PaddleOCR] warming up models (first init downloads det/rec/cls) ...")
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        _ = ocr  # keep reference briefly
        print("[PaddleOCR] ready")
    except Exception as e:
        print(f"[PaddleOCR] warmup failed (non-fatal): {e}")


def main() -> int:
    print(f"ROOT={ROOT}")
    print(f"CACHE={CACHE}")
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print("=" * 60)

    check_florence()
    ensure_clip_from_local_copy()
    download_blip()
    download_faster_whisper("base.en")  # already used by AudioTranscriber
    download_faster_whisper("large-v3")  # listed in worker_settings.WHISPERX_MODEL
    warmup_paddleocr()

    print("=" * 60)
    print("SUMMARY")
    for name in [
        "faster-whisper-base.en",
        "faster-whisper-large-v3",
        "clip-vit-base-patch32",
        "blip-image-captioning-base",
        "Florence-2-base",
    ]:
        p = CACHE / name
        print(f"  {name}: {mb(p):.1f} MB | exists={p.exists()}")
    florence = ROOT / "ai_workers" / "modules" / "visual_v2" / "florence2_vendor" / "model.safetensors"
    print(f"  florence2_vendor/model.safetensors: {mb(florence):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
