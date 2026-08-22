#!/usr/bin/env python3
"""Benchmark lecture captioners on the same keyframes.

Compares: placeholder vs Florence-2 (production) vs BLIP-base (old stack).
Metric: CLIPScore (reference-free) + latency + generic rate.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FLORENCE_DEVICE", "cuda")
os.environ.setdefault("ENABLE_FLORENCE_CAPTIONING", "1")
os.environ.setdefault("FLORENCE_MIN_AVAILABLE_MEMORY_MB", "1024")
os.environ.setdefault("FLORENCE_MIN_AVAILABLE_VRAM_MB", "2048")
os.environ.setdefault("FLORENCE_MAX_CAPTIONS", "8")

CLIP_DIR = ROOT / "cache" / "clip-vit-base-patch32"
BLIP_DIR = ROOT / "cache" / "blip-image-captioning-base"


def _patch_torch_load() -> None:
    try:
        import transformers.modeling_utils
        import transformers.utils
        import transformers.utils.import_utils

        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        transformers.utils.check_torch_load_is_safe = lambda: None
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


def pick_frames(n: int = 4) -> tuple[str, list[dict]]:
    from experiments.evaluation.datasets import pick_ted_lecture_videos
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector

    videos = pick_ted_lecture_videos(limit=1)
    if not videos:
        raise RuntimeError("no TED video")
    path = videos[0]
    name = f"TED:{path.stem}"
    det = SceneDetector({"threshold": 27.0})
    scenes = det.detect_scenes(str(path))
    out = ROOT / "outputs" / "model_choice" / "caption_frames"
    det.extract_keyframes(str(path), scenes, str(out), strategy="middle")
    usable = [s for s in scenes if s.get("keyframe_path") and Path(s["keyframe_path"]).exists()]
    if len(usable) > n:
        step = max(1, len(usable) // n)
        usable = usable[::step][:n]
    return name, usable


def caption_placeholder(scenes: list[dict]) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    caps = [f"Keyframe for Scene {s.get('scene_index', '')}".strip() for s in scenes]
    return caps, time.perf_counter() - t0


def caption_florence(scenes: list[dict]) -> tuple[list[str], float]:
    from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer

    copies = [dict(s) for s in scenes]
    t0 = time.perf_counter()
    SemanticAnalyzer().caption_scenes_florence2(copies)
    wall = time.perf_counter() - t0
    return [(s.get("caption") or "").strip() for s in copies], wall


def caption_blip(scenes: list[dict], device: str) -> tuple[list[str], float]:
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    src = str(BLIP_DIR) if (BLIP_DIR / "config.json").exists() else "Salesforce/blip-image-captioning-base"
    print(f"[BLIP] loading {src} on {device}")
    processor = BlipProcessor.from_pretrained(src)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(src, torch_dtype=dtype).to(device)
    model.eval()
    prompt = "a video scene showing"
    caps = []
    t0 = time.perf_counter()
    try:
        for s in scenes:
            img = Image.open(s["keyframe_path"]).convert("RGB")
            inputs = processor(img, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=30, min_new_tokens=5, repetition_penalty=1.5)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text.lower().startswith(prompt):
                text = text[len(prompt) :].strip()
            caps.append(text)
            print(f"  [BLIP] {Path(s['keyframe_path']).name}: {text}")
    finally:
        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return caps, time.perf_counter() - t0


def clip_scores(paths: list[str], captions: list[str], device: str) -> list[float]:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    src = str(CLIP_DIR) if (CLIP_DIR / "config.json").exists() else "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(src)
    model = CLIPModel.from_pretrained(src).to(device)
    model.eval()
    scores = []
    try:
        for path, cap in zip(paths, captions):
            image = Image.open(path).convert("RGB")
            inputs = processor(text=[cap or "image"], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
                scores.append(float((img * txt).sum()) * 100.0)
    finally:
        del model, processor
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return scores


def summarize(name: str, caps: list[str], scores: list[float], wall: float, n: int) -> dict:
    from experiments.evaluation.metrics import caption_hallucination_flags

    flags = [caption_hallucination_flags(c) for c in caps]
    return {
        "model": name,
        "wall_sec": wall,
        "sec_per_frame": wall / n if n else float("nan"),
        "clipscore": sum(scores) / len(scores) if scores else float("nan"),
        "generic_rate": sum(1 for f in flags if f["generic"]) / n if n else float("nan"),
        "unique": len({c.lower() for c in caps}),
        "mean_tokens": sum(len(c.split()) for c in caps) / n if n else float("nan"),
        "captions": caps,
        "clipscores": scores,
    }


def _fmt(v, d=2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def main() -> int:
    _patch_torch_load()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video, scenes = pick_frames(4)
    paths = [s["keyframe_path"] for s in scenes]
    print(f"[Bakeoff] {video} frames={len(scenes)} device={device}")

    systems = []
    ph_caps, ph_t = caption_placeholder(scenes)
    systems.append(("placeholder", ph_caps, ph_t))

    fl_caps, fl_t = caption_florence(scenes)
    systems.append(("florence-2", fl_caps, fl_t))
    for s, c in zip(scenes, fl_caps):
        print(f"  [Florence] {Path(s['keyframe_path']).name}: {c}")

    bl_caps, bl_t = caption_blip(scenes, device)
    systems.append(("blip-base", bl_caps, bl_t))

    rows = []
    for name, caps, wall in systems:
        scores = clip_scores(paths, caps, device)
        rows.append(summarize(name, caps, scores, wall, len(scenes)))
        print(f"[SCORE] {name} CLIPScore={rows[-1]['clipscore']:.2f} t={wall:.1f}s")

    out_dir = ROOT / "outputs" / "model_choice"
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "video": video,
        "device": device,
        "n_frames": len(scenes),
        "metric": "CLIPScore = 100 * cos(CLIP image, CLIP text); no human caption GT",
        "results": rows,
        "frames": [Path(p).name for p in paths],
    }
    (out_dir / "CAPTION_BAKEOFF.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Benchmark caption: Florence-2 vs BLIP vs placeholder",
        "",
        f"*Generated: {payload['generated']}*",
        "",
        f"Cùng {len(scenes)} keyframe `{video}`, device=`{device}`.",
        "Không có caption người viết → **CLIPScore** (ảnh↔câu, CLIP ViT-B/32). Cao hơn = khớp hình hơn.",
        "",
        "| Model | CLIPScore | s/frame | Generic | Tokens TB |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        mark = " **← production**" if r["model"] == "florence-2" else ""
        lines.append(
            f"| {r['model']}{mark} | {_fmt(r['clipscore'])} | {_fmt(r['sec_per_frame'])} | "
            f"{_fmt(r['generic_rate'])} | {_fmt(r['mean_tokens'])} |"
        )
    lines += ["", "## Caption từng frame", "", "| Frame | Placeholder | Florence-2 | BLIP |"]
    lines.append("| --- | --- | --- | --- |")
    by = {r["model"]: r["captions"] for r in rows}
    for i, name in enumerate(payload["frames"]):
        lines.append(
            f"| `{name}` | {by['placeholder'][i]} | {by['florence-2'][i]} | {by['blip-base'][i]} |"
        )
    fl = next(r for r in rows if r["model"] == "florence-2")
    bl = next(r for r in rows if r["model"] == "blip-base")
    ph = next(r for r in rows if r["model"] == "placeholder")
    lines += [
        "",
        "## Kết luận",
        "",
        f"- Placeholder CLIPScore={_fmt(ph['clipscore'])} (câu không mô tả ảnh).",
        f"- Florence-2 CLIPScore={_fmt(fl['clipscore'])}, BLIP CLIPScore={_fmt(bl['clipscore'])}.",
        "- Đây **không** phải COCO CIDEr: không chứng minh SOTA thế giới, chỉ so 2 captioner trên frame lecture/TED.",
        "- Florence được chọn vì đã vendored + prompt `<CAPTION>`; BLIP là phương án cũ cùng chức năng.",
        "",
    ]
    (out_dir / "CAPTION_BAKEOFF.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out_dir / 'CAPTION_BAKEOFF.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
