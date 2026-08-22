#!/usr/bin/env python3
"""Run production Florence-2 captions on a few lecture/TVSum keyframes."""

from __future__ import annotations

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
os.environ.setdefault("FLORENCE_MAX_CAPTIONS", "4")


def main() -> int:
    from experiments.evaluation.datasets import pick_ted_lecture_videos, pick_tvsum_videos
    from experiments.evaluation.metrics import caption_hallucination_flags
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector
    from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer

    out_dir = ROOT / "outputs" / "model_choice"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "florence_frames"

    videos = pick_ted_lecture_videos(limit=1)
    video_name = ""
    video_path = None
    if videos:
        video_path = videos[0]
        video_name = f"TED:{video_path.stem}"
    else:
        tv = pick_tvsum_videos(limit=1)
        if not tv:
            print("[ERR] no TED/TVSum video")
            return 1
        video_path = tv[0]["video_path"]
        video_name = f"TVSum:{tv[0]['video_id']}"

    print(f"[Florence] video={video_name} path={video_path}")
    det = SceneDetector({"threshold": 27.0})
    scenes = det.detect_scenes(str(video_path))
    det.extract_keyframes(str(video_path), scenes, str(frame_dir), strategy="middle")
    usable = [s for s in scenes if s.get("keyframe_path") and Path(s["keyframe_path"]).exists()]
    if len(usable) > 4:
        step = max(1, len(usable) // 4)
        usable = usable[::step][:4]
    print(f"[Florence] captioning {len(usable)}/{len(scenes)} frames")

    copies = [dict(s) for s in usable]
    t0 = time.perf_counter()
    SemanticAnalyzer().caption_scenes_florence2(copies)
    wall = time.perf_counter() - t0

    rows = []
    for s in copies:
        cap = (s.get("caption") or "").strip()
        flags = caption_hallucination_flags(cap, s.get("ocr_text") or "")
        rows.append(
            {
                "keyframe": Path(s.get("keyframe_path") or "kf").name,
                "caption": cap,
                "placeholder": f"Keyframe for Scene {s.get('scene_index', '')}".strip(),
                **flags,
            }
        )
        print(f"  {rows[-1]['keyframe']}: {cap}")

    skipped = all(r["caption"].startswith("Keyframe for Scene") for r in rows) if rows else True
    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "video": video_name,
        "device": os.environ.get("FLORENCE_DEVICE"),
        "wall_sec": wall,
        "skipped_placeholder": skipped,
        "rows": rows,
    }
    (out_dir / "FLORENCE.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Florence-2 caption (production)",
        "",
        f"*Generated: {report['generated']}*",
        "",
        f"Video: `{video_name}` · device=`{report['device']}` · {len(rows)} frame · {wall:.1f}s",
        "",
        "| Keyframe | Placeholder | Florence-2 | Generic? |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['keyframe']}` | {r['placeholder']} | {r['caption'] or '(empty)'} | "
            f"{'Yes' if r.get('generic') else 'No'} |"
        )
    if skipped:
        lines += ["", "_Florence không sinh caption (skip/RAM/lỗi) — vẫn là placeholder._"]
    else:
        lines += ["", "_Florence chạy được: caption khác placeholder, dùng được để biện minh chọn model._"]
    (out_dir / "FLORENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out_dir / 'FLORENCE.md'} skipped={skipped} wall={wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
