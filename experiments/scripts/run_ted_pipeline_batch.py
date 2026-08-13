"""Run test_pipeline.ipynb sequentially on multiple TED-LIUM videos."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "experiments" / "notebooks" / "test_pipeline.ipynb"
TED_DIR = Path(r"D:\datasets\TEDLIUM\videos")
VENV_ROOT = REPO_ROOT / ".venv-florence"


def _prepend_nvidia_dll_paths(env: dict) -> None:
    """Add NVIDIA cudnn/cublas bins to PATH so torch GPU ops work on Windows."""
    candidates = [
        VENV_ROOT / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        VENV_ROOT / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        VENV_ROOT / "Lib" / "site-packages" / "torch" / "lib",
    ]
    prefix = os.pathsep.join(str(p) for p in candidates if p.is_dir())
    if prefix:
        env["PATH"] = prefix + os.pathsep + env.get("PATH", "")

# Default batch: short -> medium -> classic TED talk
DEFAULT_VIDEOS = [
    "S44.mp4",
    "Barry_Schwartz.mp4",
    "Do schools kill creativity? | Sir Ken Robinson | TED.mp4",
]


def _patch_video_cell(nb: nbformat.NotebookNode, video_name: str) -> None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        if "TED_VIDEO_NAME" in src and "TED_VIDEO_DIR" in src:
            cell.source = re.sub(
                r'TED_VIDEO_NAME\s*=\s*"[^"]*"',
                f'TED_VIDEO_NAME = "{video_name}"',
                src,
                count=1,
            )
            return
    raise RuntimeError("Could not find TED_VIDEO_NAME cell in notebook")


def _cell_text(cell: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            chunks.append("".join(out.get("text", [])))
        elif out.get("output_type") == "error":
            chunks.append(f"ERROR {out.get('ename')}: {out.get('evalue')}")
    return "".join(chunks)


def _extract_metrics(nb: nbformat.NotebookNode) -> dict:
    metrics: dict = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        txt = _cell_text(cell)
        if "Job ID:" in txt:
            m = re.search(r"Job ID:\s*(\S+)", txt)
            if m:
                metrics["job_id"] = m.group(1)
            m = re.search(r"Video Input:\s*(.+)", txt)
            if m:
                metrics["video_path"] = m.group(1).strip()
        if "Quality Status:" in txt:
            m = re.search(r"Quality Status:\s*(\S+)", txt)
            if m:
                metrics["asr_quality"] = m.group(1)
            m = re.search(r"Average Confidence:\s*([\d.]+)", txt)
            if m:
                metrics["asr_confidence"] = float(m.group(1))
        if "Total Extracted Keyframes/Scenes:" in txt:
            m = re.search(r"Total Extracted Keyframes/Scenes:\s*(\d+)", txt)
            if m:
                metrics["scenes_raw"] = int(m.group(1))
        if "Filtered Distinct Slides:" in txt:
            m = re.search(r"Filtered Distinct Slides:\s*(\d+)", txt)
            if m:
                metrics["slides"] = int(m.group(1))
        if "Video Title:" in txt:
            m = re.search(r"Video Title:\s*(.+)", txt)
            if m:
                metrics["title"] = m.group(1).strip()
        if "Exported production summary" in txt:
            m = re.search(r"Exported production summary.*?:\s*(.+)", txt)
            if m:
                metrics["export_path"] = m.group(1).strip()
        if "PyTorch:" in txt and "CUDA Available:" in txt:
            m = re.search(r"PyTorch:\s*([^\|]+)\|\s*CUDA Available:\s*(\w+)", txt)
            if m:
                metrics["torch"] = m.group(1).strip()
                metrics["cuda"] = m.group(2).strip()

    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append(f"cell{i}:{out.get('ename')}")
                break
    metrics["errors"] = errors
    metrics["status"] = "FAILED" if errors else "OK"
    return metrics


def run_one(video_name: str, use_gpu: bool) -> dict:
    if not (TED_DIR / video_name).is_file():
        return {"video": video_name, "status": "SKIPPED", "reason": "file not found"}

    nb = nbformat.read(NOTEBOOK, as_version=4)
    _patch_video_cell(nb, video_name)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["FLORENCE_DEVICE"] = "cpu"
    env["PADDLE_USE_GPU"] = "0"
    if use_gpu:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        _prepend_nvidia_dll_paths(env)
    else:
        # Hide GPU so Whisper/CLIP stay on CPU (avoids cudnn DLL crash on Windows).
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    os.environ.update(env)

    print(f"\n{'=' * 70}\nBATCH RUN: {video_name} | GPU={use_gpu}\n{'=' * 70}", flush=True)
    t0 = time.time()
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
        allow_errors=True,
    )
    try:
        client.execute()
    except Exception as exc:
        print(f"[BATCH ERROR] {video_name}: {exc}", flush=True)
        return {
            "video": video_name,
            "status": "FAILED",
            "reason": str(exc),
            "elapsed_min": round((time.time() - t0) / 60, 2),
            "gpu_requested": use_gpu,
        }
    elapsed = time.time() - t0

    metrics = _extract_metrics(nb)
    metrics.update(
        {
            "video": video_name,
            "elapsed_min": round(elapsed / 60, 2),
            "gpu_requested": use_gpu,
        }
    )
    print(
        f"-> {metrics.get('status')} | {elapsed/60:.1f} min | job={metrics.get('job_id')} | export={metrics.get('export_path')}",
        flush=True,
    )
    if metrics.get("errors"):
        print(f"   errors: {metrics['errors']}", flush=True)
    return metrics


def main() -> int:
    videos = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_VIDEOS
    # Resolve TED filenames on disk (handles unicode punctuation variants).
    resolved: list[str] = []
    for v in videos:
        direct = TED_DIR / v
        if direct.is_file():
            resolved.append(v)
            continue
        alt = next((p.name for p in TED_DIR.glob("*.mp4") if p.name.lower() == v.lower()), None)
        if alt:
            resolved.append(alt)
        else:
            resolved.append(v)
    videos = resolved
    use_gpu = os.environ.get("USE_GPU", "1").strip() not in {"0", "false", "False"}

    print(f"TED batch runner | videos={len(videos)} | USE_GPU={use_gpu}")
    print(f"Notebook: {NOTEBOOK}")

    results: list[dict] = []
    for video in videos:
        results.append(run_one(video, use_gpu=use_gpu))

    out_dir = REPO_ROOT / "outputs" / "ted_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"batch_summary_{stamp}.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\nBATCH COMPLETE\n{'=' * 70}")
    for r in results:
        print(
            f"- {r.get('video')}: {r.get('status')} | {r.get('elapsed_min', '?')} min | "
            f"cuda={r.get('cuda', '?')} | {r.get('title', r.get('reason', ''))[:60]}"
        )
    print(f"\nSummary saved: {summary_path}")
    failed = sum(1 for r in results if r.get("status") != "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
