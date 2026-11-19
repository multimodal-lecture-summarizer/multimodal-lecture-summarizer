"""Process a list of videos one after another and record per-video timings in a CSV.

Runs `ai_workers.tasks.process_video` in this process, the same pipeline the Colab worker runs, but
without a broker, Redis or R2: the caller sets CELERY_BROKER_URL=memory:// and
CELERY_RESULT_BACKEND=cache+memory://. Models are loaded once up front (the worker does the same at
start-up) and stay resident, so the per-video times match what a warm worker spends.

    python -m ai_workers.batch_runner --sources sources.json --out-csv results.csv --log-dir logs

`sources.json` is a list of {"url": ..., "group": ...}. A source is a YouTube URL (downloaded here
and timed separately), another http(s) URL (downloaded inside the pipeline, so it counts towards the
processing time) or a path to a local/Drive file. One CSV row is appended per video as soon as it
finishes, so an interrupted run loses nothing and a rerun skips videos already done.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Stage names as the pipeline logs them. audio and visual run in parallel, so their spans overlap:
# never add the stage columns up, use processing_sec for the whole job.
STAGES = (
    "preflight", "audio", "speaker", "visual", "object_tracking", "semantic",
    "timeline", "quality", "storage", "intermediate_ready", "summarizing",
)
ELEMENT_LABELS = ("table", "chart", "photo", "illustration")

COLUMNS = (
    ["index", "group", "url", "video_id", "title", "status", "error", "started_at",
     "download_sec", "processing_sec", "wall_sec", "video_sec", "realtime_factor"]
    + [f"stage_{name}_sec" for name in STAGES]
    + ["n_scenes", "n_keyframes", "n_chapters"]
    + [f"n_{label}" for label in ELEMENT_LABELS]
    + ["summary_words", "summary_method", "asr_model", "detected_language", "fallback_used"]
)

_TIMING_RE = re.compile(
    r"\[TIMING\] total=([\d.]+)s, video=([\d.]+)s(?:, realtime_factor=([\d.]+)x)? \| stage spans: (.*)"
)
_SPAN_RE = re.compile(r"(\w+)=([\d.]+)s")
# Pipeline progress lines such as "[AUDIO +12.3s] ..."; only these are echoed to the console.
_PROGRESS_RE = re.compile(r"^\[(?:[A-Z_]+ \+[\d.]+s|TIMING)\]")
_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([\w-]{11})")
_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def parse_timing(text: str) -> Optional[Dict[str, Any]]:
    """The last `[TIMING]` line of a job log: totals plus per-stage spans, or None."""
    matches = _TIMING_RE.findall(text or "")
    if not matches:
        return None
    total, video, factor, spans = matches[-1]
    return {
        "total": float(total),
        "video": float(video),
        "realtime_factor": float(factor) if factor else None,
        "spans": {name: float(value) for name, value in _SPAN_RE.findall(spans)},
    }


def youtube_id(url: str) -> str:
    match = _YOUTUBE_ID_RE.search(url or "")
    return match.group(1) if match else ""


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def is_remote_url(url: str) -> bool:
    return url.startswith(("http://", "https://"))


def count_slide_elements(scenes: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, int]:
    counts = {label: 0 for label in ELEMENT_LABELS}
    for scene in scenes or []:
        detected = scene.get("detected_objects") or {}
        for element in detected.get("slide_elements") or []:
            label = element.get("label")
            if label in counts:
                counts[label] += 1
    return counts


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def build_row(
    entry: Dict[str, Any],
    index: int,
    *,
    status: str,
    error: str = "",
    started_at: str = "",
    download_sec: Optional[float] = None,
    wall_sec: Optional[float] = None,
    result: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One CSV row. `result` is the process_video return value, `timing` the parsed [TIMING] line."""
    url = entry["url"]
    result = result or {}
    row: Dict[str, Any] = {column: "" for column in COLUMNS}
    row.update(
        index=index,
        group=entry.get("group", ""),
        url=url,
        video_id=youtube_id(url) or Path(url).stem,
        title=result.get("video_title", ""),
        status=status,
        error=error,
        started_at=started_at,
        download_sec=_round(download_sec),
        wall_sec=_round(wall_sec),
    )
    video_sec = (timing or {}).get("video") or result.get("duration")
    row["processing_sec"] = _round((timing or {}).get("total") or result.get("processing_time"))
    row["video_sec"] = _round(video_sec)
    factor = (timing or {}).get("realtime_factor")
    if factor is None and video_sec and row["processing_sec"]:
        factor = row["processing_sec"] / video_sec
    row["realtime_factor"] = _round(factor, 2)
    spans = (timing or {}).get("spans", {})
    for name in STAGES:
        if name in spans:
            row[f"stage_{name}_sec"] = _round(spans[name])
    if result:
        scenes = result.get("scenes") or []
        row["n_scenes"] = len(scenes)
        row["n_keyframes"] = len(result.get("keyframes") or [])
        row["n_chapters"] = len(result.get("chapters") or [])
        for label, count in count_slide_elements(scenes).items():
            row[f"n_{label}"] = count
        row["summary_words"] = result.get("word_count", "")
        row["summary_method"] = result.get("summary_method", "")
        row["asr_model"] = result.get("asr_model", "")
        row["detected_language"] = result.get("detected_language", "")
        row["fallback_used"] = result.get("fallback_used", "")
    return row


def load_done_urls(csv_path: Path) -> set[str]:
    """URLs that already finished with status 'done' (a rerun skips them)."""
    if not csv_path.is_file():
        return set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["url"] for row in csv.DictReader(handle) if row.get("status") == "done"}


def append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


class _Tee(io.TextIOBase):
    """Collects everything written to it; progress lines are also echoed to the real console."""

    def __init__(self, echo) -> None:
        self._echo = echo
        self._lock = threading.Lock()
        self._chunks: List[str] = []
        self._partial = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        with self._lock:
            self._chunks.append(text)
            # tqdm redraws with \r, so a progress line can precede the next log line without a \n.
            *lines, self._partial = re.split(r"[\r\n]", self._partial + text)
            self._partial = self._partial[-2000:]
            for line in lines:
                if _PROGRESS_RE.match(line):
                    self._echo.write(line + "\n")
                    self._echo.flush()
        return len(text)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        with self._lock:
            return "".join(self._chunks)


def _find_local_copy(video_dir: Optional[Path], video_id: str) -> Optional[Path]:
    """A pre-downloaded file `<video_id>.<ext>` in video_dir, for when YouTube refuses Colab."""
    if not video_dir or not video_id or not video_dir.is_dir():
        return None
    for suffix in _VIDEO_SUFFIXES:
        candidate = video_dir / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def acquire(entry: Dict[str, Any], work_dir: Path, video_dir: Optional[Path], max_duration: int) -> Tuple[str, bool]:
    """Return (path or URL for process_video, whether it is a temporary local file to delete)."""
    url = entry["url"]
    if is_youtube(url):
        from ai_workers.core.youtube_ingest import download_youtube

        video_id = youtube_id(url) or uuid.uuid4().hex[:11]
        try:
            return download_youtube(url, str(work_dir / f"{video_id}.mp4"), max_duration), True
        except Exception as err:  # noqa: BLE001 - bot checks and blocked IPs surface as many types
            local = _find_local_copy(video_dir, video_id)
            if local is None:
                raise RuntimeError(f"{type(err).__name__}: {str(err)[:300]}") from err
            print(f"[BATCH] YouTube refused the download ({type(err).__name__}); using {local.name} from the video folder.")
            source = local
    elif is_remote_url(url):
        return url, False  # process_video downloads it itself
    else:
        source = Path(url)
        if not source.is_file():
            raise FileNotFoundError(f"File not found: {url}")
    # A local source is copied next to the other temporary files: the pipeline writes its .wav beside
    # the video and removes it afterwards, which should not happen inside Drive.
    destination = work_dir / source.name
    shutil.copy2(source, destination)
    return str(destination), True


def process_one(entry: Dict[str, Any], index: int, args: argparse.Namespace, process_video) -> Dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir) if args.video_dir else None

    download_started = time.perf_counter()
    try:
        local, is_temp = acquire(entry, work_dir, video_dir, args.max_duration)
    except Exception as err:  # noqa: BLE001
        return build_row(entry, index, status="download_failed", error=str(err)[:500], started_at=started_at)
    download_sec = time.perf_counter() - download_started if is_temp else None

    job_id = str(uuid.uuid4())
    log_path = Path(args.log_dir) / f"{index:02d}_{youtube_id(entry['url']) or Path(entry['url']).stem}.log"
    tee = _Tee(sys.__stdout__)
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            outcome = process_video.apply(
                args=[job_id, local],
                kwargs={
                    "output_language": args.output_language,
                    "source_language": args.source_language,
                    "summary_length": args.summary_length,
                },
                task_id=job_id,
                throw=False,
            )
        wall_sec = time.perf_counter() - started
    finally:
        if is_temp and os.path.isfile(local):
            os.remove(local)
        shutil.rmtree(f"./outputs/{job_id}", ignore_errors=True)

    text = tee.getvalue()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    timing = parse_timing(text)

    if outcome.successful():
        result = outcome.result
        return build_row(
            entry, index,
            status=result.get("status", "done"),
            error=str(result.get("llm_error") or "")[:500],
            started_at=started_at, download_sec=download_sec, wall_sec=wall_sec,
            result=result, timing=timing,
        )
    last_line = (outcome.traceback or "").strip().splitlines()[-1:] or [repr(outcome.result)]
    return build_row(
        entry, index, status="error", error=last_line[0][:500],
        started_at=started_at, download_sec=download_sec, wall_sec=wall_sec, timing=timing,
    )


def run(args: argparse.Namespace) -> int:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    if args.limit:
        sources = sources[: args.limit]
    out_csv = Path(args.out_csv)
    done = load_done_urls(out_csv)

    # Heavy imports stay here so the helpers above can be tested without the pipeline's dependencies.
    from ai_workers.core.model_prefetch import run_prefetch
    from ai_workers.tasks import process_video

    started = time.perf_counter()
    print(f"[BATCH] Loading models once for {len(sources)} videos ({len(done)} already done)...", flush=True)
    run_prefetch(log=lambda message: print(f"[BATCH] {message}", flush=True))
    print(f"[BATCH] Models ready in {time.perf_counter() - started:.0f}s.", flush=True)

    ok = failed = 0
    for index, entry in enumerate(sources, start=1):
        if entry["url"] in done:
            print(f"[BATCH] {index}/{len(sources)} skip (already done): {entry['url']}", flush=True)
            continue
        print(f"[BATCH] {index}/{len(sources)} start: {entry['url']}", flush=True)
        try:
            row = process_one(entry, index, args, process_video)
        except Exception as err:  # noqa: BLE001 - one bad video must not stop the batch
            traceback.print_exc()
            row = build_row(entry, index, status="error", error=f"{type(err).__name__}: {str(err)[:300]}")
        append_row(out_csv, row)
        if row["status"] == "done":
            ok += 1
        else:
            failed += 1
        print(
            f"[BATCH] {index}/{len(sources)} {row['status']}: processing={row['processing_sec']}s "
            f"download={row['download_sec']}s video={row['video_sec']}s {row['error'][:120]}",
            flush=True,
        )
    print(f"[BATCH] Finished: {ok} done, {failed} not done, {time.perf_counter() - started:.0f}s in total.", flush=True)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", required=True, help="JSON list of {url, group}")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--log-dir", required=True, help="one full pipeline log per video")
    parser.add_argument("--work-dir", default="./storage/batch_tmp")
    parser.add_argument("--video-dir", default="", help="folder with pre-downloaded <video_id>.mp4 files")
    parser.add_argument("--max-duration", type=int, default=3600, help="reject YouTube videos longer than this (s)")
    parser.add_argument("--output-language", default="en")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--summary-length", default="medium")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N sources (smoke test)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
