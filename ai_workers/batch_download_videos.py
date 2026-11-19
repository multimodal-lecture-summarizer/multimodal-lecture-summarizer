"""Download the batch videos to this machine and pack them into one zip for Google Drive.

Run it from the repository root on a machine whose IP YouTube does not block (your own PC):

    pip install yt-dlp
    python -m ai_workers.batch_download_videos

By default the links come from the VIDEOS list in ai_workers/colab_batch_30_videos.ipynb, so the zip
always matches what the notebook will process. Each video is saved as `<video_id>.mp4`, the name the
notebook looks for. Upload `videos.zip` to MyDrive/mls-batch/; the notebook unpacks it into
MyDrive/mls-batch/videos by itself (videos already there are not copied again).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

# Run as a plain script or as a module: either way the package root must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_workers.batch_runner import youtube_id  # noqa: E402  (stdlib-only module)

NOTEBOOK = Path(__file__).with_name("colab_batch_30_videos.ipynb")
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".avi")
_LIST_RE = re.compile(r"VIDEOS\s*=\s*\[(.*?)\n\]", re.S)
_URL_RE = re.compile(r"https?://[^\s'\"]+")


def read_notebook_urls(notebook: Path = NOTEBOOK) -> List[str]:
    """The URLs in the notebook's VIDEOS list, in order and without duplicates."""
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell["source"])
        match = _LIST_RE.search(source) if cell["cell_type"] == "code" else None
        if match:
            urls = [m.group(0) for m in _URL_RE.finditer(match.group(1))]
            return list(dict.fromkeys(urls))
    raise ValueError(f"No VIDEOS list found in {notebook}")


def read_urls_file(path: Path) -> List[str]:
    """One URL per line; blank lines and # comments are ignored."""
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return list(dict.fromkeys(line for line in lines if line))


def find_video(out_dir: Path, video_id: str) -> Optional[Path]:
    for suffix in VIDEO_SUFFIXES:
        candidate = out_dir / f"{video_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def download_all(urls: List[str], out_dir: Path, max_duration: int) -> List[str]:
    """Download every URL not already in out_dir; returns the URLs that failed.

    Uses the pipeline's own download_youtube (yt-dlp with the android/web/mweb clients and IPv4), the
    configuration known to work for it. Plain yt-dlp defaults get "HTTP Error 403" on every video.
    """
    from ai_workers.core.youtube_ingest import download_youtube

    out_dir.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    for number, url in enumerate(urls, start=1):
        video_id = youtube_id(url)
        existing = find_video(out_dir, video_id)
        if existing:
            print(f"[{number}/{len(urls)}] {video_id}: already downloaded ({existing.stat().st_size / 1e6:.1f} MB)")
            continue
        try:
            saved = Path(download_youtube(url, str(out_dir / f"{video_id}.mp4"), max_duration))
        except Exception as err:  # noqa: BLE001 - keep going, report at the end
            print(f"[{number}/{len(urls)}] {video_id}: FAILED {type(err).__name__}: {str(err)[:200]}", flush=True)
            failed.append(url)
            continue
        if saved.is_file():
            print(f"[{number}/{len(urls)}] {video_id}: ok ({saved.stat().st_size / 1e6:.1f} MB)", flush=True)
        else:
            print(f"[{number}/{len(urls)}] {video_id}: FAILED no file was written", flush=True)
            failed.append(url)
    return failed


def make_zip(urls: List[str], out_dir: Path, zip_path: Path) -> int:
    """Zip the downloaded files flat (no folders). Videos are already compressed, so they are stored."""
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for url in urls:
            video = find_video(out_dir, youtube_id(url))
            if video:
                archive.write(video, arcname=video.name)
                count += 1
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urls-file", type=Path, help="text file with one link per line (default: the notebook's list)")
    parser.add_argument("--out-dir", type=Path, default=Path("batch_videos"))
    parser.add_argument("--zip", type=Path, default=Path("videos.zip"), dest="zip_path")
    parser.add_argument("--max-duration", type=int, default=3600, help="skip videos longer than this (seconds)")
    parser.add_argument("--limit", type=int, default=0, help="download only the first N links")
    args = parser.parse_args(argv)

    urls = read_urls_file(args.urls_file) if args.urls_file else read_notebook_urls()
    if args.limit:
        urls = urls[: args.limit]
    print(f"{len(urls)} videos -> {args.out_dir}/")

    failed = download_all(urls, args.out_dir, args.max_duration)
    packed = make_zip(urls, args.out_dir, args.zip_path)
    size_mb = args.zip_path.stat().st_size / 1e6
    print(f"\nZip: {args.zip_path} ({packed} videos, {size_mb:.0f} MB)")
    if failed:
        print(f"{len(failed)} not downloaded:")
        for url in failed:
            print("  ", url)
        print("Rerun the same command to retry them (finished videos are kept).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
