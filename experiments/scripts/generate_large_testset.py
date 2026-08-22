"""Generate a large local testset + benchmark manifest for pipeline stress tests.

This script creates multiple lecture-like videos by transforming the existing
demo sample video. It is designed for offline, reproducible load testing.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


def run_ffmpeg(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def ensure_demo_sample(repo_root: Path) -> Path:
    sample = repo_root / "experiments" / "notebooks" / "demo_data" / "sample.mp4"
    if not sample.exists():
        raise FileNotFoundError(f"Missing demo sample video: {sample}")
    return sample


def ensure_ffmpeg_available() -> bool:
    return run_ffmpeg(["ffmpeg", "-version"])


def build_variant(
    sample_path: Path,
    output_path: Path,
    variant: str,
    duration_sec: int,
    ffmpeg_ok: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not ffmpeg_ok:
        # Fallback: exact copy, still useful for queue/load test.
        shutil.copy2(sample_path, output_path)
        return

    vf_parts: list[str] = []
    if variant == "blur":
        vf_parts.append("boxblur=3:1")
    elif variant == "lowres":
        vf_parts.append("scale=640:360")
    elif variant == "contrast":
        vf_parts.append("eq=contrast=1.25:brightness=0.02:saturation=1.12")
    elif variant == "noisy":
        vf_parts.append("noise=alls=12:allf=t")

    vf_chain = ",".join(vf_parts) if vf_parts else "null"
    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(sample_path),
        "-t",
        str(duration_sec),
        "-vf",
        vf_chain,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    if not run_ffmpeg(cmd):
        # If transform fails, still emit usable file.
        shutil.copy2(sample_path, output_path)


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "lecture_id",
        "video_path",
        "duration_min",
        "language",
        "domain",
        "reference_transcript",
        "reference_rttm",
        "reference_slides_dir",
        "reference_summary",
        "reference_chapters",
        "notes",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12, help="Number of videos")
    parser.add_argument(
        "--durations",
        type=str,
        default="15,30,45,60",
        help="Comma-separated durations in minutes",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    sample_path = ensure_demo_sample(repo_root)
    ffmpeg_ok = ensure_ffmpeg_available()

    out_dir = repo_root / "benchmarks" / "data_large_local"
    manifest_path = repo_root / "benchmarks" / "manifest_large_local.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    durations_min = [int(x.strip()) for x in args.durations.split(",") if x.strip()]
    if not durations_min:
        raise ValueError("No durations provided")

    variants = ["normal", "blur", "lowres", "contrast", "noisy"]
    languages = ["vi", "en", "mixed"]
    domains = ["computer_science", "mathematics", "soft_skill", "data_science"]

    rows: list[dict[str, str]] = []
    for i in range(args.count):
        dmin = durations_min[i % len(durations_min)]
        variant = variants[i % len(variants)]
        lang = languages[i % len(languages)]
        domain = domains[i % len(domains)]
        lecture_id = f"stress_{i+1:02d}"
        file_name = f"{lecture_id}_{variant}_{dmin}m.mp4"
        output_path = out_dir / file_name
        build_variant(
            sample_path=sample_path,
            output_path=output_path,
            variant=variant,
            duration_sec=dmin * 60,
            ffmpeg_ok=ffmpeg_ok,
        )

        rows.append(
            {
                "lecture_id": lecture_id,
                "video_path": f"./benchmarks/data_large_local/{file_name}",
                "duration_min": str(dmin),
                "language": lang,
                "domain": domain,
                "reference_transcript": "",
                "reference_rttm": "",
                "reference_slides_dir": "",
                "reference_summary": "",
                "reference_chapters": "",
                "notes": f"synthetic_{variant}_from_demo_sample",
            }
        )

    write_manifest(rows, manifest_path)
    print(f"Generated {len(rows)} videos at: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"ffmpeg_available={ffmpeg_ok}")


if __name__ == "__main__":
    main()
