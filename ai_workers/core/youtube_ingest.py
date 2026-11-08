"""Fetch YouTube videos and publish them to R2.

Shared by the pipeline, which downloads on the worker that runs it, and by the pre-download task,
which downloads on a residential IP because YouTube often blocks datacenter IPs such as Colab.
"""

from __future__ import annotations

import glob

from ai_workers.core.config import worker_settings


def download_youtube(url: str, dest_path: str, max_duration: int = 0) -> str:
    """Download a video and return its real path, since yt-dlp may pick another extension.

    A positive max_duration (seconds) rejects longer videos before any media is transferred.
    """
    import yt_dlp

    ydl_opts = {
        "outtmpl": dest_path,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "retries": 10,
        "fragment_retries": 10,
        "source_address": "0.0.0.0",
        "extractor_args": {"youtube": {"player_client": ["android", "web", "mweb"]}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        if max_duration > 0:
            info = ydl.extract_info(url, download=False) or {}
            duration = float(info.get("duration") or 0)
            if duration > max_duration:
                raise ValueError(
                    f"Video dài {duration:.0f} giây, vượt giới hạn {max_duration} giây."
                )
        ydl.download([url])

    base_without_ext = dest_path.rsplit(".", 1)[0]
    matching_files = glob.glob(base_without_ext + ".*")
    return matching_files[0] if matching_files else dest_path


def upload_video_to_r2(local_path: str, object_key: str) -> str:
    """Upload a video to R2 and return its public URL. Upload errors propagate to the caller."""
    import boto3

    s3_client = boto3.client(
        "s3",
        endpoint_url=f"https://{worker_settings.CF_R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=worker_settings.CF_R2_ACCESS_KEY_ID,
        aws_secret_access_key=worker_settings.CF_R2_SECRET_ACCESS_KEY,
    )
    s3_client.upload_file(
        local_path,
        worker_settings.CF_R2_BUCKET_NAME,
        object_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    public_base = worker_settings.CF_R2_PUBLIC_URL.rstrip("/")
    return f"{public_base}/{object_key}"
