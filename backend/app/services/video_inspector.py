"""Video inspection and validation service using ffprobe.

Provides stream extraction (codecs, resolution, fps, bitrate) and compliance checking
against system video standards before processing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_fraction(val: Any) -> float:
    """Parse frame rate or ratio string like '30/1' or '30000/1001' into float."""
    if not val:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if "/" in val:
            try:
                num, den = val.split("/", 1)
                num_f, den_f = float(num), float(den)
                return num_f / den_f if den_f != 0 else 0.0
            except (ValueError, ZeroDivisionError):
                return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


def normalize_codec_name(codec: str | None) -> str:
    """Normalize video codec naming (e.g., avc1 -> h264, h265 -> hevc)."""
    if not codec:
        return ""
    c = str(codec).strip().lower()
    if c in ("avc1", "avc", "x264"):
        return "h264"
    if c in ("h265", "x265", "hev1"):
        return "hevc"
    if c in ("vp09", "vp90"):
        return "vp9"
    if c in ("av01",):
        return "av1"
    return c


def inspect_video_streams(file_path: str) -> dict[str, Any]:
    """Inspect a video file using ffprobe and extract structured stream and format metadata.

    Returns a dictionary with keys:
      - valid: bool
      - error: Optional[str]
      - video: Optional[dict] with keys:
          codec_name, width, height, fps, bitrate, frame_count, duration, pix_fmt
      - audio: Optional[dict] with keys:
          codec_name, sample_rate, channels, bitrate
      - format: Optional[dict] with keys:
          format_name, duration, size_bytes, size_mb, bitrate
    """
    if not file_path or not os.path.exists(file_path):
        return {
            "valid": False,
            "error": f"Tệp video không tồn tại: {file_path}",
            "video": None,
            "audio": None,
            "format": None,
        }

    ffprobe_cmd = shutil.which("ffprobe") or "ffprobe"

    cmd = [
        ffprobe_cmd,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        file_path,
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "valid": False,
            "error": "Quá thời gian (timeout) khi kiểm tra tệp video bằng ffprobe.",
            "video": None,
            "audio": None,
            "format": None,
        }
    except Exception as exc:
        return {
            "valid": False,
            "error": f"Lỗi thực thi ffprobe: {exc}",
            "video": None,
            "audio": None,
            "format": None,
        }

    if res.returncode != 0:
        err_msg = res.stderr.strip() or f"ffprobe thoát với mã lỗi {res.returncode}"
        return {
            "valid": False,
            "error": f"Tệp video hỏng hoặc định dạng không đọc được: {err_msg}",
            "video": None,
            "audio": None,
            "format": None,
        }

    try:
        probe_data = json.loads(res.stdout)
    except json.JSONDecodeError as json_err:
        return {
            "valid": False,
            "error": f"Không thể giải mã dữ liệu ffprobe JSON: {json_err}",
            "video": None,
            "audio": None,
            "format": None,
        }

    streams = probe_data.get("streams", [])
    format_info = probe_data.get("format", {})

    video_stream: dict[str, Any] | None = None
    audio_stream: dict[str, Any] | None = None

    for s in streams:
        c_type = s.get("codec_type")
        if c_type == "video" and video_stream is None:
            video_stream = s
        elif c_type == "audio" and audio_stream is None:
            audio_stream = s

    if not video_stream:
        return {
            "valid": False,
            "error": "Tệp không chứa luồng video (video stream) hợp lệ.",
            "video": None,
            "audio": None,
            "format": None,
        }

    # Format calculations
    try:
        fmt_duration = float(format_info.get("duration", 0.0) or 0.0)
    except (ValueError, TypeError):
        fmt_duration = 0.0

    try:
        fmt_size = int(format_info.get("size", 0) or os.path.getsize(file_path))
    except (ValueError, TypeError, OSError):
        fmt_size = 0
    size_mb = round(fmt_size / (1024 * 1024), 2)

    try:
        fmt_bitrate = int(format_info.get("bit_rate", 0) or 0)
    except (ValueError, TypeError):
        fmt_bitrate = None

    # Video stream calculations
    v_codec = normalize_codec_name(video_stream.get("codec_name"))
    try:
        v_width = int(video_stream.get("width", 0))
        v_height = int(video_stream.get("height", 0))
    except (ValueError, TypeError):
        v_width, v_height = 0, 0

    fps = parse_fraction(video_stream.get("r_frame_rate"))
    if fps <= 0:
        fps = parse_fraction(video_stream.get("avg_frame_rate"))

    try:
        v_bitrate = int(video_stream.get("bit_rate", 0) or fmt_bitrate or 0)
    except (ValueError, TypeError):
        v_bitrate = fmt_bitrate

    try:
        v_frames = int(video_stream.get("nb_frames", 0) or 0)
    except (ValueError, TypeError):
        v_frames = 0
    if v_frames <= 0 and fmt_duration > 0 and fps > 0:
        v_frames = int(round(fmt_duration * fps))

    try:
        v_duration = float(video_stream.get("duration", 0.0) or fmt_duration)
    except (ValueError, TypeError):
        v_duration = fmt_duration

    video_meta = {
        "codec_name": v_codec,
        "raw_codec_name": video_stream.get("codec_name", ""),
        "width": v_width,
        "height": v_height,
        "fps": round(fps, 3),
        "bitrate": v_bitrate if v_bitrate and v_bitrate > 0 else None,
        "frame_count": v_frames,
        "duration": round(v_duration, 3),
        "pix_fmt": video_stream.get("pix_fmt", ""),
    }

    # Audio stream calculations
    audio_meta = None
    if audio_stream:
        a_codec = str(audio_stream.get("codec_name", "")).strip().lower()
        try:
            sample_rate = int(audio_stream.get("sample_rate", 0) or 0)
        except (ValueError, TypeError):
            sample_rate = None
        try:
            channels = int(audio_stream.get("channels", 0) or 0)
        except (ValueError, TypeError):
            channels = None
        try:
            a_bitrate = int(audio_stream.get("bit_rate", 0) or 0)
        except (ValueError, TypeError):
            a_bitrate = None

        audio_meta = {
            "codec_name": a_codec,
            "sample_rate": sample_rate,
            "channels": channels,
            "bitrate": a_bitrate if a_bitrate and a_bitrate > 0 else None,
        }

    return {
        "valid": True,
        "error": None,
        "video": video_meta,
        "audio": audio_meta,
        "format": {
            "format_name": format_info.get("format_name", ""),
            "duration": round(fmt_duration, 3),
            "size_bytes": fmt_size,
            "size_mb": size_mb,
            "bitrate": fmt_bitrate,
        },
    }


def validate_video_standards(
    file_metadata: dict[str, Any],
    active_standard: Any,
) -> Tuple[bool, Optional[str]]:
    """Validate probed video streams metadata against active VideoStandard configuration.

    Checks:
      1. Inspection validity & presence of video stream
      2. Video codec matches allowed_codecs
      3. Resolution within [min_width, min_height] and [max_width, max_height]
      4. Duration within max_duration
      5. File size within max_file_size

    Returns:
      (True, None) if compliant, or (False, "Chi tiết lỗi") if non-compliant.
    """
    if not file_metadata or not file_metadata.get("valid"):
        err = file_metadata.get("error") if file_metadata else "Không có siêu dữ liệu"
        return False, f"Không thể thẩm định video hoặc file bị lỗi: {err}"

    video = file_metadata.get("video")
    if not video:
        return False, "Tệp không chứa luồng video hợp lệ."

    codec = normalize_codec_name(video.get("codec_name"))
    width = video.get("width") or 0
    height = video.get("height") or 0
    fmt = file_metadata.get("format") or {}
    duration = fmt.get("duration") or video.get("duration") or 0.0
    size_mb = fmt.get("size_mb") or 0.0

    # 1. Validate Codec
    raw_allowed = getattr(active_standard, "allowed_codecs", ["h264", "hevc"])
    if isinstance(raw_allowed, str):
        allowed_codecs = [
            normalize_codec_name(c.strip())
            for c in raw_allowed.split(",")
            if c.strip()
        ]
    elif isinstance(raw_allowed, (list, tuple)):
        allowed_codecs = [normalize_codec_name(c) for c in raw_allowed if c]
    else:
        allowed_codecs = ["h264", "hevc"]

    if codec not in allowed_codecs:
        return (
            False,
            f"Codec '{codec}' không được hỗ trợ. Các codec hợp lệ: {allowed_codecs}",
        )

    # 2. Validate Resolution
    min_width = getattr(active_standard, "min_width", 640) or 640
    min_height = getattr(active_standard, "min_height", 360) or 360
    max_width = getattr(active_standard, "max_width", 1920) or 1920
    max_height = getattr(active_standard, "max_height", 1080) or 1080

    if width < min_width or height < min_height:
        return (
            False,
            f"Độ phân giải video ({width}x{height}) thấp hơn mức tối thiểu cho phép "
            f"({min_width}x{min_height}).",
        )

    if width > max_width or height > max_height:
        return (
            False,
            f"Độ phân giải video ({width}x{height}) vượt quá mức tối đa cho phép "
            f"({max_width}x{max_height}).",
        )

    # 3. Validate Duration
    max_duration = getattr(active_standard, "max_duration", 3600) or 3600
    if duration > 0 and duration > max_duration:
        return (
            False,
            f"Thời lượng video ({duration:.1f} giây) vượt quá giới hạn cho phép ({max_duration} giây).",
        )

    # 4. Validate File Size
    max_file_size = getattr(active_standard, "max_file_size", 500) or 500
    if size_mb > 0 and size_mb > max_file_size:
        return (
            False,
            f"Dung lượng video ({size_mb:.2f} MB) vượt quá giới hạn cho phép ({max_file_size} MB).",
        )

    return True, None
