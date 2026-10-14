import os
import unittest
from unittest.mock import patch, MagicMock

from app.services.video_inspector import (
    inspect_video_streams,
    validate_video_standards,
    normalize_codec_name,
    parse_fraction,
)
from app.models.video import VideoStandard
from app.schemas.video import VideoStandardBase, VideoStandardDTO


class TestVideoInspectorService(unittest.TestCase):
    def test_parse_fraction(self):
        self.assertEqual(parse_fraction("25/1"), 25.0)
        self.assertAlmostEqual(parse_fraction("30000/1001"), 29.970, places=2)
        self.assertEqual(parse_fraction("30"), 30.0)
        self.assertEqual(parse_fraction(24.0), 24.0)
        self.assertEqual(parse_fraction("invalid"), 0.0)
        self.assertEqual(parse_fraction(""), 0.0)
        self.assertEqual(parse_fraction(None), 0.0)

    def test_normalize_codec_name(self):
        self.assertEqual(normalize_codec_name("avc1"), "h264")
        self.assertEqual(normalize_codec_name("H264"), "h264")
        self.assertEqual(normalize_codec_name("hev1"), "hevc")
        self.assertEqual(normalize_codec_name("h265"), "hevc")
        self.assertEqual(normalize_codec_name("vp09"), "vp9")
        self.assertEqual(normalize_codec_name("av01"), "av1")
        self.assertEqual(normalize_codec_name("prores"), "prores")
        self.assertEqual(normalize_codec_name(None), "")

    def test_inspect_missing_file(self):
        result = inspect_video_streams("non_existent_file_xyz.mp4")
        self.assertFalse(result["valid"])
        self.assertIn("không tồn tại", result["error"])

    @patch("subprocess.run")
    def test_inspect_corrupt_file(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "Invalid data found when processing input"
        mock_run.return_value = mock_proc

        with patch("os.path.exists", return_value=True):
            result = inspect_video_streams("dummy_corrupt.mp4")
            self.assertFalse(result["valid"])
            self.assertIn("Tệp video hỏng", result["error"])

    @patch("subprocess.run")
    def test_inspect_valid_mock_stream(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = """{
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "avc1",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "bit_rate": "4000000",
                    "nb_frames": "300",
                    "duration": "10.0"
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "channels": 2,
                    "bit_rate": "128000"
                }
            ],
            "format": {
                "format_name": "mov,mp4",
                "duration": "10.0",
                "size": "5000000",
                "bit_rate": "4128000"
            }
        }"""
        mock_run.return_value = mock_proc

        with patch("os.path.exists", return_value=True):
            res = inspect_video_streams("dummy_valid.mp4")
            self.assertTrue(res["valid"])
            self.assertIsNone(res["error"])
            self.assertEqual(res["video"]["codec_name"], "h264")
            self.assertEqual(res["video"]["width"], 1920)
            self.assertEqual(res["video"]["height"], 1080)
            self.assertEqual(res["video"]["fps"], 30.0)
            self.assertEqual(res["audio"]["codec_name"], "aac")
            self.assertEqual(res["audio"]["sample_rate"], 48000)

    def test_validate_video_standards_success(self):
        metadata = {
            "valid": True,
            "video": {
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "duration": 120.0,
            },
            "format": {
                "duration": 120.0,
                "size_mb": 50.0,
            },
        }
        std = VideoStandard(
            allowed_codecs=["h264", "hevc"],
            min_width=640,
            min_height=360,
            max_width=1920,
            max_height=1080,
            max_duration=3600,
            max_file_size=500,
        )
        valid, err = validate_video_standards(metadata, std)
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_validate_codec_rejection(self):
        metadata = {
            "valid": True,
            "video": {
                "codec_name": "mpeg4",
                "width": 1280,
                "height": 720,
            },
            "format": {"duration": 10.0, "size_mb": 5.0},
        }
        std = VideoStandard(allowed_codecs=["h264", "hevc"])
        valid, err = validate_video_standards(metadata, std)
        self.assertFalse(valid)
        self.assertIn("Codec 'mpeg4' không được hỗ trợ", err)

    def test_validate_resolution_too_small(self):
        metadata = {
            "valid": True,
            "video": {
                "codec_name": "h264",
                "width": 480,
                "height": 270,
            },
            "format": {"duration": 10.0, "size_mb": 5.0},
        }
        std = VideoStandard(min_width=640, min_height=360, allowed_codecs=["h264"])
        valid, err = validate_video_standards(metadata, std)
        self.assertFalse(valid)
        self.assertIn("thấp hơn mức tối thiểu", err)

    def test_validate_resolution_too_large(self):
        metadata = {
            "valid": True,
            "video": {
                "codec_name": "h264",
                "width": 3840,
                "height": 2160,
            },
            "format": {"duration": 10.0, "size_mb": 5.0},
        }
        std = VideoStandard(max_width=1920, max_height=1080, allowed_codecs=["h264"])
        valid, err = validate_video_standards(metadata, std)
        self.assertFalse(valid)
        self.assertIn("vượt quá mức tối đa", err)

    def test_validate_duration_and_size_limits(self):
        metadata = {
            "valid": True,
            "video": {"codec_name": "h264", "width": 1280, "height": 720},
            "format": {"duration": 4000.0, "size_mb": 10.0},
        }
        std = VideoStandard(max_duration=3600, max_file_size=500, allowed_codecs=["h264"])
        valid, err = validate_video_standards(metadata, std)
        self.assertFalse(valid)
        self.assertIn("Thời lượng video", err)

        # Test size limit
        metadata["format"]["duration"] = 100.0
        metadata["format"]["size_mb"] = 600.0
        valid, err = validate_video_standards(metadata, std)
        self.assertFalse(valid)
        self.assertIn("Dung lượng video", err)

    def test_schema_codecs_normalization(self):
        # Test comma-separated string normalization
        s1 = VideoStandardBase(allowed_codecs="h264, hevc, av1")
        self.assertEqual(s1.allowed_codecs, ["h264", "hevc", "av1"])

        # Test list input
        s2 = VideoStandardBase(allowed_codecs=["H264", " VP9 "])
        self.assertEqual(s2.allowed_codecs, ["h264", "vp9"])

        dto = VideoStandardDTO(standard_id=1, allowed_codecs=["h264", "hevc"])
        data = dto.model_dump(by_alias=True)
        self.assertIn("allowedCodecs", data)
        self.assertEqual(data["allowedCodecs"], ["h264", "hevc"])
        self.assertIn("minWidth", data)
        self.assertIn("maxWidth", data)


if __name__ == "__main__":
    unittest.main()
