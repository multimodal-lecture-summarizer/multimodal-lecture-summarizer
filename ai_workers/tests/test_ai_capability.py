"""Phase-3 capability contract. No Whisper or pyannote model download."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_workers.core.checkpoint import (
    CheckpointManager,
    STAGE_QUALITY_COMPLETE,
    STATUS_INTERMEDIATE_READY,
    build_config_fingerprint,
)
from ai_workers.modules.audio_v2.language import normalize_source_language, whisper_language_arg
from ai_workers.modules.audio_v2.speaker import SpeakerDiarizer, assign_speakers


def _checkpoint_data():
    return {
        "utterances": [{"start": 0.0, "end": 4.0, "text": "Xin chào"}],
        "chapters": [{"title": "Mở đầu", "startTime": 0.0, "endTime": 4.0, "summary": ""}],
        "keyframes": [{"timestamp": 0.0, "ocr_text": "Tiêu đề"}],
    }


class LanguageContractTests(unittest.TestCase):
    def test_source_language_reaches_whisper_as_vi_en_or_auto(self):
        self.assertEqual(normalize_source_language("vi"), "vi")
        self.assertEqual(normalize_source_language("EN"), "en")
        self.assertEqual(normalize_source_language("auto"), "auto")
        self.assertEqual(normalize_source_language("fr"), "auto")
        self.assertIsNone(whisper_language_arg("auto"))
        self.assertEqual(whisper_language_arg("vi"), "vi")
        self.assertEqual(whisper_language_arg("en"), "en")

    def test_transcribe_passes_selected_language(self):
        from ai_workers.modules.audio_v2.transcriber import AudioTranscriber

        fake_model = MagicMock()
        fake_model.transcribe.return_value = (
            iter(()),
            SimpleNamespace(language="vi", language_probability=0.99),
        )
        with patch("ai_workers.modules.audio_v2.transcriber.WhisperModel", return_value=fake_model):
            with patch.object(AudioTranscriber, "__init__", lambda self, config=None: None):
                transcriber = AudioTranscriber()
                transcriber.model_size = "small"
                transcriber.device = "cpu"
                transcriber.compute_type = "int8"
                transcriber.cache_dir = MagicMock()
                transcriber._resolve_model_source = lambda: "small"
                transcriber.reduce_noise = lambda path: path
                result = transcriber.transcribe("clip.wav", source_language="vi")

        kwargs = fake_model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["language"], "vi")
        self.assertTrue(kwargs["word_timestamps"])
        self.assertEqual(result["source_language"], "vi")
        self.assertEqual(result["asr_model"], "small")


class DiarizationContractTests(unittest.TestCase):
    def test_overlap_assigns_majority_speaker_and_unknown_below_threshold(self):
        utterances = [
            {"start": 0.0, "end": 10.0, "text": "Phần của người A"},
            {"start": 10.0, "end": 12.0, "text": "Câu ngắn không đủ overlap"},
        ]
        speakers = [
            {"speaker_id": "SPEAKER_00", "start_sec": 0.0, "end_sec": 9.0},
            {"speaker_id": "SPEAKER_01", "start_sec": 11.8, "end_sec": 12.0},
        ]
        merged = assign_speakers(utterances, speakers)
        self.assertEqual(merged[0]["speaker_id"], "SPEAKER_00")
        self.assertEqual(merged[1]["speaker_id"], "UNKNOWN")
        self.assertNotIn("speaker_id", utterances[0])

    def test_missing_token_skips_without_fake_speaker(self):
        diarizer = SpeakerDiarizer({"hf_token": ""})
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
            with patch("ai_workers.modules.audio_v2.speaker.worker_settings") as settings:
                settings.HF_TOKEN = ""
                result = diarizer.process("clip.wav", [{"start": 0.0, "end": 1.0, "text": "Xin chào"}])
        self.assertEqual(diarizer.status, "skipped")
        self.assertNotIn("speaker_id", result[0])


class CheckpointFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mls_ckpt_fp_")
        self.manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        self.source = os.path.join(self.temp_dir, "lecture.mp4")
        with open(self.source, "wb") as handle:
            handle.write(b"lecture-bytes-v1")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save(self, source_hash: str, fingerprint: str):
        return self.manager.save_checkpoint(
            job_id="job-1",
            stage=STAGE_QUALITY_COMPLETE,
            status=STATUS_INTERMEDIATE_READY,
            data=_checkpoint_data(),
            source_sha256=source_hash,
            config_fingerprint=fingerprint,
        )

    def test_checkpoint_rejected_when_model_or_source_changes(self):
        from ai_workers.core.checkpoint import sha256_file

        source_hash = sha256_file(self.source)
        original = build_config_fingerprint(
            asr_model="small",
            source_language="vi",
            florence_device="cpu",
            sprint_stack="recommended",
        )
        changed_model = build_config_fingerprint(
            asr_model="large-v3",
            source_language="vi",
            florence_device="cpu",
            sprint_stack="recommended",
        )
        self.assertTrue(self._save(source_hash, original))
        self.assertIsNotNone(self.manager.load_checkpoint(
            "job-1",
            expected_source_sha256=source_hash,
            expected_config_fingerprint=original,
        ))
        self.assertIsNone(self.manager.load_checkpoint(
            "job-1",
            expected_source_sha256=source_hash,
            expected_config_fingerprint=changed_model,
        ))
        with open(self.source, "wb") as handle:
            handle.write(b"lecture-bytes-v2")
        self.assertIsNone(self.manager.load_checkpoint(
            "job-1",
            expected_source_sha256=sha256_file(self.source),
            expected_config_fingerprint=original,
        ))

    def test_tampered_payload_is_rejected(self):
        fingerprint = build_config_fingerprint(
            asr_model="small",
            source_language="auto",
            florence_device="cpu",
            sprint_stack="recommended",
        )
        self.assertTrue(self._save("abc", fingerprint))
        path = os.path.join(self.temp_dir, "job-1.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["data"]["utterances"][0]["text"] = "đã bị sửa"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        self.assertIsNone(self.manager.load_checkpoint("job-1"))

    def test_expired_checkpoint_is_removed_and_fresh_one_stays(self):
        self.assertTrue(self._save("abc", "fp"))
        stale = os.path.join(self.temp_dir, "old-job.json")
        with open(stale, "w", encoding="utf-8") as handle:
            handle.write("{}")
        old_time = time.time() - (8 * 24 * 60 * 60)
        os.utime(stale, (old_time, old_time))
        removed = self.manager.cleanup_expired(max_age_days=7)
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(stale))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "job-1.json")))
