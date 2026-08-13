"""Run multimodal pipeline inside experiments with optional quality gates."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.pipeline import quality_gates as qg


@dataclass
class GateConfig:
    enable_gates: bool = False
    expected_lang: str = "en"
    min_keyframe_keep_ratio: float = 0.25
    min_chapter_duration_sec: float = 45.0
    min_utterance_merge_dur_sec: float = 1.5


@dataclass
class PipelineResult:
    job_id: str
    enable_gates: bool
    wall_time_sec: float
    chapters: list
    keyframes: list
    utterances: list
    summary: str = ""
    duration: float = 0.0
    model_used: str = ""
    gate_reports: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def apply_stability_patches(*, use_gpu: bool = False) -> None:
    """Apply experiment harness patches.

    use_gpu=False (default): force CPU for reproducible offline/CPU-safe runs.
    use_gpu=True: Florence/CLIP/Paddle on CUDA; Faster-Whisper stays on CPU
    because ctranslate2 4.4 needs cuDNN8 while this env has cuDNN9
    (missing cudnn_ops_infer64_8.dll → hard crash on Windows).
    """
    os.environ["DISABLE_DENOISE"] = "1"
    os.environ["CF_R2_ACCESS_KEY_ID"] = ""
    os.environ["CF_R2_SECRET_ACCESS_KEY"] = ""

    import ai_workers.modules.audio_v2.transcriber as transcriber_mod

    _orig_init = transcriber_mod.AudioTranscriber.__init__

    if use_gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ["FLORENCE_DEVICE"] = "cuda"
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Add torch/nvidia DLL dirs so Florence CUDA can resolve cuDNN9.
        try:
            import torch
            from pathlib import Path as _Path

            for rel in (
                _Path(torch.__file__).resolve().parent / "lib",
                _Path(torch.__file__).resolve().parents[1] / "nvidia" / "cudnn" / "bin",
                _Path(torch.__file__).resolve().parents[1] / "nvidia" / "cublas" / "bin",
            ):
                if rel.is_dir():
                    os.add_dll_directory(str(rel))
                    os.environ["PATH"] = str(rel) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

        def _hybrid_init(self, config=None):
            _orig_init(self, config)
            # Avoid ctranslate2 cuDNN8 crash; ASR on CPU is still fast enough.
            self.device = "cpu"
            self.compute_type = "int8"

        transcriber_mod.AudioTranscriber.__init__ = _hybrid_init  # type: ignore[method-assign]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FLORENCE_DEVICE"] = "cpu"

        import torch

        torch.cuda.is_available = lambda: False  # type: ignore[method-assign]

        def _cpu_init(self, config=None):
            _orig_init(self, config)
            self.device = "cpu"
            self.compute_type = "int8"

        transcriber_mod.AudioTranscriber.__init__ = _cpu_init  # type: ignore[method-assign]

    import ai_workers.modules.common.denoise as denoise_mod
    import librosa
    import soundfile as sf

    def _read_raw_audio_safe(audio_path: str):
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return data.astype("float32")

    denoise_mod.get_denoised_audio_array = _read_raw_audio_safe  # type: ignore[attr-defined]
    denoise_mod.denoiser.enhance_audio = _read_raw_audio_safe  # type: ignore[attr-defined]

    import ai_workers.modules.visual_v2.florence_runtime as florence_runtime_mod

    florence_runtime_mod.validate_florence_environment = lambda: None  # type: ignore[attr-defined]


def patch_skip_rag_and_celery() -> None:
    import ai_workers.tasks as tasks_mod
    from ai_workers.modules.fusion.summarizer import Summarizer

    tasks_mod.process_video.update_state = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    Summarizer.build_rag_index = lambda *args, **kwargs: False  # type: ignore[method-assign]


def _build_keyframes(
    scenes: list,
    utterances: list,
    timeline_result: dict,
) -> list[dict]:
    scene_utterance_lists = {id(scene): [] for scene in scenes}
    aligned_segments = timeline_result.get("aligned_segments", [])

    if aligned_segments:
        for item in aligned_segments:
            scene_id = item.get("scene_id")
            utt_text = item.get("utterance", {}).get("text", "")
            if scene_id in scene_utterance_lists and utt_text:
                scene_utterance_lists[scene_id].append(utt_text)
    else:
        for utt in utterances:
            u_start = utt.get("start", 0.0)
            u_end = utt.get("end", 0.0)
            best_scene = None
            max_overlap = 0.0
            for scene in scenes:
                scene_start = scene.get("start_seconds", 0.0)
                scene_end = scene.get("end_seconds", 0.0)
                overlap = min(u_end, scene_end) - max(u_start, scene_start)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_scene = scene
            if best_scene is not None and max_overlap > 0:
                scene_utterance_lists[id(best_scene)].append(utt.get("text", ""))

    for scene in scenes:
        scene["script"] = " ".join(scene_utterance_lists[id(scene)]).strip()

    keyframes = []
    for scene in scenes:
        keyframes.append(
            {
                "timestamp": scene.get("start_seconds"),
                "imageUrl": scene.get("keyframe_url"),
                "description": scene.get("caption", f"Slide at {scene.get('start_timecode')}"),
                "transcript": scene.get("script", ""),
                "importanceScore": scene.get("importanceScore", 0.8),
            }
        )
    return keyframes


def run_gated_pipeline(
    video_path: str,
    output_root: str,
    enable_gates: bool = False,
    expected_lang: str = "en",
    job_id: str | None = None,
    gate_config: GateConfig | None = None,
    use_gpu: bool = False,
) -> PipelineResult:
    """Run stages 1-6 (no RAG) with optional experimental gates."""
    apply_stability_patches(use_gpu=use_gpu)
    patch_skip_rag_and_celery()

    cfg = gate_config or GateConfig(enable_gates=enable_gates, expected_lang=expected_lang)
    cfg.enable_gates = enable_gates
    cfg.expected_lang = expected_lang

    job_id = job_id or str(uuid.uuid4())
    out_dir = Path(output_root) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_reports: dict[str, Any] = {}
    t0 = time.perf_counter()

    from ai_workers.modules.audio_v2.transcriber import AudioTranscriber
    from ai_workers.modules.audio_v2.speaker import SpeakerDiarizer
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector
    from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer
    from ai_workers.modules.fusion.timeline import TimelineBuilder
    from ai_workers.modules.fusion.summarizer import Summarizer

    # Stage 1: ASR
    transcriber = AudioTranscriber()
    audio_result = transcriber.process(video_path)

    if cfg.enable_gates:
        asr_report = qg.validate_audio_asr(audio_result, expected_lang=cfg.expected_lang)
        audio_result["safe_segments"] = qg.build_asr_safe_segments(
            audio_result.get("segments", []), asr_report
        )
    else:
        asr_report = {"quality_status": "BYPASSED_BACKEND_PARITY", "asr_weight": 1.0}
        audio_result["safe_segments"] = audio_result.get("segments", [])
    gate_reports["asr"] = asr_report

    # Stage 2: Speaker
    speaker_diarizer = SpeakerDiarizer()
    wav_path = video_path.rsplit(".", 1)[0] + ".wav"
    segments_for_diar = audio_result.get("safe_segments", audio_result.get("segments", []))
    utterances = speaker_diarizer.process(wav_path, segments_for_diar)

    duration_sec = 0.0
    if utterances:
        duration_sec = float(utterances[-1].get("end", 0.0))

    if cfg.enable_gates:
        sp_report = qg.validate_speaker_diarization(utterances, duration_sec)
        utterances = qg.stabilize_unreliable_speakers(utterances, sp_report)
    else:
        sp_report = {"reliability": "BYPASSED_BACKEND_PARITY"}
    gate_reports["speaker"] = sp_report

    # Stage 3: Scene
    scene_detector = SceneDetector()
    visual_result = scene_detector.process(video_path, str(out_dir))
    scenes = visual_result.get("scenes", [])

    # Stage 4: Semantic
    semantic_analyzer = SemanticAnalyzer()
    slides = semantic_analyzer.process(scenes)

    # Issue 7: Visual gate
    if cfg.enable_gates:
        slides, visual_stats = qg.smart_visual_quality_gate(
            slides,
            min_blur_var=30.0,
            min_keep_ratio=cfg.min_keyframe_keep_ratio,
        )
        gate_reports["visual"] = visual_stats
    else:
        gate_reports["visual"] = {"bypassed": True, "output": len(slides)}

    # Issue 8: Caption gate
    if cfg.enable_gates:
        slides, caption_stats = qg.verify_and_ground_captions(slides)
        slides = qg.prepare_slides_for_summarizer(slides)
        gate_reports["caption"] = caption_stats
    else:
        gate_reports["caption"] = {"bypassed": True}

    visual_result["scenes"] = slides

    if cfg.enable_gates:
        utterances = qg.post_process_utterances(
            utterances,
            min_dur=cfg.min_utterance_merge_dur_sec,
            lang=cfg.expected_lang,
            visual_dominant=bool(asr_report.get("visual_dominant_fallback")),
        )

    # Stage 5: Timeline
    timeline_builder = TimelineBuilder()
    timeline_result = timeline_builder.process(utterances, slides, slides)
    chapters = timeline_result.get("chapters", [])

    if cfg.enable_gates:
        chapters = qg.post_process_chapters(chapters, min_dur_sec=cfg.min_chapter_duration_sec)

    # Stage 6: Summarizer (no RAG)
    summarizer = Summarizer()
    text_result = summarizer.process(utterances, slides, chapters)
    chapters = text_result.get("chapters", chapters)
    keyframes = _build_keyframes(slides, utterances, timeline_result)

    wall = time.perf_counter() - t0

    result = PipelineResult(
        job_id=job_id,
        enable_gates=enable_gates,
        wall_time_sec=round(wall, 2),
        chapters=chapters,
        keyframes=keyframes,
        utterances=utterances,
        summary=text_result.get("summary", ""),
        duration=duration_sec,
        model_used=text_result.get("model_used", ""),
        gate_reports=gate_reports,
        raw={"timeline": timeline_result, "summary": text_result},
    )

    payload = {
        "job_id": job_id,
        "enable_gates": enable_gates,
        "wall_time_sec": result.wall_time_sec,
        "chapter_count": len(chapters),
        "keyframe_count": len(keyframes),
        "utterance_count": len(utterances),
        "summary": result.summary,
        "duration": result.duration,
        "model_used": result.model_used,
        "gate_reports": gate_reports,
        "chapters": chapters,
        "keyframes": keyframes,
    }
    (out_dir / "pipeline_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_backend_baseline(
    video_path: str,
    output_root: str,
    job_id: str | None = None,
    use_gpu: bool = False,
) -> PipelineResult:
    """Baseline via production process_video (no experimental gates)."""
    apply_stability_patches(use_gpu=use_gpu)
    patch_skip_rag_and_celery()

    from ai_workers.tasks import process_video

    job_id = job_id or str(uuid.uuid4())
    out_dir = Path(output_root) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result_dict = process_video.run(job_id, video_path, "hybrid")
    wall = time.perf_counter() - t0

    pr = PipelineResult(
        job_id=job_id,
        enable_gates=False,
        wall_time_sec=round(wall, 2),
        chapters=result_dict.get("chapters", []),
        keyframes=result_dict.get("keyframes", []),
        utterances=result_dict.get("transcript_segments", []),
        summary=result_dict.get("summary", ""),
        duration=float(result_dict.get("duration") or 0.0),
        model_used=result_dict.get("model_used", ""),
        gate_reports={"mode": "backend_worker"},
        raw=result_dict,
    )

    payload = {
        "job_id": job_id,
        "enable_gates": False,
        "wall_time_sec": pr.wall_time_sec,
        "chapter_count": len(pr.chapters),
        "keyframe_count": len(pr.keyframes),
        "utterance_count": len(pr.utterances),
        "summary": pr.summary,
        "duration": pr.duration,
        "model_used": pr.model_used,
        "gate_reports": pr.gate_reports,
    }
    (out_dir / "pipeline_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pr
