"""Celery Tasks — kết nối 3 modules (audio, visual, fusion).

Migrated from: src/mls/pipeline.py
Định nghĩa các Celery Tasks orchestrating the processing pipeline.
"""

from __future__ import annotations

from ai_workers.core.celery_app import app
from ai_workers.modules.audio.transcriber import AudioTranscriber
from ai_workers.modules.audio.speaker import SpeakerDiarizer
from ai_workers.modules.visual.scene_detector import SceneDetector
from ai_workers.modules.visual.semantic import SemanticAnalyzer
from ai_workers.modules.fusion.timeline import TimelineBuilder
from ai_workers.modules.fusion.summarizer import Summarizer


@app.task(bind=True, name="ai_workers.process_video")
def process_video(self, job_id: str, video_path: str, config_stack: str = "hybrid"):
    """Main Celery task: orchestrate full multimodal pipeline.

    Pipeline stages:
        1. Audio extraction + ASR (WhisperX)
        2. Speaker diarization (pyannote)
        3. Scene detection + keyframe extraction (PySceneDetect)
        4. Semantic analysis: OCR + CLIP + BLIP-2
        5. Timeline alignment + chapter segmentation
        6. LLM summarization + RAG index

    Stages 1-2 and 3-4 run in parallel where possible.
    """
    output_dir = f"./outputs/{job_id}"

    # Stage 1: Audio
    self.update_state(state="PROGRESS", meta={"stage": "audio", "progress": 10})
    audio = AudioTranscriber()
    audio_result = audio.process(video_path)

    # Stage 2: Speaker diarization
    self.update_state(state="PROGRESS", meta={"stage": "speaker", "progress": 25})
    speaker = SpeakerDiarizer()
    utterances = speaker.process(
        video_path.replace(".mp4", ".wav"),
        audio_result.get("segments", []),
    )

    # Stage 3: Visual
    self.update_state(state="PROGRESS", meta={"stage": "visual", "progress": 40})
    visual = SceneDetector()
    visual_result = visual.process(video_path, output_dir)

    # Stage 4: Semantic
    self.update_state(state="PROGRESS", meta={"stage": "semantic", "progress": 60})
    semantic = SemanticAnalyzer()
    slides = semantic.process(visual_result.get("keyframes", []))

    # Stage 5: Timeline
    self.update_state(state="PROGRESS", meta={"stage": "timeline", "progress": 75})
    timeline = TimelineBuilder()
    timeline_result = timeline.process(
        utterances, visual_result.get("scenes", []), slides,
    )

    # Stage 6: Summarization
    self.update_state(state="PROGRESS", meta={"stage": "text", "progress": 90})
    summarizer = Summarizer()
    text_result = summarizer.process(
        utterances, slides, timeline_result.get("chapters", []),
    )

    # TODO: save results to PostgreSQL via backend_api
    return {
        "job_id": job_id,
        "status": "done",
        "summary": text_result.get("summary", ""),
        "chapters": timeline_result.get("chapters", []),
        "utterance_count": len(utterances),
        "scene_count": len(visual_result.get("scenes", [])),
    }


@app.task(name="ai_workers.process_audio")
def process_audio(job_id: str, video_path: str):
    """Subtask: audio extraction + ASR only."""
    audio = AudioTranscriber()
    return audio.process(video_path)


@app.task(name="ai_workers.process_visual")
def process_visual(job_id: str, video_path: str, output_dir: str):
    """Subtask: scene detection + keyframe extraction only."""
    visual = SceneDetector()
    return visual.process(video_path, output_dir)
