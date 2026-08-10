"""Generate a clearer local demo lecture video for pipeline smoke tests."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
VIDEO_NO_AUDIO = DEMO_DIR / "sample_noaudio.mp4"
WAV_PATH = DEMO_DIR / "sample_narration.wav"
OUTPUT = DEMO_DIR / "sample.mp4"


def generate_slides_video(path: Path, duration_sec: int = 40, fps: int = 10) -> None:
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # Strong color/content changes so ContentDetector(threshold=27) can cut scenes.
    slides = [
        (10, "Slide 1: Intro to Multimodal Summarization", (40, 40, 200), [
            "Topic: Lecture Pipeline",
            "Audio ASR + Visual Slides",
            "Goal: Reliable Summary",
        ]),
        (20, "Slide 2: Scene Detection with PySceneDetect", (40, 180, 40), [
            "Detect scene cuts",
            "Extract keyframes",
            "Keep important slides",
        ]),
        (30, "Slide 3: Keyframe Selection with CLIP", (200, 40, 40), [
            "CLIP embeddings",
            "Semantic filtering",
            "Deduplicate similar frames",
        ]),
        (duration_sec, "Slide 4: Captioning OCR and LLM Summary", (20, 180, 200), [
            "Florence caption",
            "PaddleOCR evidence",
            "Grounded LLM summary",
        ]),
    ]

    for frame_idx in range(duration_sec * fps):
        t = frame_idx / fps
        title = "Slide"
        color = (50, 50, 50)
        bullets: list[str] = []
        for end, text, bgr, lines in slides:
            if t < end:
                title = text
                color = bgr
                bullets = lines
                break

        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = color
        # White content card to create OCR-friendly contrast
        cv2.rectangle(img, (60, 90), (900, 460), (245, 245, 245), -1)
        cv2.putText(img, title, (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        y = 210
        for line in bullets:
            cv2.putText(img, f"- {line}", (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
            y += 50
        cv2.putText(img, f"time={t:.1f}s", (80, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2, cv2.LINE_AA)
        writer.write(img)

    writer.release()


def generate_narration_wav(path: Path, duration_sec: int = 40) -> None:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.save_to_file(
            "Welcome to multimodal lecture summarization. "
            "This is slide one about the lecture pipeline and audio visual fusion. "
            "Next, scene detection with PySceneDetect extracts keyframes. "
            "Then CLIP selects important keyframes and filters duplicates. "
            "Finally Florence captions and OCR evidence ground the LLM summary.",
            str(path),
        )
        engine.runAndWait()
    except Exception:
        # Fallback: padded tone so mux keeps full video duration
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_sec}",
                "-ar", "16000", "-ac", "1",
                str(path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    # Pad audio to video length instead of truncating with -shortest
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-af", "apad",
            "-shortest",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    generate_slides_video(VIDEO_NO_AUDIO, duration_sec=40, fps=10)
    generate_narration_wav(WAV_PATH, duration_sec=40)
    mux_video_audio(VIDEO_NO_AUDIO, WAV_PATH, OUTPUT)

    cap = cv2.VideoCapture(str(OUTPUT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    print(f"Created demo video: {OUTPUT} ({OUTPUT.stat().st_size} bytes), duration≈{frames/fps:.1f}s")
