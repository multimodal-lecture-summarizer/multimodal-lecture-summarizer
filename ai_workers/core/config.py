"""Worker configuration — shared settings for AI workers."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    """AI Worker settings loaded from environment variables."""

    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    # Celery remote control (inspect/revoke/ping) costs one extra Redis client per worker. Keep it on
    # for any worker the backend inspects (the GPU/Colab worker); the all-in-one container turns it
    # off when it shares a client-capped Redis.
    WORKER_REMOTE_CONTROL: bool = True

    # GPU
    CUDA_VISIBLE_DEVICES: str = "0"
    PROCESS_MIN_AVAILABLE_MEMORY_MB: int = 1024
    PROCESS_SOFT_MIN_AVAILABLE_MEMORY_MB: int = 700
    PROCESS_MEMORY_RETRY_SECONDS: int = 30
    PROCESS_MEMORY_RETRY_INTERVAL_SECONDS: int = 5

    # Run the audio branch (ASR + diarization) and visual branch (scenes + semantic) of one job
    # concurrently. Needs enough VRAM/RAM for both model sets; on OOM the failed branch re-runs
    # sequentially. Off by default; the Colab notebook turns it on.
    ENABLE_PARALLEL_BRANCHES: bool = False

    # Audio module. ASR is Faster-Whisper. WHISPERX_* is unused and kept so old env files still load.
    FASTER_WHISPER_MODEL: str = "small"
    WHISPERX_MODEL: str = "large-v3"
    WHISPERX_BATCH_SIZE: int = 16
    WHISPERX_COMPUTE_TYPE: str = "float16"
    # "qwen3" (default) or "whisper". qwen3 applies only when source_language == "en";
    # vi/auto always use Faster-Whisper (the Qwen aligner has no Vietnamese).
    ASR_ENGLISH_BACKEND: str = "qwen3"
    QWEN_ASR_MODEL: str = "Qwen/Qwen3-ASR-0.6B"
    QWEN_ALIGNER_MODEL: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    QWEN_ASR_MAX_NEW_TOKENS: int = 1024
    # Download model weights in the background at worker start: "auto" (only on Colab), "true", "false".
    PREFETCH_MODELS: str = "auto"
    # Load Qwen3-ASR onto the GPU once (at worker start) and keep it for every job instead of
    # reloading it each time (~2 min on Colab): "auto" (only on Colab), "true", "false".
    QWEN_KEEP_RESIDENT: str = "auto"
    # Same for the visual models (CLIP, Florence-2, PaddleOCR): "auto" (only on Colab), "true", "false".
    VISUAL_KEEP_RESIDENT: str = "auto"
    # Same for the LED summarizer (~1 GB of VRAM in fp16): "auto" (only on Colab), "true", "false".
    LED_KEEP_RESIDENT: str = "auto"
    # Audio chunks transcribed per forward pass. 1 = one chunk at a time (original behavior);
    # raise on GPUs with spare VRAM (Colab T4: 4) since decoding is memory-bound.
    QWEN_ASR_BATCH_SIZE: int = 1
    # DeepFilterNet device: "cpu" (original) or "cuda". Colab has 2 vCPU, so cuda is much faster.
    DENOISE_DEVICE: str = "cpu"

    # Visual module
    SCENE_THRESHOLD: float = 27.0
    KEYFRAME_STRATEGY: str = "sharpest"
    # Candidate frames sampled per scene when picking the sharpest keyframe. Each is a video seek,
    # so fewer candidates extract faster (Colab notebook: 3); 7 keeps the original behavior.
    KEYFRAME_CANDIDATES: int = 7
    FLORENCE_DEVICE: str = "cpu"
    ENABLE_FLORENCE_CAPTIONING: bool = True
    FLORENCE_MAX_BATCH_SIZE: int = 4
    FLORENCE_MIN_AVAILABLE_MEMORY_MB: int = 1024
    FLORENCE_MIN_AVAILABLE_VRAM_MB: int = 1024
    FLORENCE_CPU_NUM_BEAMS: int = 1
    FLORENCE_CUDA_NUM_BEAMS: int = 1
    SEMANTIC_CLIP_DEVICE: str = "auto"
    PADDLEOCR_USE_GPU: bool = False
    # Slide text language; English by default for the English-lecture scope ("vi" for Vietnamese slides).
    PADDLEOCR_LANG: str = "en"

    # Object Detection & Tracking module: RT-DETR-R18vd (detector) + BoT-SORT (tracker), the models
    # the benchmark selected (benchmark/docs/BENCHMARK.md, 03a detection and 03b tracking). On by
    # default; the detector weights are fetched and loaded at worker start-up (resident with the
    # other visual models) so no job pays for them. Set ENABLE_OBJECT_TRACKING=false to skip it.
    ENABLE_OBJECT_TRACKING: bool = True
    OBJECT_DETECTOR_MODEL: str = "PekingU/rtdetr_r18vd"
    # Boxes reported downstream need this confidence (some classes need more, see HIGH_CONF_CLASSES).
    OBJECT_CONF_THRESHOLD: float = 0.40
    # Floor for the boxes handed to the tracker: BoT-SORT's second association pass uses
    # low-confidence boxes (it ignores anything at or below 0.1).
    OBJECT_DETECTOR_MIN_SCORE: float = 0.15
    # Frames per second fed to the detector and tracker. Lectures change slowly, so 1 is enough and
    # costs a third of 3; raise it for fast-moving footage.
    OBJECT_TRACKING_SAMPLE_FPS: int = 1
    # BoT-SORT camera-motion compensation runs on frames shrunk by this factor (benchmark: 2).
    # Larger is faster; it is the main per-frame tracker cost on 1080p video.
    OBJECT_TRACKING_CMC_DOWNSCALE: int = 2
    OBJECT_TRACKING_DEVICE: str = "auto"

    # English summaries: "led" (default) runs the benchmarked LED map-reduce summarizer when
    # output_language == "en" and a CUDA GPU exists; "llm" keeps the Groq/OpenRouter path only.
    SUMMARIZER_ENGLISH_BACKEND: str = "led"
    # Set false to never call OpenRouter/Groq; non-English output then uses the extractive summary.
    ENABLE_LLM_SUMMARIZER: bool = True
    LED_MODEL: str = "allenai/led-large-16384-arxiv"
    LED_MODEL_REVISION: str = "0f8b26971c44af9d4e21edd17ccb5e000f22dac1"

    # Fusion module
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    CHROMADB_PERSIST_DIR: str = "./cache/chroma"

    # Quality post-process (sprint stack after summarizer)
    # recommended = S1+S3+S4v2+S7; sprint10/full = +S5+S8soft+S9+S10
    ENABLE_SPRINT_STACK: bool = True
    SPRINT_STACK: str = "recommended"
    MIN_CHAPTER_SEC: float = 25.0

    # API Keys (injected from .env)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "qwen/qwen-2.5-7b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    HF_TOKEN: str = ""
    ASSEMBLYAI_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""

    # Cloudflare R2 (injected from .env)
    CF_R2_ACCOUNT_ID: str = ""
    CF_R2_ACCESS_KEY_ID: str = ""
    CF_R2_SECRET_ACCESS_KEY: str = ""
    CF_R2_BUCKET_NAME: str = "lecture-summarizer-assets"
    CF_R2_PUBLIC_URL: str = "https://pub-your-bucket-id.r2.dev"

    # Paths
    OUTPUT_DIR: str = "./outputs"
    CACHE_DIR: str = "./cache"

    class Config:
        env_file = ["backend/.env", ".env"]
        case_sensitive = True
        extra = "ignore"


worker_settings = WorkerSettings()


def flag_enabled(value: str | None) -> bool:
    """Interpret a "auto" | "true" | "false" setting; "auto" means on only inside Colab."""
    import os

    text = (value or "auto").strip().lower()
    if text == "auto":
        return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    return text in ("1", "true", "yes", "on")
