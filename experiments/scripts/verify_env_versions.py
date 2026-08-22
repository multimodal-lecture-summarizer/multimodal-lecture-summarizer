"""Verify installed package versions against target lock."""
from __future__ import annotations

import importlib
import sys

TARGETS = {
    "fastapi": "0.141.1",
    "uvicorn": "0.52.0",
    "celery": "5.6.3",
    "sqlalchemy": "2.0.51",
    "pydantic": "2.13.4",
    "alembic": "1.18.5",
    "redis": "6.4.0",
    "langchain": "1.3.14",
    "langgraph": "1.2.10",
    "openai": "2.51.0",
    "anthropic": "0.120.2",
    "google.generativeai": "0.8.6",
    "chromadb": "1.5.9",
    "transformers": "4.57.6",
    "sentence_transformers": "5.6.1",
    "whisperx": "3.3.1",
    "faster_whisper": "1.1.0",
    "ctranslate2": "4.4.0",
    "pyannote.audio": "3.3.2",
    "speechbrain": "1.1.0",
    "librosa": "0.11.0",
    "df": "0.5.6",  # deepfilternet
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "torchaudio": "2.5.1",
    "cv2": "4.11.0",
    "paddleocr": "2.8.1",
    "paddle": "2.6.2",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "sklearn": "1.7.2",
    "scipy": "1.15.3",
}

IMPORT_MAP = {
    "google.generativeai": "google.generativeai",
    "sentence_transformers": "sentence_transformers",
    "faster_whisper": "faster_whisper",
    "pyannote.audio": "pyannote.audio",
    "df": "df",
    "cv2": "cv2",
    "sklearn": "sklearn",
    "paddle": "paddle",
}


def get_version(mod_name: str):
    mod = importlib.import_module(IMPORT_MAP.get(mod_name, mod_name))
    if mod_name == "cv2":
        return getattr(mod, "__version__", "?")
    if mod_name == "df":
        # deepfilternet package
        try:
            from importlib.metadata import version
            return version("deepfilternet")
        except Exception:
            return getattr(mod, "__version__", "?")
    if mod_name == "google.generativeai":
        from importlib.metadata import version
        return version("google-generativeai")
    if mod_name == "paddle":
        return getattr(mod, "__version__", "?")
    return getattr(mod, "__version__", None) or __import__("importlib.metadata", fromlist=["version"]).version(
        mod_name.replace("_", "-") if mod_name not in ("sklearn",) else "scikit-learn"
    )


def main():
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    ok = 0
    bad = 0
    rows = []
    for name, want in TARGETS.items():
        try:
            got = str(get_version(name))
            # torch may be 2.5.1+cu124
            match = got == want or got.startswith(want)
            status = "OK" if match else "MISMATCH"
            if match:
                ok += 1
            else:
                bad += 1
            rows.append((status, name, want, got))
        except Exception as e:
            bad += 1
            rows.append(("FAIL", name, want, str(e)[:80]))

    for status, name, want, got in rows:
        print(f"[{status:8}] {name:28} want={want:12} got={got}")

    import torch
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"\nSummary: OK={ok} BAD={bad} TOTAL={ok+bad}")


if __name__ == "__main__":
    main()
