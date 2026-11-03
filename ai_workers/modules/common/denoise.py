import os
import time
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

try:
    import importlib

    _df_enhance = importlib.import_module("df.enhance")
    _df_enhance.get_device = lambda: torch.device(_denoise_device())
    enhance = _df_enhance.enhance
    init_df = _df_enhance.init_df
    load_audio = _df_enhance.load_audio
    DF_AVAILABLE = True
except Exception as _df_err:
    DF_AVAILABLE = False
    print(f"[Warning] DeepFilterNet (df) import skipped ({_df_err}). Denoising will fall back to raw audio.")


def _denoise_device() -> str:
    """Resolve DENOISE_DEVICE; cuda silently degrades to cpu when no GPU is present."""
    from ai_workers.core.config import worker_settings

    wanted = (worker_settings.DENOISE_DEVICE or "cpu").strip().lower()
    if _gpu_failed:
        return "cpu"
    return "cuda" if wanted == "cuda" and torch.cuda.is_available() else "cpu"


# Set after a CUDA run fails so the rest of the process stays on CPU instead of retrying.
_gpu_failed = False


class DenoiseManager:
    def __init__(self):
        self.model = None
        self.df_state = None
        self.device = "cpu"
        self._cache = None  # ((path, size, mtime_ns), denoised array) of the last file

    def initialize(self):
        if not DF_AVAILABLE:
            return
        if self.model is None:
            device = _denoise_device()
            print(f"[Denoise] Initializing DeepFilterNet model on {device}...")
            self.model, self.df_state, _ = init_df()
            self.device = device
            if hasattr(self.model, "to"):
                self.model = self.model.to(device)

    def _read_raw_audio(self, audio_path: str) -> np.ndarray:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return data.astype(np.float32)

    def _enhance(self, audio_path: str) -> np.ndarray:
        started = time.perf_counter()
        print(f"[Denoise] Processing {os.path.basename(audio_path)} on {self.device}...")

        # Load audio (automatically resampled to DeepFilterNet's required SR, usually 48kHz)
        audio, _ = load_audio(audio_path, sr=self.df_state.sr())
        if hasattr(audio, "to"):
            audio = audio.to("cpu")

        if self.device == "cuda":
            # cuDNN rejects DeepFilterNet's layers on some torch/GPU combos (NOT_SUPPORTED);
            # native CUDA kernels still beat the 2-vCPU Colab CPU.
            with torch.backends.cudnn.flags(enabled=False):
                enhanced = enhance(self.model, self.df_state, audio.contiguous()).cpu()
        else:
            enhanced = enhance(self.model, self.df_state, audio).cpu()

        # Whisper requires 16kHz 1D array. We resample the enhanced audio.
        target_sr = 16000
        if self.df_state.sr() != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=self.df_state.sr(), new_freq=target_sr)
            enhanced = resampler(enhanced)

        # Squeeze to 1D and convert to numpy
        enhanced_np = enhanced.squeeze().numpy()
        print(f"[Denoise] Processing complete on {self.device} in {time.perf_counter() - started:.1f}s.")
        return enhanced_np.astype(np.float32)

    def enhance_audio(self, audio_path: str) -> np.ndarray:
        """
        Denoises the audio using DeepFilterNet (on CPU) and returns a 16kHz 1D numpy array 
        ready for Faster-Whisper or standard Whisper pipeline.
        """
        if not DF_AVAILABLE:
            print(f"[Denoise] Skipping DeepFilterNet enhancement. Loading raw audio: {os.path.basename(audio_path)}...")
            return self._read_raw_audio(audio_path)

        # A failed ASR engine retries with another one on the same file; don't denoise it twice.
        try:
            stat = os.stat(audio_path)
            cache_key = (audio_path, stat.st_size, stat.st_mtime_ns)
        except OSError:
            cache_key = None
        if cache_key is not None and self._cache is not None and self._cache[0] == cache_key:
            print(f"[Denoise] Reusing denoised audio for {os.path.basename(audio_path)}.")
            return self._cache[1]

        try:
            self.initialize()
            try:
                result = self._enhance(audio_path)
            except Exception as gpu_err:
                if self.device != "cuda":
                    raise
                # e.g. cuDNN "not supported" on some torch/GPU combos: redo on CPU rather than
                # silently transcribing un-denoised audio.
                global _gpu_failed
                _gpu_failed = True
                print(f"[Denoise] CUDA run failed ({gpu_err}); retrying on cpu.")
                self.model = self.model.to("cpu")
                self.device = "cpu"
                torch.cuda.empty_cache()
                result = self._enhance(audio_path)
            if cache_key is not None:
                self._cache = (cache_key, result)
            return result
        except Exception as e:
            print(f"[Warning] DeepFilterNet processing failed ({e}). Clearing CUDA cache and falling back to raw audio...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._read_raw_audio(audio_path)

# Global singleton
denoiser = DenoiseManager()

def get_denoised_audio_array(audio_path: str) -> np.ndarray:
    """Convenience function to denoise and return 16kHz numpy array."""
    return denoiser.enhance_audio(audio_path)
