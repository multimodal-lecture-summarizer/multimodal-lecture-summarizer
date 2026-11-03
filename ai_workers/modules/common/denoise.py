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
    return "cuda" if wanted == "cuda" and torch.cuda.is_available() else "cpu"


class DenoiseManager:
    def __init__(self):
        self.model = None
        self.df_state = None
        self.device = "cpu"

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

    def enhance_audio(self, audio_path: str) -> np.ndarray:
        """
        Denoises the audio using DeepFilterNet (on CPU) and returns a 16kHz 1D numpy array 
        ready for Faster-Whisper or standard Whisper pipeline.
        """
        if not DF_AVAILABLE:
            print(f"[Denoise] Skipping DeepFilterNet enhancement. Loading raw audio: {os.path.basename(audio_path)}...")
            return self._read_raw_audio(audio_path)

        try:
            self.initialize()
            started = time.perf_counter()
            print(f"[Denoise] Processing {os.path.basename(audio_path)} on {self.device}...")
            
            # Load audio (automatically resampled to DeepFilterNet's required SR, usually 48kHz)
            audio, _ = load_audio(audio_path, sr=self.df_state.sr())
            if hasattr(audio, "to"):
                audio = audio.to("cpu")
            
            enhanced = enhance(self.model, self.df_state, audio)
            
            # Whisper requires 16kHz 1D array. We resample the enhanced audio.
            target_sr = 16000
            if self.df_state.sr() != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=self.df_state.sr(), new_freq=target_sr)
                enhanced = resampler(enhanced)
                
            # Squeeze to 1D and convert to numpy
            enhanced_np = enhanced.squeeze().cpu().numpy()
            
            print(f"[Denoise] Processing complete in {time.perf_counter() - started:.1f}s.")
            return enhanced_np.astype(np.float32)
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
