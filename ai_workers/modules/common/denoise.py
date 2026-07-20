import os
import numpy as np
import torch
import torchaudio
from df.enhance import enhance, init_df, load_audio

class DenoiseManager:
    def __init__(self):
        self.model = None
        self.df_state = None

    def initialize(self):
        if self.model is None:
            print("[Denoise] Initializing DeepFilterNet model...")
            self.model, self.df_state, _ = init_df()

    def enhance_audio(self, audio_path: str) -> np.ndarray:
        """
        Denoises the audio using DeepFilterNet and returns a 16kHz 1D numpy array 
        ready for Faster-Whisper or standard Whisper pipeline.
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            np.ndarray: 1D numpy array of the denoised audio at 16kHz
        """
        self.initialize()
        print(f"[Denoise] Processing {os.path.basename(audio_path)}...")
        
        # Load audio (automatically resampled to DeepFilterNet's required SR, usually 48kHz)
        audio, _ = load_audio(audio_path, sr=self.df_state.sr())
        
        # Apply DeepFilterNet enhancement
        # Note: DeepFilterNet processes chunk by chunk internally, so it handles long files well
        enhanced = enhance(self.model, self.df_state, audio)
        
        # Whisper requires 16kHz 1D array. We resample the enhanced audio.
        target_sr = 16000
        if self.df_state.sr() != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=self.df_state.sr(), new_freq=target_sr)
            enhanced = resampler(enhanced)
            
        # Squeeze to 1D and convert to numpy
        enhanced_np = enhanced.squeeze().cpu().numpy()
        
        print("[Denoise] Processing complete.")
        return enhanced_np

# Global singleton
denoiser = DenoiseManager()

def get_denoised_audio_array(audio_path: str) -> np.ndarray:
    """Convenience function to denoise and return 16kHz numpy array."""
    return denoiser.enhance_audio(audio_path)
