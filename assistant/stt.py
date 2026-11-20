"""
Speech-to-Text (STT) transcription using Faster-Whisper.
Optimized for low-latency on NVIDIA Jetson Orin Nano with FP16/INT8 compute.
"""

from typing import Optional
import numpy as np


class SpeechToText:
    """
    Transcribes audio waveforms to text using Faster-Whisper.
    """
    def __init__(self, model_size: str = "tiny.en", device: str = "auto", compute_type: str = "int8"):
        self.model_size = model_size
        self.model = None

        try:
            from faster_whisper import WhisperModel
            import torch

            actual_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
            actual_compute = "float16" if actual_device == "cuda" else compute_type
            print(f"[*] Initializing Faster-Whisper ({model_size}) on {actual_device} ({actual_compute})...")
            self.model = WhisperModel(model_size, device=actual_device, compute_type=actual_compute)
        except Exception as e:
            print(f"[-] Faster-Whisper could not be loaded ({e}). Running in fallback mode.")
            self.model = None

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe 16kHz float32 audio array into text.
        """
        if self.model is None or audio is None or len(audio) == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text
