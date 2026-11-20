"""
Real-time audio capture and Voice Activity Detection (VAD) with echo ducking.
Captures 16kHz mono audio streams and segments user utterances.
"""

import queue
import time
from typing import Optional, Callable
import numpy as np


class AudioRecorder:
    """
    Continuous microphone stream recorder with VAD segmentation and echo ducking.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        silence_threshold_ms: int = 700,
        energy_threshold: float = 0.02
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        self.silence_frames_needed = int(silence_threshold_ms / frame_duration_ms)
        self.energy_threshold = energy_threshold

        self.audio_queue: queue.Queue = queue.Queue()
        self.is_muted: bool = False  # Set to True during TTS playback (echo ducking)
        self.stream = None

    def mute(self, muted: bool = True):
        """Suppress input during TTS speech synthesis to avoid acoustic feedback."""
        self.is_muted = muted
        if muted:
            # Drain queue to drop residual speaker audio
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

    def _audio_callback(self, indata, frames, time_info, status):
        """Sounddevice streaming callback."""
        if not self.is_muted:
            self.audio_queue.put(indata.copy())

    def start_stream(self, device_id: Optional[int] = None):
        """Initialize sounddevice InputStream."""
        try:
            import sounddevice as sd
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_size,
                device=device_id,
                callback=self._audio_callback
            )
            self.stream.start()
        except Exception as e:
            print(f"[!] Warning: Audio device stream could not be started ({e}). Running in mock/terminal mode.")
            self.stream = None

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def listen_utterance(self, timeout_sec: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Block until an utterance is spoken (VAD speech detected followed by silence).
        Returns 16kHz float32 audio array, or None if timed out.
        """
        if self.stream is None:
            return None

        speech_frames = []
        in_speech = False
        consecutive_silence = 0
        start_time = time.time()

        while True:
            if timeout_sec and (time.time() - start_time) > timeout_sec:
                return None

            try:
                frame = self.audio_queue.get(timeout=0.1).flatten()
            except queue.Empty:
                continue

            # Compute frame RMS energy as lightweight VAD indicator
            rms = np.sqrt(np.mean(frame ** 2))
            is_speech_frame = (rms >= self.energy_threshold)

            if is_speech_frame:
                in_speech = True
                consecutive_silence = 0
                speech_frames.append(frame)
            elif in_speech:
                speech_frames.append(frame)
                consecutive_silence += 1
                if consecutive_silence >= self.silence_frames_needed:
                    # Utterance finished
                    break

        if not speech_frames:
            return None

        utterance = np.concatenate(speech_frames, axis=0)
        return utterance
