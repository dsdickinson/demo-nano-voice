"""
Streaming Text-to-Speech (TTS) engine with natural female neural voices.
Supports sentence-chunked streaming playback and audio ducking hooks.
"""

import re
import queue
import threading
from typing import Optional, Callable
import numpy as np


class StreamingTTS:
    """
    Sentence-chunked streaming speech synthesis.
    Buffers tokens until sentence punctuation is encountered, then synthesizes
    and plays audio immediately while the LLM continues generating subsequent text.
    """
    SENTENCE_END_REGEX = re.compile(r'([.!?]+(?:\s+|$))')

    def __init__(
        self,
        voice_model: str = "en_US-lessac-medium",
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ):
        self.voice_model = voice_model
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.buffer = ""
        self.is_playing = False

    def synthesize_and_play(self, sentence: str):
        """Synthesize and play a complete sentence string."""
        sentence = sentence.strip()
        if not sentence:
            return

        print(f"[🔊 Nova speaking]: \"{sentence}\"")
        if self.on_speech_start:
            self.on_speech_start()

        self.is_playing = True
        try:
            # Check for piper-tts
            import shutil
            import subprocess

            piper_bin = shutil.which("piper")
            if piper_bin:
                # Piper CLI or Python wrapper
                process = subprocess.Popen(
                    [piper_bin, "--model", self.voice_model, "--output-raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                raw_audio, _ = process.communicate(input=sentence.encode("utf-8"))
                if raw_audio:
                    import sounddevice as sd
                    audio_arr = np.frombuffer(raw_audio, dtype=np.int16)
                    sd.play(audio_arr, samplerate=22050)
                    sd.wait()
            else:
                # Fallback to system say on macOS or mock audio delay
                import platform
                if platform.system() == "Darwin":
                    import subprocess
                    # Use Samantha or Victoria for natural female tone on macOS
                    subprocess.run(["say", "-v", "Samantha", sentence], check=False)
                else:
                    # Linux / Jetson fallback: simulate audio playback duration
                    import time
                    duration = max(0.5, len(sentence.split()) * 0.3)
                    time.sleep(duration)
        except Exception as e:
            print(f"[!] Warning during audio playback: {e}")
        finally:
            self.is_playing = False
            if self.on_speech_end:
                self.on_speech_end()

    def process_token(self, token_str: str):
        """
        Feed token string from LLM stream. If a sentence boundary is completed,
        synthesizes speech immediately.
        """
        self.buffer += token_str
        parts = self.SENTENCE_END_REGEX.split(self.buffer)
        
        # When a sentence boundary is found, parts will have [sentence, punctuation, remainder]
        if len(parts) > 1:
            complete_sentence = parts[0] + parts[1]
            self.buffer = "".join(parts[2:])
            self.synthesize_and_play(complete_sentence)

    def flush(self):
        """Synthesize any remaining text in the buffer."""
        if self.buffer.strip():
            remaining = self.buffer.strip()
            self.buffer = ""
            self.synthesize_and_play(remaining)
