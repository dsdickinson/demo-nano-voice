"""
Core Real-Time Voice Pipeline Coordinator.
Coordinates Audio Capture -> VAD -> STT -> RAG -> LLM Runner -> Streaming TTS.
"""

from typing import Optional
from assistant.audio import AudioRecorder
from assistant.stt import SpeechToText
from assistant.tts import StreamingTTS
from assistant.llm_runner import BaseLLMRunner, create_llm_runner
from rag.ingest import DocumentIngester
from rag.retriever import LocalVectorStore
from rag.augment import format_rag_prompt


class VoiceAssistantPipeline:
    def __init__(
        self,
        llm_type: str = "scratch",
        profile_path: str = "profiles/public",
        device_id: Optional[int] = None,
        voice_model: str = "en_US-lessac-medium"
    ):
        print("=" * 60)
        print("         INITIALIZING NANO-VOICE PIPELINE")
        print("=" * 60)

        # 1. RAG Knowledge Ingestion
        print(f"[*] Ingesting knowledge from '{profile_path}/knowledge'...")
        ingester = DocumentIngester()
        chunks = ingester.ingest_directory(f"{profile_path}/knowledge")
        self.vector_store = LocalVectorStore()
        self.vector_store.add_chunks(chunks)
        print(f"[✓] Indexed {len(chunks)} knowledge chunks.")

        # 2. Audio Recorder with Echo Ducking
        self.audio_recorder = AudioRecorder()
        self.audio_recorder.start_stream(device_id=device_id)

        # 3. Speech-to-Text
        self.stt = SpeechToText()

        # 4. Text-to-Speech with Ducking Hooks
        self.tts = StreamingTTS(
            voice_model=voice_model,
            on_speech_start=lambda: self.audio_recorder.mute(True),
            on_speech_end=lambda: self.audio_recorder.mute(False)
        )

        # 5. Dual-Engine LLM Core
        self.llm = create_llm_runner(model_type=llm_type)
        print("[✓] Pipeline initialized and ready.")

    def process_query_text(self, query: str):
        """Process a text query, retrieve knowledge, generate response, and speak."""
        print(f"\n[👤 User]: {query}")

        # Retrieve relevant knowledge
        retrieved = self.vector_store.search(query, top_k=2)
        augmented_prompt = format_rag_prompt(query, retrieved)

        print("[🧠 Thinking & Generating...]")
        for token_chunk in self.llm.stream_generate(augmented_prompt):
            self.tts.process_token(token_chunk)
        self.tts.flush()

    def run_voice_loop(self):
        """Main real-time audio interaction loop."""
        print("\n" + "=" * 60)
        print("🎤 Nano-Voice is listening... (Speak into your microphone)")
        print("   Press Ctrl+C to stop.")
        print("=" * 60 + "\n")

        try:
            while True:
                audio = self.audio_recorder.listen_utterance(timeout_sec=None)
                if audio is not None and len(audio) > 0:
                    transcription = self.stt.transcribe(audio)
                    if transcription.strip():
                        self.process_query_text(transcription.strip())
        except KeyboardInterrupt:
            print("\n[*] Stopping voice assistant...")
        finally:
            self.audio_recorder.stop_stream()
