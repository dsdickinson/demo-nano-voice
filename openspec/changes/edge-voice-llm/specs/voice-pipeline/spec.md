## ADDED Requirements

### Requirement: Real-Time Voice Activity Detection and Audio Capture
The system SHALL capture streaming audio from the local microphone and segment speech utterances using Silero Voice Activity Detection (VAD).

#### Scenario: User starts and stops speaking
- **WHEN** audio frames exceed the speech probability threshold followed by a configurable silence duration
- **THEN** the VAD module SHALL bundle the recorded speech segment into an audio tensor and dispatch it to the STT engine

### Requirement: Low-Latency Speech-to-Text Transcription
The system SHALL transcribe captured speech audio into English text using a quantized local model (Faster-Whisper).

#### Scenario: Transcribing spoken utterance
- **WHEN** an audio buffer containing speech is received by the STT module
- **THEN** the module SHALL return the transcribed text string within 300 milliseconds on the Jetson Orin Nano

### Requirement: Dual-Engine LLM Inference Selection
The system SHALL support swapping between the custom from-scratch PyTorch model and a pre-trained compact instruction model via a runtime argument.

#### Scenario: Selecting scratch model backend
- **WHEN** the assistant is started with `--model scratch`
- **THEN** the generation pipeline SHALL load the custom model weights and generate streaming text tokens

#### Scenario: Selecting instruct model backend
- **WHEN** the assistant is started with `--model instruct`
- **THEN** the generation pipeline SHALL load the compact instruction model with female persona formatting and generate streaming text tokens

### Requirement: Streaming Neural Text-to-Speech Output
The system SHALL synthesize generated text into high-quality female speech using a lightweight neural engine (Piper TTS or Kokoro) and stream audio playback to the speaker.

#### Scenario: Sentence-by-sentence audio playback
- **WHEN** the LLM produces a completed sentence boundary punctuation mark (`.`, `!`, `?`)
- **THEN** the TTS engine SHALL immediately synthesize that sentence into audio and begin playback while subsequent tokens are still being generated

#### Scenario: Echo suppression during speech output
- **WHEN** the TTS engine is actively playing audio through the speaker
- **THEN** the microphone VAD detection SHALL be suppressed or ducked to prevent the system from re-transcribing its own speech output
