# Services package — each module handles exactly one responsibility:
#
#   stt_service          →  faster-whisper (large-v3-turbo, int8) German STT
#                            used for pronunciation feedback only;
#                            voice chat STT is handled by Deepgram Nova-3 via LiveKit
#
#   llm_service          →  'University provided LLM API (qwen3.6-35b)
#                            text chat + pronunciation feedback
#                            supports streaming via build_streaming_messages()
#
#   tts_service          →  Microsoft Edge TTS (de-DE-KatjaNeural)
#                            text chat audio — generates MP3 files in static/tts/
#                            runs concurrently per sentence for lower perceived latency
#
#   livekit_agent        →  Standalone LiveKit voice agent (moved to livekit_agent/ folder)
#                            Deepgram Nova-3 STT → LLM → Cartesia Sonic-3 TTS
#                            powers both Sprechen and Live-Anruf buttons
#
#   auth_service         →  JWT (HS256, 30-day) + bcrypt password hashing
#
#   vocab_service        →  CEFR vocabulary CSV loader; injects word-stem samples
#                            into LLM system prompts to ground responses by level
#
#   transcription_service → Legacy HuggingFace whisper-base (English) — unused by main app
