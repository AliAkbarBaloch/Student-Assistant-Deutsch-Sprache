# AI Usage Documentation — Deutsch Buddy
### Student Assistant for German Language Learning
**Project:** Deutsch Buddy | **Course:** Applied AI Lab | **Team:** Ali Akbar

---

## Entry #1 — Setting Up the Full STT → LLM → TTS Voice Pipeline

**Date:** 2026-04-15

**Team member(s):** Ali Akbar

**AI Tool used:** Claude Code

### Context

We wanted the app to let users speak German and get a spoken response back. The basic idea was simple — record audio, transcribe it, send to LLM, speak the response. But actually wiring all three stages together in FastAPI while handling browser audio formats (WebM, Opus) was a nightmare. The browser sends audio in formats Whisper doesn't understand out of the box.

### Prompt / Task

> "I have a FastAPI backend. The browser records audio using MediaRecorder which produces WebM/Opus blobs. I need to:
> 1. Accept the audio upload
> 2. Decode it to a numpy float32 array at 16kHz (what Whisper expects)
> 3. Run faster-whisper transcription
> 4. Send the transcript to an OpenAI-compatible LLM API
> 5. Convert the LLM response to speech using edge-tts and return the MP3 URL
>
> The tricky part: soundfile cannot open WebM. I don't want to install ffmpeg as a system dependency. How do I decode WebM/Opus in pure Python? Show me the full audio decoding pipeline with fallback stages."

### AI Output Summary

Claude suggested a two-stage decode: first try `soundfile` (handles WAV/FLAC), then fall back to `av` (PyAV) for WebM/Opus. The PyAV approach decodes the container frame-by-frame, resamples to 16kHz mono, and converts to float32 — all without needing system ffmpeg. Also generated the full `_read_audio()` function in `app.py` and the edge-tts async wrapper in `tts_service.py`.

### Decision

- [x] Modified before use

### Reasoning

The two-stage fallback idea was exactly what we needed. We adjusted the resampling logic slightly because the original assumed stereo input and our browser was sending mono. Also added error logging so we'd know which stage actually ran. Tested with actual browser recordings before merging.

### Impact

This was the single hardest integration problem in the project. Would have taken days without the AI suggestion about PyAV. The pipeline now reliably handles WebM from Chrome/Firefox and WAV from testing tools. Saved at least a full day of digging through audio library docs.

---

## Entry #2 — CEFR Vocabulary Grounding for the LLM

**Date:** 2026-04-22

**Team member(s):** Ali Akbar

**AI Tool used:** Claude Code

### Context

The LLM kept using vocabulary that was way too advanced for A1/A2 learners. We had a CSV with ~3000 German word stems mapped to CEFR levels. The challenge was injecting the right subset into the LLM system prompt without making the prompt so large it hurt response quality or hit token limits.

### Prompt / Task

> "I have a CSV with columns: stem, level (A1/A2/B1/B2), frequency_rank. I want to:
> 1. Load it once at startup (not on every request)
> 2. For each request, sample ~80 stems from the user's CEFR level grouped by frequency — higher frequency words first
> 3. Format them as a compact string for injection into the LLM system prompt
> 4. For B2, skip the restriction entirely (advanced learners shouldn't be limited)
> 5. The whole thing should be a stateless module with no side effects between requests
>
> What's a clean Python architecture for this? The CSV is ~200KB so loading it repeatedly is not acceptable."

### AI Output Summary

Claude designed `vocab_service.py` with a module-level `_df` variable loaded at import time via `pandas`. The `get_sample_for_prompt(level)` function returns a comma-separated string of stems, sorted by frequency rank, with B2 returning empty string (no restriction). Also suggested using `@functools.lru_cache` on the level-specific filtering to avoid re-filtering the DataFrame on every request.

### Decision

- [x] Modified before use

### Reasoning

We dropped the `lru_cache` suggestion because CEFR levels are only 4 values — just precomputing all 4 at load time was simpler and more predictable than cache invalidation. The rest was used as-is. Checked the CSV column names matched before trusting the pandas code.

### Impact

Response quality for A1/A2 learners improved noticeably. Before this, "Buddy" would casually use words like "Begeisterung" and "unvermeidlich" in A1 mode. After, it stayed within simple vocabulary. The module also turned out useful for the pronunciation feedback endpoint.

---

## Entry #3 — LLM Streaming + Concurrent TTS (The Hard One)

**Date:** 2026-05-08

**Team member(s):** Ali Akbar

**AI Tool used:** Claude Code

### Context

Text chat had a bad UX: user sends message → nothing visible for 3-5 seconds → full response appears + audio plays. We wanted ChatGPT-style streaming where text appears word by word, AND the audio delay should be minimal after the text finishes. The hard part: edge-tts is async but the LLM stream is also async — you need both running concurrently without blocking each other or playing audio out of order.

### Prompt / Task

> "I have a FastAPI SSE endpoint that streams LLM tokens using AsyncOpenAI. After all tokens arrive, I generate a single TTS file and send its URL. This creates a 1.5s audio delay after text finishes.
>
> I want to start generating TTS in the background AS the LLM streams, splitting on sentence boundaries (`.!?` with min 20 chars), so that by the time the last token arrives, most TTS chunks are already done.
>
> Requirements:
> - Use `asyncio.create_task()` for concurrent TTS generation
> - Send `{"type": "audio", "tts_url": "..."}` events AFTER the stream ends, IN ORDER (chunk 1 audio before chunk 2 audio)
> - The frontend uses `audioChain = audioChain.then(() => playAudio(url))` so order matters
> - Don't send audio events during streaming — only after `{"type": "done"}` signal from LLM
>
> Show me the complete FastAPI SSE generator function and the matching TypeScript frontend code to consume it."

### AI Output Summary

Claude wrote the full `event_stream()` async generator in `app.py`. The key insight: `asyncio.create_task()` schedules TTS generation concurrently but `await task` is deferred until after the LLM stream closes. This means audio generation overlaps with LLM streaming time. Also wrote `sendTextMessageStream()` in `api.ts` with `onToken` and `onAudio` callbacks, and the `sendTextStream()` hook method in `useChat.ts` with the audio chain pattern.

### Decision

- [x] Modified before use

### Reasoning

Had to fix a TypeScript bug: `catch {} throw;` (bare rethrow) is not valid — changed to `catch (err) { throw err; }`. Also had to add `sendTextStream` to the hook's return object (AI forgot to export it). The core async pattern was correct and worked first try after those fixes.

### Impact

Audio now starts playing within ~0.3s after text finishes (down from 1.5s). The text itself appears word-by-word in real time. This was the biggest UX improvement of the whole project. The concurrent async pattern is non-trivial — without AI help this would have taken significant research time.

---

## Entry #4 — LiveKit WebRTC Voice Agent with CEFR Context Passing

**Date:** 2026-05-14

**Team member(s):** Ali Akbar

**AI Tool used:** Claude Code

### Context

We wanted a real-time voice call feature — not push-to-talk, but actual live conversation where the AI listens continuously and responds like a phone call. LiveKit agents handle the WebRTC side, but connecting it to our custom LLM (university API, not OpenAI) and passing the user's CEFR level from the frontend to the agent was unclear from the docs.

### Prompt / Task

> "I'm building a LiveKit voice agent. The agent uses:
> - Deepgram Nova-3 for STT (German language)
> - An OpenAI-compatible LLM at a custom base_url (not OpenAI's servers)
> - Cartesia Sonic-3 for TTS
> - Silero VAD for voice activity detection
>
> The frontend generates a token via `POST /api/livekit-token` and passes the user's CEFR level (A1/A2/B1/B2) to the agent.
>
> Problems I can't figure out from docs:
> 1. How do I pass arbitrary metadata (the CEFR level) from the token endpoint to the agent process?
> 2. The agent must greet the user automatically on connect — how does `on_enter` work vs `on_user_turn_completed`?
> 3. How do I configure `lk_openai.LLM` to use a custom base_url with a non-standard model name?
> 4. How do I disable 'thinking mode' tokens in the custom LLM (it supports `chat_template_kwargs`)?
>
> Show me the complete agent file and the FastAPI token endpoint."

### AI Output Summary

Claude wrote `services/livekit_agent.py` and the `/api/livekit-token` endpoint in `app.py`. The CEFR level passes via `job.metadata` (JSON string in the dispatch call) — the agent reads it in `entrypoint()`. Used `on_enter()` with `generate_reply()` for the greeting. The `lk_openai.LLM` constructor accepts `base_url` and `api_key` directly. `enable_thinking: False` goes in `extra_body`.

### Decision

- [x] Modified before use

### Reasoning

The metadata approach (JSON in `ctx.job.metadata`) was something we wouldn't have found without AI — it's not obvious in the LiveKit docs. Adjusted the system prompt instructions per CEFR level, added the `_LEVEL_DESCRIPTIONS` dict ourselves. Also added Silero VAD prewarming in `prewarm()` to avoid cold start latency on first call.

### Impact

The live call feature works with proper CEFR-aware responses. The agent correctly greets users and adapts vocabulary to level. Passing metadata through the job context was the key unlock — worth documenting specifically for other teams using LiveKit with custom LLMs.

---

## Entry #5 — Pronunciation Feedback with Phoneme-Level Scoring

**Date:** 2026-05-10

**Team member(s):** Ali Akbar

**AI Tool used:** Claude Code

### Context

We wanted real pronunciation feedback — not just "your pronunciation was okay" but specific phoneme-level issues. The plan was: user records themselves saying a word → Whisper transcribes it → LLM analyses pronunciation → optional BFA (Bournemouth Forced Aligner) for phoneme timestamps. Getting all three layers to return a single coherent JSON response without the LLM hallucinating phoneme data was the problem.

### Prompt / Task

> "I'm building a pronunciation feedback system. The pipeline is:
> 1. User records audio
> 2. Whisper (faster-whisper, German, large-v3-turbo) transcribes it
> 3. LLM receives the transcription + the intended target text and returns structured feedback
>
> The LLM must return ONLY this JSON (no markdown, no extra text):
> `{"transcribed": "...", "score": 1-10, "overall": "...", "issues": ["..."], "tips": ["..."], "feedback_en": "..."}`
>
> Problems:
> 1. The LLM sometimes wraps JSON in markdown code fences — how to handle this robustly?
> 2. The `issues` array should contain specific phoneme-level observations in German (e.g. 'Das ch in ich klingt wie sh') — how do I prompt for this without the LLM making things up when it can't actually hear the audio?
> 3. If the LLM returns malformed JSON, what's a safe fallback that doesn't crash?
>
> Write the full LLM prompt and the Python parsing function."

### AI Output Summary

Claude wrote `_FEEDBACK_SYSTEM_PROMPT` with explicit field rules and the two-stage parser in `_parse_feedback_response()`: first try `json.loads()` on the full text, then strip markdown fences with regex and retry, then fall back to regex search for `{...}`. For the hallucination problem, it suggested framing the prompt as "analyse based on what Whisper detected" rather than "describe what you heard" — this keeps the LLM grounded in the transcription, not inventing phoneme data.

### Decision

- [x] Modified before use

### Reasoning

The two-stage fallback parser was used exactly as written. Adjusted the system prompt to make `issues` and `tips` lists in German but `feedback_en` in English (easier for non-German-speaking graders to review). Also added the `target_text` parameter to the endpoint so users can optionally specify what they intended to say — the LLM then diffs what was said vs intended.

### Impact

The feedback screen became one of the most useful features. Users get a score, specific German-language observations about their pronunciation issues, and English-language tips. The anti-hallucination framing in the prompt was a genuinely non-obvious insight — without it, the LLM fabricated specific phoneme errors that didn't match the actual audio.

---

## Entry #6 — Cleaning Git History to Remove a Leaked Secret

**Date:** 2026-05-20

**Team member(s):** Ali Akbar

**AI Tool used:** Claude Code

### Context

When trying to push to GitHub, push protection rejected the push because a GitHub Personal Access Token was present in `.claude/settings.local.json` inside commit `e44a10a`. The secret was buried in history — not in the current working tree. Normal commits couldn't fix it; the whole history needed rewriting.

### Prompt / Task

> "GitHub push protection blocked my push with: 'A secret has been detected in your repository history in commit e44a10a, file .claude/settings.local.json'
>
> I cannot use `git filter-repo` (not installed). The commit is 2nd in a 7-commit chain.
> I need to:
> 1. Create a clean history with no secrets in ANY commit
> 2. Keep all my code changes intact
> 3. Not break the working tree state
> 4. Push to a repo where someone else is the owner (I have push access but not admin)
>
> The secret is already rotated/revoked. What's the safest strategy that avoids touching every commit individually?"

### AI Output Summary

Claude suggested creating an orphan branch: `git checkout --orphan clean-main`, staging all current files in one single commit (clean snapshot, no history at all), then force-pushing to the target remote. This sidesteps filter-repo entirely and guarantees no secrets exist in any prior commit because there are no prior commits.

### Decision

- [x] Accepted as-is

### Reasoning

The orphan approach was simpler and more reliable than rebasing or cherry-picking. Since we wanted to push to a fresh GitHub repo anyway, losing the 7-commit history was acceptable. The `--allow-unrelated-histories` flag was needed later when the target repo already had commits.

### Impact

Push went through after the orphan branch strategy. Added `.claude/` to `.gitignore` immediately after. Useful lesson: local AI tool config directories (`.claude/`, `.cursor/`) should always be in `.gitignore` from day one — they frequently contain tokens.

---

## Entry #7 — Restructuring the Project into Three Separate Folders

**Date:** 2026-06-21

**Team member(s):** Muhammad Mustafa Khalid Malik

**AI Tool used:** Claude (Cursor)

### Context

The project had everything mixed together at the root level — `app.py`, `database.py`, `services/`, and the LiveKit agent all lived alongside the `frontend/` folder. For the university submission, the professor and peers needed to clearly see which code belonged to the backend, which to the frontend, and which to the voice agent. A flat structure made this impossible to read at a glance.

### Prompt / Task

> "I want to revamp and restructure my project. There are three different things running simultaneously: frontend, backend, and the LiveKit agent. I want to make sure there are 3 different folders for these 3 jobs so that it will be easy for the professor to see the code. Is it doable?"

### AI Output Summary

Claude analyzed the existing project structure, identified that `livekit_agent.py` had no local imports (fully standalone), and that `database.py` used `Path(__file__).resolve().parent` making it portable after a move. It planned a three-folder split: `backend/` (FastAPI, DB, services), `frontend/` (already separate), and `livekit_agent/` (extracted from `services/`). Confirmed that `load_dotenv()` traverses up parent directories automatically, so a single `.env` at the project root would be found by both `backend/app.py` and `livekit_agent/livekit_agent.py` without any code changes. Generated shell commands to move files and created a dedicated `livekit_agent/requirements.txt` with only the LiveKit-specific dependencies.

### Decision

- [x] Modified before use

### Reasoning

The overall plan was used as-is. We chose `livekit_agent/` as the folder name (not `agent/`) to be more explicit. We also kept `.env` at the project root so both the backend and agent share credentials without duplication. After the move, we verified that no import paths in `app.py` or `livekit_agent.py` needed updating because the relative imports still resolved correctly when running each script from its own folder.

### Impact

The repository now has a clean three-folder structure that any reviewer can navigate immediately. Each folder has its own `requirements.txt` and can be explained in under one sentence. The separation also made it clearer which Python version constraint (3.10+) applied specifically to the LiveKit agent and not the general backend.

---

## Entry #8 — Fixing the Pronunciation Feedback Pipeline (Model + Progress Bar + Recording UI)

**Date:** 2026-06-21

**Team member(s):** Muhammad Mustafa Khalid Malik

**AI Tool used:** Claude (Cursor)

### Context

The pronunciation feedback feature had three separate issues discovered during testing: (1) uploading a 27-second audio file caused the UI to hang indefinitely because the `large-v3-turbo` Whisper model was being downloaded (~1.5 GB) and then running slowly on CPU, (2) a progress bar was added to communicate wait time but it was stuck at 0% due to a stale closure bug in the animation logic, and (3) the microphone recording button was navigating to a blank white page instead of starting a recording.

### Prompt / Task

> "After uploading an audio of 27 seconds, it is still loading and it is not processing the audio at all. Can you please check why it is not working?"
>
> "The progress bar is not correct — it is showing increasing percentage up to 100% but percentage is just 0."
>
> "The Aufnehmen button is not working properly. It takes me to a new page instead of recording my voice."

### AI Output Summary

For the model issue, Claude identified that `large-v3-turbo` was too slow for CPU inference and switched to the `base` model (~145 MB, 5–10 seconds for 27-second audio). For the progress bar, it identified that a recursive `setTimeout` approach had a stale closure — the inner function always read the initial value of `current`. The fix used `window.setInterval` with React's functional state updater `setProgress(prev => prev + 1)` which always receives the latest state from React's scheduler. For the navigation bug, it identified that calling `async/await` inside a click handler can lose Chrome's "trusted event" context, causing unexpected navigation. The fix refactored `toggleRecording()` from `async/await` into a synchronous function that uses `.then()/.catch()` Promise chaining, ensuring the click event completes before any async work begins.

### Decision

- [x] Modified before use

### Reasoning

The model switch to `base` was accepted as-is — sufficient accuracy for pronunciation feedback without the multi-minute wait. The progress bar fix using the functional updater was exactly the right React pattern. For the recording button, we went further than the suggested fix by completely redesigning the recording UI into a WhatsApp-style interface (see Entry #9) rather than just patching the existing button, since the UX was confusing regardless of the navigation bug.

### Impact

Pronunciation feedback now processes a 27-second recording in approximately 8 seconds on CPU. The progress bar correctly animates through four labeled stages (upload → Whisper transcription → AI analysis → compiling feedback). The recording flow no longer causes page navigation and is significantly more intuitive for users.

---

## Entry #9 — WhatsApp-Style Recording UI for Pronunciation Feedback

**Date:** 2026-06-21

**Team member(s):** Muhammad Mustafa Khalid Malik

**AI Tool used:** Claude (Cursor)

### Context

Even after fixing the navigation bug on the record button, the original UI was confusing — one button toggled between "start" and "stop" states with no visual indication that recording was actually happening. Users had no sense of how long they had been recording or whether the mic was active. A clearer, more intuitive recording interface was needed.

### Prompt / Task

> "Let's create a new UI. Once the user presses this button, a UI will appear just like a WhatsApp audio message which will show a timer of the recording. Once the user presses that button again, the audio timer will stop and the recording will stop, and then that recording will go to the model for analysis."

### AI Output Summary

Claude redesigned the recording state into a full-screen overlay within the feedback page. The idle state shows the upload zone and a clean "Aufnehmen" button. On click, the component transitions to a recording-specific view with: a large pulsing red mic icon with two ripple ring animations (`animate-ping` outer ring, `animate-pulse` middle ring), a large monospace `mm:ss` countdown timer driven by a `setInterval` ref, and a red "Aufnahme beenden" stop button. On stop, the recording blob is sent for analysis and the progress bar state takes over. The mic timer uses a separate `recTimerRef` to avoid interfering with the analysis progress interval.

### Decision

- [x] Modified before use

### Reasoning

The WhatsApp metaphor was the right model — users universally understand that UI pattern. We kept the animation refs (`progressRef` and `recTimerRef`) separate to avoid the two timers conflicting. Also removed the optional "What were you trying to say?" text input from the UI entirely — it added friction and the backend works fine without it. The `startRecording()` function was kept as a plain synchronous function (not `async`) to avoid the Chrome trusted-click issue identified in Entry #8.

### Impact

The recording experience is now clear and satisfying — users see the mic pulsing, the timer counting up, and a clear stop button. No more accidental page navigation. The visual feedback also reassures users that their mic is active, which reduces the common confusion of "did it start recording?" The WhatsApp-style pattern was immediately recognizable in user testing.

---

## Entry #10 — Building the CEFR Evaluation Script and Comparing Gemma, Qwen3.6, and Claude

**Date:** 2026-07-12 to 2026-07-16

**Team member(s):** Muhammad Mustafa Khalid Malik

**AI Tool used:** Claude

### Context

Testing & Evaluation was one of our six project phases, but I didn't want to present it with vague claims like "the model uses appropriate vocabulary." I wanted an actual number: out of 80 real conversations across A1–B2, how often does Deutsch Buddy's reply actually stay within the vocabulary of the level it's supposed to be teaching? I also needed a fair way to compare our production model against alternatives before deciding what to actually ship with, since I was considering switching from Gemma to a Qwen model.

### Prompt / Task

> "I want to generate 20 responses for 20 different questions/conversations per CEFR level, so 80 total. Then, in a simple way, check how many words from dictionary_a1a2b1_onlystems.csv match — skip stemming, skip simple German words like und, aber, etc., and only look at main vocabulary. We'll analyse if the vocabulary the LLM outputs matches the correct CEFR level."


### AI Output Summary

Claude built `evaluate_cefr_accuracy.py`: it loads the CEFR CSV into a word-to-level lookup, tokenizes each response, strips out a ~200-word German stopword list (articles, pronouns, conjunctions, auxiliary verbs), then does a direct exact-match lookup against the dictionary — no stemming — to compute a compliance percentage per response and per CEFR level.

### Decision

- [x] Modified before use

### Reasoning

I ran it against Gemma and qwen36-35b, compared real results, and switched the project to qwen36-35b after seeing a large latency improvement.

### Impact

This produced the actual Phase 6 results slide in our final presentation, and real compliance percentages.

---

*Template for future entries below this line*

---