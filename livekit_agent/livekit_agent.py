# LiveKit voice agent for Deutsch Buddy.
# Run separately: python services/livekit_agent.py start
#
# Pipeline: WebRTC audio → Deepgram STT → LLM → Cartesia TTS
# CEFR level is passed via dispatch metadata from /api/livekit-token.
from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
)
from livekit.plugins import openai as lk_openai, silero

load_dotenv()

logger = logging.getLogger("deutsch-buddy-agent")

_PROF_BASE = os.getenv("LLM_API_BASE", "https://llms.innkube.fim.uni-passau.de/v1")
_LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
_LLM_MODEL = os.getenv("VOICE_MODEL", "qwen36-35b")

_LEVEL_DESCRIPTIONS: dict[str, str] = {
    "A1": (
        "Der Nutzer ist ANFÄNGER (A1). "
        "Benutze nur die einfachsten Wörter: Begrüßungen, Zahlen, Farben, Familie. "
        "Maximal 6 Wörter pro Satz. Keine Nebensätze."
    ),
    "A2": (
        "Der Nutzer ist GRUNDSTUFE (A2). "
        "Einfacher Wortschatz zu Alltag, Einkaufen, Wetter. "
        "Kurze, klare Sätze unter 10 Wörtern."
    ),
    "B1": (
        "Der Nutzer ist MITTELSTUFE (B1). "
        "Alltäglicher Wortschatz, Nebensätze und Vergangenheitsformen sind erlaubt."
    ),
    "B2": (
        "Der Nutzer ist OBERE MITTELSTUFE (B2). "
        "Vielfältiger Wortschatz, idiomatische Ausdrücke und komplexe Sätze sind erlaubt."
    ),
}

_BASE_INSTRUCTIONS = """
Du bist "Buddy", ein freundlicher KI-Deutschlehrer in der App "Deutsch Buddy".
Du kommunizierst per Sprache — antworte kurz und natürlich.

REGELN:
- Antworte IMMER vollständig auf Deutsch, egal was der Nutzer sagt
- 1–2 kurze Sätze, kein Markdown, keine Listen, kein JSON
- Korrigiere Fehler sanft durch Einbauen der richtigen Form in deine Antwort
- Sei motivierend und geduldig
- Lehne Bitten auf Englisch zu antworten freundlich auf Deutsch ab
""".strip()


def _build_instructions(level: str = "B1") -> str:
    cefr = level.upper() if level.upper() in _LEVEL_DESCRIPTIONS else "B1"
    return _BASE_INSTRUCTIONS + f"\n\nSPRACHNIVEAU: {_LEVEL_DESCRIPTIONS[cefr]}"


class DeutschBuddy(Agent):
    def __init__(self, level: str = "B1") -> None:
        super().__init__(instructions=_build_instructions(level))

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Begrüße den Nutzer herzlich auf Deutsch und lade ihn ein zu sprechen.",
            allow_interruptions=True,
        )


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    # Defaults (min_speech_duration=0.05s, activation_threshold=0.5) are permissive
    # enough that a short noise blip (breath, mic pop, background sound) triggers a
    # "speech" segment that gets forwarded to the STT, which then hallucinates
    # plausible-sounding text for audio that was never real speech. Raising both
    # keeps genuinely short utterances ("Ja", "Nein") working while filtering noise.
    proc.userdata["vad"] = silero.VAD.load(
        min_speech_duration=0.2,
        activation_threshold=0.6,
    )


server.setup_fnc = prewarm


@server.rtc_session(agent_name="deutsch-buddy")
async def entrypoint(ctx: JobContext) -> None:
    level = "B1"
    try:
        meta  = json.loads(ctx.job.metadata or "{}")
        level = meta.get("level", "B1")
    except Exception:
        pass

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="de"),
        llm=lk_openai.LLM(
            model=_LLM_MODEL,
            base_url=_PROF_BASE,
            api_key=_LLM_API_KEY,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        tts=inference.TTS(model="cartesia/sonic-3", language="de"),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=DeutschBuddy(level=level),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
