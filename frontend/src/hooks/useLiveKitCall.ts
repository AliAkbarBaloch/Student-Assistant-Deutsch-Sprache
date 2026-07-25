import { useRef, useState, useCallback } from "react";
import {
  Room,
  RoomEvent,
  Track,
  createLocalAudioTrack,
  type LocalAudioTrack,
  type TranscriptionSegment,
} from "livekit-client";
import type { CallState } from "../types";

/**
 * Manages a LiveKit WebRTC room for the Phone (Live-Anruf) and Sprechen buttons.
 *
 * onTranscript fires when a final transcription segment arrives:
 *   - "user"      → what the user said (Deepgram STT result)
 *   - "assistant" → what Buddy said (TTS text from the agent)
 *
 * Only Sprechen uses onTranscript; Live-Anruf passes undefined so nothing is shown in chat.
 */
export function useLiveKitCall(
  onTranscript?: (text: string, role: "user" | "assistant") => void,
) {
  const roomRef     = useRef<Room | null>(null);
  const trackRef    = useRef<LocalAudioTrack | null>(null);
  const audioElsRef = useRef<HTMLAudioElement[]>([]);

  // Keep transcript callback up-to-date without re-creating connect
  const onTranscriptRef = useRef(onTranscript);
  onTranscriptRef.current = onTranscript;

  // Accumulate agent sentences — agent may send several final segments per turn
  const agentBufRef   = useRef("");
  const agentTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // TranscriptionReceived redelivers the participant's recent segment buffer on
  // every event, not just new ones — so an already-emitted final segment id can
  // reappear and must be skipped, or its text gets shown/recorded twice.
  const seenSegmentIdsRef = useRef<Set<string>>(new Set());

  const [callState, setCallState] = useState<CallState>("idle");

  const connect = useCallback(async (url: string, token: string) => {
    setCallState("listening");
    seenSegmentIdsRef.current.clear();

    const room = new Room({
      audioCaptureDefaults: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    roomRef.current = room;

    // ── Play agent audio ─────────────────────────────────────────────────────
    room.on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind !== Track.Kind.Audio) return;
      const el = track.attach() as HTMLAudioElement;
      el.autoplay = true;
      document.body.appendChild(el);
      audioElsRef.current.push(el);
      setCallState("speaking");
      el.addEventListener("ended", () => setCallState("listening"));
    });

    room.on(RoomEvent.TrackUnsubscribed, (track) => {
      track.detach().forEach((el) => el.remove());
    });

    // ── Transcription → chat messages (only used by Sprechen) ────────────────
    room.on(
      RoomEvent.TranscriptionReceived,
      (segments: TranscriptionSegment[], participant) => {
        if (!onTranscriptRef.current) return;

        // Only process final segments not already emitted (skip partial/streaming
        // ones and re-deliveries of segments we've already handled)
        const newFinalSegments = segments.filter(
          (s) => s.final && !seenSegmentIdsRef.current.has(s.id),
        );
        if (newFinalSegments.length === 0) return;
        newFinalSegments.forEach((s) => seenSegmentIdsRef.current.add(s.id));

        const finalText = newFinalSegments
          .map((s) => s.text)
          .join(" ")
          .trim();
        if (!finalText) return;

        const isUser = participant?.isLocal ?? false;

        if (isUser) {
          // User's own speech — show immediately
          onTranscriptRef.current(finalText, "user");
        } else {
          // Agent response — accumulate and debounce (agent may split into chunks)
          agentBufRef.current += (agentBufRef.current ? " " : "") + finalText;
          if (agentTimerRef.current) clearTimeout(agentTimerRef.current);
          agentTimerRef.current = setTimeout(() => {
            if (agentBufRef.current && onTranscriptRef.current) {
              onTranscriptRef.current(agentBufRef.current, "assistant");
              agentBufRef.current = "";
            }
            agentTimerRef.current = null;
          }, 1200); // 1.2 s of silence → flush accumulated agent text
        }
      },
    );

    room.on(RoomEvent.Disconnected, () => setCallState("idle"));

    await room.connect(url, token);

    // Publish microphone so the agent hears us
    const mic = await createLocalAudioTrack({
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    });
    trackRef.current = mic;
    await room.localParticipant.publishTrack(mic);
  }, []);

  const disconnect = useCallback(() => {
    // Flush any pending agent text before disconnecting
    if (agentTimerRef.current) {
      clearTimeout(agentTimerRef.current);
      agentTimerRef.current = null;
    }
    if (agentBufRef.current && onTranscriptRef.current) {
      onTranscriptRef.current(agentBufRef.current, "assistant");
    }
    agentBufRef.current = "";
    seenSegmentIdsRef.current.clear();

    trackRef.current?.stop();
    trackRef.current = null;
    audioElsRef.current.forEach((el) => el.remove());
    audioElsRef.current = [];
    roomRef.current?.disconnect();
    roomRef.current = null;
    setCallState("idle");
  }, []);

  return { callState, connect, disconnect };
}
