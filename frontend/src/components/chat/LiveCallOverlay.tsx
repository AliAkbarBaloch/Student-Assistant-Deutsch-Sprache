import React, { useEffect, useRef } from "react";
import { PhoneOff } from "lucide-react";
import type { CallState } from "../../types";

interface TranscriptEntry {
  role: "user" | "assistant";
  content: string;
}

interface Props {
  open: boolean;
  callState: CallState;
  timerStr: string;
  transcript: TranscriptEntry[];
  onEndCall: () => void;
}

const STATE_LABEL: Record<CallState, string> = {
  idle:       "",
  listening:  "Hört zu…",
  processing: "Denkt nach…",
  speaking:   "Buddy spricht…",
};

/** Full-screen "in call" popup — mascot with animated listening rings, live transcript + end-call button. */
export function LiveCallOverlay({ open, callState, timerStr, transcript, onEndCall }: Props) {
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: "smooth" });
  }, [transcript]);

  if (!open) return null;

  const speaking = callState === "speaking";
  const processing = callState === "processing";
  const ringColor = speaking ? "border-blue-400" : "border-brand-500";

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center bg-gradient-to-b from-gray-950 via-gray-900 to-black text-white py-8 px-4 gap-3">
      {/* Top — name + call timer */}
      <div className="flex-shrink-0 flex flex-col items-center gap-1 bubble-enter">
        <p className="text-base font-bold">Deutsche Buddy</p>
        <p className="text-xs text-gray-400 tabular-nums">{timerStr}</p>
      </div>

      {/* Mascot with animated listening/speaking rings */}
      <div className="flex-shrink-0 relative flex items-center justify-center w-40 h-40">
        <span className={`absolute w-40 h-40 rounded-full border-2 opacity-30 ring-pulse ${ringColor}`} />
        <span className={`absolute w-32 h-32 rounded-full border-2 opacity-25 ring-pulse-2 ${ringColor}`} />
        <span className={`absolute w-24 h-24 rounded-full border-2 opacity-20 ring-pulse-3 ${ringColor}`} />

        <div
          className={`relative z-10 w-20 h-20 rounded-full overflow-hidden border-4 shadow-2xl transition-transform duration-300 ${
            speaking ? "border-blue-400 shadow-blue-500/30 scale-105" : "border-brand-500 shadow-brand-500/30"
          } ${processing ? "animate-pulse" : ""}`}
        >
          <img src="/static/mascot.jpg" alt="Buddy" className="w-full h-full object-cover" />
        </div>
      </div>

      <p className={`flex-shrink-0 text-xs font-semibold ${speaking ? "text-blue-400" : "text-brand-400"}`}>
        {STATE_LABEL[callState]}
      </p>

      {/* Live transcript — user in yellow, Buddy in green */}
      <div
        ref={transcriptRef}
        className="w-full max-w-md flex-1 min-h-0 overflow-y-auto flex flex-col gap-2 px-1"
      >
        {transcript.length === 0 && (
          <p className="text-center text-xs text-gray-500 mt-4">Live-Transkription erscheint hier…</p>
        )}
        {transcript.map((t, i) => (
          <div
            key={i}
            className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-sm leading-snug bubble-enter ${
              t.role === "user"
                ? "self-end bg-yellow-400/20 border border-yellow-400/40 text-yellow-100 rounded-tr-sm"
                : "self-start bg-brand-500/20 border border-brand-500/40 text-brand-100 rounded-tl-sm"
            }`}
          >
            {t.content}
          </div>
        ))}
      </div>

      {/* End call button */}
      <button
        onClick={onEndCall}
        className="flex-shrink-0 w-16 h-16 rounded-full bg-red-500 hover:bg-red-600 text-white flex items-center justify-center shadow-lg shadow-red-500/30 transition-all hover:scale-105"
        title="Anruf beenden"
      >
        <PhoneOff size={26} />
      </button>
    </div>
  );
}
