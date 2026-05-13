import React from "react";
import { Mic, MicOff, Phone, PhoneOff } from "lucide-react";
import type { MicState, CallState } from "../../types";

interface Props {
  micState: MicState;
  callState: CallState;
  onMicClick: () => void;
  onCallClick: () => void;
}

export function VoiceControls({ micState, callState, onMicClick, onCallClick }: Props) {
  const inCall = callState !== "idle";

  return (
    <div className="flex-shrink-0 flex justify-center items-end gap-10 px-6 pb-7 pt-4 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">

      {/* Tap-to-talk mic */}
      <div className="flex flex-col items-center gap-2">
        <div className="relative flex items-center justify-center w-20 h-20">
          {micState === "recording" && (
            <>
              <span className="absolute w-16 h-16 rounded-full border-2 border-red-500 opacity-50 ring-pulse" />
              <span className="absolute w-20 h-20 rounded-full border-2 border-red-500 opacity-30 ring-pulse-2" />
            </>
          )}
          <button
            onClick={onMicClick}
            disabled={inCall || micState === "processing"}
            className={`relative z-10 w-14 h-14 rounded-full flex items-center justify-center transition-all hover:scale-105 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg ${
              micState === "recording"
                ? "bg-red-500 text-white shadow-red-500/30"
                : micState === "processing"
                ? "bg-blue-500 text-white shadow-blue-500/30 animate-spin"
                : "bg-brand-500 text-black shadow-brand-500/30"
            }`}
          >
            {micState === "recording" ? <MicOff size={22} /> : <Mic size={22} />}
          </button>
        </div>
        <span className="text-xs text-gray-500 dark:text-gray-500">
          {micState === "recording" ? "Stop" : micState === "processing" ? "Verarbeitung…" : "Sprechen"}
        </span>
      </div>

      {/* Divider */}
      <div className="w-px h-12 bg-gray-200 dark:bg-gray-800 self-center" />

      {/* Live call */}
      <div className="flex flex-col items-center gap-2">
        <button
          onClick={onCallClick}
          disabled={micState !== "idle"}
          className={`w-14 h-14 rounded-full flex items-center justify-center transition-all hover:scale-105 disabled:cursor-not-allowed shadow-lg ${
            inCall
              ? "bg-red-500 text-white shadow-red-500/30"
              : "bg-brand-500 text-black shadow-brand-500/30"
          }`}
        >
          {inCall ? <PhoneOff size={22} /> : <Phone size={22} />}
        </button>
        <span className="text-xs text-gray-500 dark:text-gray-500">
          {inCall ? callStateLabel(callState) : "Live-Anruf"}
        </span>
      </div>

    </div>
  );
}

function callStateLabel(s: CallState): string {
  return { listening: "Wartet…", processing: "Denkt…", speaking: "Spricht…", idle: "Auflegen" }[s];
}
