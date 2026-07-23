import React from "react";
import { Phone, PhoneOff, History } from "lucide-react";
import type { MicState, CallState } from "../../types";

interface Props {
  micState: MicState;
  callState: CallState;
  onCallClick: () => void;
  onHistoryClick: () => void;
}

export function VoiceControls({ micState, callState, onCallClick, onHistoryClick }: Props) {
  const inCall = callState !== "idle";

  return (
    <div className="flex-shrink-0 flex justify-center items-end gap-10 px-6 pb-7 pt-4 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">

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

      {/* Divider */}
      <div className="w-px h-12 bg-gray-200 dark:bg-gray-800 self-center" />

      {/* Call history */}
      <div className="flex flex-col items-center gap-2">
        <button
          onClick={onHistoryClick}
          disabled={micState !== "idle" || inCall}
          className="w-14 h-14 rounded-full flex items-center justify-center transition-all hover:scale-105 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300"
        >
          <History size={20} />
        </button>
        <span className="text-xs text-gray-500 dark:text-gray-500">History</span>
      </div>

    </div>
  );
}

function callStateLabel(s: CallState): string {
  return { listening: "Wartet…", processing: "Denkt…", speaking: "Spricht…", idle: "Auflegen" }[s];
}
