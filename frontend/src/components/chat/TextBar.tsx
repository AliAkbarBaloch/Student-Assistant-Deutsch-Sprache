import React, { useState } from "react";
import { SendHorizonal, Mic, MicOff } from "lucide-react";
import type { MicState } from "../../types";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
  micState: MicState;
  onMicClick: () => void;
  micDisabled?: boolean;
}

export function TextBar({ onSend, disabled, micState, onMicClick, micDisabled }: Props) {
  const [value, setValue] = useState("");

  function handleSend() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  return (
    <div className="flex-shrink-0 flex items-center gap-2 px-3 py-2.5 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
        placeholder="Schreib auf Deutsch…"
        disabled={disabled}
        className="flex-1 bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full px-4 py-2.5 text-sm text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-600 outline-none focus:border-brand-500 transition-colors disabled:opacity-50"
      />

      {/* Sprechen — tap-to-talk, same size as the send button */}
      <div className="relative flex-shrink-0 flex items-center justify-center w-10 h-10">
        {micState === "recording" && (
          <>
            <span className="absolute w-12 h-12 rounded-full border-2 border-red-500 opacity-50 ring-pulse" />
            <span className="absolute w-14 h-14 rounded-full border-2 border-red-500 opacity-30 ring-pulse-2" />
          </>
        )}
        <button
          onClick={onMicClick}
          disabled={micDisabled || micState === "processing"}
          title={micState === "recording" ? "Stop" : "Sprechen"}
          className={`relative z-10 w-10 h-10 rounded-full flex items-center justify-center transition-all hover:scale-105 disabled:opacity-40 disabled:cursor-not-allowed ${
            micState === "recording"
              ? "bg-red-500 text-white"
              : micState === "processing"
              ? "bg-blue-500 text-white animate-spin"
              : "bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300"
          }`}
        >
          {micState === "recording" ? <MicOff size={16} /> : <Mic size={16} />}
        </button>
      </div>

      <button
        onClick={handleSend}
        disabled={!value.trim() || disabled}
        className="flex-shrink-0 w-10 h-10 rounded-full bg-brand-500 hover:bg-brand-600 disabled:opacity-40 disabled:cursor-not-allowed text-black flex items-center justify-center transition-all hover:scale-105"
      >
        <SendHorizonal size={18} />
      </button>
    </div>
  );
}
