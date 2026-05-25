import React, { useState } from "react";
import { SendHorizonal } from "lucide-react";

interface Props {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export function TextBar({ onSend, disabled }: Props) {
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
