import React, { useState } from "react";
import type { ChatMessage } from "../../types";

interface Props {
  message: ChatMessage;
  userName?: string;
  /** URL of the user's uploaded profile picture — shows instead of initials */
  userAvatar?: string | null;
}

export function MessageBubble({ message, userName, userAvatar }: Props) {
  const [showEn, setShowEn] = useState(false);
  const isUser = message.role === "user";

  const initials = userName
    ? userName.split(" ").map((w) => w[0]).join("").toUpperCase().slice(0, 2)
    : "Du";

  return (
    <div
      className={`flex gap-2.5 max-w-[80%] bubble-enter ${
        isUser ? "self-end flex-row-reverse" : "self-start"
      }`}
    >
      {/* Avatar */}
      {isUser ? (
        userAvatar ? (
          <div className="flex-shrink-0 w-9 h-9 rounded-full overflow-hidden border border-blue-600 dark:border-blue-700">
            <img src={userAvatar} alt={userName} className="w-full h-full object-cover" />
          </div>
        ) : (
          <div className="flex-shrink-0 w-9 h-9 rounded-full bg-blue-800 dark:bg-blue-900 border border-blue-600 dark:border-blue-700 flex items-center justify-center text-white text-xs font-bold">
            {initials}
          </div>
        )
      ) : (
        <div className="flex-shrink-0 w-9 h-9 rounded-full overflow-hidden border-2 border-brand-500">
          <img src="/static/mascot.jpg" alt="Buddy" className="w-full h-full object-cover" />
        </div>
      )}

      {/* Bubble */}
      <div
        className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-blue-600 dark:bg-blue-800/60 border border-blue-500 dark:border-blue-700 rounded-tr-sm text-white"
            : "bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-tl-sm text-gray-900 dark:text-gray-100"
        }`}
      >
        {/* Always show the German text — never auto-show English */}
        <p>{message.content_de}</p>

        {/* English toggle — only for AI messages that have a translation */}
        {!isUser && message.content_en && (
          <div className="mt-2">
            <button
              onClick={() => setShowEn((v) => !v)}
              className="text-xs text-gray-500 dark:text-gray-600 border border-gray-300 dark:border-gray-700 rounded px-1.5 py-0.5 hover:text-gray-700 dark:hover:text-gray-400 transition-colors"
            >
              {showEn ? "EN ▴ verbergen" : "EN ▾ übersetzen"}
            </button>
            {showEn && (
              <p className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-800 text-xs text-gray-500 dark:text-gray-500 italic">
                {message.content_en}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** Three-dot typing indicator */
export function TypingBubble() {
  return (
    <div className="flex gap-2.5 self-start bubble-enter">
      <div className="flex-shrink-0 w-9 h-9 rounded-full overflow-hidden border-2 border-brand-500">
        <img src="/static/mascot.jpg" alt="Buddy" className="w-full h-full object-cover" />
      </div>
      <div className="bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-1.5">
        <span className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-600 dot-1 inline-block" />
        <span className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-600 dot-2 inline-block" />
        <span className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-600 dot-3 inline-block" />
      </div>
    </div>
  );
}
