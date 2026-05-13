import React from "react";
import { Sun, Moon, Trash2, LogOut, UserCircle, Mic } from "lucide-react";
import { useAuth } from "../../contexts/AuthContext";
import { useTheme } from "../../contexts/ThemeContext";
import { useLevel, type CefrLevel } from "../../contexts/LevelContext";

const LEVELS: CefrLevel[] = ["A1", "A2", "B1", "B2"];

interface Props {
  status: string;
  onClearHistory: () => void;
  onOpenProfile: () => void;
  onOpenFeedback: () => void;
}

export function Navbar({ status, onClearHistory, onOpenProfile, onOpenFeedback }: Props) {
  const { user, logout } = useAuth();
  const { theme, toggle } = useTheme();
  const { level, setLevel } = useLevel();

  return (
    <header className="flex-shrink-0 flex items-center justify-between gap-3 px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">

      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <img
          src="/static/mascot.jpg"
          alt="Buddy"
          className="w-10 h-10 rounded-full object-cover border-2 border-brand-500 shadow-md shadow-brand-500/20"
        />
        <div>
          <p className="text-sm font-extrabold text-gray-900 dark:text-white leading-none">
            Deutsch <span className="text-brand-500">Buddy</span>
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Dein KI-Deutsch-Gesprächspartner</p>
        </div>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-2">

        {/* Status indicator */}
        <div className="hidden sm:flex items-center gap-1.5 text-xs text-gray-400 dark:text-gray-500">
          <span className={`w-2 h-2 rounded-full ${
            status === "processing" ? "bg-blue-400 animate-pulse"
            : status === "speaking"   ? "bg-yellow-400 animate-pulse"
            : status === "call"       ? "bg-brand-500 animate-pulse"
            : "bg-brand-500"
          }`} />
          <span>{statusLabel(status)}</span>
        </div>

        {/* CEFR level switcher */}
        <div className="flex items-center gap-0.5 bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-0.5">
          {LEVELS.map((l) => (
            <button
              key={l}
              onClick={() => setLevel(l)}
              className={`px-2.5 py-1 text-xs font-bold rounded-md transition-colors ${
                level === l
                  ? "bg-brand-500 text-white shadow-sm"
                  : "text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
              }`}
              title={`Sprachniveau ${l}`}
            >
              {l}
            </button>
          ))}
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggle}
          className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          title="Theme wechseln"
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* Pronunciation feedback */}
        <button
          onClick={onOpenFeedback}
          className="hidden sm:flex items-center gap-1.5 px-3 py-2 rounded-lg bg-brand-500/10 border border-brand-500/30 text-brand-500 hover:bg-brand-500/20 transition-colors text-xs font-semibold"
          title="Aussprache-Feedback"
        >
          <Mic size={14} /> Feedback
        </button>
        <button
          onClick={onOpenFeedback}
          className="sm:hidden p-2 rounded-lg bg-brand-500/10 border border-brand-500/30 text-brand-500 hover:bg-brand-500/20 transition-colors"
          title="Aussprache-Feedback"
        >
          <Mic size={16} />
        </button>

        {/* Clear history */}
        <button
          onClick={onClearHistory}
          className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-red-500 hover:border-red-300 dark:hover:border-red-800 transition-colors"
          title="Verlauf löschen"
        >
          <Trash2 size={16} />
        </button>

        {/* Profile + user chip + logout */}
        <div className="hidden sm:flex items-center gap-2 bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full pl-1.5 pr-3 py-1">
          {/* Avatar / initials button → opens profile */}
          <button
            onClick={onOpenProfile}
            className="flex-shrink-0 w-7 h-7 rounded-full overflow-hidden border border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500"
            title="Profil bearbeiten"
          >
            {user?.avatar_url ? (
              <img src={user.avatar_url} alt={user.name} className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-blue-700 text-white text-[10px] font-bold">
                {(user?.name ?? "?").split(" ").map((w) => w[0]).join("").toUpperCase().slice(0, 2)}
              </div>
            )}
          </button>
          <span className="text-xs font-semibold text-gray-900 dark:text-white">{user?.name}</span>
          <button
            onClick={logout}
            className="text-red-400 hover:text-red-500 transition-colors"
            title="Abmelden"
          >
            <LogOut size={14} />
          </button>
        </div>

        {/* Mobile: profile icon */}
        <button
          onClick={onOpenProfile}
          className="sm:hidden p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-brand-500 transition-colors"
          title="Profil"
        >
          <UserCircle size={16} />
        </button>

        {/* Mobile logout */}
        <button
          onClick={logout}
          className="sm:hidden p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-red-400 transition-colors"
        >
          <LogOut size={16} />
        </button>
      </div>
    </header>
  );
}

function statusLabel(s: string): string {
  return { idle: "Bereit", call: "Anruf aktiv", processing: "Denkt…", speaking: "Buddy spricht" }[s] ?? "Bereit";
}
