/**
 * LevelContext — stores the user's selected CEFR level globally.
 * Persists the choice in localStorage so it survives page refreshes.
 */
import React, { createContext, useContext, useState, type ReactNode } from "react";

export type CefrLevel = "A1" | "A2" | "B1" | "B2";

interface LevelContextType {
  level: CefrLevel;
  setLevel: (l: CefrLevel) => void;
}

const LevelContext = createContext<LevelContextType | null>(null);

const STORAGE_KEY = "db_cefr_level";

export function LevelProvider({ children }: { children: ReactNode }) {
  const [level, setLevelState] = useState<CefrLevel>(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    return (stored as CefrLevel) ?? "B1";
  });

  function setLevel(l: CefrLevel) {
    setLevelState(l);
    localStorage.setItem(STORAGE_KEY, l);
  }

  return (
    <LevelContext.Provider value={{ level, setLevel }}>
      {children}
    </LevelContext.Provider>
  );
}

export function useLevel() {
  const ctx = useContext(LevelContext);
  if (!ctx) throw new Error("useLevel must be inside LevelProvider");
  return ctx;
}
