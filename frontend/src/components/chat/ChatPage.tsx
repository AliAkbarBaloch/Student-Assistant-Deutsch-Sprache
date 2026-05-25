import React, { useEffect, useRef, useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { useLevel } from "../../contexts/LevelContext";
import { useChat } from "../../hooks/useChat";
import { useLiveKitCall } from "../../hooks/useLiveKitCall";
import { Navbar } from "./Navbar";
import { MessageBubble, TypingBubble } from "./MessageBubble";
import { TextBar } from "./TextBar";
import { VoiceControls } from "./VoiceControls";
import { ConfirmDialog } from "../ui/ConfirmDialog";
import { Toast, type ToastData } from "../ui/Toast";
import * as api from "../../services/api";
import type { MicState } from "../../types";

interface Props {
  onOpenProfile: () => void;
  onOpenFeedback: () => void;
}

export function ChatPage({ onOpenProfile, onOpenFeedback }: Props) {
  const { user } = useAuth();
  const { level } = useLevel();
  const chat = useChat();

  const [micState, setMicState] = useState<MicState>("idle");
  const [status,   setStatus]   = useState("idle");
  const [callTime, setCallTime] = useState(0);

  // Dialog + toast state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleteLoading,    setDeleteLoading]    = useState(false);
  const [toast,            setToast]            = useState<ToastData | null>(null);

  const callTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  // "sprechen" = mic button started the call, "anruf" = phone button started it
  const callSourceRef = useRef<"sprechen" | "anruf" | null>(null);
  const scrollRef     = useRef<HTMLDivElement>(null);

  // onTranscript fires for every call — only add to chat when Sprechen started it
  const lk = useLiveKitCall((text, role) => {
    if (callSourceRef.current === "sprechen") {
      chat.addMessage({ role, content_de: text, content_en: "" });
    }
  });

  function showToast(message: string, type: ToastData["type"]) {
    setToast({ id: Date.now(), message, type });
  }

  // ── Reset all call state when LiveKit disconnects (from any source) ────────
  useEffect(() => {
    if (lk.callState === "idle") {
      setMicState("idle");
      if (callTimerRef.current) clearInterval(callTimerRef.current);
      setCallTime(0);
      setStatus("idle");
      callSourceRef.current = null;
    }
  }, [lk.callState]);

  // ── Load history on mount ──────────────────────────────────────────────────
  useEffect(() => {
    api.fetchHistory().then((data) => {
      if (data.messages.length) {
        chat.loadHistory(data.messages);
      } else {
        chat.addMessage({
          role: "assistant",
          content_de: `Hallo, ${user?.name}! Ich bin Buddy 🎙 Schreib mir auf Deutsch oder ruf mich an!`,
          content_en: `Hello, ${user?.name}! I'm Buddy 🎙 Write to me in German or call me!`,
        });
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Auto-scroll ────────────────────────────────────────────────────────────
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chat.messages, chat.isProcessing]);

  // ── Text chat — user message appears immediately ───────────────────────────
  async function handleTextSend(text: string) {
    if (chat.isProcessing) return;
    chat.addMessage({ role: "user", content_de: text, content_en: "" });
    setStatus("processing");
    try {
      await chat.sendTextStream(text, level);
    } catch {
      showToast("Fehler beim Senden der Nachricht.", "error");
    } finally {
      setStatus("idle");
    }
  }

  // ── Delete history ─────────────────────────────────────────────────────────
  async function handleDeleteConfirm() {
    setDeleteLoading(true);
    try {
      await api.deleteHistory();
      chat.clearMessages();
      chat.addMessage({
        role: "assistant",
        content_de: `Verlauf gelöscht! Wie kann ich dir helfen, ${user?.name}?`,
        content_en: `History cleared! How can I help you, ${user?.name}?`,
      });
      showToast("Verlauf erfolgreich gelöscht ✓", "success");
    } catch {
      showToast("Fehler beim Löschen des Verlaufs.", "error");
    } finally {
      setDeleteLoading(false);
      setShowDeleteDialog(false);
    }
  }

  // ── Shared: connect to a LiveKit room and start the call timer ────────────
  async function _startCall() {
    setStatus("call");
    callTimerRef.current = setInterval(() => setCallTime((t) => t + 1), 1000);
    const { token, url } = await api.getLiveKitToken(level);
    await lk.connect(url, token);
  }

  // ── Sprechen button — now uses LiveKit (same speed as Live-Anruf) ─────────
  async function toggleRecording() {
    // If a Sprechen-initiated call is active, stop it
    if (lk.callState !== "idle") {
      lk.disconnect(); // useEffect handles cleanup
      return;
    }

    callSourceRef.current = "sprechen";
    setMicState("recording"); // show red mic + pulse animation

    try {
      await _startCall();
    } catch {
      showToast("Verbindungsfehler. Bitte erneut versuchen.", "error");
      lk.disconnect();
    }
  }

  // ── Live-Anruf button — same pipeline ─────────────────────────────────────
  async function toggleCall() {
    // If any call is active (sprechen or anruf), stop it
    if (lk.callState !== "idle") {
      lk.disconnect(); // useEffect handles cleanup
      return;
    }

    callSourceRef.current = "anruf";

    try {
      await _startCall();
    } catch {
      showToast("Live-Anruf fehlgeschlagen. Bitte erneut versuchen.", "error");
      lk.disconnect();
    }
  }

  const callState = lk.callState;
  const timerStr  = `${Math.floor(callTime / 60)}:${String(callTime % 60).padStart(2, "0")}`;

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-black overflow-hidden">

      <Navbar status={status} onClearHistory={() => setShowDeleteDialog(true)} onOpenProfile={onOpenProfile} onOpenFeedback={onOpenFeedback} />

      {/* Call banner */}
      {callState !== "idle" && (
        <div className="flex-shrink-0 flex items-center justify-center gap-3 px-4 py-2 bg-brand-500/10 border-b border-brand-500/20 text-brand-600 dark:text-brand-400 text-sm font-semibold">
          <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse" />
          <span>
            {callState === "listening"  ? "Warte auf Sprache…"
              : callState === "processing" ? "Verarbeitung…"
              : "Buddy spricht…"}
          </span>
          <span className="ml-auto text-gray-400 font-normal">{timerStr}</span>
        </div>
      )}

      {/* Chat messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3"
      >
        {chat.messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} userName={user?.name} userAvatar={user?.avatar_url} />
        ))}
        {chat.isProcessing && <TypingBubble />}
      </div>

      <TextBar onSend={handleTextSend} disabled={chat.isProcessing} />

      <VoiceControls
        micState={micState}
        callState={callState}
        onMicClick={toggleRecording}
        onCallClick={toggleCall}
      />

      <ConfirmDialog
        open={showDeleteDialog}
        title="Verlauf löschen"
        message="Möchtest du deinen gesamten Gesprächsverlauf dauerhaft löschen? Diese Aktion kann nicht rückgängig gemacht werden."
        confirmLabel="Ja, löschen"
        loading={deleteLoading}
        onConfirm={handleDeleteConfirm}
        onCancel={() => !deleteLoading && setShowDeleteDialog(false)}
      />

      <Toast toast={toast} onClose={() => setToast(null)} />

    </div>
  );
}
