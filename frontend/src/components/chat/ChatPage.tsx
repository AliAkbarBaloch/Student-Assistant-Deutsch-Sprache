import React, { useEffect, useRef, useState } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { useLevel } from "../../contexts/LevelContext";
import { useChat } from "../../hooks/useChat";
import { useLiveKitCall } from "../../hooks/useLiveKitCall";
import { Navbar } from "./Navbar";
import { MessageBubble, TypingBubble } from "./MessageBubble";
import { TextBar } from "./TextBar";
import { VoiceControls } from "./VoiceControls";
import { LiveCallOverlay } from "./LiveCallOverlay";
import { ConfirmDialog } from "../ui/ConfirmDialog";
import { Toast, type ToastData } from "../ui/Toast";
import * as api from "../../services/api";
import type { MicState } from "../../types";

interface Props {
  onOpenProfile: () => void;
  onOpenFeedback: () => void;
  onOpenHistory: () => void;
}

export function ChatPage({ onOpenProfile, onOpenFeedback, onOpenHistory }: Props) {
  const { user } = useAuth();
  const { level } = useLevel();
  const chat = useChat();

  const [micState, setMicState] = useState<MicState>("idle");
  const [status,   setStatus]   = useState("idle");
  const [callTime, setCallTime] = useState(0);

  const [interimTranscript, setInterimTranscript] = useState("");
  const recognitionRef = useRef<any>(null);

  // Dialog + toast state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleteLoading,    setDeleteLoading]    = useState(false);
  const [toast,            setToast]            = useState<ToastData | null>(null);

  const callTimerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const callSourceRef = useRef<"sprechen" | "anruf" | null>(null);
  const scrollRef     = useRef<HTMLDivElement>(null);
  const isRecordingRef = useRef(false);

  // Accumulates the transcript of the current Live-Anruf call, saved on hangup
  const callTranscriptRef = useRef<{ role: "user" | "assistant"; content: string }[]>([]);
  const callStartRef      = useRef<string | null>(null);
  // Mirrors callTranscriptRef in state so the live-call popup can render it as it comes in
  const [liveCallTranscript, setLiveCallTranscript] = useState<{ role: "user" | "assistant"; content: string }[]>([]);

  // onTranscript fires for every call — Sprechen shows it in chat, Anruf records it for history
  const lk = useLiveKitCall((text, role) => {
    if (callSourceRef.current === "sprechen") {
      chat.addMessage({ role, content_de: text, content_en: "" });
    } else if (callSourceRef.current === "anruf") {
      callTranscriptRef.current.push({ role, content: text });
      setLiveCallTranscript((prev) => [...prev, { role, content: text }]);
    }
  });

  function showToast(message: string, type: ToastData["type"]) {
    setToast({ id: Date.now(), message, type });
  }

  // ── Reset all call state when LiveKit disconnects (from any source) ────────
  useEffect(() => {
    if (lk.callState === "idle") {
      if (micState === "recording" && callSourceRef.current === "anruf") {
        setMicState("idle");
      }
      if (callTimerRef.current) clearInterval(callTimerRef.current);
      setCallTime(0);
      if (status === "call") setStatus("idle");

      // Persist the Live-Anruf transcript so it shows up in the history screen
      if (callSourceRef.current === "anruf" && callTranscriptRef.current.length > 0) {
        const messages  = callTranscriptRef.current;
        const startedAt = callStartRef.current ?? new Date().toISOString();
        const endedAt   = new Date().toISOString();
        callTranscriptRef.current = [];
        api.saveCall(messages, startedAt, endedAt).catch(() => {
          showToast("Anruf-Transkript konnte nicht gespeichert werden.", "error");
        });
      }
      callSourceRef.current = null;
    }
  }, [lk.callState]);

  // ── Initialize Speech Recognition for "Sprechen" ───────────────────────────
  useEffect(() => {
    // @ts-ignore
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const rec = new SpeechRecognition();
      rec.continuous = true;
      rec.interimResults = true;
      rec.lang = "de-DE";

      rec.onresult = (event: any) => {
        if (!isRecordingRef.current) return;
        let fullTranscript = "";
        for (let i = 0; i < event.results.length; ++i) {
          fullTranscript += event.results[i][0].transcript;
        }
        setInterimTranscript(fullTranscript);
      };
      
      rec.onerror = (e: any) => {
        console.error("Speech rec error:", e);
        setMicState("idle");
      };

      recognitionRef.current = rec;
    }
  }, []);

  // ── Load history on mount ──────────────────────────────────────────────────
  useEffect(() => {
    api.fetchHistory().then((data) => {
      if (data.messages.length) {
        chat.loadHistory(data.messages);
      } else {
        chat.addMessage({
          role: "assistant",
          content_de: `Hallo, ${user?.name}! Ich bin Buddy 🎙 Schreib mir auf Deutsche oder ruf mich an!`,
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
    callTranscriptRef.current = [];
    setLiveCallTranscript([]);
    callStartRef.current = new Date().toISOString();
    setStatus("call");
    callTimerRef.current = setInterval(() => setCallTime((t) => t + 1), 1000);
    const { token, url } = await api.getLiveKitToken(level);
    await lk.connect(url, token);
  }

  // ── Sprechen button — Voice to Text using Web Speech API ─────────
  async function toggleRecording() {
    if (lk.callState !== "idle") {
      lk.disconnect();
      return;
    }

    if (micState === "recording") {
      isRecordingRef.current = false;
      recognitionRef.current?.stop();
      setMicState("processing");
      const text = interimTranscript.trim();
      setInterimTranscript("");
      
      if (text) {
        setMicState("idle");
        handleTextSend(text).catch(console.error);
      } else {
        setMicState("idle");
      }
      return;
    }

    if (!recognitionRef.current) {
      showToast("Dein Browser unterstützt keine Spracherkennung.", "error");
      return;
    }

    callSourceRef.current = "sprechen";
    isRecordingRef.current = true;
    setInterimTranscript("");
    setMicState("recording");
    try {
      recognitionRef.current.start();
    } catch (e) {
      console.error(e);
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

      {/* Live-Anruf popup — mascot with animated listening rings + end call button */}
      <LiveCallOverlay
        open={callState !== "idle" && callSourceRef.current === "anruf"}
        callState={callState}
        timerStr={timerStr}
        transcript={liveCallTranscript}
        onEndCall={toggleCall}
      />

      {/* Chat messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3"
      >
        {chat.messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} userName={user?.name} userAvatar={user?.avatar_url} />
        ))}
        {interimTranscript && (
          <MessageBubble 
            message={{ role: "user", content_de: interimTranscript, content_en: "" }} 
            userName={user?.name} 
            userAvatar={user?.avatar_url} 
          />
        )}
        {chat.isProcessing && <TypingBubble />}
      </div>

      <TextBar
        onSend={handleTextSend}
        disabled={chat.isProcessing}
        micState={micState}
        onMicClick={toggleRecording}
        micDisabled={callState !== "idle" && micState !== "recording"}
      />

      <VoiceControls
        micState={micState}
        callState={callState}
        onCallClick={toggleCall}
        onHistoryClick={onOpenHistory}
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
