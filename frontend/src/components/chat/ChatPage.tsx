import React, { useEffect, useRef, useState, useCallback } from "react";
import { useAuth } from "../../contexts/AuthContext";
import { useLevel } from "../../contexts/LevelContext";
import { useChat } from "../../hooks/useChat";
import { useVAD } from "../../hooks/useVAD";
import { Navbar } from "./Navbar";
import { MessageBubble, TypingBubble } from "./MessageBubble";
import { TextBar } from "./TextBar";
import { VoiceControls } from "./VoiceControls";
import { ConfirmDialog } from "../ui/ConfirmDialog";
import { Toast, type ToastData } from "../ui/Toast";
import * as api from "../../services/api";
import type { MicState, CallState } from "../../types";

const MIN_SPEECH_MS = 400;

interface Props {
  onOpenProfile: () => void;
  onOpenFeedback: () => void;
}

export function ChatPage({ onOpenProfile, onOpenFeedback }: Props) {
  const { user } = useAuth();
  const { level } = useLevel();
  const chat = useChat();
  const vad  = useVAD();

  const [micState,  setMicState]  = useState<MicState>("idle");
  const [callState, setCallState] = useState<CallState>("idle");
  const [status,    setStatus]    = useState("idle");
  const [callTime,  setCallTime]  = useState(0);

  // Dialog + toast state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleteLoading,    setDeleteLoading]    = useState(false);
  const [toast,            setToast]            = useState<ToastData | null>(null);

  // Refs for live call
  const callStreamRef    = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef        = useRef<Blob[]>([]);
  const speechStartRef   = useRef<number>(0);
  const callTimerRef     = useRef<ReturnType<typeof setInterval> | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);

  function showToast(message: string, type: ToastData["type"]) {
    setToast({ id: Date.now(), message, type });
  }

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

    // 1. Show user message right away — no waiting for API
    chat.addMessage({ role: "user", content_de: text, content_en: "" });
    setStatus("processing");

    try {
      // 2. Call API — sendTextToAPI adds only the AI reply
      await chat.sendTextToAPI(text, level);
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

  // ── Tap-to-talk ───────────────────────────────────────────────────────────
  async function toggleRecording() {
    if (chat.isProcessing || callState !== "idle") return;

    if (micState === "recording") {
      mediaRecorderRef.current?.stop();
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      showToast("Mikrofon-Zugriff verweigert.", "error");
      return;
    }

    chunksRef.current = [];
    const rec = new MediaRecorder(stream);
    mediaRecorderRef.current = rec;

    rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setMicState("processing");
      setStatus("processing");
      try {
        const result = await chat.sendVoice(new Blob(chunksRef.current, { type: "audio/webm" }), level);
        await chat.playAudio(result.tts_audio_url);
      } catch {
        showToast("Spracherkennung fehlgeschlagen. Bitte erneut versuchen.", "error");
      } finally {
        setMicState("idle");
        setStatus("idle");
      }
    };

    rec.start();
    setMicState("recording");
  }

  // ── Live call ─────────────────────────────────────────────────────────────
  async function toggleCall() {
    if (callState !== "idle") { endCall(); return; }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      showToast("Mikrofon-Zugriff verweigert.", "error");
      return;
    }

    callStreamRef.current = stream;
    setCallState("listening");
    setStatus("call");
    callTimerRef.current = setInterval(() => setCallTime((t) => t + 1), 1000);

    vad.start({
      stream,
      onSpeechStart: () => {
        speechStartRef.current = Date.now();
        chunksRef.current = [];
        const rec = new MediaRecorder(stream);
        mediaRecorderRef.current = rec;

        rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
        rec.onstop = async () => {
          if (Date.now() - speechStartRef.current < MIN_SPEECH_MS) {
            if (callStreamRef.current) setCallState("listening");
            return;
          }
          setCallState("processing");
          setStatus("processing");

          try {
            const data = await api.sendVoiceMessage(new Blob(chunksRef.current, { type: "audio/webm" }), [], level);
            chat.addMessage({ role: "user",      content_de: data.user_text,  content_en: "" });
            chat.addMessage({ role: "assistant", content_de: data.ai_text_de, content_en: data.ai_text_en });
            setCallState("speaking");
            setStatus("speaking");
            await chat.playAudio(data.tts_audio_url);
          } catch { /* continue call */ }

          if (callStreamRef.current) { setCallState("listening"); setStatus("call"); }
        };

        rec.start();
      },
      onSpeechEnd: () => {
        if (mediaRecorderRef.current?.state === "recording") {
          mediaRecorderRef.current.stop();
        }
      },
    });
  }

  const endCall = useCallback(() => {
    vad.stop();
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
    callStreamRef.current?.getTracks().forEach((t) => t.stop());
    callStreamRef.current = null;
    if (callTimerRef.current) clearInterval(callTimerRef.current);
    setCallTime(0);
    setCallState("idle");
    setStatus("idle");
  }, [vad]);

  const timerStr = `${Math.floor(callTime / 60)}:${String(callTime % 60).padStart(2, "0")}`;

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

      {/* Delete confirmation dialog */}
      <ConfirmDialog
        open={showDeleteDialog}
        title="Verlauf löschen"
        message="Möchtest du deinen gesamten Gesprächsverlauf dauerhaft löschen? Diese Aktion kann nicht rückgängig gemacht werden."
        confirmLabel="Ja, löschen"
        loading={deleteLoading}
        onConfirm={handleDeleteConfirm}
        onCancel={() => !deleteLoading && setShowDeleteDialog(false)}
      />

      {/* Toast notifications */}
      <Toast toast={toast} onClose={() => setToast(null)} />

    </div>
  );
}
