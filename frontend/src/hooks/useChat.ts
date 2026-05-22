import { useState, useRef, useCallback } from "react";
import type { ChatMessage, ChatResponse } from "../types";
import * as api from "../services/api";

export function useChat() {
  const [messages, setMessages]       = useState<ChatMessage[]>([]);
  const [isProcessing, setProcessing] = useState(false);

  // LLM context window — kept in sync with displayed messages
  const historyRef = useRef<{ role: string; content: string }[]>([]);

  // ── Message management ────────────────────────────────────────────────────

  const addMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
    historyRef.current.push({ role: msg.role, content: msg.content_de });
  }, []);

  function loadHistory(msgs: ChatMessage[]) {
    setMessages(msgs);
    historyRef.current = msgs.map((m) => ({ role: m.role, content: m.content_de }));
  }

  function clearMessages() {
    setMessages([]);
    historyRef.current = [];
  }

  // ── Audio playback ────────────────────────────────────────────────────────

  const playAudio = useCallback((url: string): Promise<void> => {
    return new Promise((resolve) => {
      const audio = new Audio(url);
      audio.onended = () => resolve();
      audio.onerror = () => resolve();
      audio.play().catch(() => resolve());
    });
  }, []);

  // ── Text send (streaming) ─────────────────────────────────────────────────
  /**
   * Streams the LLM reply token-by-token.
   * User message must already be added (and in historyRef) by the caller.
   * Shows typing indicator until the first token, then switches to
   * the live-updating bubble. TTS plays once the full text is ready.
   */
  const sendTextStream = useCallback(async (text: string, level = "B1"): Promise<void> => {
    setProcessing(true);
    let accumulated = "";
    let firstToken  = true;

    // Audio chain: each chunk plays only after the previous one finishes.
    // Because TTS generates concurrently on the server, chunks arrive quickly.
    let audioChain = Promise.resolve();

    try {
      await api.sendTextMessageStream(
        text,
        historyRef.current,
        level,
        (token) => {
          accumulated += token;
          if (firstToken) {
            firstToken = false;
            setProcessing(false);
            setMessages((prev) => [
              ...prev,
              { role: "assistant" as const, content_de: accumulated, content_en: "" },
            ]);
          } else {
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = {
                role: "assistant",
                content_de: accumulated,
                content_en: "",
              };
              return next;
            });
          }
        },
        (audioUrl) => {
          // Chain each audio clip — they play in order, back-to-back
          audioChain = audioChain.then(() => playAudio(audioUrl));
        },
      );

      historyRef.current.push({ role: "assistant", content: accumulated });
      await audioChain;
    } catch (err) {
      throw err;
    } finally {
      setProcessing(false);
    }
  }, [playAudio]);

  // ── Text send (legacy, non-streaming) ────────────────────────────────────
  const sendTextToAPI = useCallback(async (text: string, level = "B1"): Promise<void> => {
    setProcessing(true);
    try {
      const data = await api.sendTextMessage(text, historyRef.current, level);
      addMessage({ role: "assistant", content_de: data.ai_text_de, content_en: data.ai_text_en });
      await playAudio(data.tts_audio_url);
    } finally {
      setProcessing(false);
    }
  }, [addMessage, playAudio]);

  // ── Voice send ────────────────────────────────────────────────────────────
  /**
   * Full voice pipeline: sends audio, adds BOTH user + AI messages when done.
   * (We can't show user text immediately since it must be transcribed first.)
   */
  const sendVoice = useCallback(async (blob: Blob, level = "B1"): Promise<ChatResponse> => {
    setProcessing(true);
    try {
      const data = await api.sendVoiceMessage(blob, historyRef.current, level);
      addMessage({ role: "user",      content_de: data.user_text,  content_en: "" });
      addMessage({ role: "assistant", content_de: data.ai_text_de, content_en: data.ai_text_en });
      return data;
    } finally {
      setProcessing(false);
    }
  }, [addMessage]);

  return {
    messages,
    isProcessing,
    setProcessing,
    addMessage,
    loadHistory,
    clearMessages,
    sendTextStream,
    sendTextToAPI,
    sendVoice,
    playAudio,
  };
}
