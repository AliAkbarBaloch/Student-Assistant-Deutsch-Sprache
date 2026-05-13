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

  // ── Text send ─────────────────────────────────────────────────────────────
  /**
   * 1. User message is added IMMEDIATELY in ChatPage before this is called.
   * 2. This function only handles the API call and adding the AI response.
   */
  const sendTextToAPI = useCallback(async (text: string, level = "B1"): Promise<void> => {
    setProcessing(true);
    try {
      const data = await api.sendTextMessage(text, historyRef.current, level);
      // Add only AI message — user message was already added by caller
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
  const sendVoice = useCallback(async (blob: Blob, level = "B1"): Promise<ChatResponse | null> => {
    setProcessing(true);
    try {
      const data = await api.sendVoiceMessage(blob, historyRef.current, level);
      addMessage({ role: "user",      content_de: data.user_text,  content_en: "" });
      addMessage({ role: "assistant", content_de: data.ai_text_de, content_en: data.ai_text_en });
      return data;
    } catch {
      return null;
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
    sendTextToAPI,
    sendVoice,
    playAudio,
  };
}
