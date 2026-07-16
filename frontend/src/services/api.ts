/**
 * API service — all HTTP calls to the FastAPI backend.
 * Auth token is automatically injected when available.
 */
import type { AuthResponse, ChatResponse, HistoryResponse, ProfileResponse } from "../types";

const getToken = () => localStorage.getItem("db_token");

function authHeaders(): HeadersInit {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handleResponse<T>(res: Response): Promise<T> {
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Server error");
  return data as T;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function login(email: string, password: string): Promise<AuthResponse> {
  const fd = new FormData();
  fd.append("email", email);
  fd.append("password", password);
  const res = await fetch("/api/auth/login", { method: "POST", body: fd });
  return handleResponse<AuthResponse>(res);
}

export async function register(
  name: string,
  email: string,
  password: string
): Promise<AuthResponse> {
  const fd = new FormData();
  fd.append("name", name);
  fd.append("email", email);
  fd.append("password", password);
  const res = await fetch("/api/auth/register", { method: "POST", body: fd });
  return handleResponse<AuthResponse>(res);
}

// ── Profile ───────────────────────────────────────────────────────────────────

export async function updateProfile(
  name: string,
  avatar?: File | null,
): Promise<ProfileResponse> {
  const fd = new FormData();
  fd.append("name", name);
  if (avatar) fd.append("avatar", avatar);
  const res = await fetch("/api/auth/profile", {
    method: "PUT",
    body: fd,
    headers: authHeaders(),
  });
  return handleResponse<ProfileResponse>(res);
}

// ── History ───────────────────────────────────────────────────────────────────

export async function fetchHistory(): Promise<HistoryResponse> {
  const res = await fetch("/api/history", { headers: authHeaders() });
  return handleResponse<HistoryResponse>(res);
}

export async function deleteHistory(): Promise<void> {
  await fetch("/api/history", { method: "DELETE", headers: authHeaders() });
}

// ── Chat (text) ───────────────────────────────────────────────────────────────

export async function sendTextMessage(
  message: string,
  history: { role: string; content: string }[],
  level = "B1",
): Promise<ChatResponse> {
  const fd = new FormData();
  fd.append("message", message);
  fd.append("history", JSON.stringify(history));
  fd.append("level", level);
  const res = await fetch("/api/chat-text", {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handleResponse<ChatResponse>(res);
}

/**
 * Streaming text chat via SSE.
 * - onToken fires for every LLM token as it arrives (text appears live).
 * - onAudio fires for each TTS chunk URL in order (play them sequentially).
 * TTS generation runs concurrently with LLM streaming on the server, so
 * the first audio chunk is usually ready the moment text finishes.
 */
export async function sendTextMessageStream(
  message: string,
  history: { role: string; content: string }[],
  level = "B1",
  onToken: (token: string) => void,
  onAudio: (url: string) => void,
): Promise<void> {
  const fd = new FormData();
  fd.append("message", message);
  fd.append("history", JSON.stringify(history));
  fd.append("level", level);

  const res = await fetch("/api/chat-text-stream", {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });

  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? "Stream error");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      let event: { type: string; text?: string; tts_url?: string; message?: string };
      try { event = JSON.parse(part.slice(6)); } catch { continue; }
      if (event.type === "token" && event.text) {
        onToken(event.text);
      } else if (event.type === "audio" && event.tts_url) {
        onAudio(event.tts_url);
      } else if (event.type === "error") {
        throw new Error(event.message ?? "Streaming error");
      }
    }
  }
}

// ── Pronunciation feedback ────────────────────────────────────────────────────

export interface FeedbackResponse {
  transcribed: string;
  score: number;
  overall: string;
  issues: string[];
  tips: string[];
  feedback_en: string;
}

export async function getPronunciationFeedback(
  audio: Blob | File | null,
  targetText = "",
): Promise<FeedbackResponse> {
  const fd = new FormData();
  if (audio) {
    fd.append("audio", audio, audio instanceof File ? audio.name : "recording.webm");
  }
  fd.append("target_text", targetText);
  const res = await fetch("/api/pronunciation-feedback", {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handleResponse<FeedbackResponse>(res);
}

// ── LiveKit token (Phone / Live-Anruf button) ─────────────────────────────────

export interface LiveKitTokenResponse {
  token: string;
  url:   string;
  room:  string;
}

export async function getLiveKitToken(level = "B1"): Promise<LiveKitTokenResponse> {
  const fd = new FormData();
  fd.append("level", level);
  const res = await fetch("/api/livekit-token", {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handleResponse<LiveKitTokenResponse>(res);
}

// ── Chat (voice) ──────────────────────────────────────────────────────────────

export async function sendVoiceMessage(
  blob: Blob,
  history: { role: string; content: string }[],
  level = "B1",
): Promise<ChatResponse> {
  const fd = new FormData();
  fd.append("audio", blob, "recording.webm");
  fd.append("history", JSON.stringify(history));
  fd.append("level", level);
  const res = await fetch("/api/chat", {
    method: "POST",
    body: fd,
    headers: authHeaders(),
  });
  return handleResponse<ChatResponse>(res);
}
