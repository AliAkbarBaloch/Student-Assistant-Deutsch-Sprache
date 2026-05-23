export interface User {
  id: number;
  name: string;
  email: string;
  avatar_url?: string | null;
}

export interface ProfileResponse {
  user: User;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content_de: string;
  content_en: string;
}

export interface ChatResponse {
  user_text: string;
  ai_text_de: string;
  ai_text_en: string;
  tts_audio_url: string;
}

export interface HistoryResponse {
  messages: ChatMessage[];
}

// UI state for the mic / live call
export type MicState = "idle" | "recording" | "processing";
export type CallState = "idle" | "listening" | "processing" | "speaking";
