/**
 * FeedbackPage — German pronunciation feedback tool.
 * User uploads an MP3/WAV file OR records via microphone.
 * The AI transcribes the audio and returns a detailed pronunciation analysis.
 */
import React, { useEffect, useRef, useState } from "react";
import {
  ArrowLeft, Upload, Mic, MicOff, CheckCircle,
  AlertCircle, Lightbulb, BarChart2, RefreshCw,
} from "lucide-react";
import { getPronunciationFeedback, type FeedbackResponse } from "../../services/api";

interface Props {
  onBack: () => void;
}

type PageState = "idle" | "recording" | "uploading" | "analysing" | "done" | "error";

export function FeedbackPage({ onBack }: Props) {
  const [pageState,  setPageState]  = useState<PageState>("idle");
  const [result,     setResult]     = useState<FeedbackResponse | null>(null);
  const [errorMsg,   setErrorMsg]   = useState("");
  const [targetText, setTargetText] = useState("");
  const [fileName,   setFileName]   = useState("");
  const [showEn,     setShowEn]     = useState(false);
  const [elapsed,    setElapsed]    = useState(0);

  // Tick elapsed seconds while analysing so the user knows it's still working
  useEffect(() => {
    if (pageState !== "analysing") { setElapsed(0); return; }
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [pageState]);

  const fileRef          = useRef<HTMLInputElement>(null);
  const mediaRecRef      = useRef<MediaRecorder | null>(null);
  const chunksRef        = useRef<Blob[]>([]);

  // ── File upload ─────────────────────────────────────────────────────────────

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    analyseAudio(file);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    setFileName(file.name);
    analyseAudio(file);
  }

  // ── Microphone recording ────────────────────────────────────────────────────

  async function toggleRecording() {
    if (pageState === "recording") {
      mediaRecRef.current?.stop();
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setErrorMsg("Mikrofon-Zugriff verweigert.");
      setPageState("error");
      return;
    }

    chunksRef.current = [];
    const rec = new MediaRecorder(stream);
    mediaRecRef.current = rec;

    rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      setFileName("Aufnahme.webm");
      analyseAudio(blob);
    };

    rec.start();
    setPageState("recording");
  }

  // ── Core analysis ───────────────────────────────────────────────────────────

  async function analyseAudio(audio: Blob | File) {
    setPageState("analysing");
    setResult(null);
    setErrorMsg("");
    setShowEn(false);
    try {
      const data = await getPronunciationFeedback(audio, targetText);
      setResult(data);
      setPageState("done");
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Analyse fehlgeschlagen.");
      setPageState("error");
    }
  }

  async function testTextOnly() {
    if (!targetText.trim()) return;
    setPageState("analysing");
    setResult(null);
    setErrorMsg("");
    setShowEn(false);
    try {
      const data = await getPronunciationFeedback(null, targetText);
      setResult(data);
      setPageState("done");
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Analyse fehlgeschlagen.");
      setPageState("error");
    }
  }

  function reset() {
    setPageState("idle");
    setResult(null);
    setErrorMsg("");
    setFileName("");
    setShowEn(false);
    if (fileRef.current) fileRef.current.value = "";
  }

  // ── Score colour ────────────────────────────────────────────────────────────

  function scoreColor(s: number) {
    if (s >= 8) return "text-green-500 border-green-500";
    if (s >= 5) return "text-yellow-500 border-yellow-500";
    return "text-red-500 border-red-500";
  }

  function scoreBg(s: number) {
    if (s >= 8) return "bg-green-500/10";
    if (s >= 5) return "bg-yellow-500/10";
    return "bg-red-500/10";
  }

  function scoreLabel(s: number) {
    if (s >= 9) return "Ausgezeichnet!";
    if (s >= 7) return "Sehr gut!";
    if (s >= 5) return "Gut — weiter üben!";
    if (s >= 3) return "Noch viel zu üben";
    return "Lass uns von vorn anfangen";
  }

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-gray-950 text-gray-900 dark:text-white">

      {/* Top bar */}
      <header className="flex-shrink-0 flex items-center gap-3 px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
        <button
          onClick={onBack}
          className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          <ArrowLeft size={18} />
        </button>
        <img src="/static/mascot.jpg" alt="Buddy" className="w-8 h-8 rounded-full object-cover border-2 border-brand-500" />
        <div>
          <p className="text-sm font-extrabold leading-none">
            Deutsche <span className="text-brand-500">Buddy</span>
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Aussprache-Feedback</p>
        </div>
        {pageState === "done" && (
          <button
            onClick={reset}
            className="ml-auto flex items-center gap-1.5 text-xs text-brand-500 hover:text-brand-400 font-semibold transition-colors"
          >
            <RefreshCw size={14} /> Neu analysieren
          </button>
        )}
      </header>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-4 py-8">
        <div className="w-full max-w-xl mx-auto space-y-6">

          {/* ── Idle / Error / Recording state ── */}
          {(pageState === "idle" || pageState === "error" || pageState === "recording") && (
            <>
              <div className="text-center space-y-1">
                <h1 className="text-xl font-extrabold">Aussprache analysieren</h1>
                <p className="text-sm text-gray-400 dark:text-gray-500">
                  Lade eine Audiodatei hoch oder nimm dich auf — Buddy gibt dir detailliertes Feedback.
                </p>
              </div>

              {/* Optional target text */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                  Was hast du versucht zu sagen? <span className="font-normal normal-case">(optional)</span>
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={targetText}
                    onChange={(e) => setTargetText(e.target.value)}
                    placeholder='z. B. "Ich möchte einen Kaffee, bitte."'
                    className="flex-1 rounded-xl px-4 py-3 text-sm bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-500 placeholder-gray-400"
                  />
                  <button 
                    onClick={testTextOnly}
                    disabled={!targetText.trim()}
                    className="px-4 py-2 bg-brand-500 hover:bg-brand-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white text-sm font-bold rounded-xl transition-colors"
                  >
                    Senden
                  </button>
                </div>
              </div>

              {/* Upload zone */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileRef.current?.click()}
                className="cursor-pointer rounded-2xl border-2 border-dashed border-gray-300 dark:border-gray-700 hover:border-brand-500 dark:hover:border-brand-500 transition-colors p-8 flex flex-col items-center gap-3 bg-gray-50 dark:bg-gray-900"
              >
                <div className="w-14 h-14 rounded-full bg-brand-500/10 flex items-center justify-center">
                  <Upload size={26} className="text-brand-500" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    MP3 oder WAV hochladen
                  </p>
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                    Datei hierher ziehen oder klicken zum Auswählen
                  </p>
                </div>
                {fileName && (
                  <p className="text-xs text-brand-500 font-medium truncate max-w-full px-2">{fileName}</p>
                )}
              </div>
              <input
                ref={fileRef}
                type="file"
                accept="audio/mpeg,audio/mp3,audio/wav,audio/wave,audio/webm"
                className="hidden"
                onChange={handleFileChange}
              />

              {/* Divider */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-gray-200 dark:bg-gray-800" />
                <span className="text-xs text-gray-400 dark:text-gray-600">oder</span>
                <div className="flex-1 h-px bg-gray-200 dark:bg-gray-800" />
              </div>

              {/* Record button */}
              <button
                onClick={toggleRecording}
                className={`w-full flex items-center justify-center gap-2.5 py-4 rounded-2xl text-sm font-bold border-2 transition-all ${
                  pageState === "recording"
                    ? "bg-red-500/10 border-red-500 text-red-500 animate-pulse"
                    : "bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-brand-500 hover:text-brand-500"
                }`}
              >
                {pageState === "recording"
                  ? <><MicOff size={20} /> Aufnahme stoppen</>
                  : <><Mic size={20} /> Aufnehmen</>
                }
              </button>

              {/* Error message */}
              {pageState === "error" && (
                <div className="flex items-start gap-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-500 text-sm">
                  <AlertCircle size={16} className="mt-0.5 flex-shrink-0" />
                  {errorMsg}
                </div>
              )}
            </>
          )}

          {/* ── Analysing spinner ── */}
          {pageState === "analysing" && (
            <div className="flex flex-col items-center gap-5 py-16">
              <div className="w-16 h-16 rounded-full border-4 border-brand-500/30 border-t-brand-500 animate-spin" />
              <div className="text-center">
                <p className="font-bold text-gray-900 dark:text-white">Buddy analysiert…</p>
                <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
                  Transkription + KI-Feedback wird erstellt
                </p>
                {elapsed > 0 && (
                  <p className="text-xs text-gray-400 dark:text-gray-600 mt-2 tabular-nums">
                    {elapsed}s — das kann auf der CPU etwas dauern…
                  </p>
                )}
              </div>
            </div>
          )}

          {/* ── Results ── */}
          {pageState === "done" && result && (
            <div className="space-y-5">

              {/* Score card */}
              <div className={`rounded-2xl p-5 flex items-center gap-5 border ${scoreBg(result.score)} border-gray-200 dark:border-gray-800`}>
                <div className={`flex-shrink-0 w-20 h-20 rounded-full border-4 flex flex-col items-center justify-center ${scoreColor(result.score)}`}>
                  <span className="text-3xl font-extrabold leading-none">{result.score}</span>
                  <span className="text-[10px] font-semibold opacity-70">/ 10</span>
                </div>
                <div>
                  <p className={`text-lg font-extrabold ${scoreColor(result.score).split(" ")[0]}`}>
                    {scoreLabel(result.score)}
                  </p>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mt-1 leading-snug">
                    {result.overall}
                  </p>
                </div>
              </div>

              {/* Transcription */}
              <div className="rounded-2xl bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-4 space-y-1.5">
                <p className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wide">
                  Was Buddy gehört hat
                </p>
                <p className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed italic">
                  „{result.transcribed}"
                </p>
              </div>

              {/* Issues */}
              {result.issues.length > 0 && (
                <div className="rounded-2xl bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-4 space-y-3">
                  <p className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wide flex items-center gap-1.5">
                    <AlertCircle size={13} /> Gefundene Probleme
                  </p>
                  <ul className="space-y-2">
                    {result.issues.map((issue, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                        <span className="flex-shrink-0 w-5 h-5 rounded-full bg-red-500/10 border border-red-500/30 text-red-500 text-[10px] font-bold flex items-center justify-center mt-0.5">
                          {i + 1}
                        </span>
                        {issue}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Tips */}
              {result.tips.length > 0 && (
                <div className="rounded-2xl bg-brand-500/5 border border-brand-500/20 p-4 space-y-3">
                  <p className="text-xs font-semibold text-brand-500 uppercase tracking-wide flex items-center gap-1.5">
                    <Lightbulb size={13} /> Tipps zur Verbesserung
                  </p>
                  <ul className="space-y-2">
                    {result.tips.map((tip, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300">
                        <CheckCircle size={15} className="flex-shrink-0 text-brand-500 mt-0.5" />
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* English translation toggle */}
              {result.feedback_en && (
                <div className="rounded-2xl bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-4 space-y-2">
                  <button
                    onClick={() => setShowEn((v) => !v)}
                    className="text-xs text-gray-400 dark:text-gray-500 border border-gray-300 dark:border-gray-700 rounded px-2 py-1 hover:text-gray-700 dark:hover:text-gray-300 transition-colors flex items-center gap-1"
                  >
                    <BarChart2 size={12} />
                    {showEn ? "EN ▴ verbergen" : "EN ▾ Englische Zusammenfassung"}
                  </button>
                  {showEn && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 italic leading-relaxed pt-1 border-t border-gray-200 dark:border-gray-800">
                      {result.feedback_en}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
