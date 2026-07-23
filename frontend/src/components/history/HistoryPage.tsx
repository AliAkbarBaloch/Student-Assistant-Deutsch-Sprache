/**
 * HistoryPage — lists past Live-Anruf calls grouped into date sections.
 * Tapping a call opens its transcript, WhatsApp-style: user messages on the
 * left, Buddy's replies on the right.
 */
import React, { useEffect, useState } from "react";
import { ArrowLeft, Phone, Clock } from "lucide-react";
import * as api from "../../services/api";
import type { CallSummary, CallTranscriptMessage } from "../../types";

interface Props {
  onBack: () => void;
}

function formatDateHeader(iso: string): string {
  return new Date(iso).toLocaleDateString("de-DE", {
    weekday: "long", day: "numeric", month: "long", year: "numeric",
  });
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function groupByDate(calls: CallSummary[]): [string, CallSummary[]][] {
  const map = new Map<string, CallSummary[]>();
  for (const c of calls) {
    const label = formatDateHeader(c.started_at);
    if (!map.has(label)) map.set(label, []);
    map.get(label)!.push(c);
  }
  return Array.from(map.entries());
}

export function HistoryPage({ onBack }: Props) {
  const [calls, setCalls] = useState<CallSummary[] | null>(null);
  const [selected, setSelected] = useState<CallSummary | null>(null);
  const [detailMessages, setDetailMessages] = useState<CallTranscriptMessage[] | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  useEffect(() => {
    api.fetchCalls().then((r) => setCalls(r.calls)).catch(() => setCalls([]));
  }, []);

  async function openCall(call: CallSummary) {
    setSelected(call);
    setDetailMessages(null);
    setLoadingDetail(true);
    try {
      const detail = await api.fetchCallDetail(call.id);
      setDetailMessages(detail.messages);
    } catch {
      setDetailMessages([]);
    } finally {
      setLoadingDetail(false);
    }
  }

  // ── Transcript detail view ──────────────────────────────────────────────
  if (selected) {
    return (
      <div className="h-screen flex flex-col bg-gray-50 dark:bg-black">
        <header className="flex-shrink-0 flex items-center gap-3 px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
          <button
            onClick={() => { setSelected(null); setDetailMessages(null); }}
            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft size={18} />
          </button>
          <div className="flex items-center gap-2.5">
            <img src="/static/mascot.jpg" alt="Buddy" className="w-9 h-9 rounded-full object-cover border-2 border-brand-500" />
            <div>
              <p className="text-sm font-bold text-gray-900 dark:text-white leading-none">Deutsche Buddy</p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
                {formatDateHeader(selected.started_at)} · {formatTime(selected.started_at)}
              </p>
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
          {loadingDetail && <p className="text-center text-sm text-gray-400 mt-8">Lädt…</p>}
          {!loadingDetail && detailMessages?.length === 0 && (
            <p className="text-center text-sm text-gray-400 mt-8">Kein Transkript verfügbar.</p>
          )}
          {detailMessages?.map((m, i) => <CallBubble key={i} message={m} />)}
        </div>
      </div>
    );
  }

  // ── Call list view ───────────────────────────────────────────────────────
  const sections = groupByDate(calls ?? []);

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-gray-950 text-gray-900 dark:text-white">
      <header className="flex items-center gap-3 px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
        <button
          onClick={onBack}
          className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          <ArrowLeft size={18} />
        </button>
        <div>
          <p className="text-sm font-extrabold leading-none">
            Anruf<span className="text-brand-500">-Verlauf</span>
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Deine Live-Anruf-Transkripte</p>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {calls === null && <p className="text-center text-sm text-gray-400 mt-8">Lädt…</p>}

        {calls?.length === 0 && (
          <div className="flex flex-col items-center justify-center gap-3 mt-16 text-center">
            <div className="w-14 h-14 rounded-full bg-brand-500/10 border border-brand-500/20 flex items-center justify-center">
              <Phone size={22} className="text-brand-500" />
            </div>
            <p className="text-sm text-gray-400 max-w-xs">
              Noch keine Live-Anrufe. Starte einen Anruf mit Buddy, um dein erstes Transkript zu sehen.
            </p>
          </div>
        )}

        {sections.map(([dateLabel, callsForDate]) => (
          <div key={dateLabel} className="mb-6">
            <h3 className="text-xs font-bold uppercase tracking-wide text-gray-400 dark:text-gray-500 mb-2 px-1">
              {dateLabel}
            </h3>
            <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800 divide-y divide-gray-200 dark:divide-gray-800">
              {callsForDate.map((c) => (
                <button
                  key={c.id}
                  onClick={() => openCall(c)}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800/60 transition-colors text-left"
                >
                  <div className="flex-shrink-0 w-10 h-10 rounded-full bg-brand-500/10 border border-brand-500/20 flex items-center justify-center">
                    <Phone size={16} className="text-brand-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                      {c.preview || "Live-Anruf"}
                    </p>
                    <p className="text-xs text-gray-400 dark:text-gray-500 flex items-center gap-1 mt-0.5">
                      <Clock size={11} /> {formatDuration(c.duration_seconds)} · {c.message_count} Nachrichten
                    </p>
                  </div>
                  <span className="text-xs text-gray-400 dark:text-gray-500 flex-shrink-0">
                    {formatTime(c.started_at)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** WhatsApp-style bubble — user on the left, Buddy on the right. */
function CallBubble({ message }: { message: CallTranscriptMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex gap-2.5 max-w-[80%] bubble-enter ${isUser ? "self-start" : "self-end flex-row-reverse"}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full overflow-hidden border-2 ${isUser ? "border-blue-600" : "border-brand-500"}`}>
        {isUser ? (
          <div className="w-full h-full flex items-center justify-center bg-blue-800 text-white text-[10px] font-bold">Du</div>
        ) : (
          <img src="/static/mascot.jpg" alt="Buddy" className="w-full h-full object-cover" />
        )}
      </div>
      <div
        className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-tl-sm text-gray-900 dark:text-gray-100"
            : "bg-brand-500/15 dark:bg-brand-500/10 border border-brand-500/30 rounded-tr-sm text-gray-900 dark:text-gray-100"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
