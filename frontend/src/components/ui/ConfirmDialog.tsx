import React from "react";
import { Trash2, X } from "lucide-react";

interface Props {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  loading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Beautiful modal confirmation dialog */
export function ConfirmDialog({
  open, title, message, confirmLabel = "Löschen",
  loading, onConfirm, onCancel,
}: Props) {
  if (!open) return null;

  return (
    // Backdrop
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onCancel}
    >
      {/* Card — stop click propagation so backdrop click closes only */}
      <div
        className="w-full max-w-sm bg-gray-900 dark:bg-gray-900 border border-gray-700 rounded-2xl p-6 shadow-2xl animate-[fadeUp_.2s_ease_both]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center">
              <Trash2 size={18} className="text-red-400" />
            </div>
            <h2 className="text-base font-bold text-white">{title}</h2>
          </div>
          <button onClick={onCancel} className="text-gray-500 hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <p className="text-sm text-gray-400 mb-6 leading-relaxed">{message}</p>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            disabled={loading}
            className="flex-1 py-2.5 rounded-xl border border-gray-700 text-sm font-semibold text-gray-300 hover:bg-gray-800 transition-colors disabled:opacity-50"
          >
            Abbrechen
          </button>
          <button
            onClick={onConfirm}
            disabled={loading}
            className="flex-1 py-2.5 rounded-xl bg-red-500 hover:bg-red-600 text-sm font-bold text-white transition-colors disabled:opacity-50"
          >
            {loading ? "Wird gelöscht…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
