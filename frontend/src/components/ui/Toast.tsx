import React, { useEffect } from "react";
import { CheckCircle, XCircle, X } from "lucide-react";

export type ToastType = "success" | "error";

export interface ToastData {
  id: number;
  message: string;
  type: ToastType;
}

interface Props {
  toast: ToastData | null;
  onClose: () => void;
}

/** Auto-dismissing toast notification shown just below the navbar */
export function Toast({ toast, onClose }: Props) {
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(onClose, 3500);
    return () => clearTimeout(t);
  }, [toast, onClose]);

  if (!toast) return null;

  const isSuccess = toast.type === "success";

  return (
    <div
      className={`fixed top-16 right-4 z-50 flex items-center gap-3 px-4 py-3 rounded-xl shadow-2xl border text-sm font-medium animate-[fadeUp_.25s_ease_both] ${
        isSuccess
          ? "bg-brand-500/10 border-brand-500/30 text-brand-400"
          : "bg-red-500/10 border-red-500/30 text-red-400"
      }`}
    >
      {isSuccess ? <CheckCircle size={18} /> : <XCircle size={18} />}
      <span>{toast.message}</span>
      <button onClick={onClose} className="ml-1 opacity-60 hover:opacity-100">
        <X size={14} />
      </button>
    </div>
  );
}
