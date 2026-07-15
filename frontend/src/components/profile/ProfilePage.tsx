/**
 * ProfilePage — lets the user update their display name and upload a profile picture.
 * The uploaded avatar replaces the initial letter shown in chat bubbles.
 */
import React, { useRef, useState } from "react";
import { ArrowLeft, Camera, CheckCircle, XCircle } from "lucide-react";
import { useAuth } from "../../contexts/AuthContext";
import { updateProfile } from "../../services/api";

interface Props {
  onBack: () => void;
}

export function ProfilePage({ onBack }: Props) {
  const { user, updateUser } = useAuth();

  const [name, setName]           = useState(user?.name ?? "");
  const [preview, setPreview]     = useState<string | null>(null);
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [saving, setSaving]       = useState(false);
  const [toast, setToast]         = useState<{ ok: boolean; msg: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  /** Show image preview when user picks a file */
  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setAvatarFile(file);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result as string);
    reader.readAsDataURL(file);
  }

  /** Submit name + optional avatar to backend */
  async function handleSave() {
    if (!name.trim()) return;
    setSaving(true);
    setToast(null);
    try {
      const res = await updateProfile(name.trim(), avatarFile);
      updateUser(res.user);
      setToast({ ok: true, msg: "Profil erfolgreich gespeichert!" });
      setAvatarFile(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Fehler beim Speichern.";
      setToast({ ok: false, msg });
    } finally {
      setSaving(false);
    }
  }

  /** Derive what to show as the current avatar */
  const currentAvatar = preview ?? user?.avatar_url ?? null;
  const initials = (user?.name ?? "?")
    .split(" ")
    .map((w) => w[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-gray-950 text-gray-900 dark:text-white">

      {/* Top bar */}
      <header className="flex items-center gap-3 px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
        <button
          onClick={onBack}
          className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
        >
          <ArrowLeft size={18} />
        </button>
        <div>
          <p className="text-sm font-extrabold leading-none">
            Deutsche <span className="text-brand-500">Buddy</span>
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Profil bearbeiten</p>
        </div>
      </header>

      {/* Body */}
      <div className="flex-1 flex flex-col items-center py-12 px-4">
        <div className="w-full max-w-md space-y-8">

          {/* Avatar section */}
          <div className="flex flex-col items-center gap-4">
            <button
              onClick={() => fileRef.current?.click()}
              className="relative group w-28 h-28 rounded-full overflow-hidden border-4 border-brand-500 shadow-lg shadow-brand-500/20 focus:outline-none"
              title="Profilbild ändern"
            >
              {currentAvatar ? (
                <img
                  src={currentAvatar}
                  alt="Profilbild"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-blue-700 dark:bg-blue-900 text-white text-3xl font-bold">
                  {initials}
                </div>
              )}
              {/* Hover overlay */}
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <Camera size={28} className="text-white" />
              </div>
            </button>
            <p className="text-xs text-gray-400 dark:text-gray-500">
              Klicke auf das Bild, um es zu ändern (JPG / PNG / WebP)
            </p>
            {/* Hidden file input */}
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={handleFileChange}
            />
          </div>

          {/* Name input */}
          <div className="space-y-1.5">
            <label className="text-sm font-semibold text-gray-600 dark:text-gray-400">
              Anzeigename
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded-xl px-4 py-3 text-sm bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-brand-500 placeholder-gray-400"
              placeholder="Dein Name"
            />
          </div>

          {/* Email (read-only) */}
          <div className="space-y-1.5">
            <label className="text-sm font-semibold text-gray-600 dark:text-gray-400">
              E-Mail (nicht änderbar)
            </label>
            <input
              type="email"
              value={user?.email ?? ""}
              readOnly
              className="w-full rounded-xl px-4 py-3 text-sm bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed"
            />
          </div>

          {/* Save button */}
          <button
            onClick={handleSave}
            disabled={saving || !name.trim()}
            className="w-full py-3 rounded-xl text-sm font-bold bg-brand-500 hover:bg-brand-600 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors shadow-md shadow-brand-500/20"
          >
            {saving ? "Speichern…" : "Änderungen speichern"}
          </button>

          {/* Toast feedback */}
          {toast && (
            <div
              className={`flex items-center gap-2 px-4 py-3 rounded-xl text-sm font-medium border ${
                toast.ok
                  ? "bg-brand-500/10 border-brand-500/30 text-brand-600 dark:text-brand-400"
                  : "bg-red-500/10 border-red-500/30 text-red-600 dark:text-red-400"
              }`}
            >
              {toast.ok ? <CheckCircle size={16} /> : <XCircle size={16} />}
              {toast.msg}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
