import React, { useState } from "react";
import { Sun, Moon } from "lucide-react";
import { useAuth } from "../../contexts/AuthContext";
import { useTheme } from "../../contexts/ThemeContext";
import * as api from "../../services/api";

type Tab = "login" | "signup";

export function AuthPage() {
  const { login } = useAuth();
  const { theme, toggle } = useTheme();

  const [tab, setTab]         = useState<Tab>("login");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  // Login fields
  const [loginEmail, setLoginEmail]       = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  // Signup fields
  const [signupName, setSignupName]         = useState("");
  const [signupEmail, setSignupEmail]       = useState("");
  const [signupPassword, setSignupPassword] = useState("");

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      const data = await api.login(loginEmail, loginPassword);
      login(data.token, data.user);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Fehler beim Anmelden");
    } finally { setLoading(false); }
  }

  async function handleSignup(e: React.FormEvent) {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      const data = await api.register(signupName, signupEmail, signupPassword);
      login(data.token, data.user);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Fehler bei der Registrierung");
    } finally { setLoading(false); }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-black flex items-center justify-center p-4 relative">

      {/* Theme toggle */}
      <button
        onClick={toggle}
        className="absolute top-4 right-4 p-2 rounded-full bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
      >
        {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
      </button>

      <div className="w-full max-w-md">
        {/* Card */}
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl p-8 shadow-2xl">

          {/* Logo */}
          <div className="flex flex-col items-center gap-3 mb-8">
            <img
              src="/static/mascot.jpg"
              alt="Buddy"
              className="w-20 h-20 rounded-full object-cover border-2 border-brand-500 shadow-lg shadow-brand-500/20"
            />
            <div className="text-center">
              <h1 className="text-2xl font-extrabold text-gray-900 dark:text-white tracking-tight">
                Deutsch <span className="text-brand-500">Buddy</span>
              </h1>
              <p className="text-sm text-gray-500 mt-0.5">Dein KI-Deutsch-Gesprächspartner</p>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-xl p-1 mb-6">
            {(["login", "signup"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => { setTab(t); setError(""); }}
                className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all ${
                  tab === t
                    ? "bg-brand-500 text-black"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                {t === "login" ? "Anmelden" : "Registrieren"}
              </button>
            ))}
          </div>

          {/* Login form */}
          {tab === "login" && (
            <form onSubmit={handleLogin} className="flex flex-col gap-4">
              <Field label="E-Mail">
                <input
                  type="email" required
                  value={loginEmail} onChange={(e) => setLoginEmail(e.target.value)}
                  placeholder="name@beispiel.de"
                  className={inputCls}
                />
              </Field>
              <Field label="Passwort">
                <input
                  type="password" required
                  value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)}
                  placeholder="••••••••"
                  className={inputCls}
                />
              </Field>
              {error && <ErrorBox msg={error} />}
              <SubmitBtn loading={loading}>Anmelden</SubmitBtn>
            </form>
          )}

          {/* Signup form */}
          {tab === "signup" && (
            <form onSubmit={handleSignup} className="flex flex-col gap-4">
              <Field label="Name">
                <input
                  type="text" required
                  value={signupName} onChange={(e) => setSignupName(e.target.value)}
                  placeholder="Dein Name"
                  className={inputCls}
                />
              </Field>
              <Field label="E-Mail">
                <input
                  type="email" required
                  value={signupEmail} onChange={(e) => setSignupEmail(e.target.value)}
                  placeholder="name@beispiel.de"
                  className={inputCls}
                />
              </Field>
              <Field label="Passwort">
                <input
                  type="password" required minLength={6}
                  value={signupPassword} onChange={(e) => setSignupPassword(e.target.value)}
                  placeholder="Mindestens 6 Zeichen"
                  className={inputCls}
                />
              </Field>
              {error && <ErrorBox msg={error} />}
              <SubmitBtn loading={loading}>Konto erstellen</SubmitBtn>
            </form>
          )}

        </div>
      </div>
    </div>
  );
}

// ── Small reusable sub-components ────────────────────────────────────────────

const inputCls =
  "w-full px-4 py-3 bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 text-sm outline-none focus:border-brand-500 transition-colors";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold uppercase tracking-widest text-gray-500 dark:text-gray-500">{label}</label>
      {children}
    </div>
  );
}

function ErrorBox({ msg }: { msg: string }) {
  return (
    <p className="text-sm text-red-400 bg-red-500/10 rounded-lg px-3 py-2 text-center">{msg}</p>
  );
}

function SubmitBtn({ loading, children }: { loading: boolean; children: React.ReactNode }) {
  return (
    <button
      type="submit" disabled={loading}
      className="mt-1 w-full py-3 bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-black font-bold rounded-xl text-sm transition-all hover:-translate-y-0.5 disabled:cursor-not-allowed"
    >
      {loading ? "Bitte warten…" : children}
    </button>
  );
}
