// ─────────────────────────────────────────────────────────────────────────────
// VAD config
// ─────────────────────────────────────────────────────────────────────────────
const SPEECH_THRESHOLD = 12;
const SILENCE_DURATION = 1500;
const MIN_SPEECH_MS    = 400;

// ─────────────────────────────────────────────────────────────────────────────
// App state
// ─────────────────────────────────────────────────────────────────────────────
let authToken   = localStorage.getItem("db_token")   || null;
let currentUser = JSON.parse(localStorage.getItem("db_user") || "null");

let mediaRecorder   = null;
let recordedChunks  = [];
let isRecording     = false;
let isProcessing    = false;
let history         = [];   // LLM context window

// Live-call state
let callActive        = false;
let callStream        = null;
let vadAudioCtx       = null;
let vadAnalyser       = null;
let vadInterval       = null;
let vadSpeaking       = false;
let vadSilenceStart   = null;
let speechStart       = null;
let callStartTime     = null;
let callTimerInterval = null;

// ─────────────────────────────────────────────────────────────────────────────
// DOM — resolved lazily to avoid null refs before appPage is shown
// ─────────────────────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);

// ─────────────────────────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  // Restore saved theme
  const saved = localStorage.getItem("db_theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  _syncThemeIcons(saved);

  if (authToken && currentUser) {
    _showApp();
  } else {
    _showAuth();
  }
});


// ─────────────────────────────────────────────────────────────────────────────
// Theme toggle
// ─────────────────────────────────────────────────────────────────────────────
function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next    = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("db_theme", next);
  _syncThemeIcons(next);
}

function _syncThemeIcons(theme) {
  const icon = theme === "dark" ? "☀️" : "🌙";
  const el1  = $("themeIcon");
  const el2  = $("themeIconAuth");
  if (el1) el1.textContent = icon;
  if (el2) el2.textContent = icon;
}


// ─────────────────────────────────────────────────────────────────────────────
// Auth — page switching
// ─────────────────────────────────────────────────────────────────────────────
function _showAuth() {
  $("authPage").classList.remove("hidden");
  $("appPage").classList.add("hidden");
}

function _showApp() {
  $("authPage").classList.add("hidden");
  $("appPage").classList.remove("hidden");
  $("userNameBadge").textContent = currentUser?.name || "User";
  _loadHistory();
}

function switchTab(tab) {
  $("loginForm").classList.toggle("hidden",  tab !== "login");
  $("signupForm").classList.toggle("hidden", tab !== "signup");
  $("tabLogin").classList.toggle("active",   tab === "login");
  $("tabSignup").classList.toggle("active",  tab === "signup");
}


// ─────────────────────────────────────────────────────────────────────────────
// Auth — login
// ─────────────────────────────────────────────────────────────────────────────
async function submitLogin(e) {
  e.preventDefault();
  const btn = $("loginBtn");
  btn.disabled = true;
  btn.textContent = "Bitte warten…";
  $("loginError").classList.add("hidden");

  const fd = new FormData();
  fd.append("email",    $("loginEmail").value);
  fd.append("password", $("loginPassword").value);

  try {
    const res  = await fetch("/api/auth/login", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Fehler beim Anmelden");
    _persistAuth(data.token, data.user);
    _showApp();
  } catch (err) {
    _showAuthError("loginError", err.message);
  } finally {
    btn.disabled    = false;
    btn.textContent = "Anmelden";
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Auth — signup
// ─────────────────────────────────────────────────────────────────────────────
async function submitSignup(e) {
  e.preventDefault();
  const btn = $("signupBtn");
  btn.disabled = true;
  btn.textContent = "Bitte warten…";
  $("signupError").classList.add("hidden");

  const fd = new FormData();
  fd.append("name",     $("signupName").value);
  fd.append("email",    $("signupEmail").value);
  fd.append("password", $("signupPassword").value);

  try {
    const res  = await fetch("/api/auth/register", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Fehler bei der Registrierung");
    _persistAuth(data.token, data.user);
    _showApp();
  } catch (err) {
    _showAuthError("signupError", err.message);
  } finally {
    btn.disabled    = false;
    btn.textContent = "Konto erstellen";
  }
}

function logout() {
  authToken   = null;
  currentUser = null;
  history     = [];
  localStorage.removeItem("db_token");
  localStorage.removeItem("db_user");
  $("chatScroll").innerHTML = "";
  _showAuth();
}

function _persistAuth(token, user) {
  authToken   = token;
  currentUser = user;
  localStorage.setItem("db_token",   token);
  localStorage.setItem("db_user",    JSON.stringify(user));
}

function _showAuthError(elId, msg) {
  const el = $(elId);
  el.textContent = msg;
  el.classList.remove("hidden");
}

function _authHeaders() {
  return authToken ? { "Authorization": `Bearer ${authToken}` } : {};
}


// ─────────────────────────────────────────────────────────────────────────────
// History — load from DB on login
// ─────────────────────────────────────────────────────────────────────────────
async function _loadHistory() {
  if (!authToken) return;
  try {
    const res  = await fetch("/api/history", { headers: _authHeaders() });
    const data = await res.json();
    if (!res.ok || !data.messages?.length) {
      _addWelcomeBubble();
      return;
    }

    // Render past messages
    data.messages.forEach((m) => {
      if (m.role === "user")      addUserBubble(m.content_de);
      else                         addAIBubble(m.content_de, m.content_en);
      // Build LLM context from history
      history.push({ role: m.role, content: m.content_de });
    });
  } catch {
    _addWelcomeBubble();
  }
}

async function clearHistory() {
  if (!confirm("Möchtest du deinen gesamten Gesprächsverlauf löschen?")) return;
  await fetch("/api/history", { method: "DELETE", headers: _authHeaders() });
  history = [];
  $("chatScroll").innerHTML = "";
  _addWelcomeBubble();
}

function _addWelcomeBubble() {
  addAIBubble(
    `Hallo, ${currentUser?.name || ""}! Ich bin Buddy, dein KI-Deutsch-Freund 🎙 Wie kann ich dir heute helfen?`,
    `Hello, ${currentUser?.name || ""}! I'm Buddy, your AI German friend 🎙 How can I help you today?`
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// TEXT CHAT
// ─────────────────────────────────────────────────────────────────────────────
function handleTextKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
}

async function sendText() {
  const msg = $("textInput").value.trim();
  if (!msg || isProcessing) return;

  $("textInput").value  = "";
  $("sendBtn").disabled = true;
  isProcessing          = true;
  setGlobalStatus("processing");

  const typingId = addTypingIndicator();

  const fd = new FormData();
  fd.append("message", msg);
  fd.append("history", JSON.stringify(history));

  try {
    const res  = await fetch("/api/chat-text", { method: "POST", body: fd, headers: _authHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Serverfehler");

    removeTypingIndicator(typingId);
    addUserBubble(msg);
    addAIBubble(data.ai_text_de, data.ai_text_en);

    history.push({ role: "user",      content: msg             });
    history.push({ role: "assistant", content: data.ai_text_de });

    $("aiAudio").src = data.tts_audio_url;
    $("aiAudio").play().catch(() => {});
  } catch (err) {
    removeTypingIndicator(typingId);
    addSystemMessage("Fehler: " + err.message);
  } finally {
    isProcessing          = false;
    $("sendBtn").disabled = false;
    setGlobalStatus("idle");
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// MANUAL TAP-TO-TALK
// ─────────────────────────────────────────────────────────────────────────────
async function toggleRecording() {
  if (isProcessing || callActive) return;

  if (isRecording) { mediaRecorder.stop(); return; }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    addSystemMessage("Mikrofon-Zugriff verweigert. Öffne http://localhost:8000");
    return;
  }

  recordedChunks = [];
  mediaRecorder  = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    await sendToAPI(new Blob(recordedChunks, { type: "audio/webm" }));
  };

  mediaRecorder.start();
  isRecording = true;
  setMicState("recording");
}


// ─────────────────────────────────────────────────────────────────────────────
// LIVE CALL MODE
// ─────────────────────────────────────────────────────────────────────────────
async function toggleCall() {
  if (isProcessing) return;
  callActive ? endCall() : await startCall();
}

async function startCall() {
  try {
    callStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    addSystemMessage("Mikrofon-Zugriff verweigert. Öffne http://localhost:8000");
    return;
  }

  callActive    = true;
  callStartTime = Date.now();

  $("callBanner").classList.remove("hidden");
  callTimerInterval = setInterval(_updateCallTimer, 1000);

  $("micBtn").disabled    = true;
  $("micBtn").style.opacity = "0.35";
  $("callBtn").classList.add("active-call");
  $("callHint").textContent = "Auflegen";

  setGlobalStatus("call-listening");
  _startVAD();
}

function endCall() {
  callActive = false;
  _stopVAD();

  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  if (callStream) { callStream.getTracks().forEach((t) => t.stop()); callStream = null; }

  clearInterval(callTimerInterval);
  $("callBanner").classList.add("hidden");
  $("micBtn").disabled      = false;
  $("micBtn").style.opacity = "";
  $("callBtn").classList.remove("active-call");
  $("callHint").textContent = "Live-Anruf";

  setGlobalStatus("idle");
  setMicState("idle");
}

function _startCallRecording() {
  if (!callActive || !callStream) return;
  recordedChunks = [];
  speechStart    = Date.now();
  mediaRecorder  = new MediaRecorder(callStream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    if (Date.now() - speechStart < MIN_SPEECH_MS) { if (callActive) _resumeCallListening(); return; }
    await sendToAPI(new Blob(recordedChunks, { type: "audio/webm" }), true);
  };
  mediaRecorder.start();
}

function _resumeCallListening() {
  if (!callActive) return;
  vadSpeaking = false; vadSilenceStart = null; isProcessing = false;
  setGlobalStatus("call-listening");
  $("callStatus").textContent = "Warte auf Sprache…";
  setMicState("idle");
}


// ─────────────────────────────────────────────────────────────────────────────
// VAD
// ─────────────────────────────────────────────────────────────────────────────
function _startVAD() {
  vadAudioCtx = new AudioContext();
  vadAnalyser = vadAudioCtx.createAnalyser();
  vadAnalyser.fftSize = 256;
  vadAudioCtx.createMediaStreamSource(callStream).connect(vadAnalyser);

  const buf = new Uint8Array(vadAnalyser.frequencyBinCount);

  vadInterval = setInterval(() => {
    if (!callActive || isProcessing) return;
    vadAnalyser.getByteFrequencyData(buf);
    const vol       = buf.reduce((a, b) => a + b, 0) / buf.length;
    const speaking  = vol > SPEECH_THRESHOLD;

    if (speaking) {
      vadSilenceStart = null;
      if (!vadSpeaking) {
        vadSpeaking = true;
        $("callStatus").textContent = "Höre zu…";
        setMicState("recording");
        _startCallRecording();
      }
    } else if (vadSpeaking) {
      if (!vadSilenceStart) { vadSilenceStart = Date.now(); }
      else if (Date.now() - vadSilenceStart >= SILENCE_DURATION) {
        vadSpeaking = false; vadSilenceStart = null;
        if (mediaRecorder?.state === "recording") mediaRecorder.stop();
      }
    }
  }, 80);
}

function _stopVAD() {
  clearInterval(vadInterval); vadInterval = null;
  if (vadAudioCtx) { vadAudioCtx.close(); vadAudioCtx = null; }
  vadSpeaking = false; vadSilenceStart = null;
}


// ─────────────────────────────────────────────────────────────────────────────
// Shared API sender
// ─────────────────────────────────────────────────────────────────────────────
async function sendToAPI(blob, fromCall = false) {
  isRecording  = false;
  isProcessing = true;
  if (!fromCall) setMicState("processing");
  if (fromCall)  { $("callStatus").textContent = "Verarbeitung…"; setGlobalStatus("processing"); }

  const typingId = addTypingIndicator();
  const fd = new FormData();
  fd.append("audio",   blob, "recording.webm");
  fd.append("history", JSON.stringify(history));

  try {
    const res  = await fetch("/api/chat", { method: "POST", body: fd, headers: _authHeaders() });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Serverfehler");

    removeTypingIndicator(typingId);
    addUserBubble(data.user_text);
    addAIBubble(data.ai_text_de, data.ai_text_en);

    history.push({ role: "user",      content: data.user_text  });
    history.push({ role: "assistant", content: data.ai_text_de });

    $("aiAudio").src = data.tts_audio_url;
    if (fromCall) {
      $("callStatus").textContent = "Buddy spricht…";
      setGlobalStatus("ai-speaking");
      $("aiAudio").onended = () => { $("aiAudio").onended = null; _resumeCallListening(); };
    }
    $("aiAudio").play().catch(() => {});
  } catch (err) {
    removeTypingIndicator(typingId);
    addSystemMessage("Fehler: " + err.message);
    if (fromCall) _resumeCallListening();
  } finally {
    if (!fromCall) { isProcessing = false; setMicState("idle"); }
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// UI state helpers
// ─────────────────────────────────────────────────────────────────────────────
function setMicState(state) {
  $("micBtn").className = "ctrl-btn mic-btn";
  $("rings").className  = "rings";
  if (state === "recording") {
    $("micBtn").classList.add("recording");
    $("rings").classList.add("recording");
    $("micHint").textContent = callActive ? "Höre zu…" : "Stop";
  } else if (state === "processing") {
    $("micBtn").classList.add("processing");
    $("micHint").textContent = "Verarbeitung…";
  } else {
    $("micHint").textContent = callActive ? "Im Anruf" : "Sprechen";
  }
}

function setGlobalStatus(state) {
  $("statusDot").className = "status-dot";
  $("statusLabel").textContent = {
    "idle":           "Bereit",
    "call-listening": "Anruf aktiv",
    "processing":     "Denkt…",
    "ai-speaking":    "Buddy spricht",
  }[state] || "Bereit";
  if (state === "call-listening") $("statusDot").classList.add("call");
  else if (state === "processing") $("statusDot").classList.add("processing");
  else if (state === "ai-speaking") $("statusDot").classList.add("speaking");
}

function _updateCallTimer() {
  const s = Math.floor((Date.now() - callStartTime) / 1000);
  $("callTimer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}


// ─────────────────────────────────────────────────────────────────────────────
// Bubble helpers
// ─────────────────────────────────────────────────────────────────────────────
function addUserBubble(text) {
  const row = document.createElement("div");
  row.className = "bubble-row user-row";
  row.innerHTML = `
    <div class="avatar user-av">${_initials(currentUser?.name)}</div>
    <div class="bubble user-bubble"><p class="bubble-de">${_esc(text)}</p></div>`;
  $("chatScroll").appendChild(row);
  _scrollBottom();
}

function addAIBubble(german, english) {
  const row = document.createElement("div");
  row.className = "bubble-row ai-row";
  const en = english
    ? `<button class="en-toggle" onclick="toggleEN(this)">EN ▾</button>
       <p class="bubble-en hidden">${_esc(english)}</p>` : "";
  row.innerHTML = `
    <div class="avatar mascot-av"><img src="/static/mascot.jpg" alt="Buddy" /></div>
    <div class="bubble ai-bubble">
      <p class="bubble-de">${_esc(german)}</p>
      ${en}
    </div>`;
  $("chatScroll").appendChild(row);
  _scrollBottom();
}

function addSystemMessage(text) {
  const el = document.createElement("p");
  el.style.cssText = "text-align:center;color:var(--muted);font-size:.78rem;padding:.4rem";
  el.textContent = text;
  $("chatScroll").appendChild(el);
  _scrollBottom();
}

function addTypingIndicator() {
  const id  = "typing-" + Date.now();
  const row = document.createElement("div");
  row.id = id; row.className = "bubble-row ai-row";
  row.innerHTML = `
    <div class="avatar mascot-av"><img src="/static/mascot.jpg" alt="Buddy" /></div>
    <div class="bubble ai-bubble">
      <p class="bubble-de"><span class="dot"></span><span class="dot"></span><span class="dot"></span></p>
    </div>`;
  $("chatScroll").appendChild(row);
  _scrollBottom();
  return id;
}

function removeTypingIndicator(id) { const el = $(id); if (el) el.remove(); }

function toggleEN(btn) {
  const en = btn.nextElementSibling;
  if (!en) return;
  btn.textContent = en.classList.toggle("hidden") ? "EN ▾" : "EN ▴";
}

function _scrollBottom() { requestAnimationFrame(() => { $("chatScroll").scrollTop = $("chatScroll").scrollHeight; }); }

function _esc(str) {
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function _initials(name) {
  if (!name) return "Du";
  return name.split(" ").map((w) => w[0]).join("").toUpperCase().slice(0, 2);
}
