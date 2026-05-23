import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "https://ai-news-checker-production.up.railway.app";

const UI = {
  TR: {
    appName: "DeepVerify AI",
    subtitle: "Canlı web, haber ve sosyal medya sinyalleriyle kanıt odaklı doğrulama.",
    newChat: "Yeni sohbet",
    search: "Sohbetlerde ara",
    history: "Geçmiş kayıtlar",
    login: "Giriş yap",
    register: "Kayıt ol",
    logout: "Oturumu kapat",
    profile: "Profil",
    settings: "Ayarlar",
    personalize: "Kişiselleştirme",
    help: "Yardım",
    email: "E-posta",
    password: "Şifre",
    name: "Ad Soyad",
    guest: "Misafir mod",
    guestNote: "Geçmiş kayıtlar için hesap aç veya giriş yap.",
    input: "İddianızı yazın, link verin veya resim yükleyin...",
    loading: "Kanıtlar en güvenli şekilde toplanıyor...",
    attach: "Dosya / görsel yükle",
    deploy: "Deploy",
    theme: "Tema",
    dark: "Karanlık",
    light: "Aydınlık",
    language: "Dil",
    clearLocal: "Ekranı temizle",
    pin: "Sohbeti sabitle",
    archive: "Arşivle",
    delete: "Sil",
    files: "Sohbetteki dosyaları görüntüle",
    group: "Grup sohbeti başlat",
    decision: "Karar",
    confidence: "Güven skoru",
    summary: "Kısa özet",
    detail: "Detaylı analiz",
    openDetail: "Aç / kapat",
    sources: "Kaynaklar",
    noSource: "Kaynak bulunamadı.",
    confidenceNote: "Bu skor, kaynaklara göre verilen kararın güven oranıdır.",
    missing: "Lütfen iddia yazın veya görsel/dosya yükleyin.",
    emptyTitle: "DeepVerify AI",
    emptyHint: "Bir iddia, haber linki veya görsel gönder. Sistem SerpAPI + Gemini + opsiyonel NewsAPI/X sinyalleriyle canlı kanıt toplar.",
    deployTitle: "Canlıya alma durumu",
    deployHint: "Frontend Vercel/Netlify, backend Render/Railway üzerinde yayınlanabilir. Bu ekran ortam değişkenlerini ve API bağlantısını kontrol eder.",
    backendOk: "Backend çalışıyor",
    backendFail: "Backend bağlantısı yok",
    envOk: "tanımlı",
    envMissing: "eksik",
    openVercel: "Vercel'i aç",
    openRender: "Render'ı aç",
    copyEnv: ".env örneğini kopyala",
    close: "Kapat",
  },
  EN: {
    appName: "DeepVerify AI",
    subtitle: "Evidence-focused verification with live web, news and social signals.",
    newChat: "New chat",
    search: "Search chats",
    history: "History",
    login: "Login",
    register: "Sign up",
    logout: "Log out",
    profile: "Profile",
    settings: "Settings",
    personalize: "Personalization",
    help: "Help",
    email: "Email",
    password: "Password",
    name: "Full name",
    guest: "Guest mode",
    guestNote: "Create an account or log in to save history.",
    input: "Write a claim, paste a link or upload an image...",
    loading: "Gathering evidence securely...",
    attach: "Upload file / image",
    deploy: "Deploy",
    theme: "Theme",
    dark: "Dark",
    light: "Light",
    language: "Language",
    clearLocal: "Clear screen",
    pin: "Pin chat",
    archive: "Archive",
    delete: "Delete",
    files: "View chat files",
    group: "Start group chat",
    decision: "Decision",
    confidence: "Confidence score",
    summary: "Short summary",
    detail: "Detailed analysis",
    openDetail: "Open / close",
    sources: "Sources",
    noSource: "No source found.",
    confidenceNote: "This score is the confidence level of the decision based on sources.",
    missing: "Please write a claim or upload an image/file.",
    emptyTitle: "DeepVerify AI",
    emptyHint: "Send a claim, news link or image. The system gathers live evidence with SerpAPI + Gemini + optional NewsAPI/X signals.",
    deployTitle: "Deployment status",
    deployHint: "Frontend can be deployed on Vercel/Netlify, backend on Render/Railway. This panel checks environment variables and API health.",
    backendOk: "Backend is running",
    backendFail: "Backend is unreachable",
    envOk: "set",
    envMissing: "missing",
    openVercel: "Open Vercel",
    openRender: "Open Render",
    copyEnv: "Copy .env sample",
    close: "Close",
  },
};

function normalizeDecision(decision = "") {
  const lower = decision.toLocaleLowerCase("tr-TR");
  if (lower.includes("yanıt") || lower.includes("yanit") || lower.includes("answered")) return "answered";
  if (lower.includes("doğru") || lower.includes("true")) return "true";
  if (lower.includes("yanlış") || lower.includes("false")) return "false";
  return "uncertain";
}

function scoreValue(score = "") {
  const match = String(score).match(/(\d{1,3})/);
  if (!match) return 0;
  return Math.max(0, Math.min(100, Number(match[1])));
}

function host(url) {
  try { return new URL(url).hostname.replace("www.", ""); } catch { return url; }
}

function cleanAnalysisText(text = "") {
  return String(text)
    .replace(/^\s*(DETAYLI ANALİZ|DETAILED ANALYSIS):?\s*/i, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function renderAnalysisText(text = "") {
  const lines = cleanAnalysisText(text).split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const blocks = [];
  let paragraph = [];
  const bulletPrefix = /^[-*•]\s+/;
  const numberedPrefix = /^\d+[.)]\s+/;
  const sourceLead = /^(?:(?:kaynak|source|kanıt|kanit|evidence)\s+\d+\s*[:.)-]?|bu\s+(?:kanıt|kanit)(?:\s+da)?\b|this\s+evidence\b)/i;
  const sourceLabel = /^((?:kaynak|source|kanıt|kanit|evidence)\s+\d+)\s*[:.)-]?\s*(.*)$/i;

  const bulletContent = (line) => {
    const cleaned = line.replace(bulletPrefix, "").replace(numberedPrefix, "");
    const match = cleaned.match(sourceLabel);
    if (!match) return cleaned;
    return (
      <>
        <strong>{match[1]}:</strong> {match[2]}
      </>
    );
  };

  const flush = () => {
    if (!paragraph.length) return;
    blocks.push(<p key={`p-${blocks.length}`}>{paragraph.join(" ")}</p>);
    paragraph = [];
  };

  lines.forEach((line) => {
    if (/^#{1,4}\s+/.test(line)) {
      flush();
      blocks.push(<h4 key={`h-${blocks.length}`}>{line.replace(/^#{1,4}\s+/, "")}</h4>);
      return;
    }

    if (bulletPrefix.test(line) || numberedPrefix.test(line) || sourceLead.test(line)) {
      flush();
      blocks.push(
        <div className="analysis-bullet" key={`b-${blocks.length}`}>
          <span />
          <p>{bulletContent(line)}</p>
        </div>
      );
      return;
    }

    paragraph.push(line);
  });

  flush();
  return blocks.length ? blocks : <p>{text}</p>;
}

async function api(path, options = {}, token) {
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "İşlem başarısız.");
  return data;
}

function DeployModal({ t, onClose }) {
  const [status, setStatus] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let alive = true;
    api("/health")
      .then((data) => alive && setStatus({ ok: true, data }))
      .catch((err) => alive && setStatus({ ok: false, error: err.message }));
    return () => { alive = false; };
  }, []);

  const envText = `GEMINI_API_KEY=YOUR_GEMINI_KEY\nSERP_API_KEY=YOUR_SERPAPI_KEY\nNEWS_API_KEY=OPTIONAL_NEWSAPI_KEY\nGEMINI_MODEL=gemini-2.5-flash\nGEMINI_EMBEDDING_MODEL=text-embedding-004\nDETAIL_LEVEL=detailed\nENABLE_GEMINI_EMBEDDING=1`;

  const copyEnv = async () => {
    await navigator.clipboard.writeText(envText);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <section className="deploy-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{t.deployTitle}</h2>
          <button onClick={onClose}>×</button>
        </div>
        <p className="muted-text">{t.deployHint}</p>
        <div className="deploy-status">
          <div className={status?.ok ? "ok" : "bad"}>{status ? (status.ok ? `✅ ${t.backendOk}` : `❌ ${t.backendFail}`) : "Kontrol ediliyor..."}</div>
          {status?.ok && (
            <>
              <div>Gemini API: <b>{status.data.gemini_key ? t.envOk : t.envMissing}</b></div>
              <div>SerpAPI: <b>{status.data.serp_key ? t.envOk : t.envMissing}</b></div>
              <div>NewsAPI: <b>{status.data.newsapi_key ? t.envOk : t.envMissing}</b></div>
              <div>Database: <b>{status.data.database}</b></div>
            </>
          )}
        </div>
        <div className="deploy-actions">
          <button onClick={() => window.open("https://vercel.com/new", "_blank")}>{t.openVercel}</button>
          <button onClick={() => window.open("https://render.com", "_blank")}>{t.openRender}</button>
          <button onClick={copyEnv}>{copied ? "Kopyalandı" : t.copyEnv}</button>
        </div>
      </section>
    </div>
  );
}

function ResultMessage({ result, t }) {
  const decisionType = useMemo(() => normalizeDecision(result?.karar), [result]);
  const score = useMemo(() => scoreValue(result?.guven_skoru), [result]);
  return (
    <article className="message ai-message">
      <div className="message-body result-body report-result">
        <div className="result-header">
          <div className="decision-block">
            <span>{t.decision}</span>
            <strong className={`decision-pill ${decisionType}`}>{result.karar}</strong>
          </div>
          <div className="score-block">
            <span>{t.confidence}</span>
            <strong>{result.guven_skoru}</strong>
            <div className="score-track"><i style={{ width: `${score}%` }} /></div>
          </div>
          {result?.pro_pipeline && <span className="pro-pill">PRO</span>}
        </div>

        <section className="answer-section">
          <h3>{t.summary}</h3>
          <p>{result.kisa_ozet}</p>
        </section>

        <details className="detail-disclosure">
          <summary>
            <span>{t.detail}</span>
            <small>{t.openDetail}</small>
          </summary>
          <div className="analysis-text">{renderAnalysisText(result.detayli_analiz)}</div>
        </details>

        <section className="answer-section">
          <h3>{t.sources}</h3>
          {result.kaynaklar?.length ? (
            <div className="source-list plain-sources">
              {result.kaynaklar.map((src, i) => (
                <a key={i} href={src} target="_blank" rel="noreferrer">
                  <b>{i + 1}</b>
                  <span>{host(src)}</span>
                  <em>{src}</em>
                </a>
              ))}
            </div>
          ) : <p>{t.noSource}</p>}
        </section>
      </div>
    </article>
  );
}

function AuthModal({ t, onClose, onAuth }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault(); setError(""); setLoading(true);
    try {
      const path = mode === "login" ? "/auth/login" : "/auth/register";
      const body = mode === "login" ? { email, password } : { name, email, password };
      const data = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      onAuth(data);
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  };

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="auth-modal" onMouseDown={(e) => e.stopPropagation()} onSubmit={submit}>
        <div className="auth-tabs">
          <button type="button" className={mode === "login" ? "active" : ""} onClick={() => setMode("login")}>{t.login}</button>
          <button type="button" className={mode === "register" ? "active" : ""} onClick={() => setMode("register")}>{t.register}</button>
        </div>
        {mode === "register" && <input value={name} onChange={(e) => setName(e.target.value)} placeholder={t.name} required />}
        <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder={t.email} type="email" required />
        <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder={t.password} type="password" required />
        {error && <div className="mini-error">{error}</div>}
        <button className="primary-auth" disabled={loading}>{loading ? "..." : (mode === "login" ? t.login : t.register)}</button>
      </form>
    </div>
  );
}

function App() {
  const [language, setLanguage] = useState(localStorage.getItem("dv_lang") || "TR");
  const [theme, setTheme] = useState(localStorage.getItem("dv_theme") || "dark");
  const [token, setToken] = useState(localStorage.getItem("dv_token") || "");
  const [user, setUser] = useState(JSON.parse(localStorage.getItem("dv_user") || "null"));
  const [authOpen, setAuthOpen] = useState(false);
  const [deployOpen, setDeployOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [topMenuOpen, setTopMenuOpen] = useState(false);
  const fileRef = useRef(null);
  const t = UI[language];

  useEffect(() => { localStorage.setItem("dv_lang", language); }, [language]);
  useEffect(() => { localStorage.setItem("dv_theme", theme); }, [theme]);

  const logout = () => {
    setToken(""); setUser(null); setChats([]); setActiveChatId(null); setMessages([]); setTopMenuOpen(false);
    localStorage.removeItem("dv_token"); localStorage.removeItem("dv_user");
  };

  const loadChats = async () => {
    if (!token) return;
    try { setChats(await api("/chats", {}, token)); } catch { logout(); }
  };

  useEffect(() => { loadChats(); }, [token]);

  const onAuth = (data) => {
    setToken(data.token); setUser(data.user);
    localStorage.setItem("dv_token", data.token);
    localStorage.setItem("dv_user", JSON.stringify(data.user));
    setAuthOpen(false);
  };

  const newChat = () => { setActiveChatId(null); setMessages([]); setText(""); setFile(null); };

  const openChat = async (id) => {
    if (!token) return;
    try {
      const data = await api(`/chats/${id}`, {}, token);
      setActiveChatId(id);
      const mapped = [];
      for (const msg of data.messages) {
        if (msg.role === "user") mapped.push({ type: "user", text: msg.content });
        if (msg.role === "assistant" && msg.result) mapped.push({ type: "result", result: msg.result });
      }
      setMessages(mapped);
    } catch (err) { setMessages([{ type: "error", text: err.message }]); }
  };

  const deleteChat = async (id, e) => {
    e.stopPropagation();
    if (!token) return;
    await api(`/chats/${id}`, { method: "DELETE" }, token);
    if (activeChatId === id) newChat();
    loadChats();
  };

  const analyze = async () => {
    if (!text.trim() && !file) {
      setMessages((prev) => [...prev, { type: "error", text: t.missing }]);
      return;
    }
    const userText = text.trim() || (language === "TR" ? "Yüklenen dosya/görseli analiz et." : "Analyze the uploaded file/image.");
    const userFile = file;
    setMessages((prev) => [...prev, { type: "user", text: userText, fileName: userFile?.name }]);
    setText(""); setFile(null); setLoading(true);
    try {
      let data; const authToken = token || undefined;
      if (userFile) {
        const form = new FormData();
        form.append("text", userText);
        form.append("language", language);
        form.append("detail_level", "detailed");
        if (activeChatId) form.append("chat_id", activeChatId);
        form.append("file", userFile);
        data = await api("/analyze-file", { method: "POST", body: form }, authToken);
      } else {
        data = await api("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: userText, language, detail_level: "detailed", chat_id: activeChatId })
        }, authToken);
      }
      const finalResult = data.analysis ? data.analysis : data;
      setMessages((prev) => [...prev, { type: "result", result: data }]);
      if (data.chat_id) { setActiveChatId(data.chat_id); loadChats(); }
    } catch (err) {
      setMessages((prev) => [...prev, { type: "error", text: err.message || "Backend connection error." }]);
    } finally { setLoading(false); }
  };

  const filteredChats = chats.filter((c) => c.title.toLocaleLowerCase("tr-TR").includes(query.toLocaleLowerCase("tr-TR")));

  return (
    <div className={`app-shell ${theme}`}>
      <aside className="leftnav">
        <div className="brand-mini"><span className="brand-shield">🛡️</span><span>DeepVerify</span></div>
        <button className="new-chat" onClick={newChat}>＋ {t.newChat}</button>
        <label className="search-box"><span>⌕</span><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t.search} /></label>
        <div className="history-title">{t.history}</div>
        <div className="history-list">
          {!token && <div className="guest-card"><b>{t.guest}</b><span>{t.guestNote}</span></div>}
          {filteredChats.map((chat) => (
            <button key={chat.id} className={`chat-row ${activeChatId === chat.id ? "active" : ""}`} onClick={() => openChat(chat.id)}>
              <p>{chat.title}</p><em onClick={(e) => deleteChat(chat.id, e)}>×</em>
            </button>
          ))}
        </div>
        <div className="account-area">
          {user ? (
            <div className="account-button static-account">
              <div className="avatar">{user.name?.[0]?.toUpperCase() || "B"}</div>
              <div><b>{user.name}</b><small>{user.email}</small></div>
            </div>
          ) : (
            <div className="auth-actions"><button onClick={() => setAuthOpen(true)}>{t.login}</button><button onClick={() => setAuthOpen(true)}>{t.register}</button></div>
          )}
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div />
          <div className="top-actions">
            <button className="lang-pill" onClick={() => setLanguage(language === "TR" ? "EN" : "TR")}>{language}</button>
            <button className="theme-pill" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "☀️" : "🌙"}</button>
            <button className="deploy-btn" onClick={() => setDeployOpen(true)}>{t.deploy}</button>
            <button className="icon-btn" onClick={() => setTopMenuOpen((v) => !v)}>⋯</button>
            {topMenuOpen && <div className="top-menu">
              <button>♙ {t.group}</button>
              <button>▥ {t.files}</button>
              <button>✧ {t.pin}</button>
              <button>▣ {t.archive}</button>
              <button onClick={() => setMessages([])}>⌫ {t.clearLocal}</button>
              {user && <button onClick={logout}>↪ {t.logout}</button>}
              <button className="danger">🗑 {t.delete}</button>
            </div>}
          </div>
        </header>

        <section className="hero-panel">
          <div className="logo-mark">🛡️</div>
          <div><h1>{t.appName}</h1><p>{t.subtitle}</p></div>
        </section>
        <div className="divider" />

        <section className="chat-area">
          {messages.length === 0 && <div className="empty-state"><h2>{t.emptyTitle}</h2><p>{t.emptyHint}</p></div>}
          {messages.map((m, idx) => {
            if (m.type === "user") return <article className="message user-message" key={idx}><div className="message-body user-bubble"><p>{m.text}</p>{m.fileName && <small>📎 {m.fileName}</small>}</div></article>;
            if (m.type === "error") return <div className="error-banner" key={idx}>{m.text}</div>;
            return <ResultMessage key={idx} result={m.result} t={t} />;
          })}
          {loading && <article className="message ai-message"><div className="message-body loading-bubble"><span className="loader" />{t.loading}</div></article>}
        </section>

        <section className="composer-wrap">
          {file && <div className="file-chip">📎 {file.name} <button onClick={() => setFile(null)}>×</button></div>}
          <div className="composer">
            <input ref={fileRef} type="file" accept="image/*,.pdf,.txt" hidden onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <button className="plus-btn" title={t.attach} onClick={() => fileRef.current?.click()}>＋</button>
            <textarea value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); analyze(); } }} placeholder={t.input} />
            <button className="send-btn" onClick={analyze} disabled={loading}>↑</button>
          </div>
        </section>
      </main>

      {authOpen && <AuthModal t={t} onClose={() => setAuthOpen(false)} onAuth={onAuth} />}
      {deployOpen && <DeployModal t={t} onClose={() => setDeployOpen(false)} />}
    </div>
  );
}

export default App;
