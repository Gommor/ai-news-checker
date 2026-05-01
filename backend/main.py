import hashlib
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from agent_logic import VerificationAgent
from utils import extract_urls, scrape_url
from evidence_enhancer import build_pro_context, recalibrate_confidence

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "deepverify.db"

app = FastAPI(title="DeepVerify Pro API", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Database
# -----------------------------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                token TEXT UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            );
            """
        )


init_db()


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()


def get_user_from_token(authorization: Optional[str]) -> sqlite3.Row:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Oturum bulunamadı. Lütfen giriş yapın.")
    token = authorization.replace("Bearer ", "", 1).strip()
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()
    if not user:
        raise HTTPException(status_code=401, detail="Oturum geçersiz. Tekrar giriş yapın.")
    return user


def get_chat_or_404(chat_id: int, user_id: int) -> sqlite3.Row:
    with db() as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id),
        ).fetchone()
    if not chat:
        raise HTTPException(status_code=404, detail="Sohbet bulunamadı.")
    return chat


def make_title(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    return (clean[:42] + "…") if len(clean) > 42 else (clean or "Yeni analiz")


# -----------------------------
# Models
# -----------------------------
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


class ChatCreateRequest(BaseModel):
    title: Optional[str] = "Yeni sohbet"


class AnalyzeRequest(BaseModel):
    text: str
    language: Optional[str] = "TR"
    detail_level: Optional[str] = "normal"
    chat_id: Optional[int] = None


class AnalyzeResponse(BaseModel):
    karar: str
    guven_skoru: str
    guven_aciklamasi: str
    kisa_ozet: str
    detayli_analiz: str
    kaynaklar: list[str]
    kaynak_sayisi: int
    analiz_modu: str
    dil: str
    generated_at: str
    raw: str
    chat_id: Optional[int] = None


# -----------------------------
# Parsing helpers
# -----------------------------
def _extract_section(text: str, start_tag: str, end_tag: str) -> str:
    if start_tag in text and end_tag in text:
        return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return ""


def _score_to_int(score: str) -> int:
    found = re.search(r"(\d{1,3})", score or "")
    if not found:
        return 0
    return max(0, min(100, int(found.group(1))))


def _confidence_text(score: str, lang: str) -> str:
    value = _score_to_int(score)
    if lang == "EN":
        if value >= 80:
            return f"High confidence based on the gathered evidence ({value}%)."
        if value >= 50:
            return f"Medium confidence based on the gathered evidence ({value}%)."
        return f"Low confidence; evidence is limited or conflicting ({value}%)."
    if value >= 80:
        return f"Toplanan kanıtlara göre güven oranı yüksek (%{value})."
    if value >= 50:
        return f"Toplanan kanıtlara göre güven oranı orta düzeyde (%{value})."
    return f"Kanıtlar sınırlı veya çelişkili olduğu için güven oranı düşük (%{value})."


def _clean_sources(raw_sources: list[str]) -> list[str]:
    cleaned = []
    for source in raw_sources:
        s = (source or "").strip().rstrip(".,)]")
        if not s.startswith(("http://", "https://")):
            continue
        cleaned.append(s)
    return list(dict.fromkeys(cleaned))[:12]


def _parse_agent_output(raw: str, language: str = "TR", detail_level: str = "normal") -> dict:
    text = raw or ""
    lang = (language or "TR").upper()

    if lang == "EN" or "[SHORT SUMMARY]" in text:
        short = _extract_section(text, "[SHORT SUMMARY]", "[SHORT SUMMARY END]")
        detail = _extract_section(text, "[DETAILS]", "[DETAILS END]")
        decision_key = "DECISION:"
        confidence_key = "CONFIDENCE SCORE:"
        briefly_key = "BRIEFLY:"
        sources_key = "SOURCES:"
        fallback_decision = "Uncertain"
        fallback_summary = "Short summary could not be generated."
        fallback_detail = "Detailed analysis could not be generated."
    else:
        short = _extract_section(text, "[KISA OZET]", "[KISA OZET SONU]")
        detail = _extract_section(text, "[DETAY]", "[DETAY SONU]")
        decision_key = "KARAR:"
        confidence_key = "GÜVEN SKORU:"
        briefly_key = "KISACA:"
        sources_key = "KAYNAKLAR:"
        fallback_decision = "Şüpheli"
        fallback_summary = "Kısa özet üretilemedi."
        fallback_detail = "Detaylı analiz üretilemedi."

    karar = ""
    guven_skoru = ""
    kisa_ozet = ""

    for line in short.splitlines():
        clean = line.strip()
        if clean.startswith(decision_key):
            karar = clean.replace(decision_key, "", 1).strip()
        elif clean.startswith(confidence_key):
            guven_skoru = clean.replace(confidence_key, "", 1).strip()
        elif clean.startswith(briefly_key):
            kisa_ozet = clean.replace(briefly_key, "", 1).strip()

    detayli_analiz = detail.strip() or text.strip()
    kaynaklar = []

    if sources_key in detayli_analiz:
        before_sources, after_sources = detayli_analiz.split(sources_key, 1)
        detayli_analiz = before_sources.strip()
        for line in after_sources.splitlines():
            clean = line.strip()
            match = re.match(r"^\d+\.\s*(https?://\S+)", clean)
            if match:
                kaynaklar.append(match.group(1).strip())
            elif clean.startswith(("http://", "https://")):
                kaynaklar.append(clean)

    if not kaynaklar:
        kaynaklar = re.findall(r"https?://[^\s)\]]+", text)

    kaynaklar = _clean_sources(kaynaklar)
    guven_skoru = guven_skoru or "%0"

    return {
        "karar": karar or fallback_decision,
        "guven_skoru": guven_skoru,
        "guven_aciklamasi": _confidence_text(guven_skoru, lang),
        "kisa_ozet": kisa_ozet or short.strip() or fallback_summary,
        "detayli_analiz": detayli_analiz or fallback_detail,
        "kaynaklar": kaynaklar,
        "kaynak_sayisi": len(kaynaklar),
        "analiz_modu": detail_level,
        "dil": lang,
        "generated_at": now_iso(),
        "raw": raw,
    }


def run_agent(user_text: str, language: str, detail_level: str, uploaded_files=None) -> dict:
    selected_language = (language or "TR").upper()
    if selected_language not in {"TR", "EN"}:
        selected_language = "TR"

    # En güvenli veri toplama için hızlı mod kaldırıldı; backend her zaman detaylı/normal çalışır.
    detail_level = "normal"

    if not user_text and not uploaded_files:
        raise HTTPException(status_code=400, detail="Analiz için metin/iddia veya dosya girilmelidir.")

    gemini_key = os.getenv("GEMINI_API_KEY")
    serp_key = os.getenv("SERP_API_KEY")

    if not gemini_key or not serp_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY ve SERP_API_KEY .env dosyasında tanımlı olmalıdır.",
        )

    previous_detail_level = os.environ.get("DETAIL_LEVEL")
    os.environ["DETAIL_LEVEL"] = detail_level

    try:
        agent = VerificationAgent(gemini_key, serp_key, selected_language)
        urls = extract_urls(user_text)
        link_data = scrape_url(urls[0]) if urls else ""

        pro_context = build_pro_context(user_text, serp_key)
        result = agent.plan_and_verify(
            user_text,
            uploaded_files=uploaded_files,
            link_content=link_data,
            conversation_context=pro_context,
        )
        parsed = _parse_agent_output(result, selected_language, detail_level)
        return recalibrate_confidence(parsed)
    finally:
        if previous_detail_level is None:
            os.environ.pop("DETAIL_LEVEL", None)
        else:
            os.environ["DETAIL_LEVEL"] = previous_detail_level


def save_analysis(user_id: Optional[int], chat_id: Optional[int], text: str, result: dict) -> Optional[int]:
    if not user_id:
        return None
    stamp = now_iso()
    with db() as conn:
        if chat_id:
            get_chat_or_404(chat_id, user_id)
            final_chat_id = chat_id
            conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (stamp, final_chat_id))
        else:
            cur = conn.execute(
                "INSERT INTO chats (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, make_title(text), stamp, stamp),
            )
            final_chat_id = cur.lastrowid
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, result_json, created_at) VALUES (?, 'user', ?, NULL, ?)",
            (final_chat_id, text, stamp),
        )
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, result_json, created_at) VALUES (?, 'assistant', ?, ?, ?)",
            (final_chat_id, result.get("kisa_ozet", ""), __import__("json").dumps(result, ensure_ascii=False), stamp),
        )
        conn.commit()
    return final_chat_id


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "DeepVerify FastAPI backend çalışıyor.", "version": "8.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_key": bool(os.getenv("GEMINI_API_KEY")),
        "serp_key": bool(os.getenv("SERP_API_KEY")),
        "newsapi_key": bool(os.getenv("NEWS_API_KEY")),
        "pro_pipeline": True,
        "database": str(DB_PATH.name),
    }


@app.post("/auth/register", response_model=AuthResponse)
def register(payload: RegisterRequest):
    name = payload.name.strip()
    email = payload.email.lower().strip()
    password = payload.password
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="İsim en az 2 karakter olmalı.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Şifre en az 6 karakter olmalı.")

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    token = secrets.token_urlsafe(32)
    try:
        with db() as conn:
            cur = conn.execute(
                "INSERT INTO users (name, email, password_hash, salt, token, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (name, email, password_hash, salt, token, now_iso()),
            )
            user_id = cur.lastrowid
            conn.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=400, detail="Bu e-posta ile hesap zaten var.") from exc
    return {"token": token, "user": {"id": user_id, "name": name, "email": email}}


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    email = payload.email.lower().strip()
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı.")
        if hash_password(payload.password, user["salt"]) != user["password_hash"]:
            raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı.")
        token = secrets.token_urlsafe(32)
        conn.execute("UPDATE users SET token = ? WHERE id = ?", (token, user["id"]))
        conn.commit()
    return {"token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"]}}


@app.get("/me")
def me(authorization: Optional[str] = Header(None)):
    user = get_user_from_token(authorization)
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


@app.post("/chats")
def create_chat(payload: ChatCreateRequest, authorization: Optional[str] = Header(None)):
    user = get_user_from_token(authorization)
    stamp = now_iso()
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO chats (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user["id"], (payload.title or "Yeni sohbet").strip(), stamp, stamp),
        )
        conn.commit()
    return {"id": cur.lastrowid, "title": payload.title or "Yeni sohbet", "created_at": stamp, "updated_at": stamp}


@app.get("/chats")
def list_chats(authorization: Optional[str] = Header(None)):
    user = get_user_from_token(authorization)
    with db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50",
            (user["id"],),
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/chats/{chat_id}")
def get_chat(chat_id: int, authorization: Optional[str] = Header(None)):
    user = get_user_from_token(authorization)
    chat = get_chat_or_404(chat_id, user["id"])
    with db() as conn:
        rows = conn.execute(
            "SELECT id, role, content, result_json, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()
    import json
    messages = []
    for row in rows:
        item = dict(row)
        if item.get("result_json"):
            item["result"] = json.loads(item["result_json"])
        item.pop("result_json", None)
        messages.append(item)
    return {"chat": dict(chat), "messages": messages}


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: int, authorization: Optional[str] = Header(None)):
    user = get_user_from_token(authorization)
    get_chat_or_404(chat_id, user["id"])
    with db() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"]))
        conn.commit()
    return {"deleted": True}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, authorization: Optional[str] = Header(None)):
    user_text = (payload.text or "").strip()
    user = None
    if authorization:
        user = get_user_from_token(authorization)
    try:
        result = run_agent(user_text, payload.language or "TR", "normal")
        chat_id = save_analysis(user["id"] if user else None, payload.chat_id, user_text, result)
        result["chat_id"] = chat_id or payload.chat_id
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(exc)}") from exc


@app.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(
    text: str = Form(""),
    language: str = Form("TR"),
    detail_level: str = Form("normal"),
    chat_id: Optional[int] = Form(None),
    file: UploadFile | None = File(None),
    authorization: Optional[str] = Header(None),
):
    user_text = (text or "").strip()
    user = None
    if authorization:
        user = get_user_from_token(authorization)

    uploaded_files = None
    if file is not None:
        await file.seek(0)
        uploaded_files = [file.file]
        if not user_text:
            user_text = "Yüklenen dosya/görseli kanıtlara göre analiz et."

    try:
        result = run_agent(user_text, language, "normal", uploaded_files=uploaded_files)
        chat_id_saved = save_analysis(user["id"] if user else None, chat_id, user_text, result)
        result["chat_id"] = chat_id_saved or chat_id
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(exc)}") from exc
