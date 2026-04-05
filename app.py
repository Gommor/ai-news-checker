import os
import re
import time

import streamlit as st
from dotenv import load_dotenv

from agent_logic import VerificationAgent
from utils import extract_urls, scrape_url

# Load environment
load_dotenv()

st.set_page_config(page_title="DeepVerify 2026", layout="wide", page_icon="🛡️")

LANGUAGES = {
    "TR": {
        "title": "🛡️ DeepVerify AI",
        "subtitle": "Haberleri ve İddiaları Doğrulayın",
        "input_placeholder": "İddianızı yazın, link verin veya resim yükleyin...",
        "analyzing": "🕵️ Kanıtlar toplanıyor ve Gemini 2.5+ tarafından analiz ediliyor...",
        "files_uploaded": "📎 dosya yüklendi.",
        "result_true": "✅ SONUÇ: DOĞRU",
        "result_false": "❌ SONUÇ: YANLIŞ",
        "result_uncertain": "⚠️ SONUÇ: ŞÜPHELİ",
        "detailed_analysis": "📋 Detaylı Analiz İçeriği",
    },
    "EN": {
        "title": "🛡️ DeepVerify AI",
        "subtitle": "Verify News and Claims",
        "input_placeholder": "Type your claim, provide a link, or upload an image...",
        "analyzing": "🕵️ Gathering evidence and analyzing with Gemini 2.5+...",
        "files_uploaded": "📎 files uploaded.",
        "result_true": "✅ RESULT: TRUE",
        "result_false": "❌ RESULT: FALSE",
        "result_uncertain": "⚠️ RESULT: UNCERTAIN",
        "detailed_analysis": "📋 Detailed Analysis",
    },
}


if "messages" not in st.session_state:
    st.session_state.messages = []


def _parse_assistant_content(content: str):
    text = content or ""
    short_summary = ""
    detailed_part = ""

    if "[KISA OZET]" in text:
        try:
            short_summary = text.split("[KISA OZET]", 1)[1].split("[KISA OZET SONU]", 1)[0].strip()
        except Exception:
            short_summary = ""
        try:
            detailed_part = text.split("[DETAY]", 1)[1].split("[DETAY SONU]", 1)[0].strip()
        except Exception:
            detailed_part = ""
    elif "[SHORT SUMMARY]" in text:
        try:
            short_summary = text.split("[SHORT SUMMARY]", 1)[1].split("[SHORT SUMMARY END]", 1)[0].strip()
        except Exception:
            short_summary = ""
        try:
            detailed_part = text.split("[DETAILS]", 1)[1].split("[DETAILS END]", 1)[0].strip()
        except Exception:
            detailed_part = ""
    else:
        short_summary = text.strip()

    return short_summary, detailed_part


def _extract_short_fields(short_summary: str):
    karar = ""
    guven = ""
    kisaca = ""

    for line in (short_summary or "").split("\n"):
        line = line.strip()
        if line.startswith("KARAR:") or line.startswith("DECISION:"):
            karar = line
        elif line.startswith("GÜVEN SKORU:") or line.startswith("CONFIDENCE SCORE:"):
            guven = line
        elif line.startswith("KISACA:") or line.startswith("BRIEFLY:"):
            kisaca = line

    return karar, guven, kisaca


def _stream_markdown_text(text: str, placeholder, delay: float = 0.01, chunk_tokens: int = 10):
    raw = text or ""
    if not raw:
        placeholder.markdown("")
        return

    # Keep whitespace/newlines so markdown headings and sections stay intact while streaming.
    tokens = re.findall(r"\S+|\s+", raw)
    built = []
    for i, token in enumerate(tokens, start=1):
        built.append(token)
        if i % chunk_tokens == 0 or i == len(tokens):
            placeholder.markdown("".join(built))
            time.sleep(delay)


def _render_assistant(content: str, lang: dict, animate: bool = False):
    short_summary, detailed_part = _parse_assistant_content(content)

    st.divider()

    lowered = (short_summary or "").lower()
    if "doğru" in lowered or "true" in lowered:
        st.success(lang["result_true"])
    elif "yanlış" in lowered or "false" in lowered:
        st.error(lang["result_false"])
    elif "şüpheli" in lowered or "uncertain" in lowered:
        st.warning(lang["result_uncertain"])

    karar, guven, kisaca = _extract_short_fields(short_summary)

    if karar:
        st.markdown(f"**{karar}**")
    if guven:
        st.markdown(f"**{guven}**")
    if kisaca:
        if animate:
            ph = st.empty()
            _stream_markdown_text(f"*{kisaca}*", ph, delay=0.01, chunk_tokens=8)
        else:
            st.markdown(f"*{kisaca}*")

    if detailed_part:
        with st.expander(lang["detailed_analysis"]):
            if animate:
                ph = st.empty()
                _stream_markdown_text(detailed_part, ph, delay=0.005, chunk_tokens=14)
            else:
                st.markdown(detailed_part)

    st.divider()


def _sanitize_for_context(text: str):
    t = re.sub(r"\s+", " ", (text or "")).strip()
    t = re.sub(r"^[^\wÇĞİÖŞÜçğıöşü]+", "", t)
    return t


def _build_conversation_context(messages, max_items: int = 6):
    if not messages:
        return ""

    rows = []
    for m in messages[-max_items:]:
        role = m.get("role", "")
        raw = m.get("content", "")

        if role == "assistant":
            short_summary, _ = _parse_assistant_content(raw)
            raw = short_summary or raw
            tag = "ASSISTANT"
        elif role == "user":
            tag = "USER"
        else:
            continue

        cleaned = _sanitize_for_context(raw)
        if not cleaned:
            continue
        rows.append(f"{tag}: {cleaned[:420]}")

    return "\n".join(rows)


gemini_key = os.getenv("GEMINI_API_KEY")
serp_key = os.getenv("SERP_API_KEY")

if not gemini_key or not serp_key:
    st.error("⚠️ .env dosyanızı kontrol edin! GEMINI_API_KEY ve SERP_API_KEY gerekli.")
    st.stop()

st.sidebar.header("⚙️ Settings")
selected_language = st.sidebar.radio("Select Language / Dil Seç", ["TR", "EN"], horizontal=True)
lang = LANGUAGES[selected_language]

try:
    agent = VerificationAgent(gemini_key, serp_key, selected_language)
except Exception as e:
    st.error(f"❌ Agent başlatma hatası: {str(e)}")
    st.stop()

st.title(lang["title"])
st.write(lang["subtitle"])
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_assistant(message.get("content", ""), lang, animate=False)
        else:
            st.markdown(message.get("content", ""))
            if "files_info" in message:
                st.caption(message["files_info"])

chat_input = st.chat_input(lang["input_placeholder"], accept_file=True)

if chat_input:
    user_message = "🔍 " + chat_input.text if chat_input.text else "📎 " + ("Dosya yüklendi" if selected_language == "TR" else "File uploaded")
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.markdown(user_message)

    # Build conversational context from previous turns (excluding the just-added user line).
    conversation_context = _build_conversation_context(st.session_state.messages[:-1], max_items=8)

    with st.spinner(lang["analyzing"]):
        urls = extract_urls(chat_input.text)
        link_data = scrape_url(urls[0]) if urls else ""

        result = agent.plan_and_verify(
            chat_input.text,
            chat_input.files,
            link_data,
            conversation_context=conversation_context,
        )

    st.session_state.messages.append({"role": "assistant", "content": result})

    with st.chat_message("assistant"):
        _render_assistant(result, lang, animate=True)
