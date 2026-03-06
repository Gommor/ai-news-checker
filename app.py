import os
import re

import streamlit as st
from dotenv import load_dotenv

from agent_logic import VerificationAgent
from utils import extract_urls, scrape_url

# Çevresel değişkenleri yükle
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
            content = message["content"]
            short_summary = ""
            detailed_part = ""

            if "[KISA OZET]" in content or "[SHORT SUMMARY]" in content:
                if "[KISA OZET]" in content:
                    short_summary = content.split("[KISA OZET]")[1].split("[KISA OZET SONU]")[0].strip()
                    detailed_part = content.split("[DETAY]")[1].split("[DETAY SONU]")[0].strip() if "[DETAY]" in content else ""
                else:
                    short_summary = content.split("[SHORT SUMMARY]")[1].split("[SHORT SUMMARY END]")[0].strip()
                    detailed_part = content.split("[DETAILS]")[1].split("[DETAILS END]")[0].strip() if "[DETAILS]" in content else ""
            else:
                short_summary = content
                detailed_part = ""

            st.divider()

            if "Doğru" in short_summary or "True" in short_summary:
                st.success(lang["result_true"])
            elif "Yanlış" in short_summary or "False" in short_summary:
                st.error(lang["result_false"])
            elif "Şüpheli" in short_summary or "Uncertain" in short_summary:
                st.warning(lang["result_uncertain"])

            karar = ""
            guven = ""
            kisaca = ""

            lines = short_summary.split("\n")
            for line in lines:
                if line.startswith("KARAR:") or line.startswith("DECISION:"):
                    karar = line
                elif line.startswith("GÜVEN SKORU:") or line.startswith("CONFIDENCE SCORE:"):
                    guven = line
                elif line.startswith("KISACA:") or line.startswith("BRIEFLY:"):
                    kisaca = line

            if karar:
                st.markdown(f"**{karar}**")
            if guven:
                st.markdown(f"**{guven}**")
            if kisaca:
                st.markdown(f"*{kisaca}*")

            if detailed_part:
                with st.expander(lang["detailed_analysis"]):
                    st.markdown(detailed_part)

            st.divider()
        else:
            st.markdown(message["content"])
            if "files_info" in message:
                st.caption(message["files_info"])

chat_input = st.chat_input(lang["input_placeholder"], accept_file=True)

if chat_input:
    user_message = "🔍 " + chat_input.text if chat_input.text else "📎 " + ("Dosya yüklendi" if selected_language == "TR" else "File uploaded")
    st.session_state.messages.append({"role": "user", "content": user_message})

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.spinner(lang["analyzing"]):
        urls = extract_urls(chat_input.text)
        link_data = scrape_url(urls[0]) if urls else ""

        if urls:
            preview = re.sub(r"\[TWEET_UTC_TIME\].*?\[/TWEET_UTC_TIME\]\s*", "", link_data or "", flags=re.IGNORECASE | re.DOTALL)
            preview = re.sub(r"\[TWEET_SOURCE\].*?\[/TWEET_SOURCE\]\s*", "", preview, flags=re.IGNORECASE | re.DOTALL)
            st.info(
                f"🔗 Link bulundu: {urls[0][:50]}...\n📄 İçerik: {preview[:220]}..."
                if len(preview) > 220
                else f"🔗 Link bulundu: {urls[0][:50]}...\n📄 İçerik: {preview}"
            )

        result = agent.plan_and_verify(chat_input.text, chat_input.files, link_data)
        st.session_state.messages.append({"role": "assistant", "content": result})

    st.rerun()
