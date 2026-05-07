"""DeepVerify Pro evidence layer.

This module does not replace agent_logic.py. It adds optional, production-oriented
signal collection before the original Gemini + SerpAPI RAG agent runs:
- NewsAPI enrichment when NEWS_API_KEY exists
- X/Twitter public-signal search through SerpAPI query operators
- lightweight fake-news pattern risk scoring
- source-domain reliability weighting
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests

TRUSTED_DOMAINS = {
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "bbc.com": 0.95,
    "bbc.co.uk": 0.95,
    "aa.com.tr": 0.92,
    "trthaber.com": 0.86,
    "dw.com": 0.84,
    "euronews.com": 0.82,
    "theguardian.com": 0.82,
    "nytimes.com": 0.82,
    "hurriyet.com.tr": 0.72,
    "sozcu.com.tr": 0.72,
    "milliyet.com.tr": 0.68,
    "haberturk.com": 0.70,
    "cnnturk.com": 0.70,
    "wikipedia.org": 0.45,
    "x.com": 0.35,
    "twitter.com": 0.35,
}

PATTERNS_PATH = Path(__file__).resolve().parent / "fake_news_patterns.json"


@dataclass
class EvidenceItem:
    title: str
    snippet: str
    link: str
    source_type: str
    weight: float


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host
    except Exception:
        return ""


def source_weight(url: str) -> float:
    host = domain_of(url)
    for domain, weight in TRUSTED_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return weight
    if any(x in host for x in ["blog", "forum", "wordpress", "medium"]):
        return 0.45
    return 0.60


def _serpapi_search(query: str, serp_key: str, limit: int = 4) -> list[EvidenceItem]:
    if not serp_key:
        return []
    try:
        resp = requests.get(
            "https://serpapi.com/search.json",
            params={"q": query, "api_key": serp_key, "num": limit, "hl": "tr", "gl": "tr"},
            timeout=12,
        )
        data = resp.json()
        items = []
        for r in (data.get("organic_results") or [])[:limit]:
            link = r.get("link") or ""
            if not link:
                continue
            items.append(EvidenceItem(
                title=(r.get("title") or "").strip(),
                snippet=(r.get("snippet") or "").strip(),
                link=link,
                source_type="serpapi_google",
                weight=source_weight(link),
            ))
        return items
    except Exception:
        return []


def collect_x_signals(claim: str, serp_key: str, limit: int = 3) -> list[EvidenceItem]:
    # Real-time X scraping is fragile/blocked; this uses Google/SerpAPI public index
    # to surface X/Twitter posts without dummy data.
    q = f'{claim} (site:x.com OR site:twitter.com)'
    items = _serpapi_search(q, serp_key, limit=limit)
    for item in items:
        item.source_type = "x_public_index"
        item.weight = min(item.weight, 0.40)
    return items


def collect_newsapi(claim: str, limit: int = 5) -> list[EvidenceItem]:
    key = os.getenv("NEWS_API_KEY", "").strip()
    if not key:
        return []
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": claim,
                "language": "tr",
                "sortBy": "publishedAt",
                "pageSize": limit,
                "apiKey": key,
            },
            timeout=12,
        )
        data = resp.json()
        items = []
        for a in (data.get("articles") or [])[:limit]:
            link = a.get("url") or ""
            if not link:
                continue
            items.append(EvidenceItem(
                title=(a.get("title") or "").strip(),
                snippet=(a.get("description") or "").strip(),
                link=link,
                source_type="newsapi_live_news",
                weight=max(source_weight(link), 0.70),
            ))
        return items
    except Exception:
        return []


def fake_news_risk_score(text: str) -> dict:
    value = (text or "").lower()
    try:
        patterns = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        patterns = []
    hits = []
    score = 0
    for p in patterns:
        pattern = p.get("pattern", "")
        if pattern and re.search(pattern, value, flags=re.IGNORECASE):
            hits.append(p.get("label", pattern))
            score += int(p.get("weight", 10))
    score = max(0, min(100, score))
    return {"risk": score, "hits": hits[:8]}


def build_pro_context(claim: str, serp_key: str) -> str:
    claim = (claim or "").strip()
    if not claim:
        return ""

    news = collect_newsapi(claim)
    x_items = collect_x_signals(claim, serp_key)
    risk = fake_news_risk_score(claim)
    combined = news + x_items

    lines = [
        "[DEEPVERIFY_PRO_EVIDENCE_LAYER]",
        "Bu ek katman, ana agent_logic.py algoritmasını değiştirmeden ona ek güven sinyalleri sağlar.",
        "Karar verirken kaynak kalitesi, güncellik, çelişki ve iddiadaki sansasyonel dil birlikte değerlendirilmelidir.",
        f"Fake-news pattern risk score: {risk['risk']}/100",
    ]
    if risk["hits"]:
        lines.append("Risk pattern hits: " + ", ".join(risk["hits"]))

    if combined:
        lines.append("Ek canlı kaynak sinyalleri:")
        for idx, item in enumerate(combined[:8], 1):
            lines.append(
                f"{idx}. [{item.source_type}] weight={item.weight:.2f} | {item.title} | {item.snippet} | {item.link}"
            )
    else:
        lines.append("Ek NewsAPI/X sinyali bulunamadı veya ilgili API anahtarı tanımlı değil.")
    lines.append("[/DEEPVERIFY_PRO_EVIDENCE_LAYER]")
    return "\n".join(lines)


def recalibrate_confidence(parsed: dict) -> dict:
    """Conservative post-processing: prevent high confidence with weak sources."""
    sources = parsed.get("kaynaklar") or []
    raw_score = parsed.get("guven_skoru") or "%0"
    m = re.search(r"(\d{1,3})", raw_score)
    score = int(m.group(1)) if m else 0
    weights = [source_weight(s) for s in sources]
    avg_weight = sum(weights) / len(weights) if weights else 0
    if not sources:
        score = min(score, 35)
    elif len(sources) < 2:
        score = min(score, 65)
    elif avg_weight < 0.55:
        score = min(score, 70)
    parsed["guven_skoru"] = f"%{max(0, min(100, score))}"
    parsed["kaynak_kalite_ortalamasi"] = round(avg_weight, 2)
    parsed["pro_pipeline"] = {
        "newsapi_enabled": bool(os.getenv("NEWS_API_KEY", "").strip()),
        "x_public_index_enabled": True,
        "dataset_assisted_scoring": True,
        "source_weighting": True,
        "confidence_recalibration": True,
    }
    return parsed
