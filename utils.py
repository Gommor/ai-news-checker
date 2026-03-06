import importlib
import os
import re
import time
import warnings
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter('ignore', InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Small in-memory cache to avoid repeated scraping costs in the same process.
_SCRAPE_CACHE = {}
_CACHE_TTL_SECONDS = 300
_MAX_RETURN_CHARS = 3500


def _cache_get(key):
    item = _SCRAPE_CACHE.get(key)
    if not item:
        return None
    value, ts = item
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _SCRAPE_CACHE.pop(key, None)
        return None
    return value


def _cache_set(key, value):
    _SCRAPE_CACHE[key] = (value, time.time())


def extract_urls(text):
    """Metin icindeki URL'leri bulur."""
    raw_urls = re.findall(r'(https?://[^\s]+)', text or "")
    return [u.rstrip('.,!?;:)]}"\'') for u in raw_urls]


def _extract_tweet_id(url):
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower().replace("www.", "")
        if host not in {"twitter.com", "x.com", "mobile.twitter.com", "m.twitter.com"}:
            return None
        match = re.search(r"/(?:i/web/)?status/(\d+)", parsed.path or "")
        return match.group(1) if match else None
    except Exception:
        return None


def _tweet_id_to_utc_datetime(tweet_id):
    try:
        snowflake = int(tweet_id)
        epoch_ms = (snowflake >> 22) + 1288834974657
        return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    except Exception:
        return None


def _is_js_block_text(text):
    if not text:
        return False
    lowered = text.lower()
    block_markers = [
        "javascript is not available",
        "please enable javascript",
        "we've detected that javascript is disabled",
        "this browser is no longer supported",
    ]
    return any(marker in lowered for marker in block_markers)


def _safe_trim(text, limit=_MAX_RETURN_CHARS):
    return (text or "").strip()[:limit]


def _fetch_tweet_text_from_syndication(tweet_id, session):
    endpoints = [
        f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=tr",
        f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=en",
    ]
    for endpoint in endpoints:
        try:
            response = session.get(endpoint, timeout=7, verify=False)
            if not response.ok:
                continue
            data = response.json() or {}
            text = (data.get("text") or "").strip()
            if text:
                return text
        except Exception:
            pass
    return None


def _fetch_tweet_text_from_alt_domains(url, session):
    candidates = [
        url.replace("://x.com/", "://fixupx.com/").replace("://twitter.com/", "://fixupx.com/"),
        url.replace("://x.com/", "://vxtwitter.com/").replace("://twitter.com/", "://vxtwitter.com/"),
        url.replace("://x.com/", "://fxtwitter.com/").replace("://twitter.com/", "://fxtwitter.com/"),
    ]
    for alt in candidates:
        try:
            response = session.get(alt, timeout=8, allow_redirects=True, verify=False)
            if not response.ok:
                continue
            soup = BeautifulSoup(response.content, "html.parser")
            for prop in ("og:description", "twitter:description"):
                meta = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
                if meta and meta.get("content"):
                    text = meta.get("content").strip()
                    if text and len(text) > 8:
                        return text
        except Exception:
            pass
    return None


def _fetch_tweet_text_from_oembed(url, session):
    try:
        canonical_url = url.replace("://x.com/", "://twitter.com/")
        endpoint = f"https://publish.twitter.com/oembed?url={quote_plus(canonical_url)}&omit_script=1&dnt=true"
        response = session.get(endpoint, timeout=8, verify=False)
        if not response.ok:
            return None
        data = response.json() or {}
        html = (data.get("html") or "").strip()
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")
        p = soup.find("p")
        if p:
            text = p.get_text(" ", strip=True)
            if text and len(text) > 8:
                return text
    except Exception:
        return None
    return None


def _fetch_tweet_text_from_nitter(url, session):
    parsed = urlparse(url)
    path = (parsed.path or "").split("?")[0].rstrip("/")
    if "/status/" not in path:
        return None

    instances = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
    ]
    for base in instances:
        try:
            response = session.get(f"{base}{path}", timeout=8, verify=False)
            if not response.ok:
                continue
            soup = BeautifulSoup(response.content, "html.parser")
            node = soup.select_one(".main-tweet .tweet-content") or soup.select_one(".tweet-content")
            if node:
                text = node.get_text(" ", strip=True)
                if text and len(text) > 8:
                    return text
        except Exception:
            pass
    return None


def _fetch_tweet_text_with_playwright(url):
    """Slow fallback. Runs only when fast methods fail."""
    username = (os.getenv("X_USERNAME") or "").strip()
    password = (os.getenv("X_PASSWORD") or "").strip()
    storage_state_path = os.getenv("X_PLAYWRIGHT_STORAGE_STATE", "x_storage_state.json")
    handle = (os.getenv("X_HANDLE") or os.getenv("X_LOGIN_HANDLE") or "").strip()

    if not (os.path.exists(storage_state_path) or (username and password)):
        return None

    try:
        pw_sync_api = importlib.import_module("playwright.sync_api")
        sync_playwright = pw_sync_api.sync_playwright
        PlaywrightTimeoutError = pw_sync_api.TimeoutError
    except Exception:
        return None

    headless_env = os.getenv("X_PLAYWRIGHT_HEADLESS")
    if headless_env is None:
        headless = not (username and password and not os.path.exists(storage_state_path))
    else:
        headless = headless_env.strip() != "0"
    timeout_ms = int(os.getenv("X_PLAYWRIGHT_TIMEOUT_MS", "30000"))

    def _extract_from_page(page):
        selectors = [
            'article [data-testid="tweetText"]',
            'div[data-testid="tweetText"]',
            'article div[lang]',
        ]
        for selector in selectors:
            try:
                locator = page.locator(selector)
                if locator.count() == 0:
                    continue
                parts = []
                for i in range(min(locator.count(), 4)):
                    t = locator.nth(i).inner_text(timeout=4000).strip()
                    if t:
                        parts.append(t)
                text = " ".join(parts).strip()
                if text and not _is_js_block_text(text):
                    return text
            except Exception:
                pass
        return None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        try:
            context_kwargs = {}
            if os.path.exists(storage_state_path):
                context_kwargs["storage_state"] = storage_state_path
            context = browser.new_context(**context_kwargs)
            page = context.new_page()

            if (not os.path.exists(storage_state_path)) and username and password:
                try:
                    page.goto("https://x.com/i/flow/login", wait_until="domcontentloaded", timeout=timeout_ms)
                    page.locator('input[name="text"]').first.fill(username, timeout=10000)
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(1600)

                    if handle and page.locator('input[name="password"]').count() == 0 and page.locator('input[name="text"]').count() > 0:
                        page.locator('input[name="text"]').first.fill(handle, timeout=7000)
                        page.keyboard.press("Enter")
                        page.wait_for_timeout(1300)

                    page.locator('input[name="password"]').first.fill(password, timeout=12000)
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(2600)
                    context.storage_state(path=storage_state_path)
                except Exception:
                    pass

            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            text = _extract_from_page(page)
            if text:
                return text

            canonical_url = url.replace("://x.com/", "://twitter.com/")
            if canonical_url != url:
                page.goto(canonical_url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(1200)
                text = _extract_from_page(page)
                if text:
                    return text
        except PlaywrightTimeoutError:
            return None
        except Exception:
            return None
        finally:
            browser.close()

    return None


def _fetch_via_jina_reader(url, session):
    endpoints = []
    if url.startswith("https://"):
        endpoints.append("https://r.jina.ai/http://" + url[len("https://"):])
    elif url.startswith("http://"):
        endpoints.append("https://r.jina.ai/http://" + url[len("http://"):])
    else:
        endpoints.append("https://r.jina.ai/http://" + url)

    for endpoint in endpoints:
        try:
            response = session.get(endpoint, timeout=10, verify=False)
            if not response.ok:
                continue
            text = (response.text or "").strip()
            if text and len(text) > 20:
                return text
        except Exception:
            pass
    return None


def _decorate_tweet_text(text, tweet_id, source, created_at=None):
    tweet_dt = None
    if created_at:
        try:
            tweet_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            tweet_dt = None
    if not tweet_dt:
        tweet_dt = _tweet_id_to_utc_datetime(tweet_id)

    prefix = []
    if tweet_dt:
        prefix.append(f"[TWEET_UTC_TIME] {tweet_dt.strftime('%Y-%m-%d %H:%M:%S')} [/TWEET_UTC_TIME]")
    prefix.append(f"[TWEET_SOURCE] {source} [/TWEET_SOURCE]")
    return _safe_trim("\n".join(prefix + [text]))


def _scrape_tweet_url(url, tweet_id, session):
    attempts = []

    quick_sources = [
        ("syndication", lambda: _fetch_tweet_text_from_syndication(tweet_id, session)),
        ("alt_domains", lambda: _fetch_tweet_text_from_alt_domains(url, session)),
        ("oembed", lambda: _fetch_tweet_text_from_oembed(url, session)),
        ("nitter", lambda: _fetch_tweet_text_from_nitter(url, session)),
    ]

    for name, fn in quick_sources:
        text = None
        try:
            text = fn()
        except Exception:
            text = None
        attempts.append(f"{name}:ok" if text else f"{name}:miss")
        if text and not _is_js_block_text(text):
            return _decorate_tweet_text(text, tweet_id, name)

    pw_text = _fetch_tweet_text_with_playwright(url)
    attempts.append("playwright:ok" if pw_text else "playwright:miss")
    if pw_text and not _is_js_block_text(pw_text):
        return _decorate_tweet_text(pw_text, tweet_id, "playwright")

    jina_text = _fetch_via_jina_reader(url, session)
    attempts.append("jina:ok" if jina_text else "jina:miss")
    if jina_text and not _is_js_block_text(jina_text):
        return _decorate_tweet_text(jina_text, tweet_id, "jina")

    return "Tweet metni alinamadi. " + " | ".join(attempts)


def _scrape_regular_url(url, session):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            try:
                downloaded = trafilatura.fetch_url(url, timeout=10)
                if downloaded:
                    content = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        favor_precision=True,
                        output_format='txt',
                    )
                    if content and len(content.strip()) > 50 and not _is_js_block_text(content):
                        return _safe_trim(content)
            except Exception:
                pass

            response = session.get(url, timeout=10, allow_redirects=True, verify=False)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'

            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'noscript', 'meta', 'link', 'header', 'iframe']):
                element.decompose()

            content_div = None
            for selector in ['article', 'main', 'div[role="main"]', '.post', '.content']:
                try:
                    content_div = soup.select_one(selector)
                    if content_div:
                        break
                except Exception:
                    pass

            content = content_div.get_text(separator='\n', strip=True) if content_div else soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in content.split('\n') if line.strip() and len(line.strip()) > 3]
            clean_content = '\n'.join(lines[:80])
            if clean_content and len(clean_content) > 50:
                return _safe_trim(clean_content)

        except requests.Timeout:
            if attempt < max_retries - 1:
                time.sleep(0.6)
                continue
            return "Site gec cevap verdi. Lutfen linki tekrar deneyin."
        except requests.HTTPError as e:
            return f"Site hatasi: HTTP {e.response.status_code}"
        except requests.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(0.6)
                continue
            return "Baglanti hatasi. Interneti kontrol edin."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.6)
                continue
            return f"Link acilamadi: {str(e)[:80]}"

    return "Icerik cikarilamadi. Linki kopyalayip tarayicida acin."


def scrape_url(url):
    """Linkteki icerigi ceker ve teknik gurultuyu temizler."""
    cache_key = f"scrape::{url}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    session = requests.Session()
    session.headers.update(headers)

    tweet_id = _extract_tweet_id(url)
    if tweet_id:
        result = _scrape_tweet_url(url, tweet_id, session)
    else:
        result = _scrape_regular_url(url, session)

    _cache_set(cache_key, result)
    return result


def search_web(query, api_key):
    """SerpAPI kullanarak Google aramasi yapar."""
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}"
    try:
        response = requests.get(url, timeout=12).json()
        results = response.get("organic_results", [])
        return "\n".join([f"- {r.get('title')}: {r.get('snippet')} ({r.get('link')})" for r in results[:3]])
    except Exception:
        return "Arama motoruna ulasilamadi."
