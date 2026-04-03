import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import PIL.Image
from google import genai

from utils import search_web


class VerificationAgent:
    def __init__(self, gemini_key, serp_key, language="TR"):
        self.client = genai.Client(api_key=gemini_key)
        self.model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.embedding_model_id = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self.embedding_enabled = os.getenv("ENABLE_GEMINI_EMBEDDING", "1").strip() != "0"

        self.serp_key = serp_key
        self.language = (language or "TR").upper()

        self.max_link_chars = 2400

        self.detail_level = (os.getenv("DETAIL_LEVEL", "normal") or "normal").strip().lower()
        if self.detail_level not in {"fast", "normal"}:
            self.detail_level = "normal"

        if self.detail_level == "fast":
            default_max_queries = 3
            default_detail_min_chars = 700
            default_embedding_top_k = 4
            default_main_tokens = 1600
            default_url_only_tokens = 2000
            default_rewrite_tokens = 2000
            default_planner_tokens = 120
        else:
            default_max_queries = 4
            default_detail_min_chars = 1100
            default_embedding_top_k = 6
            default_main_tokens = 2200
            default_url_only_tokens = 2600
            default_rewrite_tokens = 2600
            default_planner_tokens = 180

        def _env_int(name, fallback):
            try:
                return int(os.getenv(name, str(fallback)))
            except Exception:
                return fallback

        self.max_queries = _env_int("MAX_QUERIES", default_max_queries)
        self.detail_min_chars = _env_int("DETAIL_MIN_CHARS", default_detail_min_chars)
        self.embedding_top_k = _env_int("EMBEDDING_TOP_K", default_embedding_top_k)

        self.url_only_max_output_tokens = _env_int("URL_ONLY_MAX_OUTPUT_TOKENS", default_url_only_tokens)
        self.rewrite_max_output_tokens = _env_int("REWRITE_MAX_OUTPUT_TOKENS", default_rewrite_tokens)
        self.planner_max_output_tokens = _env_int("PLANNER_MAX_OUTPUT_TOKENS", default_planner_tokens)

        self.generation_config = {
            "temperature": 0.2,
            "max_output_tokens": _env_int("MAIN_MAX_OUTPUT_TOKENS", default_main_tokens),
        }

        if self.detail_level == "normal":
            # Guardrails: keep normal mode truly detailed even if env values are too low.
            self.detail_min_chars = max(self.detail_min_chars, 900)
            self.generation_config["max_output_tokens"] = max(self.generation_config["max_output_tokens"], 2600)
            self.url_only_max_output_tokens = max(self.url_only_max_output_tokens, 3000)
            self.rewrite_max_output_tokens = max(self.rewrite_max_output_tokens, 3200)

        self._embedding_cache = {}

    def _response_text(self, response):
        text = getattr(response, "text", None)
        if text:
            return text
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                return ""
            parts = candidates[0].content.parts
            out = "".join([getattr(p, "text", "") for p in parts if getattr(p, "text", "")]).strip()
            return out
        except Exception:
            return ""

    def _generate_content(self, prompt, sys_instr, generation_config=None, extra_contents=None):
        contents = [prompt]
        if extra_contents:
            contents.extend(extra_contents)

        config = dict(generation_config or {})
        if sys_instr:
            config["system_instruction"] = sys_instr

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config if config else None,
        )
        return self._response_text(response)

    def _is_link_content_usable(self, link_content):
        text = (link_content or "").strip()
        if not text:
            return False

        lowered = text.lower()
        error_markers = [
            "site hatasi:",
            "baglanti hatasi",
            "link acilamadi",
            "icerik cikarilamadi",
            "site gec cevap verdi",
            "tweet metni alinamadi",
            "analysis error:",
        ]
        return not any(marker in lowered for marker in error_markers)

    def _clean_link_content(self, link_content):
        text = (link_content or "").strip()
        if not text:
            return text
        text = re.sub(r"\[TWEET_UTC_TIME\].*?\[/TWEET_UTC_TIME\]\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\[TWEET_SOURCE\].*?\[/TWEET_SOURCE\]\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    def _is_url_only_input(self, user_input):
        text = (user_input or "").strip()
        if not text:
            return False
        return re.fullmatch(r"https?://\S+", text) is not None

    def _contains_twitter_url(self, text):
        value = (text or "").strip()
        if not value:
            return False
        return re.search(
            r"https?://(?:www\.)?(?:x\.com|twitter\.com|mobile\.twitter\.com|m\.twitter\.com)/",
            value,
            flags=re.IGNORECASE,
        ) is not None

    def _extract_explicit_dates(self, text):
        if not text:
            return []

        months = {
            "ocak": 1,
            "subat": 2,
            "şubat": 2,
            "mart": 3,
            "nisan": 4,
            "mayis": 5,
            "mayıs": 5,
            "haziran": 6,
            "temmuz": 7,
            "agustos": 8,
            "ağustos": 8,
            "eylul": 9,
            "eylül": 9,
            "ekim": 10,
            "kasim": 11,
            "kasım": 11,
            "aralik": 12,
            "aralık": 12,
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "sept": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        found = []

        for y, m, d in re.findall(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
            try:
                found.append(datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d"))
            except ValueError:
                pass

        for d, m, y in re.findall(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", text):
            try:
                found.append(datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d"))
            except ValueError:
                pass

        for d, mon, y in re.findall(r"\b(\d{1,2})\s+([A-Za-zçğıöşüÇĞİÖŞÜ]+),?\s+(\d{4})\b", text, flags=re.IGNORECASE):
            month_num = months.get(mon.lower())
            if not month_num:
                continue
            try:
                found.append(datetime(int(y), int(month_num), int(d)).strftime("%Y-%m-%d"))
            except ValueError:
                pass

        return sorted(set(found))

    def _build_date_guardrails(self, user_input, link_content):
        combined = f"{user_input or ''}\n{link_content or ''}"
        detected_dates = self._extract_explicit_dates(combined)
        today = datetime.now().strftime("%Y-%m-%d")

        if self.language == "EN":
            lines = [
                f"DATE RULES:",
                f"- Today is {today}.",
                "- NEVER replace an explicitly written year with another year.",
                "- If year is missing, do not assume current year automatically.",
            ]
            if detected_dates:
                lines.append(f"- Explicitly detected dates: {', '.join(detected_dates)}")
            return "\n".join(lines)

        lines = [
            "TARIH KURALLARI:",
            f"- Bugunun tarihi {today}.",
            "- Acikca yazilan yil asla degistirilmez.",
            "- Yil yoksa otomatik mevcut yil varsayma.",
        ]
        if detected_dates:
            lines.append(f"- Acikca tespit edilen tarihler: {', '.join(detected_dates)}")
        return "\n".join(lines)

    def _complete_brief(self, text, min_sentences=1, max_sentences=3, max_chars=520):
        raw = " ".join((text or "").strip().split())
        if not raw:
            return ""

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        if not sentences:
            out = raw[:max_chars].strip()
            if out and out[-1] not in ".!?":
                out += "."
            return out

        selected = []
        current_len = 0

        for s in sentences:
            if len(selected) >= max_sentences:
                break
            
            cost = len(s) + (1 if selected else 0)
            if current_len + cost > max_chars:
                if selected:
                    break
                selected.append(s)
                current_len += cost
                break
            
            selected.append(s)
            current_len += cost

        if len(selected) < min_sentences:
            filler = (
                "This conclusion is based on available evidence consistency checks."
                if self.language == "EN"
                else "Bu degerlendirme eldeki bulgularin tutarlilik kontrolune dayanmaktadir."
            )
            while len(selected) < min_sentences and len(selected) < max_sentences:
                cost = len(filler) + (1 if selected else 0)
                if current_len + cost > max_chars:
                    break
                selected.append(filler)
                current_len += cost

        brief = " ".join(selected).strip()
        if len(brief) > max_chars:
            cut = brief[:max_chars].strip()
            punct = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
            if punct > 0:
                cut = cut[: punct + 1].strip()
            else:
                sp = cut.rfind(" ")
                if sp > 0:
                    cut = cut[:sp].strip()
                if cut and cut[-1] not in ".!?":
                    cut += "."
            return cut
        
        return brief

    def _extract_links(self, text):
        return re.findall(r"https?://[^\s)\]]+", text or "")

    def _extract_raw_urls(self, text):
        return re.findall(r"https?://[^\s]+", text or "")

    def _normalize_links(self, links):
        out = []
        seen = set()
        for link in links or []:
            cleaned = (link or "").strip().rstrip('.,!?;:\'\")')
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out

    def _format_bibliography_block(self, links):
        unique = self._normalize_links(links)
        if not unique:
            return ""
        return "\n".join([f"{i + 1}. {url}" for i, url in enumerate(unique[:6])])

    def _append_bibliography_if_missing(self, content, links):
        text = (content or "").strip()
        block = self._format_bibliography_block(links)
        if not block:
            return text

        source_label = "SOURCES:" if self.language == "EN" else "KAYNAKLAR:"
        end_tag = "[DETAILS END]" if self.language == "EN" else "[DETAY SONU]"

        # Remove all existing source sections (EN/TR) inside detail before reinserting one.
        for lbl in ("SOURCES:", "KAYNAKLAR:"):
            pattern = rf"(?is)\n?\s*{re.escape(lbl)}\s*.*?(?=\n\s*{re.escape(end_tag)})"
            text = re.sub(pattern, "", text)

        if end_tag in text:
            return text.replace(end_tag, f"{source_label}\n{block}\n{end_tag}")

        return f"{text}\n\n{source_label}\n{block}"

    def _extract_section(self, text, start_tag, end_tag):
        src = text or ""
        if start_tag in src and end_tag in src:
            try:
                return src.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
            except Exception:
                return ""
        return ""

    def _detail_looks_like_summary(self, detail_text):
        txt = (detail_text or "").strip()
        if not txt:
            return True

        lowered = txt.lower()
        summary_markers = [
            "[kisa ozet]",
            "[short summary]",
            "karar:",
            "decision:",
            "kisaca:",
            "briefly:",
        ]
        hits = sum(1 for marker in summary_markers if marker in lowered)
        if hits >= 2:
            return True

        return len(txt) < 240

    def _required_detail_headings(self):
        if self.language == "EN":
            return [
                "## Claim Overview",
                "## Supporting vs Conflicting Evidence",
                "## Source Reliability",
                "## Synthesized Conclusion (short)",
            ]
        return [
            "## Iddianin Cercevesi",
            "## Destekleyen ve Celisen Bulgular",
            "## Kaynak ve Guvenilirlik",
            "## Derlenmis Sonuc (kisa)",
        ]

    def _build_detail_skeleton(self):
        if self.language == "EN":
            return (
                "DETAILED ANALYSIS:\n"
                "## Claim Overview\n"
                "## Supporting vs Conflicting Evidence\n"
                "## Source Reliability\n"
                "## Synthesized Conclusion (short)"
            )
        return (
            "DETAYLI ANALIZ:\n"
            "## Iddianin Cercevesi\n"
            "## Destekleyen ve Celisen Bulgular\n"
            "## Kaynak ve Guvenilirlik\n"
            "## Derlenmis Sonuc (kisa)"
        )

    def _detail_has_required_headings(self, detail_text):
        lowered = (detail_text or "").lower()
        return all(h.lower() in lowered for h in self._required_detail_headings())

    def _replace_detail_section(self, content, detail_text):
        body = (detail_text or "").strip()
        if self.language == "EN":
            pattern = r"\[DETAILS\]\s*.*?\s*\[DETAILS END\]"
            repl = f"[DETAILS]\n{body}\n[DETAILS END]"
            if re.search(pattern, content or "", flags=re.DOTALL):
                return re.sub(pattern, repl, content or "", count=1, flags=re.DOTALL)
            return f"{(content or '').strip()}\n\n{repl}".strip()

        pattern = r"\[DETAY\]\s*.*?\s*\[DETAY SONU\]"
        repl = f"[DETAY]\n{body}\n[DETAY SONU]"
        if re.search(pattern, content or "", flags=re.DOTALL):
            return re.sub(pattern, repl, content or "", count=1, flags=re.DOTALL)
        return f"{(content or '').strip()}\n\n{repl}".strip()

    def _build_fallback_detail(self, context_text):
        ctx = " ".join((context_text or "").split())
        ctx_excerpt = ctx[:900] if ctx else "Yeterli kanit metni otomatik olarak sinirli kaldi."

        if self.language == "EN":
            return (
                "DETAILED ANALYSIS:\\n"
                "## Claim Overview\\n"
                "The claim was evaluated by combining the user input with the collected evidence. "
                "The goal is to test factual consistency, timeline coherence, and source alignment rather than rely on a single statement. "
                "When evidence is incomplete, the analysis remains cautious and explicitly marks uncertainty.\\n\\n"
                "## Supporting vs Conflicting Evidence\\n"
                f"Evidence snapshot: {ctx_excerpt}. "
                "Supporting signals and conflicting signals are read together to avoid confirmation bias. "
                "If multiple sources repeat similar points independently, confidence increases; if they diverge materially, confidence is reduced.\\n\\n"
                "## Source Reliability\\n"
                "Source quality is assessed by consistency, verifiability, and whether links are traceable. "
                "Unattributed or weakly sourced claims are treated as lower reliability. "
                "Directly linked evidence is prioritized over paraphrased or ambiguous statements.\\n\\n"
                "## Synthesized Conclusion (short)\\n"
                "The combined evidence provides a structured but provisional conclusion. "
                "This conclusion reflects currently available signals and should be updated if stronger or newer sources appear."
            )

        return (
            "DETAYLI ANALIZ:\\n"
            "## Iddianin Cercevesi\\n"
            "Bu iddia, kullanici girdisi ve toplanan kanitlar birlikte ele alinarak degerlendirildi. "
            "Amac tek bir ifadeyi tekrar etmek degil, iddianin olgusal tutarliligini ve zaman baglamini kontrol etmektir. "
            "Kanitlar sinirli oldugunda sonuc temkinli bicimde sunulur.\\n\\n"
            "## Destekleyen ve Celisen Bulgular\\n"
            f"Kanit ozeti: {ctx_excerpt}. "
            "Destekleyen noktalar ile celisen noktalar ayni anda okunarak dengeli bir degerlendirme yapildi. "
            "Farkli kaynaklarda bagimsiz benzerlik arttikca guven artar; ciddi uyumsuzlukta guven dusurulur.\\n\\n"
            "## Kaynak ve Guvenilirlik\\n"
            "Kaynaklarin guvenilirligi; baglanti verilebilirlik, icerik tutarliligi ve dogrulanabilirlik kriterleriyle ele alindi. "
            "Kaynak gostermeyen veya zayif dayanakli ifadeler daha dusuk guvenle yorumlandi. "
            "Dogrudan baglantili bulgular, dolayli ve belirsiz ifadelere gore daha yuksek agirlik aldi.\\n\\n"
            "## Derlenmis Sonuc (kisa)\\n"
            "Elde edilen bulgular bir arada degerlendirildiginde, sonuc mevcut kanitlarla sinirli ama tutarli bir cercevede verildi. "
            "Daha guclu veya yeni kaynaklar geldikce sonucun guncellenmesi gerekir."
        )

    def _ensure_detail_quality(self, normalized_text, context_text, links):
        out = self._append_bibliography_if_missing(normalized_text, links)
        if self._needs_detail_rewrite(out):
            out = self._replace_detail_section(out, self._build_fallback_detail(context_text))
            out = self._append_bibliography_if_missing(out, links)
        return out

    def _normalize_model_output(self, text):
        content = (text or "").strip()
        if not content:
            if self.language == "EN":
                return (
                    "[SHORT SUMMARY]\n"
                    "DECISION: Uncertain\n"
                    "CONFIDENCE SCORE: 0%\n"
                    "BRIEFLY: No analysis output was produced.\n"
                    "[SHORT SUMMARY END]\n\n"
                    "[DETAILS]\n"
                    "DETAILED ANALYSIS:\nNo details.\n"
                    "[DETAILS END]"
                )
            return (
                "[KISA OZET]\n"
                "KARAR: Şüpheli\n"
                "GÜVEN SKORU: %0\n"
                "KISACA: Analiz cikti uretemedi.\n"
                "[KISA OZET SONU]\n\n"
                "[DETAY]\n"
                "DETAYLI ANALIZ:\nDetay uretilemedi.\n"
                "[DETAY SONU]"
            )

        if self.language == "EN":
            short = self._extract_section(content, "[SHORT SUMMARY]", "[SHORT SUMMARY END]")
            detail = self._extract_section(content, "[DETAILS]", "[DETAILS END]")

            decision = re.search(r"^\s*DECISION:\s*(.+)$", short, flags=re.MULTILINE)
            confidence = re.search(r"^\s*CONFIDENCE SCORE:\s*(.+)$", short, flags=re.MULTILINE)
            briefly = re.search(r"^\s*BRIEFLY:\s*(.*)$", short, flags=re.MULTILINE)

            decision_text = decision.group(1).strip() if decision else "Uncertain"
            conf_text = confidence.group(1).strip() if confidence else "50%"
            brief_seed = (briefly.group(1).strip() if briefly else "")
            if not brief_seed:
                brief_seed = (short or "").strip()
            if not brief_seed:
                brief_seed = (detail or content or "").strip()
            brief_text = self._complete_brief(brief_seed, min_sentences=2, max_sentences=3, max_chars=420)
            if not brief_text:
                brief_text = "Evidence is mixed across available sources. The claim remains uncertain until stronger confirmation appears."
            detail_text = (detail or "").strip()
            if self._detail_looks_like_summary(detail_text):
                detail_text = self._build_detail_skeleton()
            elif not re.search(r"DETAILED\s+ANALYSIS", detail_text, flags=re.IGNORECASE):
                detail_text = f"DETAILED ANALYSIS:\n{detail_text}"

            return (
                "[SHORT SUMMARY]\n"
                f"DECISION: {decision_text}\n"
                f"CONFIDENCE SCORE: {conf_text}\n"
                f"BRIEFLY: {brief_text}\n"
                "[SHORT SUMMARY END]\n\n"
                "[DETAILS]\n"
                f"{detail_text.strip()}\n"
                "[DETAILS END]"
            )

        short = self._extract_section(content, "[KISA OZET]", "[KISA OZET SONU]")
        detail = self._extract_section(content, "[DETAY]", "[DETAY SONU]")

        karar = re.search(r"^\s*KARAR:\s*(.+)$", short, flags=re.MULTILINE)
        guven = re.search(r"^\s*G[ÜU]VEN SKORU:\s*(.+)$", short, flags=re.MULTILINE)
        kisaca = re.search(r"^\s*KISACA:\s*(.*)$", short, flags=re.MULTILINE)

        karar_text = karar.group(1).strip() if karar else "Şüpheli"
        guven_text = guven.group(1).strip() if guven else "%50"
        kisaca_seed = (kisaca.group(1).strip() if kisaca else "")
        if not kisaca_seed:
            kisaca_seed = (short or "").strip()
        if not kisaca_seed:
            kisaca_seed = (detail or content or "").strip()
        kisaca_text = self._complete_brief(kisaca_seed, min_sentences=2, max_sentences=3, max_chars=420)
        if not kisaca_text:
            kisaca_text = "Eldeki bulgular birbiriyle tam uyumlu degil. Bu nedenle iddia su an supheli degerlendirildi."
        detail_text = (detail or "").strip()
        if self._detail_looks_like_summary(detail_text):
            detail_text = self._build_detail_skeleton()
        elif not re.search(r"DETAYLI\s+ANALIZ", detail_text, flags=re.IGNORECASE):
            detail_text = f"DETAYLI ANALIZ:\n{detail_text}"

        return (
            "[KISA OZET]\n"
            f"KARAR: {karar_text}\n"
            f"GÜVEN SKORU: {guven_text}\n"
            f"KISACA: {kisaca_text}\n"
            "[KISA OZET SONU]\n\n"
            "[DETAY]\n"
            f"{detail_text.strip()}\n"
            "[DETAY SONU]"
        )

    def _needs_detail_rewrite(self, content):
        if self.language == "EN":
            detail = self._extract_section(content, "[DETAILS]", "[DETAILS END]")
        else:
            detail = self._extract_section(content, "[DETAY]", "[DETAY SONU]")

        detail_text = (detail or "").strip()
        if self._detail_looks_like_summary(detail_text):
            return True
        if not self._detail_has_required_headings(detail_text):
            return True
        return len(detail_text) < self.detail_min_chars

    def _rewrite_for_long_detail(self, sys_instr, current_output, context_text):
        if self.language == "EN":
            fix_prompt = f"""
Expand ONLY the [DETAILS] section and keep [SHORT SUMMARY] as-is.
Current output:
{current_output}

Context:
{context_text}

Requirements:
- Keep all section tags exactly.
- Use these exact headings in this order:
  - ## Claim Overview
  - ## Supporting vs Conflicting Evidence
  - ## Source Reliability
  - ## Synthesized Conclusion (short)
- [DETAILS] must be rich, logical and at least {self.detail_min_chars} characters.
- End with a short synthesized conclusion.
"""
        else:
            fix_prompt = f"""
Sadece [DETAY] bolumunu genislet ve [KISA OZET] bolumunu oldugu gibi koru.
Mevcut cikti:
{current_output}

Baglam:
{context_text}

Kurallar:
- Tum bolum etiketleri ayni kalsin.
- Asagidaki basliklari bu sira ile aynen kullan:
  - ## Iddianin Cercevesi
  - ## Destekleyen ve Celisen Bulgular
  - ## Kaynak ve Guvenilirlik
  - ## Derlenmis Sonuc (kisa)
- [DETAY] bolumu mantiksal akisla en az {self.detail_min_chars} karakter olsun.
- Sonunda kisa bir derlenmis sonuc ver.
"""

        return self._generate_content(
            fix_prompt,
            sys_instr,
            generation_config={"temperature": 0.15, "max_output_tokens": self.rewrite_max_output_tokens},
        )

    def _parse_queries(self, text, fallback):
        raw = (text or "").strip()
        if not raw:
            return [fallback]

        parts = re.split(r"[\n,;]+", raw)
        queries = []
        seen = set()

        for p in parts:
            q = re.sub(r"^\s*[-*\d\.\)]\s*", "", p).strip()
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(q)
            if len(queries) >= self.max_queries:
                break

        return queries or [fallback]

    def _build_evidence_block(self, queries, search_results):
        sections = []
        all_links = []

        for i, res in enumerate(search_results):
            q = queries[i] if i < len(queries) else f"query_{i + 1}"
            links = self._extract_links(res)
            all_links.extend(links)
            if links:
                source_lines = "\n".join([f"  - {lnk}" for lnk in links[:5]])
            else:
                source_lines = "  - (link yok)"

            sections.append(
                f"[EVIDENCE {i + 1}] QUERY: {q}\n"
                f"RESULTS:\n{res}\n"
                f"EXTRACTED LINKS:\n{source_lines}"
            )

        merged = "\n\n".join(sections)
        unique_links = self._normalize_links(all_links)
        links_block = "\n".join([f"- {lnk}" for lnk in unique_links[:12]]) if unique_links else "- (yok)"
        return merged, links_block, unique_links

    def _parse_search_candidates(self, queries, search_results):
        candidates = []

        for i, res in enumerate(search_results):
            query = queries[i] if i < len(queries) else f"query_{i + 1}"
            lines = [ln.strip() for ln in (res or "").splitlines() if ln.strip()]
            parsed_any = False

            for ln in lines:
                # Expected format from utils.search_web:
                # - title: snippet (link)
                m = re.match(r"^-\s*(.+?):\s*(.+?)\s*\((https?://[^\s)]+)\)\s*$", ln)
                if not m:
                    continue

                title = m.group(1).strip()
                snippet = m.group(2).strip()
                link = m.group(3).strip()
                text = f"{title}. {snippet}"[:900]
                candidates.append({
                    "query": query,
                    "text": text,
                    "link": link,
                })
                parsed_any = True

            if not parsed_any and res:
                first_link = self._extract_links(res)
                candidates.append(
                    {
                        "query": query,
                        "text": (res or "")[:900],
                        "link": first_link[0] if first_link else "",
                    }
                )

        return candidates

    def _cosine_similarity(self, vec_a, vec_b):
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def _extract_embedding_vectors(self, response):
        raw = None
        if hasattr(response, "embeddings"):
            raw = response.embeddings
        elif isinstance(response, list):
            raw = response
        else:
            raw = []

        vectors = []
        for emb in raw:
            values = getattr(emb, "values", None)
            if values is None and isinstance(emb, dict):
                values = emb.get("values")
            vectors.append(values if isinstance(values, list) else None)
        return vectors

    def _embed_texts(self, texts):
        if not self.embedding_enabled:
            return None

        clean_texts = [(t or "").strip()[:1200] for t in (texts or [])]
        if not clean_texts:
            return []

        vectors = [None] * len(clean_texts)
        to_fetch = []
        fetch_idx = []

        for i, txt in enumerate(clean_texts):
            if not txt:
                continue
            key = txt
            cached = self._embedding_cache.get(key)
            if cached is not None:
                vectors[i] = cached
            else:
                to_fetch.append(txt)
                fetch_idx.append(i)

        if to_fetch:
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model_id,
                    contents=to_fetch,
                )
                fetched_vectors = self._extract_embedding_vectors(response)
                for j, vec in enumerate(fetched_vectors):
                    idx = fetch_idx[j]
                    vectors[idx] = vec
                    if vec is not None:
                        self._embedding_cache[to_fetch[j]] = vec
            except Exception:
                return None

        return vectors

    def _lexical_overlap_score(self, claim, text):
        claim_tokens = set(re.findall(r"\w+", (claim or "").lower()))
        text_tokens = set(re.findall(r"\w+", (text or "").lower()))
        if not claim_tokens or not text_tokens:
            return 0.0
        inter = len(claim_tokens.intersection(text_tokens))
        return inter / max(1, len(claim_tokens))

    def _rank_candidates_with_embedding(self, claim, candidates, top_k):
        if not candidates:
            return []

        claim_text = (claim or "").strip()[:1200]
        candidate_texts = [c["text"] for c in candidates]

        claim_vecs = self._embed_texts([claim_text])
        cand_vecs = self._embed_texts(candidate_texts)

        ranked = []
        use_embedding = bool(claim_vecs and cand_vecs and claim_vecs[0] is not None)

        for i, cand in enumerate(candidates):
            if use_embedding and i < len(cand_vecs) and cand_vecs[i] is not None:
                score = self._cosine_similarity(claim_vecs[0], cand_vecs[i])
            else:
                score = self._lexical_overlap_score(claim_text, cand["text"])

            row = dict(cand)
            row["score"] = float(score)
            ranked.append(row)

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def _build_semantic_evidence_block(self, claim, queries, search_results):
        candidates = self._parse_search_candidates(queries, search_results)
        ranked = self._rank_candidates_with_embedding(claim, candidates, self.embedding_top_k)
        if not ranked:
            return "", []

        lines = []
        links = []
        for i, item in enumerate(ranked, start=1):
            link = item.get("link", "")
            if link:
                links.append(link)
            lines.append(
                f"[SEMANTIC EVIDENCE {i}] SCORE: {item['score']:.4f}\n"
                f"QUERY: {item['query']}\n"
                f"SNIPPET: {item['text']}\n"
                f"LINK: {link if link else '-'}"
            )

        return "\n\n".join(lines), self._normalize_links(links)

    def _build_conversation_block(self, conversation_context):
        ctx = (conversation_context or "").strip()
        if not ctx:
            return ""

        if self.language == "EN":
            return (
                "CONVERSATION CONTEXT (use only if relevant to the current question):\n"
                f"{ctx}\n"
            )

        return (
            "SOHBET BAGLAMI (yalnizca mevcut soruyla ilgiliyse kullan):\n"
            f"{ctx}\n"
        )

    def _system_instruction(self):
        en_detail_mode = (
            "Keep [DETAILS] concise and focused; avoid unnecessary repetition."
            if self.detail_level == "fast"
            else "Put long reasoning only inside [DETAILS]."
        )
        tr_detail_mode = (
            "[DETAY] bolumunu daha kisa ve odakli tut; gereksiz tekrar yapma."
            if self.detail_level == "fast"
            else "Uzun aciklama sadece [DETAY] bolumunde olsun."
        )

        if self.language == "EN":
            return (
                "You are a professional fact-checking analyst. "
                "Use provided evidence and dates carefully. "
                "Never alter explicitly written years. "
                "If year is missing, do not assume current year. "
                "BRIEFLY must be 2-3 sentences with a short reason. "
                "Use conversation context when the user asks a follow-up; ignore unrelated past turns. "
                f"{en_detail_mode}"
            )

        return (
            "Sen profesyonel bir teyit uzmansin. "
            "Verilen kanitlari ve tarihleri dikkatle kullan. "
            "Acikca yazili yillari asla degistirme. "
            "Yil yoksa mevcut yili otomatik varsayma. "
            "KISACA bolumu 2-3 cumle ve kisa gerekce icersin. "
            "Kullanici onceki soruya atif yapiyorsa sohbet baglamini kullan; ilgisizse gecmisi yoksay. "
            f"{tr_detail_mode}"
        )

    def plan_and_verify(self, user_input, uploaded_files=None, link_content="", conversation_context=""):
        try:
            original_user_input = (user_input or "").strip()
            link_content = self._clean_link_content(link_content)
            if len(link_content) > self.max_link_chars:
                link_content = link_content[: self.max_link_chars]

            url_only_input = self._is_url_only_input(original_user_input)
            twitter_url_input = self._contains_twitter_url(original_user_input)
            force_search_mode = self._is_link_content_usable(link_content) and (url_only_input or twitter_url_input)

            effective_input = link_content if force_search_mode else original_user_input
            if (not effective_input) and uploaded_files:
                effective_input = "Gorsel uzerindeki iddia"
            date_guardrails = self._build_date_guardrails(effective_input, link_content)
            conversation_block = self._build_conversation_block(conversation_context)
            sys_instr = self._system_instruction()

            selected_generation_config = dict(self.generation_config)
            if url_only_input:
                selected_generation_config["max_output_tokens"] = self.url_only_max_output_tokens

            source_links = self._normalize_links(self._extract_raw_urls(original_user_input))

            if self._is_link_content_usable(link_content) and not force_search_mode:
                if self.language == "EN":
                    final_prompt = f"""
PRIMARY SOURCE - ANALYZE ONLY THIS:
CLAIM/QUESTION: {effective_input}
LINKED CONTENT: {link_content}

{date_guardrails}

{conversation_block}TASK: Return EXACT format.

[SHORT SUMMARY]
DECISION: (True / False / Uncertain)
CONFIDENCE SCORE: (0-100%)
BRIEFLY: (2-3 short sentences, include short why)
[SHORT SUMMARY END]

[DETAILS]
DETAILED ANALYSIS:
## Claim Overview
## Supporting and Conflicting Evidence
## Source Reliability
## Synthesized Conclusion (short)
SOURCES: Link provided above
[DETAILS END]
"""
                else:
                    final_prompt = f"""
ASIL KAYNAK - SADECE BUNU ANALIZ ET:
IDDIA/SORU: {effective_input}
LINK ICERIGI: {link_content}

{date_guardrails}

{conversation_block}GOREV: Tam olarak asagidaki formati dondur.

[KISA OZET]
KARAR: (Doğru / Yanlış / Şüpheli)
GÜVEN SKORU: (%0-100)
KISACA: (2-3 kisa cumle, kisa neden belirt)
[KISA OZET SONU]

[DETAY]
DETAYLI ANALIZ:
## Iddianin Cercevesi
## Destekleyen ve Celisen Bulgular
## Kaynak ve Guvenilirlik
## Derlenmis Sonuc (kisa)
KAYNAKLAR: Yukaridaki link
[DETAY SONU]
"""

                final_res_text = self._generate_content(
                    final_prompt,
                    sys_instr,
                    generation_config=selected_generation_config,
                )
                normalized = self._normalize_model_output(final_res_text)
                normalized = self._append_bibliography_if_missing(normalized, source_links)
                if self._needs_detail_rewrite(normalized):
                    rewritten = self._rewrite_for_long_detail(
                        sys_instr,
                        normalized,
                        f"IDDIA/SORU: {effective_input}\nICERIK:\n{link_content}",
                    )
                    normalized = self._normalize_model_output(rewritten)
                    normalized = self._append_bibliography_if_missing(normalized, source_links)
                    if self._needs_detail_rewrite(normalized):
                        rewritten2 = self._rewrite_for_long_detail(
                            sys_instr,
                            normalized,
                            f"IDDIA/SORU: {effective_input}\nICERIK:\n{link_content}",
                        )
                        normalized = self._normalize_model_output(rewritten2)
                        normalized = self._append_bibliography_if_missing(normalized, source_links)
                normalized = self._ensure_detail_quality(
                    normalized,
                    f"IDDIA/SORU: {effective_input}\nICERIK:\n{link_content}",
                    source_links,
                )
                return normalized

            if self.language == "EN":
                planner_prompt = f"""
{conversation_block}Claim: {effective_input}
Generate up to {self.max_queries} effective Google search queries to verify this claim.
Return only queries separated by commas.
"""
            else:
                planner_prompt = f"""
{conversation_block}Girdi: {effective_input}
Bu iddiayi teyit etmek icin Google'da aranabilecek en etkili en fazla {self.max_queries} sorguyu uret.
Sadece sorgulari virgulle ayirarak yaz.
"""

            image_parts = None
            if uploaded_files:
                image_parts = []
                for f in uploaded_files:
                    try:
                        image_parts.append(PIL.Image.open(f))
                    except Exception:
                        pass

            plan_text = self._generate_content(
                planner_prompt,
                sys_instr,
                generation_config={"temperature": 0.1, "max_output_tokens": self.planner_max_output_tokens},
                extra_contents=image_parts,
            )
            queries = self._parse_queries(plan_text, effective_input)

            with ThreadPoolExecutor(max_workers=min(len(queries), self.max_queries)) as executor:
                search_results = list(executor.map(lambda q: search_web(q, self.serp_key), queries[: self.max_queries]))

            all_evidence, all_links, unique_links = self._build_evidence_block(queries, search_results)
            semantic_evidence, semantic_links = self._build_semantic_evidence_block(effective_input, queries, search_results)

            merged_links = self._normalize_links(source_links + unique_links + semantic_links)
            links_block = "\n".join([f"- {lnk}" for lnk in merged_links[:12]]) if merged_links else "- (yok)"

            if self.language == "EN":
                final_prompt = f"""
CLAIM: {effective_input}
WEB EVIDENCE:
{all_evidence}

SEMANTICALLY RANKED EVIDENCE (Gemini Embedding):
{semantic_evidence if semantic_evidence else '(none)'}

{date_guardrails}

{conversation_block}TASK: Analyze all evidence and produce the exact format below.

[SHORT SUMMARY]
DECISION: (True / False / Uncertain)
CONFIDENCE SCORE: (0-100%)
BRIEFLY: (2-3 short sentences, include short why)
[SHORT SUMMARY END]

[DETAILS]
DETAILED ANALYSIS:
## Claim Overview
## Supporting vs Conflicting Evidence
## Source Reliability
## Synthesized Conclusion (short)
SOURCES: (Use only links from AVAILABLE LINKS)
[DETAILS END]

AVAILABLE LINKS:
{links_block}
"""
            else:
                final_prompt = f"""
IDDIA: {effective_input}
WEB KANITLARI:
{all_evidence}

ANLAMSAL OLARAK SIRALANMIS KANITLAR (Gemini Embedding):
{semantic_evidence if semantic_evidence else '(yok)'}

{date_guardrails}

{conversation_block}GOREV: Tum kanitlari analiz ederek tam olarak asagidaki formati dondur.

[KISA OZET]
KARAR: (Doğru / Yanlış / Şüpheli)
GÜVEN SKORU: (%0-100)
KISACA: (2-3 kisa cumle, neden belirt)
[KISA OZET SONU]

[DETAY]
DETAYLI ANALIZ:
## Iddianin Cercevesi
## Destekleyen ve Celisen Bulgular
## Kaynak ve Guvenilirlik
## Derlenmis Sonuc (kisa)
KAYNAKLAR: (Sadece MEVCUT LINKLER listesinden sec, uydurma link yazma)
[DETAY SONU]

MEVCUT LINKLER:
{links_block}
"""

            final_res_text = self._generate_content(
                final_prompt,
                sys_instr,
                generation_config=selected_generation_config,
            )

            normalized = self._normalize_model_output(final_res_text)
            normalized = self._append_bibliography_if_missing(normalized, merged_links)

            if self._needs_detail_rewrite(normalized):
                rewritten = self._rewrite_for_long_detail(
                    sys_instr,
                    normalized,
                    f"IDDIA/SORU: {effective_input}\nKANITLAR:\n{all_evidence}\n\nSEMANTIK KANITLAR:\n{semantic_evidence}",
                )
                normalized = self._normalize_model_output(rewritten)
                normalized = self._append_bibliography_if_missing(normalized, merged_links)
                if self._needs_detail_rewrite(normalized):
                    rewritten2 = self._rewrite_for_long_detail(
                        sys_instr,
                        normalized,
                        f"IDDIA/SORU: {effective_input}\nKANITLAR:\n{all_evidence}\n\nSEMANTIK KANITLAR:\n{semantic_evidence}",
                    )
                    normalized = self._normalize_model_output(rewritten2)
                    normalized = self._append_bibliography_if_missing(normalized, merged_links)

            normalized = self._ensure_detail_quality(
                normalized,
                f"IDDIA/SORU: {effective_input}\nKANITLAR:\n{all_evidence}\n\nSEMANTIK KANITLAR:\n{semantic_evidence}",
                merged_links,
            )
            return normalized

        except Exception as exc:
            error_msg = str(exc)
            if self.language == "EN":
                if "429" in error_msg:
                    return "Analysis failed: API quota exceeded. Please try again later."
                return f"Analysis error: {error_msg}"
            if "429" in error_msg:
                return "Analiz basarisiz: API kotasi asildi. Lutfen daha sonra tekrar deneyin."
            return f"Analiz hatasi: {error_msg}"
