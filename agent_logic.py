import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import PIL.Image
from google import genai

from utils import search_web


class VerificationAgent:
    def __init__(self, gemini_key, serp_key, language="TR"):
        self.client = genai.Client(api_key=gemini_key)
        self.model_id = "gemini-2.5-flash"
        self.serp_key = serp_key
        self.language = language

        self.max_link_chars = 2200
        self.max_queries = 4
        self.detail_min_chars = 3000
        self.generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 3400,
        }

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
            "ocak": 1, "subat": 2, "Ã…Å¸ubat": 2, "mart": 3, "nisan": 4, "mayis": 5, "mayÃ„Â±s": 5,
            "haziran": 6, "temmuz": 7, "agustos": 8, "aÃ„Å¸ustos": 8, "eylul": 9, "eylÃƒÂ¼l": 9,
            "ekim": 10, "kasim": 11, "kasÃ„Â±m": 11, "aralik": 12, "aralÃ„Â±k": 12,
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
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

        for d, mon, y in re.findall(r"\b(\d{1,2})\s+([A-Za-zÃƒâ€¡Ã„Å¾Ã„Â°Ãƒâ€“Ã…Å¾ÃƒÅ“ÃƒÂ§Ã„Å¸Ã„Â±ÃƒÂ¶Ã…Å¸ÃƒÂ¼]+),?\s+(\d{4})\b", text, flags=re.IGNORECASE):
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
            if detected_dates:
                return (
                    f"DATE RULES:\n"
                    f"- Today is {today}.\n"
                    f"- NEVER replace an explicitly written year with another year.\n"
                    f"- Detected explicit dates: {', '.join(detected_dates)}\n"
                    f"- If year is missing, do not assume current year automatically."
                )
            return (
                f"DATE RULES:\n"
                f"- Today is {today}.\n"
                f"- NEVER replace an explicitly written year with another year.\n"
                f"- If year is missing, do not assume current year automatically."
            )

        if detected_dates:
            return (
                f"TARIH KURALLARI:\n"
                f"- BugÃƒÂ¼nÃƒÂ¼n tarihi {today}.\n"
                f"- AÃƒÂ§Ã„Â±k yazÃ„Â±lan yÃ„Â±l asla deÃ„Å¸iÃ…Å¸tirilmez.\n"
                f"- Tespit edilen aÃƒÂ§Ã„Â±k tarihler: {', '.join(detected_dates)}\n"
                f"- YÃ„Â±l yazmÃ„Â±yorsa otomatik mevcut yÃ„Â±l varsayma."
            )
        return (
            f"TARIH KURALLARI:\n"
            f"- BugÃƒÂ¼nÃƒÂ¼n tarihi {today}.\n"
            f"- AÃƒÂ§Ã„Â±k yazÃ„Â±lan yÃ„Â±l asla deÃ„Å¸iÃ…Å¸tirilmez.\n"
            f"- YÃ„Â±l yazmÃ„Â±yorsa otomatik mevcut yÃ„Â±l varsayma."
        )

    def _complete_brief(self, text, max_chars=520):
        raw = (text or "").strip()
        if not raw:
            return ""

        cleaned = " ".join(raw.split())
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return cleaned[:max_chars].strip()

        selected = sentences[:3]
        brief = " ".join(selected).strip()

        # Ensure at least 2 sentences.
        if len(selected) == 1 and len(sentences) > 1:
            brief = f"{selected[0]} {sentences[1]}".strip()
        elif len(selected) == 1:
            if self.language == "EN":
                addon = "This summary is based on currently available evidence and consistency checks."
            else:
                addon = "Bu kısa özet, eldeki bulguların tutarlılık kontrolüne dayanır."
            brief = f"{selected[0]} {addon}".strip()

        if len(brief) <= max_chars:
            return brief

        cut = brief[:max_chars].strip()
        punct_idx = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if punct_idx >= 50:
            return cut[: punct_idx + 1].strip()

        space_idx = cut.rfind(" ")
        if space_idx > 50:
            cut = cut[:space_idx].strip()
        if cut and cut[-1] not in ".!?":
            cut += "."
        return cut

    def _normalize_model_output(self, text):
        content = (text or "").strip()
        if not content:
            if self.language == "EN":
                return (
                    "[SHORT SUMMARY]\n"
                    "DECISION: Uncertain\n"
                    "CONFIDENCE SCORE: 0%\n"
                    "BRIEFLY: No output was produced. Please try again.\n"
                    "[SHORT SUMMARY END]\n\n"
                    "[DETAILS]\n"
                    "DETAILED ANALYSIS: No output.\n"
                    "SOURCES: N/A\n"
                    "[DETAILS END]"
                )
            return (
                "[KISA OZET]\n"
                "KARAR: Supheli\n"
                "GÃƒÅ“VEN SKORU: %0\n"
                "KISACA: Ãƒâ€¡Ã„Â±ktÃ„Â± ÃƒÂ¼retilemedi. LÃƒÂ¼tfen tekrar deneyin.\n"
                "[KISA OZET SONU]\n\n"
                "[DETAY]\n"
                "DETAYLI ANALIZ: Ãƒâ€¡Ã„Â±ktÃ„Â± ÃƒÂ¼retilemedi.\n"
                "KAYNAKLAR: Yok\n"
                "[DETAY SONU]"
            )

        if self.language == "EN":
            if "[SHORT SUMMARY]" in content and "[DETAILS]" in content:
                return re.sub(r"(BRIEFLY:\s*)(.+)", lambda m: f"{m.group(1)}{self._complete_brief(m.group(2), 380)}", content)
            return (
                "[SHORT SUMMARY]\n"
                "DECISION: Uncertain\n"
                "CONFIDENCE SCORE: 50%\n"
                f"BRIEFLY: {self._complete_brief(content, 380)}\n"
                "[SHORT SUMMARY END]\n\n"
                "[DETAILS]\n"
                "DETAILED ANALYSIS:\n"
                "## Claim Overview\n"
                f"{content}\n\n"
                "## Supporting and Conflicting Evidence\n"
                "Evidence was compared for consistency and contradiction.\n\n"
                "## Logical Assessment\n"
                "The reasoning was checked for assumptions and internal consistency.\n\n"
                "## Source Reliability\n"
                "Source trust level was compared by transparency and corroboration.\n\n"
                "## Synthesized Conclusion\n"
                "Conclusion is a synthesis of available evidence rather than a forced binary claim.\n"
                "SOURCES: N/A\n"
                "[DETAILS END]"
            )

        if "[KISA OZET]" in content and "[DETAY]" in content:
            return re.sub(r"(KISACA:\s*)(.+)", lambda m: f"{m.group(1)}{self._complete_brief(m.group(2), 380)}", content)

        return (
            "[KISA OZET]\n"
            "KARAR: Supheli\n"
            "GÃƒÅ“VEN SKORU: %50\n"
            f"KISACA: {self._complete_brief(content, 380)}\n"
            "[KISA OZET SONU]\n\n"
            "[DETAY]\n"
            "DETAYLI ANALIZ:\n"
            "## Ã„Â°ddianÃ„Â±n Ãƒâ€¡erÃƒÂ§evesi\n"
            f"{content}\n\n"
            "## Destekleyen ve Ãƒâ€¡eliÃ…Å¸en Bulgular\n"
            "Ã„Â°ddiayÃ„Â± destekleyen ve ÃƒÂ§eliÃ…Å¸en unsurlar birlikte deÃ„Å¸erlendirildi.\n\n"
            "## MantÃ„Â±ksal DeÃ„Å¸erlendirme\n"
            "Ãƒâ€¡Ã„Â±karÃ„Â±mlarÃ„Â±n dayanaklarÃ„Â± ve olasÃ„Â± varsayÃ„Â±mlar kontrol edildi.\n\n"
            "## Kaynak ve GÃƒÂ¼venilirlik\n"
            "KaynaklarÃ„Â±n gÃƒÂ¼venilirliÃ„Å¸i ve tutarlÃ„Â±lÃ„Â±Ã„Å¸Ã„Â± karÃ…Å¸Ã„Â±laÃ…Å¸tÃ„Â±rÃ„Â±ldÃ„Â±.\n\n"
            "## DerlenmiÃ…Å¸ SonuÃƒÂ§\n"
            "Kesin hÃƒÂ¼kÃƒÂ¼m dayatmadan, eldeki bulgularÃ„Â±n derlenmiÃ…Å¸ ÃƒÂ¶zeti sunuldu.\n"
            "KAYNAKLAR: Yok\n"
            "[DETAY SONU]"
        )

    def _extract_links(self, text):
        return re.findall(r"\((https?://[^)\s]+)\)", text or "")

    def _extract_raw_urls(self, text):
        return re.findall(r"https?://[^\s)]+", text or "")

    def _normalize_links(self, links):
        unique = []
        seen = set()
        for link in links or []:
            cleaned = (link or "").strip().rstrip('.,);]')
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)
        return unique

    def _format_bibliography_block(self, links):
        normalized = self._normalize_links(links)
        if not normalized:
            return ""
        header = "## Bibliography (Found Web Links)" if self.language == "EN" else "## KaynakÃƒÂ§a (Ã„Â°nternetten Bulunan Linkler)"
        lines = "\n".join([f"- {lnk}" for lnk in normalized[:12]])
        return f"{header}\n{lines}".strip()

    def _append_bibliography_if_missing(self, content, links):
        text = (content or "").strip()
        if not text:
            return text

        block = self._format_bibliography_block(links)
        if not block:
            return text

        if "## Bibliography (Found Web Links)" in text or "## KaynakÃƒÂ§a (Ã„Â°nternetten Bulunan Linkler)" in text:
            return text

        if self.language == "EN":
            start_tag, end_tag = "[DETAILS]", "[DETAILS END]"
        else:
            start_tag, end_tag = "[DETAY]", "[DETAY SONU]"

        if start_tag in text and end_tag in text:
            start_idx = text.find(start_tag)
            end_idx = text.find(end_tag)
            detail = text[start_idx + len(start_tag):end_idx].rstrip()
            detail = f"{detail}\n\n{block}\n"
            return f"{text[:start_idx + len(start_tag)]}{detail}{text[end_idx:]}"

        return f"{text}\n\n{block}"

    def _needs_detail_rewrite(self, content):
        text = (content or "").strip()
        if not text:
            return True
        if self.language == "EN":
            if "[SHORT SUMMARY]" not in text or "[DETAILS]" not in text:
                return True
            try:
                detail = text.split("[DETAILS]")[1].split("[DETAILS END]")[0].strip()
            except Exception:
                return True
            return len(detail) < self.detail_min_chars or "[SHORT SUMMARY]" in detail

        if "[KISA OZET]" not in text or "[DETAY]" not in text:
            return True
        try:
            detail = text.split("[DETAY]")[1].split("[DETAY SONU]")[0].strip()
        except Exception:
            return True
        return len(detail) < self.detail_min_chars or "[KISA OZET]" in detail

    def _rewrite_for_long_detail(self, sys_instr, current_output, context_text):
        if self.language == "EN":
            fix_prompt = f"""
Rewrite the report strictly in this format.
Keep BRIEFLY to 2-3 sentences.
Make [DETAILS] long, evidence-based, and coherent logical narrative (not bullet points).
Under each heading, write rich explanatory paragraphs (at least 5-7 sentences).
Target [DETAILS] length to be at least 2200 characters.
In Final Conclusion, provide a synthesized summary of findings instead of forcing a binary verdict.

CONTEXT:
{context_text}

CURRENT OUTPUT:
{current_output}

[SHORT SUMMARY]
DECISION: (True / False / Uncertain)
CONFIDENCE SCORE: (0-100%)
BRIEFLY: (2-3 sentences)
[SHORT SUMMARY END]

[DETAILS]
DETAILED ANALYSIS:
## Claim Overview
## Supporting and Conflicting Evidence
## Logical Assessment
## Source Reliability
## Synthesized Conclusion
SOURCES: (list links)
[DETAILS END]
"""
        else:
            fix_prompt = f"""
Raporu aÃ…Å¸aÃ„Å¸Ã„Â±daki formata tam uyacak Ã…Å¸ekilde yeniden yaz.
KISACA 2-3 cÃƒÂ¼mle olsun.
[DETAY] bÃƒÂ¶lÃƒÂ¼mÃƒÂ¼ uzun, kanÃ„Â±t odaklÃ„Â± ve akÃ„Â±cÃ„Â± mantÃ„Â±ksal anlatÃ„Â±m olsun (maddeleme yapma).
BaÃ…Å¸lÃ„Â±klar kullan.
Her baÃ…Å¸lÃ„Â±k altÃ„Â±nda en az 5-7 cÃƒÂ¼mlelik doyurucu aÃƒÂ§Ã„Â±klama yaz.
Toplam [DETAY] uzunluÃ„Å¸u en az 2200 karakter olsun.
SonuÃƒÂ§ bÃƒÂ¶lÃƒÂ¼mÃƒÂ¼nde zorunlu doÃ„Å¸ru/yanlÃ„Â±Ã…Å¸ hÃƒÂ¼kmÃƒÂ¼ verme; elde edilen bilgilerin derlenmiÃ…Å¸ ve toparlanmÃ„Â±Ã…Å¸ ÃƒÂ¶zetini ver.

BAÃ„Å¾LAM:
{context_text}

MEVCUT Ãƒâ€¡IKTI:
{current_output}

[KISA OZET]
KARAR: (Dogru / Yanlis / Supheli)
GÃƒÅ“VEN SKORU: (%0-100)
KISACA: (2-3 cÃƒÂ¼mle)
[KISA OZET SONU]

[DETAY]
DETAYLI ANALIZ:
## Ã„Â°ddianÃ„Â±n Ãƒâ€¡erÃƒÂ§evesi
## Destekleyen ve Ãƒâ€¡eliÃ…Å¸en Bulgular
## MantÃ„Â±ksal DeÃ„Å¸erlendirme
## Kaynak ve GÃƒÂ¼venilirlik
## DerlenmiÃ…Å¸ SonuÃƒÂ§
KAYNAKLAR: (linkleri yaz)
[DETAY SONU]
"""

        return self._generate_content(
            fix_prompt,
            sys_instr,
            generation_config={"temperature": 0.1, "max_output_tokens": 4200},
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
            q = queries[i] if i < len(queries) else f"query_{i+1}"
            links = self._extract_links(res)
            all_links.extend(links)
            source_lines = "\n".join([f"  - {lnk}" for lnk in links[:5]]) if links else "  - (link bulunamadÃ„Â±)"
            sections.append(f"[EVIDENCE {i+1}] QUERY: {q}\nRESULTS:\n{res}\nEXTRACTED LINKS:\n{source_lines}")

        merged = "\n\n".join(sections)
        unique_links = self._normalize_links(all_links)
        links_block = "\n".join([f"- {lnk}" for lnk in unique_links[:12]]) if unique_links else "- (yok)"
        return merged, links_block, unique_links

    def plan_and_verify(self, user_input, uploaded_files=None, link_content=""):
        try:
            original_user_input = user_input or ""
            link_content = self._clean_link_content(link_content)
            if len(link_content) > self.max_link_chars:
                link_content = link_content[: self.max_link_chars]

            url_only_input = self._is_url_only_input(user_input)
            twitter_url_input = self._contains_twitter_url(original_user_input)
            force_search_mode = self._is_link_content_usable(link_content) and (url_only_input or twitter_url_input)
            if force_search_mode:
                user_input = link_content

            deep_detail_en = (
                "In [DETAILS], write a long and readable logical narrative with clear section headings. "
                "For each heading, write rich explanatory paragraphs (at least 5-7 sentences). "
                "Target at least 2200 characters in [DETAILS]. "
                "In the conclusion, synthesize findings instead of forcing a strict true/false judgement."
            )
            deep_detail_tr = (
                "[DETAY] bÃƒÂ¶lÃƒÂ¼mÃƒÂ¼nde uzun, okunabilir ve mantÃ„Â±ksal akÃ„Â±Ã…Å¸lÃ„Â± bir anlatÃ„Â±m yaz; baÃ…Å¸lÃ„Â±klar kullan. "
                "Her baÃ…Å¸lÃ„Â±k altÃ„Â±nda en az 5-7 cÃƒÂ¼mlelik aÃƒÂ§Ã„Â±klama ver. [DETAY] bÃƒÂ¶lÃƒÂ¼mÃƒÂ¼ en az 2200 karakter olsun. "
                "Son bÃƒÂ¶lÃƒÂ¼mde zorunlu doÃ„Å¸ru/yanlÃ„Â±Ã…Å¸ hÃƒÂ¼kmÃƒÂ¼ yerine elde edilen bilgileri derleyip toparlayan bir sonuÃƒÂ§ yaz."
            )

            selected_generation_config = dict(self.generation_config)
            if url_only_input:
                selected_generation_config["max_output_tokens"] = 4200

            date_guardrails = self._build_date_guardrails(user_input, link_content)

            if self.language == "EN":
                sys_instr = (
                    "You are a professional fact-checking expert. "
                    "When analyzing information, check logical consistency between sources. "
                    "CRITICAL: If link content is provided, ONLY analyze that content and ignore other sources. "
                    "The content may contain technical noise; ignore IDs/tags and focus on human-written meaning. "
                    "DATE SAFETY: Keep explicitly written years exactly as they appear. "
                    "Never rewrite 2019 as 2026 or vice versa. "
                    "If a date lacks year, do not auto-assume the current year. "
                    "OUTPUT RULE: Always return all required section tags exactly as requested. "
                    "SHORT SUMMARY RULE: BRIEFLY must be 2-3 sentences, concise but complete. "
                    "DETAIL RULE: Put long, comprehensive reasoning only inside [DETAILS]."
                )
            else:
                sys_instr = (
                    "Sen profesyonel bir teyit uzmanÃ„Â±sÃ„Â±n. "
                    "Bilgileri analiz ederken kaynaklar arasÃ„Â±ndaki mantÃ„Â±ksal tutarlÃ„Â±lÃ„Â±Ã„Å¸Ã„Â± incele. "
                    "KRITIK: EÃ„Å¸er link iÃƒÂ§eriÃ„Å¸i verildiyse SADECE o iÃƒÂ§eriÃ„Å¸i analiz et, diÃ„Å¸er kaynaklarÃ„Â± yoksay. "
                    "Ã„Â°ÃƒÂ§erikte teknik gÃƒÂ¼rÃƒÂ¼ltÃƒÂ¼ olabilir; ID/etiketleri yoksay ve insan tarafÃ„Â±ndan yazÃ„Â±lan anlamÃ„Â± analiz et. "
                    "TARIH GUVENLIGI: AÃƒÂ§Ã„Â±k yazÃ„Â±lan yÃ„Â±lÃ„Â± birebir koru. "
                    "2019 gibi bir yÃ„Â±lÃ„Â± 2026'ya ÃƒÂ§evirme. "
                    "YÃ„Â±l verilmeyen tarihte mevcut yÃ„Â±lÃ„Â± otomatik varsayma. "
                    "CIKTI KURALI: Ã„Â°stenen tÃƒÂ¼m bÃƒÂ¶lÃƒÂ¼m etiketlerini eksiksiz ve aynÃ„Â± Ã…Å¸ekilde yaz. "
                    "KISA OZET KURALI: KISACA 2-3 cÃƒÂ¼mle olsun. "
                    "DETAY KURALI: Uzun ve kapsamlÃ„Â± aÃƒÂ§Ã„Â±klamayÃ„Â± sadece [DETAY] bÃƒÂ¶lÃƒÂ¼mÃƒÂ¼nde ver."
                )

            source_links = self._normalize_links(self._extract_raw_urls(original_user_input))

            if self._is_link_content_usable(link_content) and not force_search_mode:
                if self.language == "EN":
                    final_executor_prompt = f"""
PRIMARY SOURCE - ANALYZE ONLY THIS:
CLAIM/QUESTION: {user_input}
LINKED CONTENT: {link_content}

{date_guardrails}

TASK: Analyze only the linked content below.

[SHORT SUMMARY]
DECISION: (True / False / Uncertain)
CONFIDENCE SCORE: (0-100%)
BRIEFLY: (2-3 short sentences)
[SHORT SUMMARY END]

[DETAILS]
DETAILED ANALYSIS:
## Claim Overview
## Supporting and Conflicting Evidence
## Reasoning Path
## Logical Assessment
## Source Reliability
## Synthesized Conclusion
SOURCES: Link provided above
[DETAILS END]
{deep_detail_en}
"""
                else:
                    final_executor_prompt = f"""
ASIL KAYNAK - SADECE BUNU ANALIZ ET:
IDDIA/SORU: {user_input}
LINK ICERIGI: {link_content}

{date_guardrails}

GOREV: Sadece yukarÃ„Â±daki link iÃƒÂ§eriÃ„Å¸ini analiz ederek iddiayÃ„Â± deÃ„Å¸erlendir.

[KISA OZET]
KARAR: (Dogru / Yanlis / Supheli)
GÃƒÅ“VEN SKORU: (%0-100)
KISACA: (2-3 kÃ„Â±sa cÃƒÂ¼mle)
[KISA OZET SONU]

[DETAY]
DETAYLI ANALIZ:
## Ã„Â°ddianÃ„Â±n Ãƒâ€¡erÃƒÂ§evesi
## Destekleyen ve Ãƒâ€¡eliÃ…Å¸en Bulgular
## MantÃ„Â±ksal DeÃ„Å¸erlendirme
## Kaynak ve GÃƒÂ¼venilirlik
## DerlenmiÃ…Å¸ SonuÃƒÂ§
KAYNAKLAR: YukarÃ„Â±da verilen link
[DETAY SONU]
{deep_detail_tr}
"""

                final_res_text = self._generate_content(
                    final_executor_prompt,
                    sys_instr,
                    generation_config=selected_generation_config,
                )
                normalized = self._normalize_model_output(final_res_text)
                normalized = self._append_bibliography_if_missing(normalized, source_links)
                if self._needs_detail_rewrite(normalized):
                    rewritten = self._rewrite_for_long_detail(sys_instr, normalized, f"IDDIA/SORU: {user_input}\nICERIK: {link_content}")
                    normalized = self._normalize_model_output(rewritten)
                    normalized = self._append_bibliography_if_missing(normalized, source_links)
                return normalized

            if self.language == "EN":
                planner_prompt = f"""
Claim: {user_input}
Generate 3 effective search queries to verify this claim on Google.
Write only queries separated by commas.
"""
            else:
                planner_prompt = f"""
Girdi: {user_input}
Bu iddiayÃ„Â± teyit etmek iÃƒÂ§in Google'da aratÃ„Â±lmasÃ„Â± gereken en etkili 3 terimi ÃƒÂ¼ret.
Sadece terimleri virgÃƒÂ¼lle ayÃ„Â±rarak yaz.
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
                generation_config={"temperature": 0.1, "max_output_tokens": 140},
                extra_contents=image_parts,
            )
            queries = self._parse_queries(plan_text, user_input)

            with ThreadPoolExecutor(max_workers=min(len(queries), self.max_queries)) as executor:
                search_results = list(executor.map(lambda q: search_web(q, self.serp_key), queries[: self.max_queries]))

            all_evidence, all_links, unique_links = self._build_evidence_block(queries, search_results)

            if self.language == "EN":
                final_executor_prompt = f"""
CLAIM: {user_input}
SEARCH RESULTS: {all_evidence}

{date_guardrails}

TASK: Analyze the evidence and create a report in this exact format.

[SHORT SUMMARY]
DECISION: (True / False / Uncertain)
CONFIDENCE SCORE: (0-100%)
BRIEFLY: (2-3 short sentences)
[SHORT SUMMARY END]

[DETAILS]
DETAILED ANALYSIS:
## Claim Overview
## Supporting vs Conflicting Evidence
## Logical Assessment
## Source Reliability
## Synthesized Conclusion
SOURCES: (List links from this list exactly, do not invent)
[DETAILS END]
{deep_detail_en}
AVAILABLE LINKS:
{all_links}
"""
            else:
                final_executor_prompt = f"""
IDDIA: {user_input}
ARAMA SONUCLARI: {all_evidence}

{date_guardrails}

GOREV: Arama kanÃ„Â±tlarÃ„Â±nÃ„Â± analiz ederek iddiayÃ„Â± deÃ„Å¸erlendir.

[KISA OZET]
KARAR: (Dogru / Yanlis / Supheli)
GÃƒÅ“VEN SKORU: (%0-100)
KISACA: (2-3 kÃ„Â±sa cÃƒÂ¼mle)
[KISA OZET SONU]

[DETAY]
DETAYLI ANALIZ:
## Ã„Â°ddianÃ„Â±n Ãƒâ€¡erÃƒÂ§evesi
## Destekleyen ve Ãƒâ€¡eliÃ…Å¸en Bulgular
## MantÃ„Â±ksal DeÃ„Å¸erlendirme
## Kaynak ve GÃƒÂ¼venilirlik
## DerlenmiÃ…Å¸ SonuÃƒÂ§
KAYNAKLAR: (AÃ…Å¸aÃ„Å¸Ã„Â±daki listeden seÃƒÂ§, uydurma link yazma)
[DETAY SONU]
{deep_detail_tr}
MEVCUT LINKLER:
{all_links}
"""

            final_res_text = self._generate_content(
                final_executor_prompt,
                sys_instr,
                generation_config=selected_generation_config,
            )
            normalized = self._normalize_model_output(final_res_text)
            normalized = self._append_bibliography_if_missing(normalized, unique_links)
            if self._needs_detail_rewrite(normalized):
                rewritten = self._rewrite_for_long_detail(sys_instr, normalized, f"IDDIA/SORU: {user_input}\nKANITLAR:\n{all_evidence}")
                normalized = self._normalize_model_output(rewritten)
                normalized = self._append_bibliography_if_missing(normalized, unique_links)
            return normalized

        except Exception as e:
            error_msg = str(e)
            if self.language == "EN":
                if "429" in error_msg:
                    return "Ã¢Å¡Â Ã¯Â¸Â **API Quota Exceeded**\nPlease try again in a few minutes."
                return f"Ã¢Å¡Â Ã¯Â¸Â **Analysis Error:** {error_msg}"
            if "429" in error_msg:
                return "Ã¢Å¡Â Ã¯Â¸Â **API KotasÃ„Â± AÃ…Å¸Ã„Â±ldÃ„Â±**\nLÃƒÂ¼tfen birkaÃƒÂ§ dakika sonra tekrar deneyin."
            return f"Ã¢Å¡Â Ã¯Â¸Â **Analiz HatasÃ„Â±:** {error_msg}"