"""Фундаментално потвърждение на техническите сетъпи:
- ниво 1 (автоматично, Yahoo): консенсус на анализаторите, потенциал до целевата
  цена, очакван ръст на печалбата, дата на следващия отчет;
- ниво 2 (с бутон, Claude + web search или Gemini + Google Search): свежи
  новини, рейтинг промени от големите банки, отчети/guidance - оценка с
  линкове към източниците.
Само потвърждава и подрежда - не филтрира."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone

import streamlit as st
import yfinance as yf
from anthropic import Anthropic

from ai_client import AI_PROVIDERS, CLAUDE_MODEL, GEMINI_DEFAULT_MODEL, gemini_thinking_config

FUND_CONFIRMED = "✅ Потвърден"
FUND_NEUTRAL = "➖ Неутрален"
FUND_AGAINST = "⚠️ Против"
FUND_NO_DATA = "— няма данни"

MIN_ANALYSTS = 5          # по-малко анализатори = консенсусът не е представителен
MIN_UPSIDE_PCT = 10       # мин. потенциал до средната целева цена за "потвърден"
EARNINGS_WARN_DAYS = 14   # отчет до толкова дни = риск от гап
NEWS_MAX_SEARCHES = 3     # web търсения на компания (всяко се таксува)
NEWS_WORKERS = 6
NEWS_PROVIDERS = AI_PROVIDERS  # общият превключвател на приложението

BUY_KEYS = {"strong_buy", "buy"}
SELL_KEYS = {"sell", "strong_sell", "underperform"}
REC_LABELS = {"strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold", "underperform": "Underperform",
              "sell": "Sell", "strong_sell": "Strong Sell"}


# ---------------------------------------------------------------- ниво 1: Yahoo

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _yahoo_info(symbol: str) -> dict:
    """Yahoo .info с кеш 6 ч. Празен отговор хвърля грешка - неуспехите НЕ се
    кешират (иначе едно временно блокиране от Yahoo оставя "няма данни" за часове)."""
    info = yf.Ticker(symbol).info or {}
    if not info.get("quoteType"):
        raise ValueError("Yahoo върна празен отговор")
    return info


def _analyst_info(symbol: str) -> dict:
    """Анализаторските полета от Yahoo .info; до 3 опита с пауза (Yahoo
    ограничава честите заявки). При неуспех - {"error": причина}."""
    info, error = None, None
    for attempt in range(3):
        try:
            info = _yahoo_info(symbol)
            break
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
    if info is None:
        return {"error": error}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    target = info.get("targetMeanPrice")
    fwd, trailing = info.get("forwardEps"), info.get("trailingEps")
    earnings_ts = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
    earnings = datetime.fromtimestamp(earnings_ts, tz=timezone.utc).date() if earnings_ts else None
    return {
        "quote_type": info.get("quoteType"),
        "rec": info.get("recommendationKey"),
        "analysts": info.get("numberOfAnalystOpinions") or 0,
        "upside": round(100 * (target / price - 1), 1) if price and target else None,
        "eps_growth": round(100 * (fwd / trailing - 1), 1) if fwd and trailing and trailing > 0 else None,
        "earnings": earnings if earnings and earnings >= date.today() else None,
    }


def fetch_analyst_data(symbols) -> dict:
    """{symbol: анализаторски данни или {"error": ...}} - по 3 паралелно
    (повече нишки = по-често блокиране от Yahoo)."""
    symbols = list(symbols)
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(symbols, pool.map(_analyst_info, symbols)))


def failed_symbols(fund_data: dict) -> list:
    return [s for s, d in fund_data.items() if "error" in d]


def fundamental_verdict(data: dict | None) -> str:
    if not data or "error" in data or data.get("quote_type") == "ETF":
        return FUND_NO_DATA
    rec, analysts, upside = data.get("rec"), data.get("analysts") or 0, data.get("upside")
    if not rec or rec == "none" or analysts == 0:
        return FUND_NO_DATA
    if rec in SELL_KEYS or (upside is not None and upside < 0):
        return FUND_AGAINST
    if rec in BUY_KEYS and analysts >= MIN_ANALYSTS and upside is not None and upside >= MIN_UPSIDE_PCT:
        return FUND_CONFIRMED
    return FUND_NEUTRAL


def fundamental_columns(data: dict | None) -> dict:
    """Колоните за таблиците на скрийнъра."""
    data = data or {}
    rec = data.get("rec")
    earnings = data.get("earnings")
    earnings_txt = ""
    if earnings:
        days = (earnings - date.today()).days
        earnings_txt = ("⚠️ " if days <= EARNINGS_WARN_DAYS else "") + earnings.strftime("%d.%m")
    return {
        "📊 Фундамент": fundamental_verdict(data),
        "Анализатори": f"{REC_LABELS.get(rec, rec)} ({data.get('analysts')})" if rec and rec != "none" and data.get("analysts") else "",
        "Потенциал до целта (%)": data.get("upside"),
        "Ръст EPS (%)": data.get("eps_growth"),
        "Отчет": earnings_txt,
    }


# ---------------------------------------------------------------- ниво 2: новини (Claude + web search)

NEWS_POSITIVE, NEWS_NEUTRAL, NEWS_NEGATIVE = "🟢 Положително", "⚪ Неутрално", "🔴 Отрицателно"
_NEWS_LABELS = {"positive": NEWS_POSITIVE, "neutral": NEWS_NEUTRAL, "negative": NEWS_NEGATIVE}

NEWS_PROMPT = """Ти си финансов анализатор. Трябва да провериш дали има фундаментално
потвърждение за СУИНГ ПОКУПКА (long, държане дни до седмици) на:

{kind}: {name} (Yahoo символ: {symbol})

Потърси в интернет (най-много {max_searches} търсения) информация от последните ~30 дни:
- промени в рейтинга / целевата цена от големи банки и анализаторски къщи
  (Goldman Sachs, JPMorgan, Morgan Stanley, UBS, Deutsche Bank, BofA, Barclays,
  Jefferies, Citi, BNP Paribas, Berenberg, Kepler Cheuvreux...);
- последен отчет и guidance: над/под очакванията;
- значими новини: поръчки, сделки, регулации, съдебни дела, смяна на ръководство;
- дата на следващия отчет.
{etf_hint}
Използвай САМО намереното - не измисляй. Ако няма свежа информация, кажи го.
В текстовете не слагай двойни кавички " (ползвай „ “ или единични ').

Отговори САМО с JSON (без друг текст), на български:
{{"verdict": "positive" | "neutral" | "negative",
  "summary": "2-3 изречения защо",
  "analyst_actions": "конкретни рейтинг промени (банка, рейтинг, цел) или празно",
  "next_earnings": "ГГГГ-ММ-ДД или null",
  "sources": [{{"title": "...", "url": "..."}}]}}"""

ETF_HINT = """Това е ETF - вместо анализатори оцени сектора/темата/базовия актив
(тенденции, макро фактори, потоци към фонда)."""


_JSON_FIELDS = ("verdict", "summary", "analyst_actions", "next_earnings", "sources")


def _extract_json(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError("моделът не върна JSON")
    raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _salvage_json(raw)


def _salvage_json(raw: str) -> dict:
    """Резервно четене на полетата, когато JSON-ът е невалиден (най-често
    неекранирани кавички в текста, напр. "Buy" вътре в summary): всяко поле е
    текстът между неговия ключ и ключа на следващото."""
    data = {}
    for i, key in enumerate(_JSON_FIELDS):
        nxt = "|".join(re.escape(f'"{k}"') for k in _JSON_FIELDS[i + 1:]) or r"\}\s*$"
        m = re.search(rf'"{key}"\s*:\s*(.*?)\s*,?\s*(?={nxt})', raw, re.DOTALL)
        if not m:
            continue
        value = m.group(1).strip().rstrip(",").strip()
        if key == "sources":
            data[key] = [{"title": t, "url": u} for t, u in
                         re.findall(r'"title"\s*:\s*"(.*?)"\s*,\s*"url"\s*:\s*"(.*?)"', value)]
        elif value.startswith('"') and value.endswith('"'):
            data[key] = value[1:-1].replace('\\"', '"')
        elif value != "null":
            data[key] = value
    if "verdict" not in data:
        raise ValueError("невалиден JSON в отговора на модела")
    return data


def _build_prompt(name: str, symbol: str, is_etf: bool) -> str:
    return NEWS_PROMPT.format(
        kind="ETF" if is_etf else "Акция", name=name, symbol=symbol,
        max_searches=NEWS_MAX_SEARCHES, etf_hint=ETF_HINT if is_etf else "",
    )


def _news_result(data: dict, sources: list, searches: int) -> dict:
    return {
        "verdict": _NEWS_LABELS.get(str(data.get("verdict", "")).lower(), NEWS_NEUTRAL),
        "summary": data.get("summary") or "",
        "analyst_actions": data.get("analyst_actions") or "",
        "next_earnings": data.get("next_earnings"),
        "sources": sources[:5],
        "searches": searches,
    }


def research_news_gemini(name: str, symbol: str, is_etf: bool, api_key: str, model: str = GEMINI_DEFAULT_MODEL) -> dict:
    """Една компания: Gemini с Google Search grounding. Източниците са от
    grounding метаданните (реално намерените страници), не от текста на модела."""
    try:
        # импортът е тук, а не най-горе: без инсталиран google-genai приложението
        # трябва да работи (Gemini е само по избор)
        from google import genai
        from google.genai import types as genai_types
        # клиентът трябва да е в променлива: временен обект се затваря (garbage
        # collection) още преди заявката -> "the client has been closed"
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model, contents=_build_prompt(name, symbol, is_etf),
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                thinking_config=gemini_thinking_config(model),
            ),
        )
        meta = response.candidates[0].grounding_metadata if response.candidates else None
        data = _extract_json(response.text or "")
    except Exception as e:
        return {"error": str(e), "searches": 0}
    sources, seen = [], set()
    for chunk in (meta.grounding_chunks or []) if meta else []:
        web = chunk.web
        if web and web.uri and web.uri not in seen:
            seen.add(web.uri)
            sources.append({"title": web.title or web.domain or web.uri, "url": web.uri})
    searches = len(meta.web_search_queries or []) if meta else 0
    return _news_result(data, sources, searches)


def research_news(name: str, symbol: str, is_etf: bool, api_key: str) -> dict:
    """Една компания: Claude с web search. Връща {verdict, summary, analyst_actions,
    next_earnings, sources, searches} или {error}. Линковете се пазят само ако са
    от реално намерените резултати (без измислени URL-и)."""
    # таймаут на заявка: web search отнема десетки секунди, но не бива да виси безкрай
    # 1 опит до 4 мин.: при таймаут повторният клик на бутона проверява само неуспелите
    client = Anthropic(api_key=api_key, timeout=240.0, max_retries=0)
    messages = [{"role": "user", "content": _build_prompt(name, symbol, is_etf)}]
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": NEWS_MAX_SEARCHES}]
    found_urls, searches, response = {}, 0, None
    try:
        for _ in range(3):  # pause_turn: сървърът спира дълги търсения - продължаваме
            response = client.messages.create(model=CLAUDE_MODEL, max_tokens=4096, tools=tools, messages=messages)
            usage = getattr(response.usage, "server_tool_use", None)
            searches += getattr(usage, "web_search_requests", 0) or 0
            for block in response.content:
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    for r in block.content:
                        found_urls.setdefault(r.url, r.title)
            if response.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": response.content}]
        text = "".join(b.text for b in response.content if b.type == "text")
        data = _extract_json(text)
    except Exception as e:
        return {"error": str(e), "searches": searches}
    sources = [s for s in data.get("sources") or [] if isinstance(s, dict) and s.get("url") in found_urls]
    if not sources:
        sources = [{"title": t, "url": u} for u, t in list(found_urls.items())[:3]]
    return _news_result(data, sources, searches)


def research_news_many(items: list, provider: str, api_key: str, gemini_model: str = GEMINI_DEFAULT_MODEL, on_done=None) -> dict:
    """items = [(name, symbol, is_etf)] -> {symbol: резултат}; паралелно по NEWS_WORKERS.
    provider = "Claude" или "Gemini" (api_key е ключът на съответния доставчик)."""
    def one(name, symbol, is_etf):
        if provider == "Gemini":
            return research_news_gemini(name, symbol, is_etf, api_key, gemini_model)
        return research_news(name, symbol, is_etf, api_key)

    out = {}
    with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as pool:
        futures = {pool.submit(one, n, s, e): s for n, s, e in items}
        for i, fut in enumerate(as_completed(futures), 1):  # прогресът расте с всеки готов
            out[futures[fut]] = fut.result()
            if on_done:
                on_done(i, len(items))
    return out


# ---------------------------------------------------------------- подреждане

def confirmation_score(fund: str, news: str | None) -> int:
    """По-високо = по-силно потвърждение; ползва се за подреждане на таблиците."""
    score = {FUND_CONFIRMED: 2, FUND_NEUTRAL: 1, FUND_NO_DATA: 1, FUND_AGAINST: 0}.get(fund, 1)
    return score + {NEWS_POSITIVE: 1, NEWS_NEGATIVE: -1}.get(news, 0)
