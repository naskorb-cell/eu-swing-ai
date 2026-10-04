"""Справедлива цена на акциите (за „⭐ Селектирани“ в секцията „📈 Възходящ тренд“):
  А) безплатно, от Yahoo `.info`: няколко независими оценки на справедливата цена +
     PEG, а накрая присъда „подценена / около справедливата / надценена“ и колко
     от методите са съгласни;
  Б) по желание AI проверка в интернет (Gemini / Claude, като новините): готови оценки
     от Morningstar, Simply Wall St, GuruFocus, анализатори - с линкове само към реално
     намерени страници. Платено, затова с потвърждение и общ дневен склад.
Само за акции - при ETF-ите няма „справедлива цена“."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import median

import fundamentals as fund

VALUE_UNDER, VALUE_FAIR, VALUE_OVER = "🟢 Подценена", "⚪ Около справедливата", "🔴 Надценена"
VALUE_NO_DATA = "— няма данни"
VALUE_ETF = "— ETF"
_VALUE_LABELS = {"undervalued": VALUE_UNDER, "fair": VALUE_FAIR, "overvalued": VALUE_OVER}

MARGIN_PCT = 10        # ±10% от цената = „около справедливата“
MIN_ANALYSTS = 3       # целевата цена на анализаторите се брои при поне толкова мнения
REQUIRED_FCF_YIELD = 0.05   # справедлива цена по свободния паричен поток: 5% доходност
LYNCH_GROWTH_RANGE = (5, 25)  # Линч: справедливо P/E = ръстът на печалбата в %, в тези граници
GRAHAM_FACTOR = 22.5   # √(22.5 × EPS × BVPS) - P/E 15 × P/B 1.5


def _num(info: dict, *keys):
    for k in keys:
        v = info.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _info(symbol: str):
    """Yahoo .info (кеш 6 ч в fundamentals) с до 3 опита."""
    for attempt in range(3):
        try:
            return fund._yahoo_info(symbol)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def fair_value_methods(info: dict) -> dict:
    """{метод: справедлива цена} + показателите, от които са сметнати."""
    price = _num(info, "currentPrice", "regularMarketPrice")
    eps, fwd_eps = _num(info, "trailingEps"), _num(info, "forwardEps")
    bvps = _num(info, "bookValue")
    analysts = _num(info, "numberOfAnalystOpinions") or 0
    target = _num(info, "targetMeanPrice")
    fcf, mcap = _num(info, "freeCashflow"), _num(info, "marketCap")
    growth = _num(info, "earningsGrowth")  # дял, напр. 0.12
    if growth is None and eps and fwd_eps and eps > 0:
        growth = fwd_eps / eps - 1
    methods = {}
    if target and analysts >= MIN_ANALYSTS:
        methods["Анализатори"] = target
    if eps and bvps and eps > 0 and bvps > 0:
        methods["Греъм"] = math.sqrt(GRAHAM_FACTOR * eps * bvps)
    if fwd_eps and fwd_eps > 0 and growth is not None and growth > 0:
        lo, hi = LYNCH_GROWTH_RANGE
        methods["Линч (P/E = ръст)"] = min(max(growth * 100, lo), hi) * fwd_eps
    if fcf and mcap and price and fcf > 0:
        methods[f"FCF {REQUIRED_FCF_YIELD:.0%}"] = price * (fcf / REQUIRED_FCF_YIELD) / mcap
    peg = _num(info, "trailingPegRatio", "pegRatio")
    return {
        "price": price, "methods": methods, "peg": peg,
        "pe": _num(info, "trailingPE"), "fwd_pe": _num(info, "forwardPE"), "pb": _num(info, "priceToBook"),
        "ev_ebitda": _num(info, "enterpriseToEbitda"),
        "fcf_yield": round(100 * fcf / mcap, 1) if fcf and mcap else None,
        "growth": round(100 * growth, 1) if growth is not None else None,
        "currency": info.get("currency"), "sector": info.get("sector"),
    }


def verdict_from(data: dict) -> dict:
    """Гласове на методите (+ PEG) и обща присъда по медианата на потенциала."""
    price, methods = data["price"], data["methods"]
    upsides = {m: 100 * (v / price - 1) for m, v in methods.items() if price and v}
    votes = {"under": 0, "fair": 0, "over": 0}
    for u in upsides.values():
        votes["under" if u > MARGIN_PCT else "over" if u < -MARGIN_PCT else "fair"] += 1
    peg = data.get("peg")
    if peg is not None and peg > 0:
        votes["under" if peg < 1 else "over" if peg > 2 else "fair"] += 1
    total = sum(votes.values())
    if not upsides:
        return {"verdict": VALUE_NO_DATA, "median_upside": None, "votes": votes, "total": total, "upsides": {}}
    med = median(upsides.values())
    if med > MARGIN_PCT and votes["under"] >= total / 2:
        verdict = VALUE_UNDER
    elif med < -MARGIN_PCT and votes["over"] >= total / 2:
        verdict = VALUE_OVER
    else:
        verdict = VALUE_FAIR
    return {"verdict": verdict, "median_upside": round(med, 1), "votes": votes, "total": total, "upsides": upsides}


def valuation_one(symbol: str) -> dict:
    info = _info(symbol)
    if info is None:
        return {"verdict": VALUE_NO_DATA, "error": "Yahoo не върна данни"}
    if info.get("quoteType") == "ETF":
        return {"verdict": VALUE_ETF}
    data = fair_value_methods(info)
    return {**data, **verdict_from(data)}


def valuation_many(symbols) -> dict:
    """{symbol: оценка} - по 3 паралелно (Yahoo ограничава честите заявки)."""
    symbols = list(symbols)
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(symbols, pool.map(valuation_one, symbols)))


def votes_text(v: dict) -> str:
    if not v.get("total"):
        return ""
    votes = v["votes"]
    return f"🟢{votes['under']} ⚪{votes['fair']} 🔴{votes['over']} от {v['total']}"


# ---------------------------------------------------------------- Б) AI проверка в интернет

VALUATION_PROMPT = """Ти си финансов анализатор. Трябва да прецениш дали акцията е ПОДЦЕНЕНА
спрямо справедливата си цена:

Акция: {name} (Yahoo символ: {symbol})

Потърси в интернет (най-много {max_searches} търсения) ГОТОВИ оценки на справедливата цена:
Morningstar (fair value и звезди), Simply Wall St (DCF fair value), GuruFocus (GF Value),
InvestingPro Fair Value, Alpha Spread, както и последните целеви цени на анализаторите.
Използвай САМО намереното - не измисляй числа. Ако нищо не намериш, кажи го.
В текстовете не слагай двойни кавички " (ползвай „ “ или единични ').

Отговори САМО с JSON (без друг текст), на български:
{{"verdict": "undervalued" | "fair" | "overvalued",
  "summary": "2-3 изречения: колко е справедливата цена според източниците и спрямо текущата",
  "estimates": [{{"source": "Morningstar", "fair_value": 123.4, "currency": "EUR"}}],
  "sources": [{{"title": "...", "url": "..."}}]}}"""


def valuation_prompt(name: str, symbol: str, is_etf: bool) -> str:
    return VALUATION_PROMPT.format(name=name, symbol=symbol, max_searches=fund.NEWS_MAX_SEARCHES)


def parse_valuation(data: dict, sources: list, searches: int) -> dict:
    estimates = []
    for e in data.get("estimates") or []:
        if isinstance(e, dict) and isinstance(e.get("fair_value"), (int, float)):
            estimates.append({"source": str(e.get("source") or "?"), "fair_value": float(e["fair_value"]),
                              "currency": e.get("currency") or ""})
    return {
        "verdict": _VALUE_LABELS.get(str(data.get("verdict", "")).lower(), VALUE_FAIR),
        "summary": data.get("summary") or "",
        "estimates": estimates[:6],
        "sources": sources[:5],
        "searches": searches,
    }


def new_ai_store() -> dict:
    """Празен склад за AI оценките (Streamlit слоят го пази с st.cache_resource за всички сесии)."""
    return {"lock": threading.Lock(), "date": None, "by_provider": {}, "checks": {}}


def ai_store_today(store: dict) -> dict:
    today = datetime.now().date().isoformat()
    with store["lock"]:
        if store["date"] != today:
            store.update(date=today, by_provider={}, checks={})
    return store
