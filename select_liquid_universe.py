"""
select_liquid_universe.py

Месечен pre-screening на ЦЕЛИЯ eu_instruments.json универс: оценява
ликвидност (среден дневен оборот) и моментум (3-месечна доходност),
взима капитализация (акции) / AUM (ETF) от Yahoo и записва в
curated_universe.json ВСИЧКИ инструменти, които минават критериите от
universe_rules.py (без ливъриджнати/short ETP). Това е файлът, който
Streamlit приложението реално сканира; slider-ите там само стесняват още.

Пуска се веднъж месечно от GitHub Actions (.github/workflows/monthly_curate.yml).
Тежка операция (тегли данни за хиляди тикери) - затова НЕ се пуска на всеки
дневен fetch, а отделно, рядко.

Бел.: пробвахме да добавим FMP DCF Fair Value/Ratings Snapshot enrichment
тук, но FMP free tier връща 402 Payment Required за всички европейски
тикери (.DE/.PA/.MI/.AS) - покритието за EU борси изисква платен план.
Премахнато - виж историята на repo-то, ако решиш да платиш за EODHD или
FMP по-нататък и искаш да го върнем.

Изисква: pip install yfinance pandas anthropic
"""

import json
import os
import time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf
from anthropic import Anthropic

import universe_rules as rules

INSTRUMENTS_FILE = "eu_instruments.json"
OUTPUT_FILE = "curated_universe.json"
CHUNK_SIZE = 50
INFO_WORKERS = 4  # паралелни заявки за капитализация/AUM (повече = риск от 429 от Yahoo)

# Gettex (.MU) листванията нямат използваеми данни в Yahoo - за тях търсим по ISIN
# основното листване. Предпочитан суфикс на Yahoo символа по държава от ISIN
# ("" = US листване без суфикс).
GETTEX_SUFFIX = ".MU"
PRIMARY_SUFFIX_BY_COUNTRY = {
    "US": ("",), "CA": (".TO", ".V", ".CN", ""), "SE": (".ST",), "NO": (".OL",), "FI": (".HE",),
    "DK": (".CO",), "JP": (".T",), "GB": (".L",), "CH": (".SW",), "DE": (".DE",), "FR": (".PA",),
    "NL": (".AS",), "IT": (".MI",), "ES": (".MC",), "BE": (".BR",), "AT": (".VI",), "PT": (".LS",),
    "IE": (".IR", ".L", ""), "AU": (".AX",), "HK": (".HK",), "IL": (".TA", ""),
}

EXCHANGE_NAME_TO_YAHOO_SUFFIX = [
    ("XETRA", ".DE"), ("FRANKFURT", ".DE"), ("DEUTSCHE", ".DE"), ("GETTEX", ".MU"),
    ("PARIS", ".PA"), ("AMSTERDAM", ".AS"), ("MILAN", ".MI"), ("BORSA ITALIANA", ".MI"),
]


def exchange_to_yahoo_suffix(exchange_name: str):
    name_upper = (exchange_name or "").upper()
    for keyword, suffix in EXCHANGE_NAME_TO_YAHOO_SUFFIX:
        if keyword in name_upper:
            return suffix
    return None


def normalize_company_name(name: str) -> str:
    """Опростена нормализация за съпоставяне на имена от новини с нашите."""
    name = name.upper()
    for suffix in [" PLC", " SE", " AG", " SA", " NV", " AB", " ASA", " GROUP",
                   " HOLDING", " HOLDINGS", " CORPORATION", " CORP", " INC.",
                   " INC", " LTD", " CO.", " CO", " N.V.", " S.A.", " GMBH"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    for ch in [".", ",", "'", "-", "&"]:
        name = name.replace(ch, " ")
    return " ".join(name.split()).strip()


def fetch_trending_names(api_key: str) -> set:
    """Вика Claude с вграден web search tool да претърси финансови медии,
    анализаторски бележки и trading форуми за компании, за които в момента
    се говори най-много - връща множество нормализирани имена за
    кръстосване с нашия ликвиден пул. Мека добавка, не твърд филтър:
    ако заявката се провали, продължаваме само с ликвидност+моментум."""
    try:
        client = Anthropic(api_key=api_key)
        prompt = """Претърси актуални финансови новини, анализаторски доклади и
популярни trading форуми (напр. финансови секции на големи медии,
Bloomberg/Reuters/CNBC отразяване, обсъждания в инвеститорски общности)
за КОМПАНИИ И ETF-и, търгувани на европейски борси (Германия, Франция,
Италия, Нидерландия) или големи американски компании, търгувани в евро
там, които са особено активно обсъждани, анализирани или споменавани
през последния месец - независимо дали заради ръст, спад, нови продукти,
регулаторни новини или друга причина.

СТРИКТЕН ФОРМАТ НА ОТГОВОРА (много важно, спазвай точно):
- Отговори САМО с списъка, без увод, без заключение, без обяснения преди
  или след него.
- Точно един ред на компания.
- Всеки ред трябва да съдържа САМО името на компанията, нищо друго -
  без номерация (1. 2. 3.), без тирета, без звездички, без markdown,
  без коментар защо е избрана.
- Пример за целия очакван отговор (само този формат, нищо друго):
ASML Holding
Siemens Energy
LVMH

До 60 реда общо."""

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=8192,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        full_text = "\n".join(text_parts)

        names = set()
        for line in full_text.splitlines():
            line = line.strip()
            # маха водещи номера/тирета/звездички от различни формати списъци
            line = line.lstrip("0123456789.)-•*: ").strip()
            # маха markdown удебеляване **Компания**
            line = line.strip("*").strip()
            if not line or len(line) > 100:
                continue
            # прескачаме очевидни заглавия/секции, не имена на компании
            if line.endswith(":") or line.lower().startswith(("категория", "списък", "топ ")):
                continue
            names.add(normalize_company_name(line))

        print(f"Намерени {len(names)} трендиращи имена от медиен анализ. (stop_reason: {response.stop_reason})")
        if not names:
            # debug: показваме суровия отговор, за да разберем защо parsing-ът е дал 0
            snippet = full_text[:1500]
            print(f"--- Суров отговор от Claude (за диагностика) ---\n{snippet}\n--- край на суровия отговор ---")
        return names
    except Exception as e:
        print(f"Предупреждение: неуспешно теглене на трендиращи имена ({e}), продължавам без тях.")
        return set()


def build_candidate_list():
    data = json.loads(Path(INSTRUMENTS_FILE).read_text(encoding="utf-8"))
    instruments = data.get("instruments", [])

    candidates, skipped_leveraged = [], 0
    for inst in instruments:
        if inst.get("type") not in ("STOCK", "ETF"):
            continue
        suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
        if suffix is None:
            continue
        symbol = f"{inst.get('shortName', '')}{suffix}"
        company_name = inst["name"]
        label = f"{inst.get('shortName', inst['ticker'])} ({company_name})"
        if rules.is_leveraged_or_short_etp(label):
            skipped_leveraged += 1
            continue
        candidates.append({
            "name": label, "company_name": company_name, "symbol": symbol,
            "type": inst["type"], "isin": inst.get("isin", ""),
        })
    print(f"Изключени ливъриджнати/short ETP: {skipped_leveraged}")
    return candidates


def pick_primary_symbol(quotes: list, isin: str):
    """Избира основното листване от резултатите на Yahoo search по държавата от ISIN."""
    symbols = [q.get("symbol", "") for q in quotes if q.get("quoteType") == "EQUITY" and q.get("symbol")]
    if not symbols:
        return None
    for suffix in PRIMARY_SUFFIX_BY_COUNTRY.get(isin[:2].upper(), ("",)):
        for sym in symbols:
            if (suffix == "" and "." not in sym) or (suffix and sym.endswith(suffix)):
                return sym
    no_suffix = [sym for sym in symbols if "." not in sym]
    return no_suffix[0] if no_suffix else symbols[0]


def resolve_primary_symbol(item: dict) -> dict:
    """За Gettex инструмент: търси по ISIN основното листване в Yahoo и подменя
    symbol-а (оригиналът остава в t212_symbol). При неуспех оставя .MU."""
    for attempt in range(3):
        try:
            quotes = yf.Search(item["isin"], max_results=10, news_count=0).quotes
            primary = pick_primary_symbol(quotes, item["isin"])
            if primary:
                item["t212_symbol"] = item["symbol"]
                item["symbol"] = primary
            return item
        except Exception:
            time.sleep(2 * (attempt + 1))
    return item


def eur_rates(currencies: set) -> dict:
    """Курс 1 EUR = X <валута> за всяка валута (GBp = пенсове)."""
    rates = {"EUR": 1.0}
    for cur in currencies - {"EUR", None}:
        base, factor = ("GBP", 100) if cur in ("GBp", "GBX") else (cur, 1)
        try:
            df = yf.download(f"EUR{base}=X", period="5d", interval="1d", progress=False, auto_adjust=True)
            close = df["Close"].dropna()
            last = close.iloc[-1] if close.ndim == 1 else close.iloc[-1, 0]
            rates[cur] = float(last) * factor
        except Exception as e:
            print(f"  Няма курс EUR/{cur}: {e}")
    return rates


def convert_to_eur(items: list) -> list:
    """Оборотът и капитализацията/AUM са във валутата на листването - превръщаме
    ги в €, за да важат праговете. Без курс инструментът отпада."""
    rates = eur_rates({x.get("currency") or "EUR" for x in items})
    converted = []
    for item in items:
        cur = item.get("currency") or "EUR"
        rate = rates.get(cur)
        if not rate:
            continue
        item["currency"] = cur
        for field in ("avg_dollar_volume", "market_cap", "aum"):
            if item.get(field):
                item[field] = round(item[field] / rate, 0)
        converted.append(item)
    return converted


def score_in_batches(candidates: list) -> list:
    scored = []
    by_symbol = {c["symbol"]: c for c in candidates}
    symbols = list(by_symbol.keys())

    for i in range(0, len(symbols), CHUNK_SIZE):
        chunk = symbols[i : i + CHUNK_SIZE]
        try:
            df = yf.download(
                chunk, period="3mo", interval="1d", group_by="ticker",
                progress=False, auto_adjust=True, threads=True,
            )
        except Exception as e:
            print(f"  Грешка при батч {i}-{i+len(chunk)}: {e}")
            continue

        for symbol in chunk:
            try:
                sub = df[symbol] if isinstance(df.columns, pd.MultiIndex) else df
                if sub.empty or len(sub) < 40 or sub["Close"].isna().all():
                    continue
                sub = sub.dropna(subset=["Close"])

                avg_dollar_volume = float((sub["Volume"] * sub["Close"]).tail(20).mean())
                momentum_3m_pct = float((sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1) * 100)

                if avg_dollar_volume <= 0:
                    continue

                scored.append({
                    **by_symbol[symbol],
                    "avg_dollar_volume": round(avg_dollar_volume, 0),
                    "momentum_3m_pct": round(momentum_3m_pct, 2),
                })
            except Exception:
                continue

        print(f"  обработени {min(i + CHUNK_SIZE, len(symbols))}/{len(symbols)} ...")

    return scored


def min_possible_turnover(item: dict) -> float:
    """Най-ниският праг за оборот, който инструментът изобщо може да мине -
    ползва се за евтин pre-filter преди бавните заявки за капитализация/AUM."""
    if item["type"] == "ETF":
        return rules.ETF_MIN_TURNOVER
    return rules.STOCK_MIN_TURNOVER if rules.is_home_listing(item["isin"]) else rules.STOCK_FOREIGN_MIN_TURNOVER


def fetch_size(item: dict) -> dict:
    """Капитализация (акции) или AUM (ETF) от Yahoo .info; None при липса/грешка.
    До 3 опита с пауза при rate limit."""
    for attempt in range(3):
        try:
            info = yf.Ticker(item["symbol"]).info or {}
            item["currency"] = info.get("currency") or "EUR"
            if item["type"] == "STOCK":
                item["market_cap"] = info.get("marketCap")
            else:
                item["aum"] = info.get("totalAssets") or info.get("netAssets")
            return item
        except Exception:
            time.sleep(2 * (attempt + 1))
    return item


def main():
    candidates = build_candidate_list()
    print(f"Общо кандидати за оценка: {len(candidates)}")

    gettex = [c for c in candidates if c["symbol"].endswith(GETTEX_SUFFIX)]
    if gettex and not hasattr(yf, "Search"):
        print("Предупреждение: yfinance няма yf.Search - Gettex инструментите остават с .MU")
    elif gettex:
        with ThreadPoolExecutor(max_workers=INFO_WORKERS) as pool:
            list(pool.map(resolve_primary_symbol, gettex))
        resolved = sum(1 for c in gettex if "t212_symbol" in c)
        print(f"Gettex: намерено основно листване за {resolved}/{len(gettex)}")

    scored = score_in_batches(candidates)
    print(f"Успешно оценени (с валидни данни): {len(scored)}")

    if not scored:
        print("Няма оценени инструменти - прекратявам без запис.")
        return

    # 1) евтин pre-filter по оборот, 2) капитализация/AUM само за минелите, 3) пълните критерии
    prefiltered = [x for x in scored if x["avg_dollar_volume"] >= min_possible_turnover(x)]
    print(f"След pre-filter по оборот: {len(prefiltered)} - тегля капитализация/AUM...")
    with ThreadPoolExecutor(max_workers=INFO_WORKERS) as pool:
        prefiltered = list(pool.map(fetch_size, prefiltered))
    # pre-filter-ът е по оборот във валутата на листването - за валути, по-слаби от
    # еврото (USD, SEK, JPY...), той е по-хлабав, така че нищо валидно не отпада;
    # истинската проверка е след превръщането в €
    prefiltered = convert_to_eur(prefiltered)

    liquid_pool, reject_counts, rejected = [], {}, []
    for item in prefiltered:
        ok, why = rules.passes_liquidity(item)
        if ok:
            liquid_pool.append(item)
        else:
            reject_counts[why] = reject_counts.get(why, 0) + 1
            rejected.append({
                "name": item["name"], "symbol": item["symbol"], "reason": why,
                "avg_dollar_volume": item.get("avg_dollar_volume"),
                "market_cap": item.get("market_cap"), "aum": item.get("aum"),
            })
    print(f"Минали критериите: {len(liquid_pool)}; отпаднали: {reject_counts}")
    etf_no_aum = sum(1 for x in liquid_pool if x["type"] == "ETF" and not x.get("aum"))
    if etf_no_aum:
        print(f"  (от тях {etf_no_aum} ETF без данни за AUM в Yahoo - проверени само по оборот)")

    # --- Медиен/аналитичен "buzz" сигнал (мека добавка) ---
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    trending_names = fetch_trending_names(anthropic_api_key) if anthropic_api_key else set()

    def is_trending_match(company_name: str) -> bool:
        norm = normalize_company_name(company_name)
        if not norm:
            return False
        if norm in trending_names:
            return True
        for t in trending_names:
            if len(t) >= 3 and (t in norm or norm in t):
                return True
        return False

    for item in liquid_pool:
        item["media_trending"] = is_trending_match(item.get("company_name", ""))

    trending_matches = [x for x in liquid_pool if x["media_trending"]]
    non_trending = [x for x in liquid_pool if not x["media_trending"]]
    print(f"Съвпадения с медийно трендиращи имена: {len(trending_matches)}")

    # Записваме всички ликвидни; редът (трендиращи първо, после по моментум)
    # има значение само за фалбек сценарии с лимит на броя в UI.
    trending_matches.sort(key=lambda x: x["momentum_3m_pct"], reverse=True)
    non_trending.sort(key=lambda x: x["momentum_3m_pct"], reverse=True)
    top = trending_matches + non_trending

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": len(scored),
        "count": len(top),
        "stock_count": sum(1 for x in top if x["type"] == "STOCK"),
        "etf_count": sum(1 for x in top if x["type"] == "ETF"),
        "media_trending_count": len(trending_matches),
        # минали pre-filter-а по оборот, но отпаднали на капитализация/AUM - за
        # проверка в UI дали Yahoo не е дал грешни числа
        "rejected": sorted(rejected, key=lambda x: -(x["avg_dollar_volume"] or 0)),
        "criteria": {
            "stock_min_market_cap": rules.STOCK_MIN_MARKET_CAP,
            "stock_min_turnover": rules.STOCK_MIN_TURNOVER,
            "stock_foreign_min_turnover": rules.STOCK_FOREIGN_MIN_TURNOVER,
            "etf_min_aum": rules.ETF_MIN_AUM,
            "etf_min_turnover": rules.ETF_MIN_TURNOVER,
        },
        "instruments": top,
    }

    Path(OUTPUT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано {len(top)} инструмента в {OUTPUT_FILE} ({len(trending_matches)} медийно трендиращи)")


if __name__ == "__main__":
    main()
