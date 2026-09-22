"""
select_liquid_universe.py

Месечен pre-screening на ЦЕЛИЯ eu_instruments.json универс: оценява
ликвидност (среден дневен оборот) и моментум (3-месечна доходност),
маха най-неликвидните, класира по моментум и записва топ N (по
подразбиране 500) в curated_universe.json - това е файлът, който
Streamlit приложението реално ползва за сканиране.

Пуска се веднъж месечно от GitHub Actions (.github/workflows/monthly_curate.yml).
Тежка операция (тегли данни за хиляди тикери) - затова НЕ се пуска на всеки
дневен fetch, а отделно, рядко.

FMP fundamentals (ново): за първите FMP_MAX_INSTRUMENTS от финалния избран
списък (не за целия универс - FMP free tier е 250 заявки/ден общо, а DCF+
съотношения искат 2 заявки/инструмент) добавяме DCF Fair Value upside % и
проста оценка за финансово здраве (Debt/Equity + Current Ratio) - най-
близкото безплатно съответствие на InvestingPro Fair Value/Health Score.
Мека добавка (като trending сигнала) - при грешка полето просто остава
празно, скриптът продължава без да спира.

Изисква: pip install yfinance pandas anthropic requests
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from anthropic import Anthropic

INSTRUMENTS_FILE = "eu_instruments.json"
OUTPUT_FILE = "curated_universe.json"
TOP_N = 500
LIQUID_KEEP_FRACTION = 0.6
CHUNK_SIZE = 50
TRENDING_RESERVED_SLOTS = 80  # колко от TOP_N места пазим за медийно/аналитично "трендиращи" имена

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_MAX_INSTRUMENTS = 100  # 2 заявки/инструмент => до 200 от 250-те дневни FMP заявки, с буфер

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


def fetch_fmp_fundamentals(symbol: str, api_key: str):
    """Тегли DCF Fair Value + основни съотношения за ЕДИН инструмент от FMP
    (2 заявки). Връща dict с dcf_upside_pct и financially_healthy, или None
    при грешка/липсващи данни - никога не гърми скрипта, само пропуска."""
    try:
        dcf_resp = requests.get(
            f"{FMP_BASE_URL}/discounted-cash-flow",
            params={"symbol": symbol, "apikey": api_key}, timeout=10,
        )
        dcf_resp.raise_for_status()
        dcf_data = dcf_resp.json()
        if not dcf_data:
            return None
        dcf_row = dcf_data[0] if isinstance(dcf_data, list) else dcf_data
        dcf_value = dcf_row.get("dcf")
        stock_price = dcf_row.get("Stock Price") or dcf_row.get("stockPrice")

        dcf_upside_pct = None
        if dcf_value and stock_price and stock_price > 0:
            dcf_upside_pct = round((float(dcf_value) - float(stock_price)) / float(stock_price) * 100, 1)

        ratios_resp = requests.get(
            f"{FMP_BASE_URL}/ratios-ttm",
            params={"symbol": symbol, "apikey": api_key}, timeout=10,
        )
        ratios_resp.raise_for_status()
        ratios_data = ratios_resp.json()
        ratios_row = (ratios_data[0] if isinstance(ratios_data, list) else ratios_data) if ratios_data else {}

        debt_equity = ratios_row.get("debtToEquityRatioTTM", ratios_row.get("debtEquityRatioTTM"))
        current_ratio = ratios_row.get("currentRatioTTM")

        financially_healthy = None
        if debt_equity is not None and current_ratio is not None:
            financially_healthy = bool(debt_equity < 2 and current_ratio > 1)

        if dcf_upside_pct is None and financially_healthy is None:
            return None

        return {
            "dcf_upside_pct": dcf_upside_pct,
            "financially_healthy": financially_healthy,
        }
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None


def enrich_with_fmp_fundamentals(top: list, api_key: str):
    """Обогатява първите FMP_MAX_INSTRUMENTS от финалния избран списък с
    DCF upside % и финансово здраве. НЕ променя кой е избран - само добавя
    информация, за да можеш после да приоритизираш/филтрираш по нея в
    приложението или AI анализа. При липсващ ключ прескача изцяло."""
    if not api_key:
        print("Предупреждение: липсва FMP_API_KEY - пропускам fundamentals enrichment.")
        return 0

    enriched_count = 0
    subset = top[:FMP_MAX_INSTRUMENTS]
    for i, item in enumerate(subset):
        fundamentals = fetch_fmp_fundamentals(item["symbol"], api_key)
        if fundamentals:
            item.update(fundamentals)
            enriched_count += 1
        if (i + 1) % 20 == 0:
            print(f"  FMP fundamentals: {i + 1}/{len(subset)} проверени...")
        time.sleep(0.25)  # леко забавяне, за да не гърмим rate limit-а на FMP

    return enriched_count


def build_candidate_list():
    data = json.loads(Path(INSTRUMENTS_FILE).read_text(encoding="utf-8"))
    instruments = data.get("instruments", [])

    candidates = []
    for inst in instruments:
        suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
        if suffix is None:
            continue
        symbol = f"{inst.get('shortName', '')}{suffix}"
        company_name = inst["name"]
        label = f"{inst.get('shortName', inst['ticker'])} ({company_name})"
        candidates.append({"name": label, "company_name": company_name, "symbol": symbol})
    return candidates


def score_in_batches(candidates: list) -> list:
    scored = []
    symbol_to_name = {c["symbol"]: c["name"] for c in candidates}
    symbol_to_company = {c["symbol"]: c["company_name"] for c in candidates}
    symbols = list(symbol_to_name.keys())

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
                    "name": symbol_to_name[symbol],
                    "company_name": symbol_to_company.get(symbol, ""),
                    "symbol": symbol,
                    "avg_dollar_volume": round(avg_dollar_volume, 0),
                    "momentum_3m_pct": round(momentum_3m_pct, 2),
                })
            except Exception:
                continue

        print(f"  обработени {min(i + CHUNK_SIZE, len(symbols))}/{len(symbols)} ...")

    return scored


def main():
    candidates = build_candidate_list()
    print(f"Общо кандидати за оценка: {len(candidates)}")

    scored = score_in_batches(candidates)
    print(f"Успешно оценени (с валидни данни): {len(scored)}")

    if not scored:
        print("Няма оценени инструменти - прекратявам без запис.")
        return

    # Ликвиден филтър: пазим само топ 60% по среден дневен оборот
    scored.sort(key=lambda x: x["avg_dollar_volume"], reverse=True)
    liquid_cutoff = max(int(len(scored) * LIQUID_KEEP_FRACTION), TOP_N)
    liquid_pool = scored[:liquid_cutoff]
    print(f"След ликвиден филтър: {len(liquid_pool)}")

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

    # Класация по моментум във всяка от двете групи поотделно
    trending_matches.sort(key=lambda x: x["momentum_3m_pct"], reverse=True)
    non_trending.sort(key=lambda x: x["momentum_3m_pct"], reverse=True)

    # Запазваме до TRENDING_RESERVED_SLOTS места за трендиращи (ако толкова има),
    # остатъкът от TOP_N се запълва по чист моментум от останалите ликвидни.
    reserved = trending_matches[:TRENDING_RESERVED_SLOTS]
    remaining_slots = TOP_N - len(reserved)
    fill = non_trending[:remaining_slots]
    top = reserved + fill

    # --- FMP DCF Fair Value + финансово здраве (мека добавка, само за
    # първите FMP_MAX_INSTRUMENTS - виж коментара при константата защо) ---
    fmp_api_key = os.environ.get("FMP_API_KEY")
    enriched_count = enrich_with_fmp_fundamentals(top, fmp_api_key)
    print(f"FMP fundamentals добавени за {enriched_count} инструмента.")

    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": len(scored),
        "count": len(top),
        "media_trending_count": len(reserved),
        "fmp_fundamentals_count": enriched_count,
        "instruments": top,
    }

    Path(OUTPUT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано {len(top)} инструмента в {OUTPUT_FILE} ({len(reserved)} медийно трендиращи, {enriched_count} с FMP fundamentals)")


if __name__ == "__main__":
    main()
