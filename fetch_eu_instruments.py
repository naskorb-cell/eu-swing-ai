"""
fetch_eu_instruments.py

Изтегля пълния списък инструменти от Trading 212 Public API,
маха US тикери и филтрира само европейски акции/ETF-и.

Изисква следните environment променливи:
    T212_API_KEY
    T212_API_SECRET

Използване:
    export T212_API_KEY="..."
    export T212_API_SECRET="..."
    python fetch_eu_instruments.py

Резултат:
    eu_instruments.json  -- филтриран списък, готов за screening скрипта
"""

import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

# --- Настройки -------------------------------------------------------------

# "live" за реалната ти сметка, "demo" за paper trading
ENVIRONMENT = os.environ.get("T212_ENV", "live")
BASE_URL = f"https://{ENVIRONMENT}.trading212.com/api/v0"

# Кои валути да пазим. Само EUR - сметката в Trading 212 е в евро, а търговия
# в друга валута (SEK, NOK, DKK, PLN, GBP, CHF) минава през конвертиране с
# такса от Trading 212, което не е желателно.
ALLOWED_CURRENCIES = {"EUR"}

# Суфикси на тикери, които T212 използва за US борси -- изключваме ги
US_SUFFIXES = ("_US_EQ",)

CACHE_FILE = "t212_instruments_raw.json"
OUTPUT_FILE = "eu_instruments.json"


def get_auth_header() -> str:
    key = os.environ.get("T212_API_KEY")
    secret = os.environ.get("T212_API_SECRET")
    if not key or not secret:
        sys.exit("Липсват T212_API_KEY / T212_API_SECRET в environment.")
    token = base64.b64encode(f"{key}:{secret}".encode()).decode()
    return f"Basic {token}"


def fetch_json(path: str, auth_header: str):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"Authorization": auth_header})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        sys.exit(f"HTTP {e.code} при заявка към {path}: {body}")
    except urllib.error.URLError as e:
        sys.exit(f"Мрежова грешка при {path}: {e}")


def load_instruments(auth_header: str):
    # Кешираме локално, защото endpoint-ът е rate-limited (~1 заявка / 50s)
    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        if age < 6 * 3600:  # ползвай кеша ако е по-нов от 6 часа
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

    print("Тегля инструменти от Trading 212 API ...")
    data = fetch_json("/equity/metadata/instruments", auth_header)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


# Ликвидността на акциите НЕ се филтрира тук: месечният select_liquid_universe.py
# проверява капитализация и оборот за всички. (Досега тук имаше whitelist от
# STOXX 600 + S&P 500 - статичният STOXX списък беше непълен и изрязваше акции
# като Nordex, Generali, DWS, De'Longhi още преди подбора.)


EXCLUDED_EXCHANGES = {
    "London Stock Exchange",
    "London Stock Exchange AIM",
    "SIX Swiss Exchange",
}


def is_european(instrument: dict) -> bool:
    ticker = instrument.get("ticker", "")
    currency = instrument.get("currencyCode", "")
    exchange_name = instrument.get("exchangeName", "")

    if ticker.endswith(US_SUFFIXES):
        return False
    if currency not in ALLOWED_CURRENCIES:
        return False
    if exchange_name in EXCLUDED_EXCHANGES:
        return False
    return True


EXCHANGES_CACHE_FILE = "t212_exchanges_raw.json"


def load_exchanges(auth_header: str):
    """Връща речник workingScheduleId -> име на борсата (реалната, не гадана)."""
    if os.path.exists(EXCHANGES_CACHE_FILE):
        age = time.time() - os.path.getmtime(EXCHANGES_CACHE_FILE)
        if age < 24 * 3600:
            with open(EXCHANGES_CACHE_FILE, "r", encoding="utf-8") as f:
                exchanges = json.load(f)
        else:
            exchanges = None
    else:
        exchanges = None

    if exchanges is None:
        print("Тегля списък с борси от Trading 212 API ...")
        exchanges = fetch_json("/equity/metadata/exchanges", auth_header)
        with open(EXCHANGES_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(exchanges, f, ensure_ascii=False, indent=2)

    schedule_to_exchange = {}
    for exch in exchanges:
        exch_name = exch.get("name", "UNKNOWN")
        for schedule in exch.get("workingSchedules", []):
            schedule_to_exchange[schedule["id"]] = exch_name
    return schedule_to_exchange


def main():
    auth_header = get_auth_header()
    instruments = load_instruments(auth_header)
    schedule_to_exchange = load_exchanges(auth_header)

    print(f"Общо инструменти от T212: {len(instruments)}")
    print(f"Намерени борси: {len(set(schedule_to_exchange.values()))}")

    # Прикачваме истинското име на борсата към всеки инструмент
    for inst in instruments:
        inst["exchangeName"] = schedule_to_exchange.get(
            inst.get("workingScheduleId"), "UNKNOWN"
        )

    filtered = [i for i in instruments if is_european(i)]
    print(f"След филтър за европейски борси/валути: {len(filtered)}")

    distinct_exchanges = sorted(set(i["exchangeName"] for i in filtered))
    print("Борси в резултата:", ", ".join(distinct_exchanges))

    # Разделяме на акции и ETF-и за по-лесна обработка по-нататък
    etfs = [i for i in filtered if i.get("type") == "ETF"]
    stocks = [i for i in filtered if i.get("type") == "STOCK"]
    print(f"  -> ETF-и: {len(etfs)}")
    print(f"  -> Акции: {len(stocks)}")

    result = {
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(filtered),
        "instruments": filtered,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Записано в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
