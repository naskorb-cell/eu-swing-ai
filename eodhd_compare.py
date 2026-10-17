"""Проба: EODHD срещу Yahoo за няколко от ⭐ Селектирани (ръчно от GitHub Actions - eodhd_test.yml).

Сравнява дневните цени (1 година, коригирани за дивиденти и сплитове) и показва новините с
sentiment от EODHD за последните 30 дни. Резултатът отива в Telegram (и в лога - тикерите са
публични, няма позиции/суми). Безплатният план: 20 заявки/ден + 500 бонус; пробата харчи ~60
(дневни цени = 1, новини = 5 на инструмент).

    python eodhd_compare.py [--count 10] [--dry-run]
"""

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

from alerts import selected_items, send_telegram

API = "https://eodhd.com/api"
# Yahoo суфикс -> код на борсата в EODHD (без суфикс = САЩ)
EXCHANGES = {
    "DE": "XETRA", "F": "F", "MU": "MU", "PA": "PA", "AS": "AS", "MI": "MI", "MC": "MC", "BR": "BR",
    "VI": "VI", "LS": "LS", "HE": "HE", "ST": "ST", "CO": "CO", "OL": "OL", "SW": "SW", "L": "LSE",
    "IR": "IR", "WA": "WAR",
}


def eodhd_symbol(yahoo: str) -> str:
    base, _, suffix = yahoo.rpartition(".")
    if not base or suffix not in EXCHANGES:
        return f"{yahoo.replace('.', '-')}.US"
    return f"{base}.{EXCHANGES[suffix]}"


def pick(items: list, count: int) -> list:
    """Предимно европейски, по един от всяка борса, после допълва по реда в списъка."""
    eu = [x for x in items if "." in x["symbol"]]
    chosen, seen = [], set()
    for x in eu:
        ex = x["symbol"].rsplit(".", 1)[1]
        if ex not in seen:
            seen.add(ex)
            chosen.append(x)
    rest = [x for x in eu + items if x not in chosen]
    return (chosen + rest)[:count]


def get(path: str, token: str, **params):
    r = requests.get(f"{API}/{path}", params={"api_token": token, "fmt": "json", **params}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()


def compare_prices(item: dict, token: str) -> str:
    start = date.today() - timedelta(days=365)
    try:
        rows = get(f"eod/{eodhd_symbol(item['symbol'])}", token, **{"from": start.isoformat()})
        e = pd.Series({pd.Timestamp(r["date"]): r["adjusted_close"] for r in rows}, dtype=float)
    except Exception as ex:  # noqa: BLE001
        return f"EODHD: {str(ex)[:120]}"
    try:
        y = yf.Ticker(item["symbol"]).history(start=start.isoformat(), auto_adjust=True)["Close"]
        y.index = pd.DatetimeIndex(y.index.tz_localize(None).normalize())
    except Exception as ex:  # noqa: BLE001
        return f"EODHD: {len(e)} дни; Yahoo: грешка {type(ex).__name__}"
    if e.empty or y.empty:
        return f"дни: EODHD {len(e)} / Yahoo {len(y)} - няма какво да се сравни"
    both = e.index.intersection(y.index)
    diff = ((e[both] - y[both]).abs() / y[both] * 100) if len(both) else pd.Series(dtype=float)
    text = (f"дни: EODHD {len(e)} / Yahoo {len(y)} (само в EODHD {len(e.index.difference(y.index))}, "
            f"само в Yahoo {len(y.index.difference(e.index))})")
    if len(diff):
        text += f"; разлика в цената: медиана {diff.median():.2f}%, макс. {diff.max():.2f}%"
    text += f"; последен ден: EODHD {e.index.max():%d.%m} / Yahoo {y.index.max():%d.%m}"
    return text


def news(item: dict, token: str) -> str:
    try:
        rows = get("news", token, s=eodhd_symbol(item["symbol"]), limit=20,
                   **{"from": (date.today() - timedelta(days=30)).isoformat()})
    except Exception as ex:  # noqa: BLE001
        return f"новини: EODHD {str(ex)[:120]}"
    pol = [r["sentiment"]["polarity"] for r in rows if isinstance(r.get("sentiment"), dict)
           and r["sentiment"].get("polarity") is not None]
    if not rows:
        return "новини (30 дни): няма"
    text = f"новини (30 дни): {len(rows)}"
    if pol:
        avg = sum(pol) / len(pol)
        dot = "🟢" if avg >= 0.3 else "🔴" if avg < 0 else "⚪"
        text += f", sentiment {dot} {avg:+.2f}"
    title = (rows[0].get("title") or "").strip()
    return text + (f"\n   последна: {title[:90]}" if title else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("EODHD_API_KEY")
    if not token:
        sys.exit("Липсва EODHD_API_KEY в secrets.")
    items = pick(selected_items(), args.count)
    lines = [f"🔬 Проба EODHD срещу Yahoo ({len(items)} от ⭐ Селектирани, дневни цени за 1 г.)"]
    for x in items:
        lines.append(f"\n• {x['name']} ({eodhd_symbol(x['symbol'])})\n   {compare_prices(x, token)}\n   {news(x, token)}")
    try:
        user = get("user", token)
        lines.append(f"\nЗаявки днес: {user.get('apiRequests')} от {user.get('dailyRateLimit')}")
    except Exception:  # noqa: BLE001
        pass
    text = "\n".join(lines)
    print(text)
    if not args.dry_run and not send_telegram([text]):
        sys.exit(1)


if __name__ == "__main__":
    main()
