"""Telegram известия (пуска се от GitHub Actions - `.github/workflows/alerts.yml`, в работни дни в 10:00, 13:00
и 16:30 централноевропейско време - 1 ч след отварянето, обед и 1 ч преди затварянето на борсите).

Какво проверява:
  1. Photon „✅ Готов“ (Phase A/B, с хоризонта по подразбиране ~1 месец) за инструментите от ⭐ Селектирани -
     известие, когато инструмент СТАНЕ готов (не всеки път, докато е готов);
  2. отворените позиции в двата T212 акаунта (без пайовете): P&L минава ±100 / ±200 / ±300 € (всяко ниво
     веднъж, докато позицията е отворена) и позицията влиза в 4-та / 8-ма седмица;
  3. само с `--trend` (проверката в 16:30): нови инструменти в „🤝 И двата“ (метод А + Б,
     праговете по подразбиране) - пълен скан на универса.

Паметта (какво вече е пратено) е в alerts_state.json, който workflow-ът пази в кеша на Actions, НЕ в repo-то:
repo-то е публично, затова ключовете на позициите са хеширани с тайна, а в лога не се печатат позиции и суми.

Secrets (GitHub Actions): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, T212_API_KEY / T212_API_SECRET и по избор
T212_API_KEY_WIFE / T212_API_SECRET_WIFE. `--dry-run` - само печата съобщенията (без Telegram и без запис)."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

STATE_FILE = Path("alerts_state.json")
SELECTED_FILE = Path("selected_trends.json")
INSTRUMENTS_FILE = Path("eu_instruments.json")
ACCOUNTS = [("N", "T212_API_KEY", "T212_API_SECRET"), ("T", "T212_API_KEY_WIFE", "T212_API_SECRET_WIFE")]
PNL_LEVELS = (100, 200, 300)  # € - известие при първото минаване на всяко ниво (нагоре и надолу)
AGE_WEEKS = (4, 8)            # седмица 1 = първите 7 дни (като колоната „Седмица“ в Портфолио)
TREND_THRESHOLDS = (0.6, 10, (0.0, 10.0))  # мин. R², мин. наклон %/год, ATR % - по подразбиране в приложението
TREND_LIST_MAX = 15           # най-много толкова имена в едно известие за „И двата“
TELEGRAM_LIMIT = 3800         # знака на съобщение (лимитът на Telegram е 4096)
# часовете на проверките по централноевропейско време (борсите: 9:00-17:30): 1 ч след отварянето, обед,
# 1 ч преди затварянето (тогава и скана за „И двата“). GitHub графикът е в UTC, затова workflow-ът пуска и
# двата варианта (лятно/зимно време), а тук се изпълнява само този, който съвпада с часа в Берлин.
MARKET_TZ = ZoneInfo("Europe/Berlin")
SLOTS = {(10, 0): False, (13, 0): False, (16, 30): True}  # (час, минута): с ли е скана за „И двата“
SLOT_EARLY_MIN = 5            # пускане до 5 мин. преди часа още се брои за него


def current_slot(state: dict, now: datetime = None):
    """(час, минута) на проверката, която трябва да се изпълни сега, или None.

    GitHub пуска насрочените задачи с голямо закъснение (понякога часове) или изобщо не ги пуска, затова
    графикът е на всеки 30 мин. в работното време, а тук се взима последният настъпил час за проверка
    (по берлинско време), който днес още не е изпълнен (`state["slots"]`). Пропуснатите по-ранни часове не се
    наваксват поотделно - последният покрива и тях."""
    now = (now or datetime.now(timezone.utc)).astimezone(MARKET_TZ)
    if now.weekday() >= 5:
        return None
    today = now.date().isoformat()
    done = (state.get("slots") or {}).get(today, [])
    due = [(h, m) for h, m in SLOTS
           if now >= now.replace(hour=h, minute=m, second=0, microsecond=0) - timedelta(minutes=SLOT_EARLY_MIN)]
    if not due:
        return None
    slot = max(due)
    return None if f"{slot[0]:02d}:{slot[1]:02d}" in done else slot


def mark_slot_done(state: dict, slot, now: datetime = None):
    today = (now or datetime.now(timezone.utc)).astimezone(MARKET_TZ).date().isoformat()
    slots = state.get("slots") or {}
    state["slots"] = {today: sorted(set(slots.get(today, [])) | {f"{slot[0]:02d}:{slot[1]:02d}"})}  # само днешните


class _NoProgress:
    """Заместител на st.progress за скановете извън Streamlit."""

    def progress(self, *args, **kwargs):
        pass


def secret_hash(text: str) -> str:
    """Ключ за паметта без името на позицията (repo-то и кешът са публично достъпни)."""
    salt = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    return hashlib.sha256(f"{salt}|{text}".encode()).hexdigest()[:16]


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict):
    state["updated"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def send_telegram(messages: list) -> bool:
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Липсват TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID в secrets - не пращам.")
        return False
    chunks, current = [], ""
    for m in messages:
        if current and len(current) + len(m) + 2 > TELEGRAM_LIMIT:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{m}" if current else m
    if current:
        chunks.append(current)
    ok = True
    for text in chunks:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=20,
                          json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True})
        ok = ok and r.ok
        if not r.ok:
            print(f"Telegram върна {r.status_code}")
    return ok


def selected_items() -> list:
    try:
        return json.loads(SELECTED_FILE.read_text(encoding="utf-8")).get("items", [])
    except (OSError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------- 1. Photon „Готов“ за Селектираните

def photon_params() -> dict:
    import photon
    d = photon.PROFILE_DEFAULTS  # профил „Стандартен“, както по подразбиране в приложението
    return {
        "swing_order_weekly": d["ph_swo_w"], "swing_order_daily": d["ph_swo_d"], "min_range_atr": d["ph_min_range"],
        "closed_weeks_only": True, "choch_max_age": d["ph_choch_age"], "poi_atr_mult": d["ph_poi_atr"],
        "stop_atr_buffer": d["ph_stop_buf"], "min_rr": d["ph_min_rr"], "choch_1h": False,
    }


STATUS = []  # кратко обобщение на всяка проверка - за „✅ Проверката мина“ при ръчно пускане


def check_photon(state: dict, messages: list):
    import photon
    from universe import curated_file_mtime, load_curated_symbol_info
    items = selected_items()
    if not items:
        STATUS.append("Photon: няма ⭐ Селектирани")
        return
    currencies, _, _ = load_curated_symbol_info(curated_file_mtime())
    results, watch, *_ = photon.run_photon_scan({x["name"]: x["symbol"] for x in items}, photon_params(),
                                                _NoProgress(), currencies)
    results, _ = photon.split_by_horizon(results, watch)  # като в таба „✅ Готови“ (хоризонт ~1 месец)
    ready = {x.symbol: x for x in results}
    before = set(state.get("ready", []))
    for symbol, x in ready.items():
        if symbol in before:
            continue
        plan = photon.trade_plan(x)
        kind = "Лимит" if plan["is_limit"] else "Buy stop"
        rr = f" · R/R {plan['rr']:.1f}" if plan["rr"] else ""
        messages.append(f"🟢 Photon: {x.name} - Phase {x.phase} готов\n"
                        f"{kind} {plan['entry']:.2f} · Stop {x.stop:.2f} · Цел 1 {x.daily_resistance:.2f} {x.currency}{rr}\n"
                        f"{x.note}")
    state["ready"] = sorted(ready)
    STATUS.append(f"Photon: {len(items)} ⭐ Селектирани, {len(ready)} готови, {len(set(ready) - before)} нови")
    print(f"Photon: {len(items)} селектирани, {len(ready)} готови, {len(set(ready) - before)} нови")


# ---------------------------------------------------------------- 2. Отворени позиции

def instrument_names() -> dict:
    try:
        data = json.loads(INSTRUMENTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {i["ticker"]: i.get("name") or i["ticker"] for i in data.get("instruments", []) if i.get("ticker")}


def ordinal(n: int) -> str:
    """4 -> „4-та“, 8 -> „8-ма“, 1 -> „1-ва“, 2 -> „2-ра“ (за „седмица“)."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n}-ва"
    if n % 10 == 2 and n % 100 != 12:
        return f"{n}-ра"
    if n % 10 in (7, 8) and n % 100 not in (17, 18):
        return f"{n}-ма"
    return f"{n}-та"


def pnl_level(pnl: float) -> int:
    """+300 / +200 / +100 / 0 / -100 ... - най-високото минато ниво."""
    crossed = [lvl for lvl in PNL_LEVELS if abs(pnl) >= lvl]
    return (max(crossed) if crossed else 0) * (1 if pnl > 0 else -1)


def error_reason(e: Exception) -> str:
    """Кратка причина за грешка - за Telegram (личен чат); в публичния лог се печата само видът."""
    response = getattr(e, "response", None)
    if response is not None and getattr(response, "status_code", None):
        return f"HTTP {response.status_code} от {getattr(response, 'url', '').split('?')[0]}"
    return f"{type(e).__name__}: {str(e)[:160]}"


def check_positions(state: dict, messages: list):
    import t212_portfolio as t212
    names = instrument_names()
    pnl_seen, age_seen = state.get("pnl", {}), state.get("age", {})
    open_keys, now, accounts, failed, per_account = set(), pd.Timestamp.now(tz="UTC"), 0, [], []
    for label, key_env, secret_env in ACCOUNTS:
        key, secret = os.environ.get(key_env), os.environ.get(secret_env)
        if not key or not secret:
            continue
        accounts += 1
        try:  # грешка в единия акаунт не спира другия
            auth = t212.build_auth_header(key, secret)
            df = t212.exclude_pies(t212.fetch_open_positions(t212.T212_ENV_TO_BASE_URL["live"], auth))
        except Exception as e:  # noqa: BLE001
            failed.append(label)
            print(f"Позиции ({label}): грешка {type(e).__name__}")
            messages.append(f"⚠️ Позиции ({label}): T212 не върна позициите - {error_reason(e)}")
            continue
        per_account.append(f"{label} - {len(df)}")
        for _, row in df.iterrows():
            k = secret_hash(f"{label}:{row['Тикер']}")
            open_keys.add(k)
            name = names.get(row["Тикер"], row["Тикер"])
            pnl = float(row["P&L (€)"] or 0)
            level = pnl_level(pnl)
            seen = pnl_seen.setdefault(k, {"up": 0, "down": 0})
            if level > seen["up"]:
                seen["up"] = level
                messages.append(f"💰 {name} ({label}): печалба над +{level} € (сега {pnl:+.0f} €, {row['P&L (%)']:+.1f}%)")
            elif level < 0 and -level > seen["down"]:
                seen["down"] = -level
                messages.append(f"🔻 {name} ({label}): загуба над -{-level} € (сега {pnl:+.0f} €, {row['P&L (%)']:+.1f}%)")
            opened = row.get("Отворена")
            if opened is not None and pd.notna(opened):
                week = (now - opened).days // 7 + 1
                done = age_seen.setdefault(k, [])
                due = [w for w in AGE_WEEKS if week >= w and w not in done]
                if due:
                    done.extend(due)
                    icon = "🟥" if max(due) >= 8 else "🟧"
                    messages.append(f"{icon} {name} ({label}): {ordinal(week)} седмица отворена")
    # затворените позиции се забравят - при ново отваряне известията започват отначало (но не и ако
    # някой акаунт не е отговорил - тогава не знаем кои негови позиции са затворени)
    state["pnl"] = {k: v for k, v in pnl_seen.items() if k in open_keys or failed}
    state["age"] = {k: v for k, v in age_seen.items() if k in open_keys or failed}
    STATUS.append("Позиции: " + (", ".join(per_account) + " отворени" if per_account else "няма отговорил акаунт")
                  + (f" (грешка: {', '.join(failed)})" if failed else ""))
    print(f"Позиции: {accounts} акаунта ({len(failed)} с грешка), {len(open_keys)} отворени позиции")


# ---------------------------------------------------------------- 3. Нови в „🤝 И двата“ (вечер)

def check_trend(state: dict, messages: list):
    import trend_scanner as ts
    raw = ts.run_trend_scan(ts.trend_universe(), _NoProgress())
    if raw.empty:
        print("Тренд: Yahoo не върна данни - пропускам")
        return
    df = ts.classify(raw, *TREND_THRESHOLDS)
    both = df[df["А"] & df["Б"]]
    current = dict(zip(both["Тикер"], both["Име"]))
    selected = {x["symbol"] for x in selected_items()}
    before = state.get("both")
    if before is None:
        messages.append(f"📈 „🤝 И двата“ (А + Б): {len(current)} инструмента днес. Оттук нататък - известие за "
                        f"всеки НОВ в групата (проверка всяка вечер).")
    else:
        new = [s for s in current if s not in set(before)]
        if new:
            lines = [f"• {current[s]}" + (" ⭐" if s in selected else "") for s in new[:TREND_LIST_MAX]]
            more = f"\n... и още {len(new) - TREND_LIST_MAX}" if len(new) > TREND_LIST_MAX else ""
            messages.append(f"📈 Нови в „🤝 И двата“ (А + Б) - {len(new)}:\n" + "\n".join(lines) + more)
    state["both"] = sorted(current)
    STATUS.append(f"„🤝 И двата“: {len(current)} инструмента")
    print(f"Тренд: {len(raw)} сканирани, {len(current)} в „И двата“")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trend", action="store_true", help="и пълния скан за нови в „И двата“ (вечерта)")
    parser.add_argument("--dry-run", action="store_true", help="само печата съобщенията, без Telegram и запис")
    parser.add_argument("--summary", action="store_true",
                        help="при ръчно пускане: ако няма нищо ново, праща „✅ Проверката мина“ с кратко обобщение")
    parser.add_argument("--scheduled", action="store_true",
                        help="от графика: само ако е час за проверка по европейско време (скана за тренда - по часа)")
    args = parser.parse_args()
    state, messages, failed = load_state(), [], []
    slot = None
    if args.scheduled:
        slot = current_slot(state)
        if slot is None:
            print("Няма неизпълнена проверка за сега - пропускам.")
            return
        # скана за „И двата“ - на проверката в 16:30 (и ако тя е закъсняла)
        args.trend = args.trend or SLOTS[slot]
        print(f"Проверка за {slot[0]:02d}:{slot[1]:02d} европейско време" + (" + тренд" if args.trend else ""))

    checks = [("Photon", check_photon), ("Позиции", check_positions)] + ([("Тренд", check_trend)] if args.trend else [])
    for title, check in checks:
        try:
            check(state, messages)
        except Exception as e:  # noqa: BLE001 - една счупена проверка не спира останалите
            failed.append(f"{title} ({error_reason(e)})")
            print(f"{title}: грешка {type(e).__name__}")
    if failed:
        messages.append(f"⚠️ Проверката не мина за: {', '.join(failed)} (виж лога в GitHub Actions)")

    if args.summary and not messages:
        messages.append("✅ Проверката мина - няма нищо ново.\n" + "\n".join(f"• {x}" for x in STATUS))

    if args.dry_run:
        print("\n\n".join(messages) or "(няма известия)")
        return
    if slot is not None:
        mark_slot_done(state, slot)
    if messages and not send_telegram(messages):
        sys.exit(1)  # паметта не се записва - следващият път се опитва пак
    save_state(state)
    print(f"Известия: {len(messages)}")


if __name__ == "__main__":
    main()
