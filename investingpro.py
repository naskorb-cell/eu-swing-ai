"""Данни от InvestingPro (Fair Value, Financial Health...) от експорта на Screener-а.

InvestingPro няма API - Наско експортира Screener-а (изгледа „Nasko“) 1-2 пъти в месеца като Excel/CSV
(до 1000 реда на файл: Европа + Великобритания над 2 млрд., отделно САЩ над 20 млрд.) и го качва тук.
Редовете се съпоставят с нашите символи (Yahoo) по „Full Ticker“ (`ENXTPA:SAF` -> `SAF.PA`), резервно
по името. Данните са платени/лицензирани, затова НЕ се пазят в публичното repo, а в частно -
`IP_DATA_REPO` в Streamlit secrets (напр. „naskorb-cell/eu-swing-data“), със същия `GITHUB_TOKEN`.
"""

import io
import json
import re
from collections import Counter
from datetime import date

import pandas as pd
import streamlit as st

from universe import github_get_file, github_write_file

DATA_FILE = "investingpro.json"
STALE_DAYS = 35  # след толкова дни - напомняне за нов експорт

# борсата в „Full Ticker“ (кодовете на S&P Capital IQ, които ползва InvestingPro) -> суфикс в Yahoo
EXCHANGE_SUFFIX = {
    "XTRA": ".DE", "DB": ".F", "ENXTPA": ".PA", "ENXTAM": ".AS", "ENXTBR": ".BR", "ENXTLS": ".LS",
    "ISE": ".IR", "ENXTDU": ".IR", "BIT": ".MI", "BME": ".MC", "WBAG": ".VI", "LSE": ".L", "HLSE": ".HE",
    "OM": ".ST", "CPSE": ".CO", "OB": ".OL", "SWX": ".SW", "ATSE": ".AT", "WSE": ".WA",
    "NYSE": "", "NASDAQGS": "", "NASDAQGM": "", "NASDAQCM": "", "NYSEAM": "", "NYSEARCA": "", "ARCA": "",
    "BATS": "",
}
# същото по пълното име на борсата („Stock Exchange Name“) - ако няма „Full Ticker“
EXCHANGE_NAME_SUFFIX = {
    "xetra": ".DE", "frankfurt": ".F", "euronext paris": ".PA", "euronext amsterdam": ".AS",
    "euronext brussels": ".BR", "euronext lisbon": ".LS", "euronext dublin": ".IR", "irish": ".IR",
    "borsa italiana": ".MI", "milan": ".MI", "madrid": ".MC", "bme": ".MC", "bolsas y mercados": ".MC", "wiener": ".VI", "vienna": ".VI",
    "london": ".L", "helsinki": ".HE", "stockholm": ".ST", "copenhagen": ".CO", "oslo": ".OL", "six swiss": ".SW",
    "new york": "", "nyse": "", "nasdaq": "",
}
EMPTY = {"", "-", "—", "nm", "n/a", "na", "#restricted!", "unavailable", "none", "nan"}

# Fair Value Label -> точка
FV_DOTS = {"bargain": "🟢", "undervalued": "🟢", "fair": "⚪", "overvalued": "🔴", "premium": "🔴"}
FV_BG = {"Bargain": "Изгодна", "Undervalued": "Подценена", "Fair": "Справедлива", "Overvalued": "Надценена",
         "Premium": "Скъпа"}
# Overall Health Label -> точка (скала на InvestingPro: Weak < Fair < Good < Great < Excellent)
HEALTH_DOTS = {"excellent": "🟢", "great": "🟢", "good": "🟢", "fair": "⚪", "weak": "🔴"}
HEALTH_BG = {"Excellent": "Отлично", "Great": "Много добро", "Good": "Добро", "Fair": "Средно", "Weak": "Слабо"}
LEGAL_WORDS = {"sa", "se", "nv", "ag", "plc", "spa", "inc", "corp", "corporation", "co", "ltd", "limited", "group",
               "holding", "holdings", "the", "class", "a", "b", "company", "oyj", "ab", "asa", "as", "kgaa", "nom",
               "reg", "de", "cv", "sca", "bv", "oy", "aps", "publ", "s", "p", "l"}


# ---------------------------------------------------------------- четене на експорта

def _norm_header(h) -> str:
    return re.sub(r"\s+", " ", str(h)).strip().lower()


def _field(header: str):
    """Името на полето ни по заглавието на колоната (останалите се пазят в „extra“ с оригиналното име)."""
    h = _norm_header(header)
    if h == "name":
        return "name"
    if h == "full ticker":
        return "full_ticker"
    if h in ("ticker", "symbol"):
        return "ticker"
    if "exchange" in h and "name" in h or h in ("exchange", "stock exchange"):
        return "exchange"
    if h.startswith("fair value upside"):
        return "fv_upside"
    if h.startswith("fair value label"):
        return "fv_label"
    if h.startswith("fair value") and "analyst" in h:
        return "analyst_target"
    if h in ("fair value", "fair value (investingpro)"):
        return "fair_value"
    if "upside" in h and "analyst" in h:
        return "analyst_upside"
    if h == "overall health score":
        return "health_score"
    if h == "overall health label":
        return "health_label"
    if h == "market cap":
        return "market_cap"
    if h in ("price, current", "price", "last price"):
        return "price"
    return None


def _number(v, percent: bool = False):
    """Число от клетката: 23.5 / „23,5%“ / „1.2B“; „NM“, „#RESTRICTED!“, „-“ -> None.
    percent - връща (стойност, дали е дошла с „%“)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return (None, False) if percent else None
    if isinstance(v, (int, float)):
        return (float(v), False) if percent else float(v)
    s = str(v).strip()
    if s.lower() in EMPTY:
        return (None, False) if percent else None
    had_pct = "%" in s
    s = s.replace("%", "").replace("$", "").replace("€", "").replace(" ", "").replace(" ", "")
    mult = 1.0
    if s and s[-1].upper() in "KMBT" and len(s) > 1:
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[s[-1].upper()]
        s = s[:-1]
    if "," in s and "." in s:  # 10.604,83 или 10,604.83
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        x = float(s) * mult
    except ValueError:
        return (None, False) if percent else None
    return (x, had_pct) if percent else x


def _text(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return None if s.lower() in EMPTY else s


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Таблицата от файла: заглавният ред е този с „Name“ (Excel експортът има заглавие и празни редове отгоре)."""
    if filename.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(io.BytesIO(data), header=None, dtype=object)
    else:
        text = data.decode("utf-8-sig", errors="replace")
        sep = ";" if text.count(";") > text.count(",") else ","
        raw = pd.read_csv(io.StringIO(text), header=None, dtype=object, sep=sep)
    for i in range(min(len(raw), 40)):
        cells = [_norm_header(c) for c in raw.iloc[i].tolist()]
        if "name" in cells and ("ticker" in cells or "full ticker" in cells):
            df = raw.iloc[i + 1:].copy()
            df.columns = [str(c).strip() for c in raw.iloc[i].tolist()]
            df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan"]]
            return df.dropna(how="all")
    raise ValueError("Не намерих заглавен ред с колони „Name“ и „Ticker“ - това експорт от InvestingPro ли е?")


def yahoo_symbol(full_ticker, ticker, exchange) -> str:
    """`ENXTPA:SAF` -> `SAF.PA`, `NYSE:BRK.B` -> `BRK-B`; без Full Ticker - по името на борсата."""
    suffix, base = None, _text(ticker)
    ft = _text(full_ticker)
    if ft and ":" in ft:
        prefix, base = ft.split(":", 1)
        suffix = EXCHANGE_SUFFIX.get(prefix.strip().upper().replace(" ", ""))
    if suffix is None and exchange:
        ex = str(exchange).lower()
        suffix = next((v for k, v in EXCHANGE_NAME_SUFFIX.items() if k in ex), None)
    if not base or suffix is None:
        return None
    return re.sub(r"[ .]+", "-", base.strip().upper()) + suffix


def name_key(name: str) -> str:
    """Името без правната форма и пунктуацията - за резервното съпоставяне („Safran SA“ = „Safran“)."""
    if not name:
        return ""
    m = re.search(r"\(([^()]+)\)\s*$", name)  # нашите имена са „SAF (Safran)“
    words = re.findall(r"[a-z0-9]+", (m.group(1) if m else name).lower())
    return " ".join(w for w in words if w not in LEGAL_WORDS)


def exchange_of(rec: dict) -> str:
    """Борсата на реда за отчета: префиксът от „Full Ticker“ (`ENXTPA`), иначе името на борсата."""
    ft = rec.get("full_ticker") or ""
    return ft.split(":", 1)[0] if ":" in ft else (rec.get("exchange") or "?")


def parse_export(data: bytes, filename: str) -> tuple:
    """-> ({символ: запис}, брой редове, [имена без символ], Counter по борси)."""
    df = read_table(data, filename)
    fields = {c: _field(c) for c in df.columns}
    pct_cols = [c for c, f in fields.items() if f in ("fv_upside", "analyst_upside")]
    # процентите: ако клетките са числа (Excel формат „%“), са дроби (0.235) -> по 100
    fraction = {}
    for c in pct_cols:
        vals = [v for v in df[c] if isinstance(v, (int, float)) and not pd.isna(v)]
        fraction[c] = bool(vals) and pd.Series(vals).abs().median() < 1.5
    records, unmatched, exchanges = {}, [], Counter()
    for _, row in df.iterrows():
        rec, extra = {}, {}
        for c, f in fields.items():
            v = row[c]
            if f in ("name", "full_ticker", "ticker", "exchange", "fv_label", "health_label"):
                rec[f] = _text(v)
            elif f in ("fv_upside", "analyst_upside"):
                x, had_pct = _number(v, percent=True)
                if x is not None and fraction[c] and not had_pct:
                    x *= 100
                rec[f] = None if x is None else round(x, 2)
            elif f:
                x = _number(v)
                rec[f] = None if x is None else round(x, 4)
            else:
                x = _number(v)
                t = _text(v)
                if x is not None:
                    extra[c] = round(x, 4)
                elif t is not None:
                    extra[c] = t
        if not rec.get("name"):
            continue
        exchanges[exchange_of(rec)] += 1
        symbol = yahoo_symbol(rec.get("full_ticker"), rec.get("ticker"), rec.get("exchange"))
        if not symbol:
            unmatched.append(rec["name"])
            continue
        if rec.get("analyst_target") is None and rec.get("analyst_upside") is not None and rec.get("price"):
            rec["analyst_target"] = round(rec["price"] * (1 + rec["analyst_upside"] / 100), 4)
        if extra:
            rec["extra"] = extra
        records[symbol] = {k: v for k, v in rec.items() if v is not None}
    return records, len(df), unmatched, exchanges


# ---------------------------------------------------------------- съхранение (частно repo)

def data_repo():
    return st.secrets.get("IP_DATA_REPO", None)


@st.cache_data(ttl=3600, show_spinner=False)
def _load(repo: str, version: int) -> dict:
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not (repo and token):
        return {}
    text, _ = github_get_file(DATA_FILE, token, repo)
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {}


def load_data() -> dict:
    """{"updated": "2026-10-18", "files": [...], "records": {символ: {...}}}; в паметта на сесията след качване."""
    if "ip_data" in st.session_state:
        return st.session_state["ip_data"]
    return _load(data_repo(), st.session_state.get("ip_version", 0))


@st.cache_data(show_spinner=False)
def _name_index(updated: str, count: int, _records: dict) -> tuple:
    """({ключ на името: символ}, {тикер без борсата: [символи]})."""
    index, dup, bases = {}, set(), {}
    for symbol, rec in _records.items():
        key = name_key(rec.get("name"))
        if key in index:
            dup.add(key)
        index[key] = symbol
        bases.setdefault(symbol.rsplit(".", 1)[0], []).append(symbol)
    return {k: v for k, v in index.items() if k and k not in dup}, bases


def lookup(symbol: str, name: str = None, data: dict = None):
    """Записът за наш символ (`SAF.PA`); резервно - по името (напр. US акция, сканирана на Xetra)."""
    data = load_data() if data is None else data
    records = data.get("records") or {}
    if symbol in records:
        return records[symbol]
    if name:
        index, bases = _name_index(data.get("updated", ""), len(records), records)
        key = name_key(name)
        found = index.get(key)
        if found:
            return records[found]
        # същият тикер на друга борса (напр. различно записана борса във файла) + същата първа дума в името
        first = key.split(" ")[0] if key else ""
        same = [s for s in bases.get(symbol.rsplit(".", 1)[0], [])
                if first and name_key(records[s].get("name")).split(" ")[0] == first]
        if len(same) == 1:
            return records[same[0]]
    return None


def save_data(new_records: dict, files: list) -> tuple:
    """Новите записи заместват старите за същите символи; останалите (напр. САЩ при качване само на
    Европа) остават. -> (ok, съобщение)."""
    repo, token = data_repo(), st.secrets.get("GITHUB_TOKEN", None)
    old = (load_data() or {}).get("records") or {}
    data = {"updated": date.today().isoformat(), "files": files, "records": {**old, **new_records}}
    st.session_state["ip_data"] = data
    if not (repo and token):
        return False, ("Данните са само за тази сесия: няма `IP_DATA_REPO` в Streamlit Secrets (частното repo "
                       "за данните от InvestingPro) или `GITHUB_TOKEN`.")
    ok, msg = github_write_file(DATA_FILE, json.dumps(data, ensure_ascii=False, separators=(",", ":")), token,
                                f"InvestingPro: {len(new_records)} инструмента", repo)
    if ok:
        _load.clear()
    return ok, msg


# ---------------------------------------------------------------- показване

def fv_summary(rec) -> str:
    """За таблицата: точката по Fair Value Label + % до справедливата („🟢 +23%“)."""
    if not rec:
        return ""
    label = (rec.get("fv_label") or "").strip()
    up = rec.get("fv_upside")
    if up is None and rec.get("fair_value") and rec.get("price"):
        up = 100 * (rec["fair_value"] / rec["price"] - 1)
    if up is None:
        return "—"
    return f"{FV_DOTS.get(label.lower(), '⚪')} {up:+.0f}%"


def health_summary(rec) -> str:
    """За таблицата: точката + оценката („🟢 2.9 Добро“)."""
    if not rec:
        return ""
    label, score = (rec.get("health_label") or "").strip(), rec.get("health_score")
    if not label and score is None:
        return "—"
    if not label:
        label = ("Excellent" if score >= 3.5 else "Great" if score >= 3 else "Good" if score >= 2.5
                 else "Fair" if score >= 1.75 else "Weak")
    dot = HEALTH_DOTS.get(label.lower(), "⚪")
    return f"{dot} " + (f"{score:.1f} " if score is not None else "") + HEALTH_BG.get(label, label)


def detail_markdown(rec) -> str:
    """Подробностите за един инструмент (прозорецът с графиката / клик по клетката)."""
    if not rec:
        return ""
    parts = []
    # сумите са във валутата на експорта (може да е USD при „US“ настройка на сайта) - затова и цената от файла
    if rec.get("fair_value") is not None:
        label = rec.get("fv_label")
        parts.append(f"**Fair Value {rec['fair_value']:.2f}**"
                     + (f" при цена {rec['price']:.2f}" if rec.get("price") else "")
                     + (f" ({rec['fv_upside']:+.1f}%)" if rec.get("fv_upside") is not None else "")
                     + (f" - {FV_BG.get(label, label)}" if label else ""))
    if rec.get("analyst_upside") is not None:
        parts.append(f"Анализатори: {rec['analyst_upside']:+.1f}% до целта")
    health = health_summary(rec)
    if health and health != "—":
        parts.append(f"Финансово здраве: {health}")
    if rec.get("market_cap"):
        parts.append(f"Капитализация: {rec['market_cap'] / 1e9:.1f} млрд.")
    return "🏛 **InvestingPro**: " + " · ".join(parts) if parts else ""


def data_age_days(data: dict = None):
    data = load_data() if data is None else data
    try:
        return (date.today() - date.fromisoformat(data["updated"])).days
    except (KeyError, TypeError, ValueError):
        return None


def render_upload(known: dict):
    """Качване на експорта (в изскачащ панел). known - {символ: име} от нашите (за отчета колко съвпадат)."""
    data = load_data()
    count = len(data.get("records") or {})
    age = data_age_days(data)
    if count:
        st.markdown(f"Сега: **{count}** инструмента, качени на **{date.fromisoformat(data['updated']):%d.%m.%Y}**"
                    + (f" (преди {age} дни)" if age else ""))
    else:
        st.markdown("Още няма качени данни.")
    st.caption("InvestingPro → Screener → изгледа „Nasko“ → **Export as… → Excel**. Лимитът е 1000 реда на файл: "
               "Европа + Великобритания (Market Cap > 2 млрд.) и отделно САЩ (> 20 млрд.). Качи двата наведнъж.")
    files = st.file_uploader("Файлове от InvestingPro", type=["xlsx", "xls", "csv"], accept_multiple_files=True,
                             key="ip_upload", label_visibility="collapsed")
    if files and st.button("💾 Обработи и запази", key="ip_save", type="primary"):
        records, total, unmatched, names, exchanges = {}, 0, [], [], Counter()
        for f in files:
            try:
                r, n, u, ex = parse_export(f.getvalue(), f.name)
            except Exception as e:  # noqa: BLE001
                st.error(f"{f.name}: {e}")
                continue
            records.update(r)
            total += n
            unmatched += u
            exchanges += ex
            names.append(f.name)
        if not records:
            return
        ok, msg = save_data(records, names)
        (st.success if ok else st.warning)(f"Прочетени {total} реда, {len(records)} с разпознат символ. "
                                           + ("Записано в частното repo." if ok else msg))
        probe = {"records": records, "updated": date.today().isoformat()}
        missing = [n for s, n in known.items() if not lookup(s, n, probe)]
        st.caption(f"От акциите в Селектирани: {len(known) - len(missing)} от {len(known)} намерени."
                   + (f" Без данни: {', '.join(missing[:15])}" + (" ..." if len(missing) > 15 else "") if missing else ""))
        st.caption("По борси: " + ", ".join(f"{k} {v}" for k, v in exchanges.most_common()))
        if unmatched:
            st.caption(f"Без разпозната борса ({len(unmatched)}): {', '.join(unmatched[:10])}"
                       + (" ..." if len(unmatched) > 10 else ""))
