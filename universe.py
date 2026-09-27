"""Универсът за сканиране: месечната селекция (curated_universe.json), ръчният списък,
CSV/Excel upload, търсене, макро сигналът и GitHub интеграцията (workflow dispatch, запис)."""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

import universe_rules as rules
from ui_common import format_eur

# ============================================================================
# ОБЩА ЧАСТ: реален универс от eu_instruments.json (Trading 212 API)
# ============================================================================

INSTRUMENTS_FILE = "eu_instruments.json"
CURATED_FILE = "curated_universe.json"

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


FALLBACK_TICKERS = {
    "SXR8 (iShares Core S&P 500)": "SXR8.DE",
    "EXH1 (iShares STOXX Europe 600)": "EXH1.DE",
    "SAP (SAP SE)": "SAP.DE",
    "ASML (ASML Holding)": "ASML.AS",
    "AIR (Airbus)": "AIR.PA",
    "ISP (Intesa Sanpaolo)": "ISP.MI",
}


@st.cache_data(ttl=6 * 3600)
def load_universe(max_instruments=500, pinned_keywords: tuple = (), liquidity: tuple = None, curated_mtime: float = 0):
    """Зарежда универса за сканиране. Приоритет:
    1) закачени (pinned_keywords) инструменти - винаги от ПЪЛНИЯ eu_instruments.json,
       за да не пропуснем нищо, дори ако не са в месечната селекция;
    2) curated_universe.json (месечна селекция по ликвидност+моментум+медиен buzz),
       ако съществува;
    3) fallback - суровият ред от eu_instruments.json, ако все още няма curated файл.
    max_instruments=None = без лимит; liquidity = (мин. капитализация на акция,
    мин. оборот на акция, мин. AUM на ETF, мин. оборот на ETF) - допълнително
    стесняване на curated файла (закачените винаги влизат). curated_mtime не се
    ползва в тялото - само е част от ключа на кеша, за да се презареди
    автоматично, когато workflow-ът запише нов curated_universe.json."""
    if max_instruments is None:
        max_instruments = float("inf")
    pinned_keywords_lower = [kw.lower() for kw in pinned_keywords if kw.strip()]
    mapped = {}

    full_path = Path(INSTRUMENTS_FILE)
    full_instruments = None
    if full_path.exists():
        full_instruments = json.loads(full_path.read_text(encoding="utf-8")).get("instruments", [])

        # при liquidity (Photon) закачените НЕ се добавят от пълния списък - иначе
        # широки макро думи ("Europe", "Dividend") вкарват стотици нисколиквидни;
        # ликвидните така или иначе са в curated файла
        if pinned_keywords_lower and liquidity is None:
            for inst in full_instruments:
                name_field = inst.get("name", "")
                if not any(kw in name_field.lower() for kw in pinned_keywords_lower):
                    continue
                suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
                if suffix is None:
                    continue
                yahoo_ticker = f"{inst.get('shortName', '')}{suffix}"
                label = f"{inst.get('shortName', inst['ticker'])} ({inst['name']})"
                mapped[label] = yahoo_ticker

    curated_path = Path(CURATED_FILE)
    if curated_path.exists():
        curated_data = json.loads(curated_path.read_text(encoding="utf-8"))
        for item in curated_data.get("instruments", []):
            if len(mapped) >= max_instruments:
                break
            label = item["name"]
            if label in mapped:
                continue
            # стар формат на файла (без type) - не филтрираме, докато месечният workflow не го обнови
            if liquidity and "type" in item and not rules.passes_liquidity(item, *liquidity)[0]:
                continue
            mapped[label] = item["symbol"]
        return mapped

    if full_instruments is not None:
        st.warning(
            "Не намерих curated_universe.json — ползвам обичайния ред от eu_instruments.json. "
            "Пусни месечния workflow 'Monthly Curate Universe' за по-качествена селекция по "
            "ликвидност/моментум/медиен интерес."
        )
        for inst in full_instruments:
            if len(mapped) >= max_instruments:
                break
            suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
            if suffix is None:
                continue
            yahoo_ticker = f"{inst.get('shortName', '')}{suffix}"
            label = f"{inst.get('shortName', inst['ticker'])} ({inst['name']})"
            if label in mapped:
                continue
            mapped[label] = yahoo_ticker
        return mapped

    if mapped:
        return mapped

    st.warning(f"Не намерих нито {INSTRUMENTS_FILE}, нито {CURATED_FILE} — ползвам малък резервен списък.")
    return FALLBACK_TICKERS


CSV_COLUMNS = {
    "isin": ["ISIN"],
    "symbol": ["Symbol", "Ticker"],
    "name": ["Name", "Company", "Company Name", "Instrument"],
}


def parse_uploaded_ticker_list(uploaded_file):
    """Чете CSV/XLSX (напр. износ от InvestingPro screener) и връща редовете
    като [{isin, symbol, name}] - от колоните ISIN / Symbol|Ticker /
    Name|Company (без значение главни/малки букви). Ако няма нито една от тях,
    първата колона се ползва като име."""
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Не успях да прочета файла: {e}")
        return []
    if df.empty:
        return []
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    found = {}
    for field, candidates in CSV_COLUMNS.items():
        for candidate in candidates:
            if candidate.lower() in by_lower:
                found[field] = by_lower[candidate.lower()]
                break
    if not found:
        found["name"] = df.columns[0]

    def cell(row, field):
        value = row.get(found[field]) if field in found else None
        return str(value).strip() if value is not None and str(value).strip().lower() not in ("", "nan") else ""

    rows = [{f: cell(r, f) for f in CSV_COLUMNS} for r in df.to_dict("records")]
    return [r for r in rows if any(r.values())]


def contains_words(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def load_universe_from_terms(rows: list):
    """Съпоставя редовете от качения файл срещу ПЪЛНИЯ eu_instruments.json:
    първо по ISIN (точно), после по тикер (точно, спрямо T212 shortName),
    накрая по име (точно, иначе подниз). Gettex акциите се пренасочват към
    основното им листване, ако е известно от месечната селекция.
    Приема и стар формат - списък от низове (имена/тикери)."""
    full_path = Path(INSTRUMENTS_FILE)
    if not full_path.exists():
        st.error(f"Не намерих {INSTRUMENTS_FILE} - не мога да съпоставя качения списък.")
        return {}
    rows = [r if isinstance(r, dict) else {"isin": "", "symbol": "", "name": str(r)} for r in rows]
    instruments = [i for i in json.loads(full_path.read_text(encoding="utf-8")).get("instruments", [])
                   if exchange_to_yahoo_suffix(i.get("exchangeName", "")) is not None]
    by_isin = {i.get("isin", "").upper(): i for i in instruments if i.get("isin")}
    by_short = {i.get("shortName", "").lower(): i for i in instruments if i.get("shortName")}
    by_name = {i.get("name", "").lower(): i for i in instruments if i.get("name")}
    _, resolved, _ = load_curated_symbol_info(curated_file_mtime())

    def find(row):
        if row["isin"] and row["isin"].upper() in by_isin:
            return by_isin[row["isin"].upper()]
        if row["symbol"] and row["symbol"].lower() in by_short:
            return by_short[row["symbol"].lower()]
        name = row["name"].lower()
        if name in by_name:
            return by_name[name]
        # по цели думи (иначе "onex" съвпада с "n-onex-istent"), само за имена от 4+ знака
        if len(name) >= 4:
            for inst in instruments:
                inst_name = inst.get("name", "").lower()
                if len(inst_name) >= 4 and (contains_words(inst_name, name) or contains_words(name, inst_name)):
                    return inst
        return None

    mapped, unmatched = {}, []
    for row in rows:
        inst = find(row)
        if inst is None:
            unmatched.append(row["name"] or row["symbol"] or row["isin"])
            continue
        yahoo_ticker = f"{inst.get('shortName', '')}{exchange_to_yahoo_suffix(inst.get('exchangeName', ''))}"
        label = f"{inst.get('shortName', inst['ticker'])} ({inst['name']})"
        symbol = resolved.get(yahoo_ticker, yahoo_ticker)
        if symbol.endswith(".MU") and inst.get("isin", "").startswith("US") and row["symbol"]:
            symbol = row["symbol"].upper()  # Gettex US акция: тикерът от файла е основното US листване
        mapped[label] = symbol

    if unmatched:
        preview = ", ".join(unmatched[:8])
        more = f" (+{len(unmatched) - 8} още)" if len(unmatched) > 8 else ""
        st.warning(
            f"{len(unmatched)} от {len(rows)} реда не намерих в ЕС/ЕИП+EUR универса на T212 "
            f"(извън обхвата или разлика в изписването): {preview}{more}"
        )
    return mapped


def render_universe_uploader(key: str):
    """Бутон за качване на фундаментално прецеден списък (напр. InvestingPro
    Fair Value/Health Score screener export, или списък от брокера). Ако
    качиш файл, той ЗАМЕСТВА обичайния универс за тази сесия - технически
    сканираме САМО компаниите от файла. Връща dict {label: ticker} или None."""
    uploaded = st.file_uploader(
        "📤 Качи фундаментален списък (CSV/Excel — напр. InvestingPro screener export)",
        type=["csv", "xlsx", "xls"], key=f"{key}_universe_upload",
        help="Заменя обичайния универс за тази сесия - сканираме технически САМО компаниите от файла.",
    )
    if uploaded is None:
        return None
    rows = parse_uploaded_ticker_list(uploaded)
    if not rows:
        st.error("Файлът изглежда празен или нечетим.")
        return None
    mapped = load_universe_from_terms(rows)
    if mapped:
        st.success(f"Качени {len(rows)} реда → {len(mapped)} съвпадения в ЕС/ЕИП+EUR универса.")
    return mapped or None


MACRO_SIGNAL_FILE = "daily_macro_signal.json"

# Данни за repo-то, нужни само за да пуснем daily_macro.yml ръчно от приложението
# (workflow_dispatch през GitHub REST API) - самото сканиране пак се изпълнява
# в GitHub Actions, с техния ANTHROPIC_API_KEY secret, не тук.
GITHUB_REPO_OWNER = "naskorb-cell"
GITHUB_REPO_NAME = "eu-swing-ai"
GITHUB_WORKFLOW_FILE = "daily_macro.yml"
GITHUB_REF = "main"


@st.cache_data(ttl=30 * 60)
def load_daily_macro_signal():
    """Чете daily_macro_signal.json, генериран от daily_macro_scan.py (виж
    .github/workflows/daily_macro.yml). Връща None, ако файлът липсва още -
    приложението работи нормално и без него."""
    path = Path(MACRO_SIGNAL_FILE)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def trigger_macro_workflow_dispatch(github_token: str, workflow_file: str = GITHUB_WORKFLOW_FILE):
    """Праща workflow_dispatch към GitHub Actions, за да пусне workflow-а
    (по подразбиране daily_macro.yml) веднага, вместо да чака cron-а.
    Връща (success, съобщение)."""
    url = (
        f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"
        f"/actions/workflows/{workflow_file}/dispatches"
    )
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        resp = requests.post(url, headers=headers, json={"ref": GITHUB_REF}, timeout=15)
    except requests.RequestException as e:
        return False, f"Грешка при връзка с GitHub: {e}"

    if resp.status_code == 204:
        return True, "Пуснато! Изчакай ~1 минута, после презареди страницата."
    if resp.status_code == 404:
        return False, "404 - провери GITHUB_REPO_OWNER/NAME/WORKFLOW_FILE или дали токенът има достъп до repo-то."
    if resp.status_code == 401:
        return False, "401 - невалиден или изтекъл GitHub token."
    return False, f"GitHub върна {resp.status_code}: {resp.text[:200]}"


CURATE_WORKFLOW_FILE = "monthly_curate.yml"


def curated_file_mtime() -> float:
    path = Path(CURATED_FILE)
    return path.stat().st_mtime if path.exists() else 0


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_workflow_run(workflow_file: str, github_token: str = None):
    """Последното пускане на workflow-а от GitHub API (кеш 60 сек.).
    Repo-то е публично, затова работи и без token (с по-нисък лимит).
    Връща dict с status/conclusion/времена/линк или None при грешка."""
    url = (
        f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"
        f"/actions/workflows/{workflow_file}/runs"
    )
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    try:
        resp = requests.get(url, headers=headers, params={"per_page": 1}, timeout=10)
        runs = resp.json().get("workflow_runs", []) if resp.status_code == 200 else []
    except (requests.RequestException, ValueError):
        return None
    if not runs:
        return None
    run = runs[0]
    return {
        "status": run.get("status"), "conclusion": run.get("conclusion"),
        "started": run.get("run_started_at") or run.get("created_at"),
        "updated": run.get("updated_at"), "url": run.get("html_url"),
    }


def format_github_time(iso_ts: str) -> str:
    """'2026-09-27T09:25:11Z' -> '27.09 12:25' (българско време)."""
    if not iso_ts:
        return "?"
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return dt.astimezone(ZoneInfo("Europe/Sofia")).strftime("%d.%m %H:%M")


def render_workflow_status(run: dict):
    if run is None:
        st.caption("Статусът на обновяването не е достъпен в момента.")
        return
    started = format_github_time(run["started"])
    if run["status"] != "completed":
        st.info(f"⏳ Обновяването върви (започнато {started}). Приложението ще вземе новия списък автоматично, щом приключи.")
    elif run["conclusion"] == "success":
        st.success(f"✅ Последното обновяване завърши успешно ({format_github_time(run['updated'])}).")
    else:
        st.error(f"❌ Последното обновяване завърши с грешка ({run['conclusion']}, {started}). [Виж лога в GitHub]({run['url']})")


def render_universe_refresh(key: str):
    """Инфо за месечната селекция (curated_universe.json) + бутон за ръчно
    обновяване (пуска monthly_curate.yml в GitHub Actions)."""
    curated_path = Path(CURATED_FILE)
    data = json.loads(curated_path.read_text(encoding="utf-8")) if curated_path.exists() else {}
    github_token = st.secrets.get("GITHUB_TOKEN", None)
    run = fetch_latest_workflow_run(CURATE_WORKFLOW_FILE, github_token)
    running = run is not None and run["status"] != "completed"
    title = "🗂️ Универс: месечна селекция по ликвидност" + (" · ⏳ обновява се" if running else "")
    with st.expander(title, expanded=running):
        render_workflow_status(run)
        if not data:
            st.info("Още няма curated_universe.json - пусни обновяване.")
        else:
            st.caption(f"Обновена: {data.get('generated_at', '?')}")
            if "stock_count" in data:
                st.markdown(
                    f"**{data.get('stock_count', 0)} акции + {data.get('etf_count', 0)} ETF/ETP** "
                    f"(от {data.get('total_evaluated', '?')} оценени; медийно трендиращи: {data.get('media_trending_count', 0)})"
                )
                c = data.get("criteria", {})
                st.caption(
                    f"Критерии: акции капитализация ≥ {format_eur(c.get('stock_min_market_cap', 0))}, "
                    f"оборот ≥ {format_eur(c.get('stock_min_turnover', 0))}/ден (родна борса); "
                    f"ETF AUM ≥ {format_eur(c.get('etf_min_aum', 0))}, оборот ≥ {format_eur(c.get('etf_min_turnover', 0))}/ден; "
                    "без ливъриджнати/short ETP."
                )
                rejected = data.get("rejected", [])
                if rejected:
                    with st.expander(f"Отпаднали при месечния подбор ({len(rejected)}) - по данни на Yahoo"):
                        st.caption(
                            "Минали са прага за оборот, но са отпаднали на капитализация/AUM. Ако числата "
                            "изглеждат грешни (Yahoo понякога греши при европейските листвания), добави "
                            "инструмента ръчно в ✏️."
                        )
                        st.dataframe(pd.DataFrame([{
                            "Инструмент": r["name"], "Символ": r["symbol"], "Причина": r["reason"],
                            "Оборот/ден": format_eur(r["avg_dollar_volume"] or 0),
                            "Капитализация": format_eur(r["market_cap"]) if r.get("market_cap") else "-",
                            "AUM": format_eur(r["aum"]) if r.get("aum") else "-",
                        } for r in rejected]), hide_index=True, width="stretch")
            else:
                st.warning(
                    "Селекцията е в стар формат (само по оборот, без капитализация/AUM). "
                    "Пусни обновяване, за да се приложат новите критерии."
                )

        if not github_token:
            github_token = st.text_input("GitHub token (за ръчно пускане)", type="password", key=f"{key}_curate_token")

        cooldown_key = "curate_trigger_last_ts"
        seconds_left = int(300 - (time.time() - st.session_state.get(cooldown_key, 0)))
        col_trigger, col_reload = st.columns(2)
        with col_trigger:
            if st.button("🔄 Обнови универса сега", key=f"{key}_curate_trigger", disabled=seconds_left > 0 or running):
                if not github_token:
                    st.error("Липсва GitHub token!")
                else:
                    ok, msg = trigger_macro_workflow_dispatch(github_token, workflow_file=CURATE_WORKFLOW_FILE)
                    st.session_state[cooldown_key] = time.time()
                    fetch_latest_workflow_run.clear()
                    if ok:
                        msg = ("Пуснато! Обновяването отнема ~10-30 мин. Натисни 'Провери статуса' след малко - "
                               "новият списък се зарежда автоматично, щом приключи.")
                    (st.success if ok else st.error)(msg)
        with col_reload:
            if st.button("🔁 Провери статуса", key=f"{key}_curate_status"):
                fetch_latest_workflow_run.clear()
                st.rerun()
        st.caption("Автоматично се обновява всяко 1-во число от месеца.")


def render_macro_section(key: str, allow_autopin: bool = True):
    """Показва дневното макро резюме + бутон за принудително сканиране.
    Връща (auto_pin_enabled, macro_keywords). allow_autopin=False скрива
    опцията за автоматично добавяне (темите само се маркират с 📰)."""
    signal = load_daily_macro_signal()
    themes = signal.get("themes", []) if signal else []
    macro_keywords = sorted({kw for t in themes for kw in t.get("keywords", [])})

    with st.expander("📰 Дневен макро преглед", expanded=False):
        if signal is None:
            st.info("Още няма записан макро сигнал за днес.")
        else:
            st.caption(f"Обновено: {signal.get('generated_at', '?')}")
            st.markdown(signal.get("summary_bg", ""))
            for t in themes:
                kws = ", ".join(t.get("keywords", []))
                st.markdown(f"- **{t.get('theme', '')}** ({kws}) — {t.get('reasoning_bg', '')}")

        if allow_autopin:
            auto_pin = st.checkbox(
                "Автоматично добавяй тези активи към скрининга за деня",
                value=True,
                key=f"{key}_macro_autopin",
            )
        else:
            auto_pin = False
            st.caption("Инструментите, свързани с тези теми, се маркират с 📰 в резултатите.")

        st.divider()
        github_token = st.secrets.get("GITHUB_TOKEN", None)
        if not github_token:
            github_token = st.text_input(
                "GitHub token (за ръчно пускане)", type="password", key=f"{key}_gh_token",
                help="Fine-grained personal access token с права 'Actions: Read and write' "
                     "само за repo-то eu-swing-ai. Може да го запишеш трайно в Streamlit "
                     "Secrets като GITHUB_TOKEN, за да не го въвеждаш всеки път.",
            )

        cooldown_key = "macro_trigger_last_ts"
        last_ts = st.session_state.get(cooldown_key, 0)
        seconds_left = int(60 - (time.time() - last_ts))

        col_trigger, col_reload = st.columns(2)
        with col_trigger:
            if st.button(
                "🔄 Изпълни макро сканиране сега",
                key=f"{key}_macro_trigger",
                disabled=seconds_left > 0,
            ):
                if not github_token:
                    st.error("Липсва GitHub token!")
                else:
                    ok, msg = trigger_macro_workflow_dispatch(github_token)
                    st.session_state[cooldown_key] = time.time()
                    (st.success if ok else st.error)(msg)
        with col_reload:
            if st.button("🧹 Изчисти кеша и презареди", key=f"{key}_macro_reload"):
                load_daily_macro_signal.clear()
                st.rerun()

        if seconds_left > 0:
            st.caption(f"Изчакай още {seconds_left} сек. преди да пуснеш пак.")
        st.caption(
            "Резюмето по-горе се кешира до 30 мин. Ако workflow-ът в GitHub Actions "
            "вече е завършил (провери в Actions таба), натисни 'Изчисти кеша и презареди', "
            "вместо да чакаш кеша сам да изтече."
        )

    return auto_pin, macro_keywords


def flag_macro_signal(df: pd.DataFrame, macro_keywords, name_col="Име", ticker_col="Тикер") -> pd.DataFrame:
    """Добавя булева колона '📰 Медиен сигнал' - True, ако името или тикерът на
    инструмента съвпада (case-insensitive substring) с някоя от темите, които
    daily_macro_scan.py е намерил за деня."""
    df = df.copy()
    if df.empty:
        df["📰 Медиен сигнал"] = pd.Series(dtype=bool)
        return df
    if not macro_keywords:
        df["📰 Медиен сигнал"] = False
        return df
    kws_lower = [k.lower() for k in macro_keywords]

    def matches(row):
        text = f"{row.get(name_col, '')} {row.get(ticker_col, '')}".lower()
        return any(kw in text for kw in kws_lower)

    df["📰 Медиен сигнал"] = df.apply(matches, axis=1)
    # премести новата колона веднага след Име, за да се вижда лесно
    cols = list(df.columns)
    cols.remove("📰 Медиен сигнал")
    insert_at = (cols.index(name_col) + 1) if name_col in cols else 1
    cols.insert(insert_at, "📰 Медиен сигнал")
    return df[cols]


@st.cache_data(ttl=6 * 3600)
@st.cache_data(ttl=6 * 3600)
def load_curated_symbol_info(curated_mtime: float = 0):
    """От curated_universe.json: ({symbol: валута}, {оригинален T212 .MU символ:
    основно листване}, {symbol: тип STOCK/ETF}) - за Gettex акциите се сканира
    основното листване (US/SE/...), защото Yahoo няма използваеми данни за .MU."""
    path = Path(CURATED_FILE)
    if not path.exists():
        return {}, {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    instruments = data.get("instruments", [])
    currencies = {x["symbol"]: x.get("currency", "EUR") for x in instruments}
    # основното листване на Gettex акциите - и за отпадналите при подбора (за CSV/ръчно добавени)
    resolved = {x["t212_symbol"]: x["symbol"] for x in instruments + data.get("rejected", []) if x.get("t212_symbol")}
    types = {x["symbol"]: x.get("type") for x in instruments}
    return currencies, resolved, types


def load_full_universe_for_search():
    """Отделна функция само за търсене - винаги чете суровия eu_instruments.json
    в пълен размер, независимо от месечната curated_universe.json селекция, за
    да можеш да провериш дали инструмент изобщо съществува в T212, дори да не
    е попаднал в тазмесечния топ 500."""
    path = Path(INSTRUMENTS_FILE)
    if not path.exists():
        return {}
    instruments = json.loads(path.read_text(encoding="utf-8")).get("instruments", [])
    _, resolved, _ = load_curated_symbol_info(curated_file_mtime())
    mapped = {}
    for inst in instruments:
        suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
        if suffix is None:
            continue
        yahoo_ticker = f"{inst.get('shortName', '')}{suffix}"
        label = f"{inst.get('shortName', inst['ticker'])} ({inst['name']})"
        mapped[label] = resolved.get(yahoo_ticker, yahoo_ticker)
    return mapped


def render_universe_search(key: str):
    """Малка помощна секция: търсене по име в ЦЕЛИЯ универс (не само
    месечната selекция от 500) - зареждането само на имена/тикери
    е бързо (чете JSON), не тегли ценови данни, затова не бави нищо."""
    with st.expander("🔎 Търси в целия универс (провери дали инструмент е наличен)"):
        query = st.text_input("Име съдържа:", key=f"{key}_search").strip().lower()
        if query:
            full_universe = load_full_universe_for_search()
            matches = {name: sym for name, sym in full_universe.items() if query in name.lower()}
            if matches:
                st.write(f"Намерени {len(matches)}:")
                for name, sym in matches.items():
                    st.write(f"- {name} → `{sym}`")
            else:
                st.info("Няма съвпадение в целия универс (провери дали правописът/името е различно в T212).")


MANUAL_UNIVERSE_FILE = "manual_universe.json"


def github_get_file(path: str, github_token: str):
    """Чете файл от GitHub Contents API. Връща (decoded_text, sha) или
    (None, None), ако файлът не съществува/грешка."""
    import base64

    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/contents/{path}"
    headers = {"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github+json"}
    try:
        resp = requests.get(url, headers=headers, params={"ref": GITHUB_REF}, timeout=15)
    except requests.RequestException:
        return None, None
    if resp.status_code != 200:
        return None, None
    data = resp.json()
    try:
        content = base64.b64decode(data["content"]).decode("utf-8")
    except Exception:
        return None, None
    return content, data.get("sha")


def github_write_file(path: str, content_str: str, github_token: str, message: str):
    """Записва (create/update) файл в repo-то през GitHub Contents API.
    Връща (success, съобщение)."""
    import base64

    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/contents/{path}"
    headers = {"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github+json"}
    _, sha = github_get_file(path, github_token)
    payload = {
        "message": message,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("utf-8"),
        "branch": GITHUB_REF,
    }
    if sha:
        payload["sha"] = sha
    try:
        resp = requests.put(url, headers=headers, json=payload, timeout=15)
    except requests.RequestException as e:
        return False, f"Грешка при връзка с GitHub: {e}"
    if resp.status_code in (200, 201):
        return True, "Записано в repo-то."
    if resp.status_code == 403:
        return False, "403 - токенът няма 'Contents: Read and write' право за repo-то."
    if resp.status_code == 401:
        return False, "401 - невалиден или изтекъл GitHub token."
    return False, f"GitHub върна {resp.status_code}: {resp.text[:200]}"


def load_manual_universe():
    """Чете manual_universe.json от локалния checkout (същия начин като
    curated_universe.json) - бързо, без GitHub API извикване, за да не
    хаби бюджет на всяко презареждане на страницата."""
    path = Path(MANUAL_UNIVERSE_FILE)
    if not path.exists():
        return {"include": [], "exclude": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"include": data.get("include", []), "exclude": data.get("exclude", [])}
    except (json.JSONDecodeError, OSError):
        return {"include": [], "exclude": []}


def add_to_manual_universe(items: dict, github_token: str):
    """Добавя {label: symbol} към "include" в manual_universe.json в repo-то
    (без дубликати по символ). Чете актуалната версия от GitHub, за да не
    презапише промени, направени междувременно. Връща (success, съобщение)."""
    content, _ = github_get_file(MANUAL_UNIVERSE_FILE, github_token)
    try:
        manual = json.loads(content) if content else load_manual_universe()
    except json.JSONDecodeError:
        manual = load_manual_universe()
    manual.setdefault("include", []); manual.setdefault("exclude", [])
    have = {x["symbol"] for x in manual["include"]}
    new = [{"name": n, "symbol": s} for n, s in items.items() if s not in have]
    if not new:
        return True, "Всички вече са в ръчния списък."
    manual["include"] += new
    ok, msg = github_write_file(
        MANUAL_UNIVERSE_FILE, json.dumps(manual, ensure_ascii=False, indent=2), github_token,
        f"Manual universe: add {len(new)} from uploaded file",
    )
    if ok:
        msg = f"Добавени {len(new)} в ръчния списък. Streamlit ще се обнови след минута-две."
    return ok, msg


def render_manual_universe_editor(key: str):
    """Трайно (записва се в repo-то) ръчно добавяне/премахване на конкретни
    активи от универса - алтернатива на CSV upload-а, когато просто искаш
    да добавиш/махнеш няколко имена. Промените остават и след presetart на
    приложението, за разлика от качения CSV файл, който важи само за
    текущата сесия."""
    manual = load_manual_universe()
    github_token = st.secrets.get("GITHUB_TOKEN", None)

    with st.expander("✏️ Ръчно добавяне/премахване на активи (трайно)", expanded=False):
        if not github_token:
            github_token = st.text_input(
                "GitHub token (за запис)", type="password", key=f"{key}_manual_gh_token",
                help="Същият token, ползван за макро бутона, но с добавено право 'Contents: Read and write'.",
            )

        scan_only_manual = st.checkbox(
            "🎯 Сканирай само ръчно добавените активи",
            value=False, key=f"{key}_manual_only",
            help="Игнорира целия универс (curated/качен файл/лимит) - сканира се САМО списъкът 'Винаги включвай' по-долу.",
        )

        def _save(new_manual, action_desc):
            content_str = json.dumps(new_manual, ensure_ascii=False, indent=2)
            ok, msg = github_write_file(MANUAL_UNIVERSE_FILE, content_str, github_token, action_desc)
            (st.success if ok else st.error)(msg if ok else f"{msg} (промяната НЕ е записана трайно)")
            if ok:
                st.session_state["manual_universe_cache"] = new_manual
                st.rerun()

        st.markdown("**➕ Винаги включвай:**")
        if manual["include"]:
            for item in list(manual["include"]):
                col1, col2 = st.columns([5, 1])
                col1.write(f"{item['name']} → `{item['symbol']}`")
                if col2.button("🗑️", key=f"{key}_rm_inc_{item['symbol']}"):
                    if not github_token:
                        st.error("Липсва GitHub token - не мога да запиша промяната.")
                    else:
                        new_manual = {
                            "include": [x for x in manual["include"] if x["symbol"] != item["symbol"]],
                            "exclude": manual["exclude"],
                        }
                        _save(new_manual, f"Manual universe: remove include {item['symbol']}")
        else:
            st.caption("Няма ръчно добавени активи.")

        add_query = st.text_input("Търси по име, за да добавиш:", key=f"{key}_manual_add_search")
        if add_query:
            full = load_full_universe_for_search()
            matches = {n: s for n, s in full.items() if add_query.lower() in n.lower()}
            for name, sym in list(matches.items())[:10]:
                already_in = any(x["symbol"] == sym for x in manual["include"])
                if st.button(f"➕ {name}" + (" (вече добавен)" if already_in else ""), key=f"{key}_add_inc_{sym}", disabled=already_in):
                    if not github_token:
                        st.error("Липсва GitHub token - не мога да запиша промяната.")
                    else:
                        new_manual = {
                            "include": manual["include"] + [{"name": name, "symbol": sym}],
                            "exclude": manual["exclude"],
                        }
                        _save(new_manual, f"Manual universe: add include {sym}")

        st.divider()
        st.markdown("**🚫 Никога не сканирай:**")
        if manual["exclude"]:
            for sym in list(manual["exclude"]):
                col1, col2 = st.columns([5, 1])
                col1.write(f"`{sym}`")
                if col2.button("🗑️", key=f"{key}_rm_exc_{sym}"):
                    if not github_token:
                        st.error("Липсва GitHub token - не мога да запиша промяната.")
                    else:
                        new_manual = {
                            "include": manual["include"],
                            "exclude": [x for x in manual["exclude"] if x != sym],
                        }
                        _save(new_manual, f"Manual universe: remove exclude {sym}")
        else:
            st.caption("Няма изключени активи.")

        exc_query = st.text_input("Търси по име, за да изключиш:", key=f"{key}_manual_exc_search")
        if exc_query:
            full = load_full_universe_for_search()
            matches = {n: s for n, s in full.items() if exc_query.lower() in n.lower()}
            for name, sym in list(matches.items())[:10]:
                already_out = sym in manual["exclude"]
                if st.button(f"🚫 {name}" + (" (вече изключен)" if already_out else ""), key=f"{key}_add_exc_{sym}", disabled=already_out):
                    if not github_token:
                        st.error("Липсва GitHub token - не мога да запиша промяната.")
                    else:
                        new_manual = {
                            "include": manual["include"],
                            "exclude": manual["exclude"] + [sym],
                        }
                        _save(new_manual, f"Manual universe: add exclude {sym}")

    return manual, scan_only_manual


def apply_manual_universe(tickers: dict, manual: dict, scan_only_manual: bool = False) -> dict:
    """Прилага ръчния include/exclude списък върху вече заредения универс.
    Ако scan_only_manual е True, целият подаден tickers се игнорира и се
    връща САМО ръчно добавеният ('include') списък."""
    if scan_only_manual:
        return {item["name"]: item["symbol"] for item in manual.get("include", [])}
    result = dict(tickers)
    for sym in manual.get("exclude", []):
        result = {n: s for n, s in result.items() if s != sym}
    for item in manual.get("include", []):
        result[item["name"]] = item["symbol"]
    return result
