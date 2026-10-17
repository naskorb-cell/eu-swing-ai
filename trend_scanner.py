"""Секция 📈 Възходящ тренд (D + W): два метода за стабилен възходящ тренд на дневната
И седмичната графика, един до друг за нагледно сравнение:
  - Метод А „📐 EMA подреждане“ (предложението на Наско): EMA20 > EMA50 > EMA200;
  - Метод Б „📈 Качество на тренда“: годишен наклон x R² на права линия през
    логаритъма на цената (метод на Андреас Кленов) - високо е само при бърз И равномерен ръст.
И при двата - ATR % (дневен ход като % от цената) за волатилността."""

import json
from datetime import date
from statistics import median
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

import fundamentals as fund
import investingpro as ip
import universe_rules as rules
import valuation as val
from ai_client import AI_KEY_SECRETS
from indicators import average_true_range, ema_alignment, resample_ohlc
from photon import (
    change_style, check_news, clicked_name_row, fetch_ohlc_many, news_checks_today, numbered, table_key, install_chart_scripts,
    instrument_dialog, news_provider, render_news_item, render_news_section, render_photon_strategy, today_news,
)
from ui_common import claude_news_model, friendly_ai_error, gemini_model, gemini_news_model, section_header
from universe import (
    apply_manual_universe, curated_file_mtime, github_get_file, github_write_file, load_curated_symbol_info,
    load_manual_universe, load_universe,
)

TREND_PERIOD = "5y"  # седмичната EMA200 иска ~4 години история
QUALITY_BARS_DAILY, QUALITY_BARS_WEEKLY = 90, 52  # ~4 месеца дневни / 1 година седмични свещи
ATR_SLIDER_MAX = 10.0  # горната граница на плъзгача = без горен лимит


SELECTED_FILE = "selected_trends.json"  # „⭐ Селектирани“ - трайно в repo-то (като manual_universe.json)


# ---------------------------------------------------------------- ⭐ Селектирани

def load_selected(force: bool = False) -> list:
    """Списъкът „⭐ Селектирани“ [{name, symbol, added, price}]. Чете се веднъж на сесия от
    GitHub (актуалният, ако е променян от друго устройство), иначе от локалния файл."""
    if "tr_selected" in st.session_state and not force:
        return st.session_state["tr_selected"]
    text = None
    token = st.secrets.get("GITHUB_TOKEN", None)
    if token:
        text, _ = github_get_file(SELECTED_FILE, token)
    if text is None and Path(SELECTED_FILE).exists():
        text = Path(SELECTED_FILE).read_text(encoding="utf-8")
    try:
        items = json.loads(text).get("items", []) if text else []
    except (json.JSONDecodeError, AttributeError):
        items = []
    st.session_state["tr_selected"] = items
    return items


def save_selected(items: list, message: str) -> bool:
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token:
        st.error("Липсва GITHUB_TOKEN в Streamlit Secrets - селектираните не могат да се запишат трайно.")
        return False
    ok, msg = github_write_file(SELECTED_FILE, json.dumps({"items": items}, ensure_ascii=False, indent=2),
                                token, message)
    if not ok:
        st.error(msg)
        return False
    st.session_state["tr_selected"] = items
    return True


# бутонът за Селектирани - само символ горе в прозореца, на реда на името (до X), а не цял ред
SELECTION_ACTION_CSS = """<style>
div[role="dialog"] { position: relative; }
.st-key-tr_sel_action { position: absolute; top: 8px; right: 64px; z-index: 50; width: auto !important; }
.st-key-tr_sel_action button { min-height: 36px; padding: 2px 10px; font-size: 1.15rem; }
</style>"""


def selection_actions(name: str, symbol: str, price=None):
    """Символът в прозореца с графиката: ⭐ добави в / 🗑 премахни от „⭐ Селектирани“."""
    items = load_selected()
    with st.container(key="tr_sel_action"):
        st.markdown(SELECTION_ACTION_CSS, unsafe_allow_html=True)  # вътре - да не заема ред в прозореца
        if any(x["symbol"] == symbol for x in items):
            if st.button("🗑", key=f"tr_unsel_{symbol}", help="Премахни от ⭐ Селектирани"):
                if save_selected([x for x in items if x["symbol"] != symbol], f"Селектирани: махнат {name}"):
                    st.toast(f"{name} е премахнат от селектираните")
                    st.rerun()
        elif st.button("⭐", key=f"tr_sel_{symbol}", type="primary", help="Добави в ⭐ Селектирани"):
            item = {"name": name, "symbol": symbol, "added": date.today().isoformat(),
                    "price": round(float(price), 4) if price is not None else None}
            if save_selected(items + [item], f"Селектирани: добавен {name}"):
                st.toast(f"⭐ {name} е добавен в селектираните")
                st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def selected_metrics(symbols: tuple) -> dict:
    """Текущите показатели само за селектираните (за всекидневния преглед без пълен скан)."""
    data = fetch_ohlc_many(list(symbols), TREND_PERIOD, "1d")
    return {s: analyze_trend(data.get(s)) for s in symbols}


# ---------------------------------------------------------------- 💎 справедлива цена

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def selected_valuations(symbols: tuple) -> dict:
    return val.valuation_many(symbols)


@st.cache_resource
def _valuation_ai_store() -> dict:
    """AI оценките за деня - общи за всички сесии (обновяване не плаща повторно)."""
    return val.new_ai_store()


def today_ai_valuations() -> dict:
    store = val.ai_store_today(_valuation_ai_store())
    return dict(store["by_provider"].get(news_provider(), {}))


def check_ai_valuations(items: list):
    provider = news_provider()
    api_key = st.secrets.get(AI_KEY_SECRETS[provider], None)
    if not api_key:
        st.error(f"Липсва {AI_KEY_SECRETS[provider]} в Streamlit Secrets.")
        return
    bar = st.progress(0.0, text=f"Търся оценки на справедливата цена за {len(items)} акции - 20-60 сек. всяка...")
    found = fund.research_news_many(
        [(x["name"], x["symbol"], False) for x in items], provider, api_key, gemini_model=gemini_news_model(),
        gemini_fallback_model=gemini_model(), claude_model=claude_news_model(),
        make_prompt=val.valuation_prompt, parse=val.parse_valuation,
        on_done=lambda i, n: bar.progress(i / n, text=f"Проверени {i}/{n}"))
    bar.empty()
    store = val.ai_store_today(_valuation_ai_store())
    with store["lock"]:
        store["by_provider"].setdefault(provider, {}).update(found)
        store["checks"][provider] = store["checks"].get(provider, 0) + len(found)


def valuation_summary(v: dict) -> str:
    """За основната таблица - кратко: точката на присъдата по методите от Yahoo + % до справедливата."""
    verdict = v.get("verdict", val.VALUE_NO_DATA)
    up = v.get("median_upside")
    if verdict == val.VALUE_ETF:
        return "ETF"
    if up is None:
        return "—"
    return f"{verdict.split()[0]} {up:+.0f}%"


AI_VERDICT_SHORT = {val.VALUE_FAIR: "⚪ Неутрална"}  # „Около справедливата“ е дълго за колоната


def ai_valuation_summary(a: dict) -> str:
    """За основната таблица: AI присъдата + медианата на намерените справедливи цени."""
    if not a:
        return ""
    if "error" in a:
        return "⚠️ Грешка"
    verdict = a.get("verdict", "")
    return AI_VERDICT_SHORT.get(verdict, verdict)


def render_valuation_table(selected: list, vals: dict, ai: dict):
    """💎 Подробната таблица на справедливата цена (показва се само при поискване)."""
    rows = []
    for x in selected:
        v = vals.get(x["symbol"]) or {}
        m = v.get("methods") or {}
        a = ai.get(x["symbol"]) or {}
        est = [e["fair_value"] for e in a.get("estimates") or []]
        rows.append({
            "Име": x["name"], "Цена": v.get("price"),
            "💎 Оценка": v.get("verdict", val.VALUE_NO_DATA), "До справедливата %": v.get("median_upside"),
            "Гласове": val.votes_text(v),
            "Анализатори": m.get("Анализатори"), "Греъм": m.get("Греъм"), "Линч": m.get("Линч (P/E = ръст)"),
            f"FCF {val.REQUIRED_FCF_YIELD:.0%}": m.get(f"FCF {val.REQUIRED_FCF_YIELD:.0%}"),
            "PEG": v.get("peg"), "P/E": v.get("pe"), "Fwd P/E": v.get("fwd_pe"), "P/B": v.get("pb"),
            "EV/EBITDA": v.get("ev_ebitda"), "FCF доходност %": v.get("fcf_yield"),
            "🤖 AI оценка": ("⚠️ Грешка" if "error" in a else a.get("verdict", "")) if a else "",
            "🤖 AI справедлива": round(median(est), 2) if est else None,
        })
    money = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Име": st.column_config.TextColumn(pinned=True),
            "💎 Оценка": st.column_config.TextColumn(help=f"По медианата на методите: над +{val.MARGIN_PCT}% до справедливата "
                                                         f"и поне половината гласове = подценена; под -{val.MARGIN_PCT}% = надценена"),
            "До справедливата %": st.column_config.NumberColumn(format="%+.1f", help="Медианата на методите спрямо цената"),
            "Гласове": st.column_config.TextColumn(help="Колко метода (и PEG) казват подценена / около / надценена"),
            "Цена": money, "Анализатори": st.column_config.NumberColumn(format="%.2f", help="Средна целева цена (поне 3 анализатора)"),
            "Греъм": st.column_config.NumberColumn(format="%.2f", help="√(22.5 × печалба на акция × балансова стойност на акция)"),
            "Линч": st.column_config.NumberColumn(format="%.2f", help="Справедливо P/E = ръстът на печалбата в % (5-25) × прогнозна печалба"),
            f"FCF {val.REQUIRED_FCF_YIELD:.0%}": st.column_config.NumberColumn(
                format="%.2f", help="Цената, при която свободният паричен поток дава 5% доходност"),
            "PEG": st.column_config.NumberColumn(format="%.2f", help="P/E ÷ ръст; под 1 = евтина, над 2 = скъпа"),
            "P/E": st.column_config.NumberColumn(format="%.1f"), "Fwd P/E": st.column_config.NumberColumn(format="%.1f"),
            "P/B": st.column_config.NumberColumn(format="%.2f"), "EV/EBITDA": st.column_config.NumberColumn(format="%.1f"),
            "FCF доходност %": st.column_config.NumberColumn(format="%.1f"),
            "🤖 AI справедлива": st.column_config.NumberColumn(format="%.2f", help="Медианата на намерените в интернет оценки"),
        },
    )
    st.caption("Цените са във валутата на акцията. Методите са ориентир - при банки, застрахователи и бързо растящи "
               "компании Греъм и FCF често подценяват; гледай съгласието между методите и AI източниците.")


def render_valuation_detail(v: dict, a: dict):
    """Подробностите за справедливата цена на един инструмент (клик по клетката в основната таблица)."""
    if v.get("verdict") == val.VALUE_ETF:
        st.markdown("ETF - за фондовете няма „справедлива цена“.")
    elif v.get("methods"):
        price = v.get("price")
        parts = [f"**{m}**: {fv:.2f}" + (f" ({100 * (fv / price - 1):+.0f}%)" if price else "")
                 for m, fv in v["methods"].items()]
        up = v.get("median_upside")
        head = f"**{v.get('verdict')}{f' {up:+.0f}%' if up is not None else ''}** · гласове {val.votes_text(v) or '—'}"
        body = (f"Цена {price:.2f} · " if price else "") + " · ".join(parts)
        if v.get("peg"):
            body += f" · PEG {v['peg']:.2f}"
        st.markdown(f"{head}  \n{body}")
    else:
        st.markdown(f"**{v.get('verdict', val.VALUE_NO_DATA)}** - Yahoo няма достатъчно данни за методите.")
    if not a:
        st.caption("🤖 AI проверка в интернет - с бутона „🤖 Справедлива цена AI“ горе.")
    elif "error" in a:
        st.error(friendly_ai_error(a["error"], news_provider()))
    else:
        st.markdown(f"**🤖 AI: {a['verdict']}** - {a['summary'] or '—'}")
        if a["estimates"]:
            st.markdown(" · ".join(f"**{e['source']}**: {e['fair_value']:.2f} {e['currency']}" for e in a["estimates"]))
        if a["sources"]:
            st.markdown("Източници: " + " · ".join(f"[{s_['title']}]({s_['url']})" for s_ in a["sources"]))


def confirm_paid_check(key: str, count: int, what: str) -> bool:
    """Потвърждение за платените AI проверки (на цяла ширина под бутоните). True = „Да, провери“."""
    if not st.session_state.get(key):
        return False
    with st.container(border=True):
        st.warning(f"Ще се направят **{count} платени проверки** ({what}) с {news_provider()}. Продължаваме ли?")
        c1, c2 = st.columns(2)
        if c1.button("✅ Да, провери", key=f"{key}_yes", type="primary", width="stretch"):
            st.session_state.pop(key, None)
            return True
        if c2.button("✖ Откажи", key=f"{key}_no", width="stretch"):
            st.session_state.pop(key, None)
            st.rerun()
    return False


# ---------------------------------------------------------------- тренд

def trend_quality(close: pd.Series, bars: int, periods_per_year: int):
    """(годишен наклон %, R², качество = наклон x R²) на права линия през ln(цена)
    за последните bars свещи; None при недостатъчно данни."""
    values = close.dropna().to_numpy()[-bars:]
    if len(values) < bars * 0.8 or (values <= 0).any():
        return None
    y = np.log(values)
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if y.std() > 0 else 0.0
    annual = float((np.exp(slope * periods_per_year) - 1) * 100)
    return annual, r2, annual * r2


def analyze_trend(daily: pd.DataFrame):
    """Показателите на двата метода за една акция или None, ако историята е твърде къса."""
    if daily is None or len(daily) < 220:
        return None
    weekly = resample_ohlc(daily, "W")
    price = float(daily["Close"].iloc[-1])
    atr = average_true_range(daily, period=14)
    ema_d = ema_alignment(daily["Close"])
    ema_w = ema_alignment(weekly["Close"])  # None при под 200 седмици (млада акция)
    q_d = trend_quality(daily["Close"], QUALITY_BARS_DAILY, 252)
    q_w = trend_quality(weekly["Close"], QUALITY_BARS_WEEKLY, 52)
    return {
        "price": price, "atr_pct": round(100 * atr / price, 2) if atr and price else None,
        "ema_d": ema_d[0] if ema_d else None, "ema_w": ema_w[0] if ema_w else None,
        "slope_d": q_d[0] if q_d else None, "r2_d": q_d[1] if q_d else None, "quality_d": q_d[2] if q_d else None,
        "slope_w": q_w[0] if q_w else None, "r2_w": q_w[1] if q_w else None, "quality_w": q_w[2] if q_w else None,
    }


def trend_universe() -> dict:
    """Същият универс като на Photon скана с настройките по подразбиране (месечният списък,
    без ливъриджнати/парични/извън Европа и САЩ, плюс ръчно добавените)."""
    liquidity = (rules.STOCK_MIN_MARKET_CAP, rules.STOCK_MIN_TURNOVER, rules.ETF_MIN_AUM, rules.ETF_MIN_TURNOVER)
    tickers = load_universe(max_instruments=None, liquidity=liquidity, curated_mtime=curated_file_mtime())
    _, resolved, types = load_curated_symbol_info(curated_file_mtime())
    tickers = {
        n: s for n, s in tickers.items()
        if not rules.is_leveraged_or_short_etp(n)
        and not (types.get(s) == "ETF" and rules.is_cash_or_bond_fund(n))
        and not rules.is_non_eu_us_listing(resolved.get(s, s))
    }
    tickers = apply_manual_universe(tickers, load_manual_universe(), False)
    return {n: resolved.get(s, s) for n, s in tickers.items()}


def run_trend_scan(tickers: dict, progress) -> pd.DataFrame:
    data = fetch_ohlc_many(list(tickers.values()), TREND_PERIOD, "1d", on_progress=lambda done, total: progress.progress(
        done / total, text=f"Дневни данни за 5 години: {done}/{total}"))
    currencies, _, types = load_curated_symbol_info(curated_file_mtime())
    rows = []
    for name, symbol in tickers.items():
        m = analyze_trend(data.get(symbol))
        if m:
            rows.append({"Име": name, "Тикер": symbol, "Вид": types.get(symbol, ""),
                         "Валута": currencies.get(symbol, "EUR"), **m})
    return pd.DataFrame(rows)


def classify(df: pd.DataFrame, min_r2: float, min_slope: float, atr_range: tuple) -> pd.DataFrame:
    """Колони „А“ / „Б“ (минава ли метода на D и на W) според праговете; ATR % филтрира и двата."""
    out = df.copy()
    out["А"] = out["ema_d"].eq(True) & out["ema_w"].eq(True)
    out["Б"] = ((out["r2_d"] >= min_r2) & (out["slope_d"] >= min_slope)
                & (out["r2_w"] >= min_r2) & (out["slope_w"] >= min_slope)).fillna(False)
    lo, hi = atr_range
    atr_ok = out["atr_pct"].ge(lo) & (out["atr_pct"].le(hi) if hi < ATR_SLIDER_MAX else True)
    out["А"] &= atr_ok
    out["Б"] &= atr_ok
    return out


def display_table(df: pd.DataFrame, news: dict) -> pd.DataFrame:
    def mark(v):
        return "✓" if v is True else ("✗" if v is False else "—")
    return pd.DataFrame({
        "Име": df["Име"], "Тикер": df["Тикер"], "Цена": df["price"].round(2), "Валута": df["Валута"],
        "Метод": np.where(df["А"] & df["Б"], "А + Б", np.where(df["А"], "А", np.where(df["Б"], "Б", "—"))),
        "📐 EMA D": df["ema_d"].map(mark), "📐 EMA W": df["ema_w"].map(mark),
        "📈 Качество D": df["quality_d"].round(0), "R² D": df["r2_d"].round(2), "Наклон D %/год": df["slope_d"].round(0),
        "📈 Качество W": df["quality_w"].round(0), "R² W": df["r2_w"].round(2), "Наклон W %/год": df["slope_w"].round(0),
        "🌊 ATR %": df["atr_pct"].round(1),
        "📰 Новини": [(news.get(s) or {}).get("verdict", "") for s in df["Тикер"]],
    }).sort_values("📈 Качество D", ascending=False, na_position="last").reset_index(drop=True)


NEWS_RANK = {fund.NEWS_POSITIVE: 0, fund.NEWS_NEUTRAL: 1, "": 1, fund.NEWS_NEGATIVE: 2}


def _open_selected(key: str, table: pd.DataFrame):
    row = clicked_name_row(key)  # клик по името (колона „Име“) отваря графиката
    if row is not None and row < len(table):
        st.session_state["tr_dialog"] = (table["Име"].iloc[row], table["Тикер"].iloc[row])


SCAN_MAIN_COLUMNS = ["⭐", "Име", "🏛 Fair Value", "❤️ Здраве", "💎 Справедлива", "🤖 AI оценка", "📰 Новини"]
SCAN_SIGNAL_COLUMNS = ["🏛 Fair Value", "❤️ Здраве", "💎 Справедлива", "🤖 AI оценка", "📰 Новини"]


def render_table(df: pd.DataFrame, key: str, news: dict, valuations: bool = False):
    """Таблица от скана: кратка (както в Селектирани) - ⭐, Име, Fair Value / Здраве (InvestingPro),
    💎 Справедлива (Yahoo методите - само при valuations=True: тегли .info за всеки, затова само за „И двата“)
    и 🤖 AI оценка (ако е проверявана днес); показателите на тренда - с „Още колони“."""
    if df.empty:
        st.info("Няма инструменти в тази група.")
        return
    table = display_table(df, news)
    selected = {x["symbol"] for x in load_selected()}
    table.insert(0, "⭐", ["⭐" if s in selected else "" for s in table["Тикер"]])
    ip_data, ai = ip.load_data(), today_ai_valuations()
    recs = [ip.lookup(s, n, ip_data) for s, n in zip(table["Тикер"], table["Име"])]
    table["🏛 Fair Value"] = [ip.fv_summary(r) for r in recs]
    table["❤️ Здраве"] = [ip.health_summary(r) for r in recs]
    if valuations:
        with st.spinner(f"Тегля справедливата цена (Yahoo) за {len(table)} инструмента..."):
            vals = selected_valuations(tuple(table["Тикер"]))
        table["💎 Справедлива"] = [valuation_summary(vals.get(s) or {}) for s in table["Тикер"]]
    table["🤖 AI оценка"] = [ai_valuation_summary(ai.get(s) or {}) for s in table["Тикер"]]
    # най-горе - с най-много 🟢 (после най-малко 🔴), при равни - положителните новини и качеството D
    signals = table[[c for c in SCAN_SIGNAL_COLUMNS if c in table.columns]].astype(str)
    table = table.assign(_g=-signals.apply(lambda c: c.str.startswith("🟢")).sum(axis=1),
                         _r=signals.apply(lambda c: c.str.startswith("🔴")).sum(axis=1),
                         _n=table["📰 Новини"].map(NEWS_RANK).fillna(1)) \
        .sort_values(["_g", "_r", "_n"], kind="stable").drop(columns=["_g", "_r", "_n"]).reset_index(drop=True)
    # „📰 Новини“ - най-отзад (в display_table е по средата)
    table = table[[c for c in table.columns if c != "📰 Новини"] + ["📰 Новини"]]
    show_all = st.toggle("Още колони", key=f"{key}_all_cols", help="Тикер, цена, методите, EMA, качество, ATR")
    order = list(table.columns) if show_all else [c for c in SCAN_MAIN_COLUMNS if c in table.columns]
    st.dataframe(
        numbered(table), hide_index=False, width="stretch", key=(tk := table_key(key)), selection_mode="single-cell",
        on_select=lambda: _open_selected(tk, table), column_order=order,
        column_config={
            "Име": st.column_config.TextColumn(pinned=True),
            "⭐": st.column_config.TextColumn("⭐", width=40, help="В ⭐ Селектирани"),
            "Метод": st.column_config.TextColumn(help="А = EMA подреждане на D и W; Б = качество на тренда на D и W"),
            "📈 Качество D": st.column_config.NumberColumn(format="%.0f", help="Годишен наклон % x R² за последните 90 дни"),
            "📈 Качество W": st.column_config.NumberColumn(format="%.0f", help="Годишен наклон % x R² за последните 52 седмици"),
            "R² D": st.column_config.NumberColumn(format="%.2f", help="0-1: колко плътно цената следва правата линия"),
            "R² W": st.column_config.NumberColumn(format="%.2f"),
            "🌊 ATR %": st.column_config.NumberColumn(format="%.1f", help="Среден дневен ход (ATR 14) като % от цената"),
            "🏛 Fair Value": st.column_config.TextColumn(
                width="small", help="InvestingPro Fair Value: 🟢 изгодна/подценена · ⚪ справедлива · 🔴 надценена/скъпа "
                                    "и % до справедливата (от експорта в „⭐ Селектирани“ → „📥 InvestingPro“)"),
            "❤️ Здраве": st.column_config.TextColumn(
                width="small", help="InvestingPro Financial Health (0-5): 🟢 добро и нагоре · ⚪ средно · 🔴 слабо"),
            "💎 Справедлива": st.column_config.TextColumn(
                width="small", help="🟢 подценена / ⚪ около / 🔴 надценена по методите от Yahoo (анализатори, Греъм, "
                                    "Линч, FCF, PEG) и % до справедливата (медианата)"),
            "🤖 AI оценка": st.column_config.TextColumn(
                width="small", help="Справедливата цена от проверката в интернет (бутона „🤖 Справедлива цена AI“ в "
                                    "⭐ Селектирани) - само ако е проверявана днес"),
            "📰 Новини": st.column_config.TextColumn(
                width="small", help="От бутона „🔎 Провери новини и анализи“ (в „И двата“ - под таблицата)"),
        },
    )
    st.caption("👆 Кликни името за графиката - там е и ⭐ за Селектирани. Най-горе - с най-много 🟢; "
               "подреждане по колона - клик върху заглавието ѝ.")


def render_trend_section():
    tickers = trend_universe()
    # чист начален екран: бутонът за скан, а праговете и обяснението на методите - в изскачащи панели
    scan_col, thr_col, info_col = st.columns([2, 1, 1], vertical_alignment="center")
    scan_clicked = scan_col.button("🔍 Сканирай за възходящ тренд", type="primary", key="tr_scan", width="stretch")
    with thr_col.popover("⚙️ Прагове", width="stretch"):
        min_r2 = st.slider("Метод Б: мин. R²", 0.3, 0.95, 0.6, 0.05, key="tr_min_r2",
                           help="Колко равномерен трябва да е ръстът (1 = идеална права линия).")
        min_slope = st.slider("Метод Б: мин. наклон (% годишно)", 0, 60, 10, 5, key="tr_min_slope")
        atr_range = st.slider("ATR % (и двата метода)", 0.0, ATR_SLIDER_MAX, (0.0, ATR_SLIDER_MAX), 0.5,
                              key="tr_atr", help="10 = без горна граница. За swing до месец ~1.5-4% е добре.")
    with info_col.popover("ℹ️ Методите", width="stretch"):
        st.markdown(
            "**📈 Възходящ тренд на дневна и седмична графика** - два метода един до друг за нагледно "
            "сравнение. Само показват тренда; входът и stop-ът са в таба „🧭 Photon вход“.\n\n"
            "**Метод А · 📐 EMA подреждане** (твоят): EMA20 > EMA50 > EMA200 и на дневната, и на седмичната.\n\n"
            "**Метод Б · 📈 Качество на тренда** (моят): годишен наклон × R² на права линия през цената - "
            "90 дни на дневната и 52 седмици на седмичната.\n\n"
            f"Универс: **{len(tickers)}** инструмента · 5 години дневни цени (седмичната EMA200 иска ~4 г.)."
        )
    if scan_clicked:
        progress = st.progress(0.0, text="Тегля дневни данни за 5 години...")
        st.session_state["tr_results"] = run_trend_scan(tickers, progress)
        progress.empty()

    news = today_news()
    selected = load_selected()
    raw = st.session_state.get("tr_results")
    scanned = raw is not None and not raw.empty
    if scanned:
        df = classify(raw, min_r2, min_slope, atr_range)
        both, only_a, only_b = df[df["А"] & df["Б"]], df[df["А"] & ~df["Б"]], df[df["Б"] & ~df["А"]]
        m = st.columns(4)
        m[0].metric("Сканирани", len(df))
        m[1].metric("📐 Метод А", int(df["А"].sum()))
        m[2].metric("📈 Метод Б", int(df["Б"].sum()))
        m[3].metric("🤝 И двата", len(both))
        young = int(df["ema_w"].isna().sum())
        if young:
            st.caption(f"{young} инструмента са с под ~4 години история - седмичната EMA200 не се смята и не минават метод А.")
    elif raw is not None:
        st.warning("Няма данни - опитай пак след малко (Yahoo понякога ограничава заявките).")

    labels = [f"⭐ Селектирани ({len(selected)})", f"🧭 Photon вход ({len(selected)})"]
    if scanned:
        labels += [f"🤝 И двата ({len(both)})", f"📐 Само А - EMA ({len(only_a)})", f"📈 Само Б - качество ({len(only_b)})"]
    tabs = st.tabs(labels)
    with tabs[0]:
        render_selected(selected, news, raw if scanned else None, (min_r2, min_slope, atr_range))
    with tabs[1]:
        # Photon Phases (BOS/CHoCH, Phase A/B) - само за селектираните: къде и кога да се влезе
        st.caption("Photon Phases (BOS/CHoCH, Phase A/B, само long) сканира **само ⭐ Селектирани** - "
                   "кога и на каква цена да се влезе в избраните трендове.")
        if selected:
            render_photon_strategy(universe={x["name"]: x["symbol"] for x in selected})
        else:
            st.info("Още няма селектирани - добави от „🤝 И двата“ (клик по ред → ⭐ Добави в Селектирани).")
    if scanned:
        with tabs[2]:
            st.caption("Минават и двата метода. Провери новините, разгледай графиката и добави най-добрите в ⭐ Селектирани.")
            render_table(both, "tr_tbl_both", news, valuations=True)
            if not both.empty:
                st.divider()
                section_header("📰 Новинарска подкрепа", status="info", subtitle="Само за групата „И двата“ (А + Б)")
                render_news_section([SimpleNamespace(name=n, symbol=sym) for n, sym in zip(both["Име"], both["Тикер"])],
                                    news, key="tr_news")
        with tabs[3]:
            st.caption("EMA-тата са подредени, но ръстът е накъсан или бавен (нисък R² или наклон).")
            render_table(only_a, "tr_tbl_a", news)
        with tabs[4]:
            st.caption("Равномерен ръст, но EMA20/50/200 не са подредени - често след скорошен пулбек или млад тренд.")
            render_table(only_b, "tr_tbl_b", news)
    elif raw is None:
        st.info("Натисни **🔍 Сканирай за възходящ тренд** за нови кандидати. Първият скан тегли 5 години история - "
                "1-3 минути. „⭐ Селектирани“ работи и без скан.")

    install_chart_scripts()  # стрелката назад затваря прозореца с графиката
    # прозорецът: от клик по ред тук или по ред в таблицата с новините (тя ползва ph_dialog)
    pending = st.session_state.pop("tr_dialog", None) or st.session_state.pop("ph_dialog", None)
    if pending:
        name, symbol = pending
        price = None
        if scanned and symbol in set(raw["Тикер"]):
            price = float(raw.loc[raw["Тикер"] == symbol, "price"].iloc[0])
        instrument_dialog(name, symbol, actions=lambda: selection_actions(name, symbol, price))


SELECTED_MAIN_COLUMNS = ["Име", "Промяна %", "📰 Новини", "💎 Справедлива", "🤖 AI оценка",
                         "🏛 Fair Value", "❤️ Здраве"]
SELECTED_DETAIL_COLUMNS = {"📰 Новини": "news", "💎 Справедлива": "value", "🤖 AI оценка": "value",
                           "🏛 Fair Value": "ip", "❤️ Здраве": "ip"}
# най-горе - с най-много 🟢 в тях
SIGNAL_COLUMNS = ["📰 Новини", "💎 Справедлива", "🤖 AI оценка", "🏛 Fair Value", "❤️ Здраве"]


def render_selected(selected: list, news: dict, raw, thresholds):
    """Таб „⭐ Селектирани“: стоят, докато не ги премахнеш ръчно; текущите показатели
    се теглят само за тях (без пълен скан), за всекидневен преглед. Горе - бутоните за новините и
    справедливата цена (AI), основната таблица е кратка (останалото с „Още колони“), клик по клетка
    с новини / справедлива цена показва подробностите под таблицата, клик по името - графиката."""
    if not selected:
        st.info("Още няма селектирани. Отвори графиката на инструмент от „🤝 И двата“ (клик по името) и натисни "
                "**⭐** горе до името - ще стои тук, докато не го премахнеш.")
        return
    provider = news_provider()
    symbols = tuple(x["symbol"] for x in selected)
    with st.spinner("Тегля актуалните цени и показателите на селектираните..."):
        metrics = selected_metrics(symbols)
        vals = selected_valuations(symbols)
    ai = today_ai_valuations()
    news_missing = [SimpleNamespace(name=x["name"], symbol=x["symbol"]) for x in selected
                    if x["symbol"] not in news or "error" in news[x["symbol"]]]
    val_missing = [x for x in selected if (vals.get(x["symbol"]) or {}).get("verdict") != val.VALUE_ETF
                   and (x["symbol"] not in ai or "error" in ai[x["symbol"]])]

    b1, b2, b4, b3 = st.columns([2, 2, 1.3, 1], vertical_alignment="center")
    if b1.button(f"🔎 Новини ({len(news_missing)})", key="tr_sel_news_btn", disabled=not news_missing, width="stretch",
                 help=f"Новини и анализи с {provider} (търсене в интернет, платено; пази се до края на деня за всички "
                      f"сесии). Днес с {provider}: {news_checks_today(provider)} проверки."):
        st.session_state["tr_sel_news_confirm"] = True
    if b2.button(f"🤖 Справедлива цена AI ({len(val_missing)})", key="tr_val_btn", disabled=not val_missing,
                 width="stretch", help="Morningstar, Simply Wall St, GuruFocus, анализатори - с линкове. Платено; "
                                       "пази се до края на деня. Без ETF-ите."):
        st.session_state["tr_val_confirm"] = True
    ip_age = ip.data_age_days()
    ip_stale = ip_age is None or ip_age > ip.STALE_DAYS
    with b4.popover("📥 InvestingPro" + (" ⚠️" if ip_stale else ""), width="stretch",
                    help="Fair Value и финансово здраве от експорта на InvestingPro Screener (1-2 пъти в месеца)"):
        types = load_curated_symbol_info(curated_file_mtime())[2]
        ip.render_upload({x["symbol"]: x["name"] for x in selected  # без ETF/ETC - за тях няма Fair Value
                          if types.get(x["symbol"]) != "ETF"
                          and (vals.get(x["symbol"]) or {}).get("verdict") != val.VALUE_ETF})
    if b3.button("🔄 Опресни", key="tr_sel_reload", width="stretch",
                 help="Презареди списъка (ако е променян от друго устройство) и показателите"):
        load_selected(force=True)
        selected_metrics.clear()
        st.rerun()
    if confirm_paid_check("tr_sel_news_confirm", len(news_missing), "новини"):
        found = check_news(news_missing, news)
        errors = sum(1 for r in found.values() if "error" in r)
        st.toast(f"Новини: {len(found) - errors} проверени" + (f", {errors} с грешка (натисни пак)" if errors else ""))
        st.rerun()
    if confirm_paid_check("tr_val_confirm", len(val_missing), "справедлива цена"):
        check_ai_valuations(val_missing)
        st.rerun()

    currencies, _, _ = load_curated_symbol_info(curated_file_mtime())
    rows = []
    for x in selected:
        m = metrics.get(x["symbol"]) or {}
        rows.append({"Име": x["name"], "Тикер": x["symbol"], "Валута": currencies.get(x["symbol"], "EUR"),
                     "added": x.get("added"), "price_added": x.get("price"), **m})
    df = pd.DataFrame(rows)
    for col in ("price", "ema_d", "ema_w", "slope_d", "r2_d", "quality_d", "slope_w", "r2_w", "quality_w", "atr_pct",
                "price_added"):
        if col not in df:
            df[col] = None
        if col not in ("ema_d", "ema_w"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = classify(df, *thresholds)
    table = display_table(df, news)
    extra = df.set_index("Тикер")
    table.insert(1, "Добавен", [extra.at[s, "added"] for s in table["Тикер"]])
    change = []
    for s in table["Тикер"]:
        p0, p1 = extra.at[s, "price_added"], extra.at[s, "price"]
        change.append(round(100 * (p1 / p0 - 1), 1) if p0 and p1 and pd.notna(p1) else None)
    table.insert(2, "Промяна %", pd.to_numeric(pd.Series(change, index=table.index), errors="coerce"))  # NaN, не „None“
    table["💎 Справедлива"] = [valuation_summary(vals.get(s) or {}) for s in table["Тикер"]]
    table["🤖 AI оценка"] = [ai_valuation_summary(ai.get(s) or {}) for s in table["Тикер"]]
    ip_data = ip.load_data()
    ip_recs = {s: ip.lookup(s, n, ip_data) for s, n in zip(table["Тикер"], table["Име"])}
    table["🏛 Fair Value"] = [ip.fv_summary(ip_recs[s]) for s in table["Тикер"]]
    table["❤️ Здраве"] = [ip.health_summary(ip_recs[s]) for s in table["Тикер"]]
    # най-горе тези с най-много зелени точки (новини + справедлива + AI), после с най-малко червени
    signals = table[SIGNAL_COLUMNS].astype(str)
    greens = signals.apply(lambda c: c.str.startswith("🟢")).sum(axis=1)
    reds = signals.apply(lambda c: c.str.startswith("🔴")).sum(axis=1)
    table = table.assign(_g=-greens, _r=reds).sort_values(["_g", "_r"], kind="stable") \
        .drop(columns=["_g", "_r"]).reset_index(drop=True)

    show_all = st.toggle("Още колони", key="tr_sel_all_cols")
    order = list(table.columns) if show_all else SELECTED_MAIN_COLUMNS
    st.dataframe(
        numbered(table).style.map(change_style, subset=["Промяна %"]).format("{:+.1f}", na_rep="", subset=["Промяна %"]),
        hide_index=False, width="stretch", key=(tk := table_key("tr_tbl_sel")), selection_mode="single-cell",
        on_select=lambda: _open_selected(tk, table), column_order=order,
        column_config={
            "Име": st.column_config.TextColumn(pinned=True),
            "Метод": st.column_config.TextColumn(help="Днес: А = EMA подреждане на D и W; Б = качество на тренда"),
            "Добавен": st.column_config.TextColumn(width="small"),
            "Промяна %": st.column_config.NumberColumn("±%", format="%+.1f", width=60,
                                                       help="Промяна на цената в % от добавянето в Селектирани "
                                                            "(датата е в „Още колони“)"),
            "🌊 ATR %": st.column_config.NumberColumn(format="%.1f"),
            "📰 Новини": st.column_config.TextColumn(width="small", help="Кликни клетката за подробностите и източниците"),
            "💎 Справедлива": st.column_config.TextColumn(
                width="small",
                help="🟢 подценена / ⚪ около / 🔴 надценена по методите от Yahoo (анализатори, Греъм, Линч, FCF, PEG) "
                     "и % до справедливата (медианата). Кликни клетката за подробностите"),
            "🤖 AI оценка": st.column_config.TextColumn(
                width="small",
                help="От проверката в интернет (бутона горе): Morningstar, Simply Wall St, GuruFocus, анализатори. "
                     "Кликни клетката за справедливите цени и източниците"),
            "🏛 Fair Value": st.column_config.TextColumn(
                width="small", help="InvestingPro Fair Value: 🟢 изгодна/подценена · ⚪ справедлива · 🔴 надценена/скъпа "
                                    "и % до справедливата. От качения експорт (бутона „📥 InvestingPro“ горе)"),
            "❤️ Здраве": st.column_config.TextColumn(
                width="small", help="InvestingPro Financial Health (0-5): 🟢 добро и нагоре · ⚪ средно · 🔴 слабо"),
        },
    )
    st.caption("👆 Името = графиката (там е и 🗑 за премахване) · клетка 📰 / 💎 / 🤖 / 🏛 / ❤️ = подробностите тук "
               "отдолу." + (f" · ⚠️ Данните от InvestingPro са от преди {ip_age} дни - качи нов експорт."
                           if ip_age is not None and ip_age > ip.STALE_DAYS else ""))

    # подробностите за кликнатата клетка с новини / справедлива цена
    cells = ((st.session_state.get(tk) or {}).get("selection") or {}).get("cells") or []
    if cells and cells[0][1] in SELECTED_DETAIL_COLUMNS and cells[0][0] < len(table):
        row, kind = cells[0][0], SELECTED_DETAIL_COLUMNS[cells[0][1]]
        name, symbol = table["Име"].iloc[row], table["Тикер"].iloc[row]
        with st.container(border=True):
            if kind == "news":
                st.markdown(f"##### 📰 {name}")
                r = news.get(symbol)
                if not r:
                    st.caption("Още няма проверка за днес - бутона „🔎 Новини“ горе.")
                elif "error" in r:
                    st.error(friendly_ai_error(r["error"], provider))
                else:
                    render_news_item(r)
            elif kind == "ip":
                st.markdown(f"##### 🏛 {name}")
                st.markdown(ip.detail_markdown(ip_recs.get(symbol)) or
                            "Няма данни от InvestingPro за него (ETF или не е в качения експорт).")
            else:
                st.markdown(f"##### 💎 {name}")
                render_valuation_detail(vals.get(symbol) or {}, ai.get(symbol) or {})

    if st.toggle("💎 Подробна таблица на справедливата цена", key="tr_val_table"):
        render_valuation_table(selected, vals, ai)
