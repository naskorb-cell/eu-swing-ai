"""Секция 📈 Възходящ тренд (D + W): два метода за стабилен възходящ тренд на дневната
И седмичната графика, един до друг за нагледно сравнение:
  - Метод А „📐 EMA подреждане“ (предложението на Наско): EMA20 > EMA50 > EMA200;
  - Метод Б „📈 Качество на тренда“: годишен наклон x R² на права линия през
    логаритъма на цената (метод на Андреас Кленов) - високо е само при бърз И равномерен ръст.
И при двата - ATR % (дневен ход като % от цената) за волатилността."""

import numpy as np
import pandas as pd
import streamlit as st

import universe_rules as rules
from indicators import average_true_range, ema_alignment, resample_ohlc
from photon import fetch_ohlc_many, install_chart_scripts, instrument_dialog
from ui_common import section_header
from universe import (
    apply_manual_universe, curated_file_mtime, load_curated_symbol_info, load_manual_universe, load_universe,
)

TREND_PERIOD = "5y"  # седмичната EMA200 иска ~4 години история
QUALITY_BARS_DAILY, QUALITY_BARS_WEEKLY = 90, 52  # ~4 месеца дневни / 1 година седмични свещи
ATR_SLIDER_MAX = 10.0  # горната граница на плъзгача = без горен лимит


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


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    def mark(v):
        return "✓" if v is True else ("✗" if v is False else "—")
    return pd.DataFrame({
        "Име": df["Име"], "Тикер": df["Тикер"], "Цена": df["price"].round(2), "Валута": df["Валута"],
        "Метод": np.where(df["А"] & df["Б"], "А + Б", np.where(df["А"], "А", np.where(df["Б"], "Б", "—"))),
        "📐 EMA D": df["ema_d"].map(mark), "📐 EMA W": df["ema_w"].map(mark),
        "📈 Качество D": df["quality_d"].round(0), "R² D": df["r2_d"].round(2), "Наклон D %/год": df["slope_d"].round(0),
        "📈 Качество W": df["quality_w"].round(0), "R² W": df["r2_w"].round(2), "Наклон W %/год": df["slope_w"].round(0),
        "🌊 ATR %": df["atr_pct"].round(1),
    }).sort_values("📈 Качество D", ascending=False, na_position="last").reset_index(drop=True)


def _open_selected(key: str, table: pd.DataFrame):
    event = st.session_state.get(key)
    rows = event.selection.rows if event and event.selection else []
    if rows and rows[0] < len(table):
        st.session_state["tr_dialog"] = (table["Име"].iloc[rows[0]], table["Тикер"].iloc[rows[0]])


def render_table(df: pd.DataFrame, key: str):
    if df.empty:
        st.info("Няма инструменти в тази група.")
        return
    table = display_table(df)
    st.dataframe(
        table, hide_index=True, width="stretch", key=key, selection_mode="single-row",
        on_select=lambda: _open_selected(key, table),
        column_config={
            "Име": st.column_config.TextColumn(pinned=True),
            "Метод": st.column_config.TextColumn(help="А = EMA подреждане на D и W; Б = качество на тренда на D и W"),
            "📈 Качество D": st.column_config.NumberColumn(format="%.0f", help="Годишен наклон % x R² за последните 90 дни"),
            "📈 Качество W": st.column_config.NumberColumn(format="%.0f", help="Годишен наклон % x R² за последните 52 седмици"),
            "R² D": st.column_config.NumberColumn(format="%.2f", help="0-1: колко плътно цената следва правата линия"),
            "R² W": st.column_config.NumberColumn(format="%.2f"),
            "🌊 ATR %": st.column_config.NumberColumn(format="%.1f", help="Среден дневен ход (ATR 14) като % от цената"),
        },
    )
    st.caption("👆 Кликни ред за графиката. Подреждане по колона - клик върху заглавието ѝ.")


def render_trend_section():
    section_header(
        "📈 Възходящ тренд на дневна и седмична графика", status="info",
        subtitle="Два метода един до друг - за нагледно сравнение. Само показва, не дава вход/stop.",
    )
    c1, c2 = st.columns(2)
    c1.markdown("**Метод А · 📐 EMA подреждане** (твоят)  \nEMA20 > EMA50 > EMA200 и на дневната, и на седмичната.")
    c2.markdown("**Метод Б · 📈 Качество на тренда** (моят)  \nГодишен наклон × R² на права линия през цената: "
                "90 дни на дневната и 52 седмици на седмичната.")

    with st.expander("⚙️ Прагове"):
        p1, p2, p3 = st.columns(3)
        min_r2 = p1.slider("Метод Б: мин. R²", 0.3, 0.95, 0.6, 0.05, key="tr_min_r2",
                           help="Колко равномерен трябва да е ръстът (1 = идеална права линия).")
        min_slope = p2.slider("Метод Б: мин. наклон (% годишно)", 0, 60, 10, 5, key="tr_min_slope")
        atr_range = p3.slider("ATR % (и двата метода)", 0.0, ATR_SLIDER_MAX, (0.0, ATR_SLIDER_MAX), 0.5,
                              key="tr_atr", help="10 = без горна граница. За swing до месец ~1.5-4% е добре.")

    tickers = trend_universe()
    scan_col, info_col = st.columns([1, 2], vertical_alignment="center")
    if scan_col.button("🔍 Сканирай за възходящ тренд", type="primary", key="tr_scan", width="stretch"):
        progress = st.progress(0.0, text="Тегля дневни данни за 5 години...")
        st.session_state["tr_results"] = run_trend_scan(tickers, progress)
        progress.empty()
    info_col.markdown(f"Универс: **{len(tickers)}** инструмента · 5 години дневни цени (седмичната EMA200 иска ~4 г.)")

    raw = st.session_state.get("tr_results")
    if raw is None:
        st.info("Натисни **🔍 Сканирай за възходящ тренд**. Първият скан тегли 5 години история - 1-3 минути.")
        return
    if raw.empty:
        st.warning("Няма данни - опитай пак след малко (Yahoo понякога ограничава заявките).")
        return

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

    t_both, t_a, t_b = st.tabs([f"🤝 И двата ({len(both)})", f"📐 Само А - EMA ({len(only_a)})",
                                f"📈 Само Б - качество ({len(only_b)})"])
    with t_both:
        render_table(both, "tr_tbl_both")
    with t_a:
        st.caption("EMA-тата са подредени, но ръстът е накъсан или бавен (нисък R² или наклон).")
        render_table(only_a, "tr_tbl_a")
    with t_b:
        st.caption("Равномерен ръст, но EMA20/50/200 не са подредени - често след скорошен пулбек или млад тренд.")
        render_table(only_b, "tr_tbl_b")

    install_chart_scripts()  # стрелката назад затваря прозореца с графиката
    pending = st.session_state.pop("tr_dialog", None)
    if pending:
        instrument_dialog(*pending)
