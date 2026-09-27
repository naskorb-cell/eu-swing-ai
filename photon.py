"""Photon Phases стратегията (SMC/MTF, Phase A/B, само long): анализ, скан, таблици, графики."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import t212_portfolio as t212
import universe_rules as rules
from ai_client import stream_claude
from indicators import (
    alternating_swings, average_true_range, detect_structure_events, drop_incomplete_week,
    find_swing_points, resample_ohlc, resample_session_halves, swing_structure,
)
from portfolio_ui import T212_ACCOUNTS
from ui_common import format_eur, section_header
from universe import (
    INSTRUMENTS_FILE, apply_manual_universe, curated_file_mtime, exchange_to_yahoo_suffix,
    flag_macro_signal, load_curated_symbol_info, load_universe, render_macro_section,
    render_manual_universe_editor, render_universe_refresh, render_universe_search,
    render_universe_uploader,
)

PHOTON_BATCH_SIZE = 50

# Причини за отпадане - ползват се във фунията на скана
REJECT_NO_DATA = "Няма/малко ценови данни"
REJECT_WEEKLY = "Седмичният тренд не е Pro (HH+HL)"
REJECT_DAILY = "Дневният тренд не е Pro (HH+HL)"
REJECT_SMALL_RANGE = "Дневният диапазон е твърде тесен (под мин. x ATR)"
REJECT_BROKEN = "Цена под дневната подкрепа (счупена структура)"
REJECT_NO_4H = "Няма 4ч данни/структура"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlc_batch(symbols: tuple, period: str, interval: str) -> dict:
    """Тегли OHLC за няколко тикера с ЕДНА заявка към yfinance (паралелно)
    вместо по една заявка на тикер. Връща {symbol: DataFrame}; липсващите
    тикери просто ги няма в резултата."""
    out = {}
    try:
        df = yf.download(
            list(symbols), period=period, interval=interval, group_by="ticker",
            progress=False, auto_adjust=True, threads=True,
        )
    except Exception:
        return out
    if df is None or df.empty:
        return out
    for symbol in symbols:
        try:
            sub = df[symbol] if isinstance(df.columns, pd.MultiIndex) else df
        except KeyError:
            continue
        sub = sub.dropna(subset=["Close"])
        if not sub.empty:
            out[symbol] = sub.copy()
    return out


def fetch_ohlc_many(symbols: list, period: str, interval: str, on_progress=None) -> dict:
    """fetch_ohlc_batch на порции по PHOTON_BATCH_SIZE (сортирани, за да са
    стабилни ключовете на кеша между сканирания)."""
    symbols = sorted(set(symbols))
    out = {}
    for i in range(0, len(symbols), PHOTON_BATCH_SIZE):
        out.update(fetch_ohlc_batch(tuple(symbols[i: i + PHOTON_BATCH_SIZE]), period, interval))
        if on_progress:
            on_progress(min(i + PHOTON_BATCH_SIZE, len(symbols)), len(symbols))
    return out


def significant_daily_structure(daily_df: pd.DataFrame, base_order: int, min_range_atr: float, atr_daily):
    """Дневна структура с минимална ширина на диапазона: ако последните swing
    high/low са по-близо от min_range_atr x дневния ATR (шум, не swing),
    повишаваме чувствителността (order) до +4, докато намерим значим диапазон.
    Връща (structure, order, None) или (None, None, причина за отпадане)."""
    min_range = min_range_atr * atr_daily if atr_daily else 0
    for order in range(base_order, base_order + 5):
        s = swing_structure(find_swing_points(daily_df, order=order))
        if s is None:
            return None, None, REJECT_DAILY
        if s["last_high"] - s["last_low"] >= min_range and s["last_high"] > s["last_low"]:
            return s, order, None
    return None, None, REJECT_SMALL_RANGE


def analyze_photon_daily(daily_df: pd.DataFrame, p: dict):
    """Стъпка 1 (без 4ч данни): седмичен и дневен тренд + дневен диапазон.
    Връща (context, None) при успех или (None, причина за отпадане)."""
    if daily_df is None or len(daily_df) < 150:
        return None, REJECT_NO_DATA

    # --- HTF (Седмичен): задължителна посока, само LONG ---
    weekly = resample_ohlc(daily_df, "W")
    if p["closed_weeks_only"]:
        weekly = drop_incomplete_week(weekly, daily_df)
    if len(weekly) < 20:
        return None, REJECT_NO_DATA
    weekly_s = swing_structure(find_swing_points(weekly, order=p["swing_order_weekly"]))
    if weekly_s is None or not weekly_s["uptrend"]:
        return None, REJECT_WEEKLY

    # --- Swing/MTF (Дневен): Pro Swing (нагоре) със значим диапазон - Phase C/D извън обхват ---
    atr_daily = average_true_range(daily_df, period=14)
    daily_s, used_order, why = significant_daily_structure(daily_df, p["swing_order_daily"], p["min_range_atr"], atr_daily)
    if daily_s is None:
        return None, why
    if not daily_s["uptrend"]:
        return None, REJECT_DAILY

    current_price = float(daily_df["Close"].iloc[-1])
    # Затваряне под последния дневен HL = дневната структура е счупена (не е discount)
    if current_price < daily_s["last_low"]:
        return None, REJECT_BROKEN

    return {
        "current_price": current_price, "atr_daily": atr_daily, "daily_order": used_order,
        "daily_support": daily_s["last_low"], "daily_resistance": daily_s["last_high"],
        "weekly_resistance": weekly_s["last_high"],
    }, None


@dataclass
class PhotonSetup:
    """Резултат от Photon анализа за един инструмент. Полетата са на английски
    (за логиката); to_row() дава реда с българските колони за таблиците/AI."""
    name: str
    symbol: str
    price: float
    phase: str                  # "A" (Pro Internal) или "B" (Counter Internal)
    zone: str                   # Discount / Premium / Над съпротивата (BOS)
    range_pos: float            # % от дневния диапазон: 0 = подкрепа, 100 = съпротива
    choch_now: bool
    choch_level: float          # 4ч swing high, над който е CHoCH
    note: str
    daily_support: float
    daily_resistance: float
    weekly_resistance: float
    width_atr: float | None
    stop: float
    rr: float | None
    rr_weekly: float | None
    ready: bool
    poi_low: float | None       # само Phase A
    poi_high: float | None
    daily_order: int            # swing чувствителност, при която е намерена значимата структура
    currency: str = "EUR"

    def to_row(self, held_by: str = "") -> dict:
        return {
            "Име": self.name, "Тикер": self.symbol, "💼 Държа": held_by,
            "Цена": round(self.price, 2), "Валута": self.currency,
            "Фаза": f"{self.phase} ({'Pro' if self.phase == 'A' else 'Counter'} Internal)",
            "Зона": self.zone, "Позиция в диапазона (%)": self.range_pos,
            "4ч CHoCH сега": self.choch_now, "Бележка": self.note,
            "Дневна подкрепа": round(self.daily_support, 2), "Дневна съпротива": round(self.daily_resistance, 2),
            "Ширина (x ATR)": self.width_atr, "Stop": round(self.stop, 2),
            "R/R (до дневна съпротива)": self.rr, "R/R (до седм. съпротива)": self.rr_weekly,
        }


def analyze_photon_intraday(name: str, symbol: str, ctx: dict, intraday: pd.DataFrame, p: dict):
    """Стъпка 2: 4ч Internal структура, фаза, stop и R/R.
    Връща (PhotonSetup, None) или (None, причина за отпадане)."""
    events, atr_4h = None, None
    if intraday is not None and not intraday.empty:
        h4 = resample_session_halves(intraday)
        if len(h4) >= 10:
            events = detect_structure_events(find_swing_points(h4, order=1), p["choch_max_age"])
            atr_4h = average_true_range(h4, period=14)
    if events is None:
        return None, REJECT_NO_4H

    current_price, atr_daily = ctx["current_price"], ctx["atr_daily"]
    daily_support, daily_resistance = ctx["daily_support"], ctx["daily_resistance"]
    daily_range = daily_resistance - daily_support
    if not atr_4h:
        atr_4h = atr_daily / 2 if atr_daily else None

    in_discount = current_price <= daily_support + daily_range / 2
    # позиция в дневния диапазон: 0% = подкрепа, 50% = equilibrium, 100% = съпротива;
    # над 100% = цената е пробила съпротивата (BOS), нов swing high още не е потвърден
    range_pos = round(100 * (current_price - daily_support) / daily_range, 1)
    above_resistance = current_price > daily_resistance

    # --- Класификация на фазата (само A и B - long-only, консервативен обхват) ---
    if events["internal_state"] == "pro":
        phase = "A"
        # POI = зоната точно над последния internal HL (до poi_atr_mult x ATR 4ч)
        poi_low = events["internal_hl"]
        poi_high = poi_low + p["poi_atr_mult"] * atr_4h if atr_4h else poi_low
        at_poi = poi_low <= current_price <= poi_high
        setup_ok = in_discount and at_poi
    else:
        phase = "B"
        poi_low = poi_high = None
        setup_ok = in_discount and events["choch_bullish_now"]

    # --- Stop с ATR буфер под reference low; минимален риск 0.5 x дневен ATR ---
    stop = events["reference_low"] - (p["stop_atr_buffer"] * atr_4h if atr_4h else 0)
    if atr_daily:
        stop = min(stop, current_price - 0.5 * atr_daily)
    risk = current_price - stop
    reward = daily_resistance - current_price
    reward_weekly = ctx["weekly_resistance"] - current_price
    rr = round(reward / risk, 2) if risk > 0 and reward > 0 else None
    rr_weekly = round(reward_weekly / risk, 2) if risk > 0 and reward_weekly > 0 else None
    rr_ok = rr is not None and rr >= p["min_rr"]
    ready = setup_ok and rr_ok

    if ready:
        note = "Вход на POI" if phase == "A" else f"CHoCH преди {events['choch_bars_ago']} свещи"
    elif setup_ok:
        note = f"R/R под {p['min_rr']}"
    elif above_resistance:
        note = "Над дневната съпротива (BOS) - чакаме нов пулбек"
    elif not in_discount:
        note = "Premium - чакаме връщане под 50%"
    elif phase == "A":
        note = f"Чакаме цената в POI ({poi_low:.2f}-{poi_high:.2f})"
    else:
        note = f"Чакаме 4ч CHoCH над {events['choch_level']:.2f}"

    return PhotonSetup(
        name=name, symbol=symbol, price=current_price, phase=phase,
        zone="Над съпротивата (BOS)" if above_resistance else ("Discount" if in_discount else "Premium"),
        range_pos=range_pos, choch_now=events["choch_bullish_now"], choch_level=events["choch_level"],
        note=note, daily_support=daily_support, daily_resistance=daily_resistance,
        weekly_resistance=ctx["weekly_resistance"],
        width_atr=round(daily_range / atr_daily, 1) if atr_daily else None,
        stop=stop, rr=rr, rr_weekly=rr_weekly, ready=ready,
        poi_low=poi_low, poi_high=poi_high, daily_order=ctx["daily_order"],
    ), None


def run_photon_scan(tickers: dict, p: dict, progress, currencies: dict = None):
    """Целият скан: пакетно теглене на дневни данни за всички, стъпка 1,
    после пакетно 60m данни САМО за оцелелите и стъпка 2.
    Връща (готови, watchlist, фуния {етап: брой}, {причина за отпадане: брой}),
    готовите и watchlist-ът са списъци от PhotonSetup."""
    items = list(tickers.items())
    funnel = {"Сканирани": len(items)}
    rejects = {}

    def reject(reason):
        rejects[reason] = rejects.get(reason, 0) + 1

    progress.progress(0.0, text="Тегля дневни данни...")
    daily_data = fetch_ohlc_many(
        [s for _, s in items], "2y", "1d",
        on_progress=lambda done, total: progress.progress(0.5 * done / total, text=f"Дневни данни: {done}/{total}"),
    )

    survivors = []
    for name, symbol in items:
        ctx, why = analyze_photon_daily(daily_data.get(symbol), p)
        if ctx is None:
            reject(why)
        else:
            survivors.append((name, symbol, ctx))
    funnel["Седмичен + дневен Pro тренд"] = len(survivors)

    intraday_data = {}
    if survivors:
        intraday_data = fetch_ohlc_many(
            [s for _, s, _ in survivors], "60d", "60m",
            on_progress=lambda done, total: progress.progress(0.5 + 0.5 * done / total, text=f"4ч данни: {done}/{total}"),
        )

    results, watch_list = [], []
    for name, symbol, ctx in survivors:
        setup, why = analyze_photon_intraday(name, symbol, ctx, intraday_data.get(symbol), p)
        if setup is None:
            reject(why)
        else:
            setup.currency = (currencies or {}).get(symbol, "EUR")
            (results if setup.ready else watch_list).append(setup)
    funnel["Watchlist"] = len(watch_list)
    funnel["Готови за вход"] = len(results)

    results.sort(key=lambda s: s.rr or 0, reverse=True)
    watch_list.sort(key=lambda s: s.range_pos)
    return results, watch_list, funnel, rejects


def generate_ai_analysis_photon(df_ready: pd.DataFrame, df_watch: pd.DataFrame, api_key: str):

    ready_text = df_ready.to_string(index=False) if not df_ready.empty else "НЯМА готови сетъпи в момента."
    watch_text = df_watch.to_string(index=False) if not df_watch.empty else "НЯМА инструменти на watchlist в момента."

    prompt = f"""
    Ти си суинг търговец, ползващ Photon Trading MTF Phases методологията
    (Smart Money Concepts, top-down: HTF Objective -> MTF POIs -> LTF Execution),
    адаптирана: HTF=Седмичен, Swing/MTF=Дневен, Internal/LTF=4ч. САМО LONG.

    Правила:
    - Седмичен тренд трябва да е Pro (възходящ) - твърд филтър.
    - Дневен Swing тренд трябва също да е Pro (възходящ) - твърд филтър.
      (Counter Swing фази C/D са изключени - твърде агресивни за системата.)
    - Phase A (Pro Swing + Pro Internal): 4ч структурата е HH+HL - влизаме
      на POI (зоната точно над последния 4ч higher low), без да чакаме CHoCH.
    - Phase B (Pro Swing + Counter Internal): 4ч е в пулбек (lower high/low) -
      влизаме на СВЕЖ CHoCH (първо затваряне над последния 4ч swing high
      преди най-много няколко свещи).
    - И двете фази: само в discount зона (под 50% от дневния диапазон),
      Stop под reference low с ATR буфер, минимален R/R спрямо дневната съпротива.
    - Колоната "Бележка" казва какво чакаме за всеки инструмент от watchlist-а.
    - Колоната "💼 Държа" (ако я има) показва инструменти, в които вече има
      отворена позиция (N/T = акаунт) - за тях коментирай управление на
      позицията (stop, частична печалба), не нов вход.
    - Колоната "Валута": цените на някои акции са в USD/SEK/... (основното им
      листване) - пиши нивата в тази валута.

    Използвай СТРИКТНО само данните по-долу.

    === ГОТОВИ ЗА ВХОД ===
    {ready_text}

    === WATCHLIST ===
    {watch_text}

    ЖЕЛЕЗНИ ПРАВИЛА:
    - Избирай ЕДИНСТВЕНО измежду инструментите по-долу.
    - Ако категория е празна, кажи го ясно.
    - За Watchlist ползвай колоната "Бележка" за конкретното условие, което чакаме.

    За "ГОТОВИ ЗА ВХОД": обясни фазата, Stop (колоната "Stop"), Target 1 на дневна
    съпротива, Target 2 на седмична съпротива.
    Бъди кратък, удобен за телефон.
    """
    yield from stream_claude(prompt, api_key, max_tokens=4096)


def render_photon_strategy():
    with st.expander("⚙️ Настройки на скрининга", expanded=False):
        st.markdown("**Ликвидност** - месечната селекция вече е филтрирана по тези прагове; тук може само да ги вдигнеш.")
        liq_cols = st.columns(2)
        with liq_cols[0]:
            stock_min_cap = st.select_slider(
                "Акции: мин. капитализация", options=[2_000_000_000, 5_000_000_000, 10_000_000_000, 20_000_000_000, 50_000_000_000],
                value=rules.STOCK_MIN_MARKET_CAP, format_func=format_eur, key="ph_stock_cap",
            )
            stock_min_turnover = st.select_slider(
                "Акции: мин. оборот/ден", options=[5_000_000, 10_000_000, 20_000_000, 50_000_000],
                value=rules.STOCK_MIN_TURNOVER, format_func=format_eur, key="ph_stock_turn",
                help="Само за акции на родна (ЕС/ЕИП) борса. US/CH/UK акции на Xetra/Gettex се гледат само по капитализация.",
            )
        with liq_cols[1]:
            etf_min_aum = st.select_slider(
                "ETF: мин. AUM", options=[100_000_000, 250_000_000, 500_000_000, 1_000_000_000, 5_000_000_000],
                value=rules.ETF_MIN_AUM, format_func=format_eur, key="ph_etf_aum",
                help="ETF-и без данни за AUM в Yahoo не отпадат - проверяват се само по оборот.",
            )
            etf_min_turnover = st.select_slider(
                "ETF: мин. оборот/ден", options=[250_000, 500_000, 1_000_000, 2_000_000, 5_000_000],
                value=rules.ETF_MIN_TURNOVER, format_func=format_eur, key="ph_etf_turn",
            )
        swing_order_weekly = st.slider("Чувствителност на седмичните swing точки", 1, 4, 2, key="ph_swo_w")
        swing_order_daily = st.slider("Чувствителност на дневните swing точки", 2, 6, 3, key="ph_swo_d")
        min_range_atr = st.slider(
            "Мин. ширина на дневния диапазон (x дневен ATR)", 1.0, 6.0, 3.0, step=0.5, key="ph_min_range",
            help="По-тесен диапазон е шум, не swing - тогава се търсят по-значими swing точки.",
        )
        closed_weeks_only = st.checkbox(
            "Седмичен тренд само по затворени седмици", value=True, key="ph_closed_w",
            help="Текущата незавършена седмица не участва в седмичните swing точки.",
        )
        choch_max_age = st.slider(
            "Свежест на CHoCH (макс. 4ч свещи назад)", 1, 6, 3, key="ph_choch_age",
            help="Phase B е 'готов' само ако пробивът над 4ч swing high е станал до толкова свещи назад.",
        )
        poi_atr_mult = st.slider(
            "Ширина на POI зоната (x ATR 4ч)", 0.5, 2.0, 1.0, step=0.25, key="ph_poi_atr",
            help="Phase A: цената трябва да е до толкова ATR над последния 4ч higher low.",
        )
        stop_atr_buffer = st.slider("Stop буфер под reference low (x ATR 4ч)", 0.0, 1.0, 0.5, step=0.1, key="ph_stop_buf")
        min_rr = st.slider("Минимален R/R (до дневна съпротива)", 1.0, 4.0, 2.0, step=0.5, key="ph_min_rr")
        exclude_leveraged = st.checkbox(
            "Изключи ливъриджнати/short ETP", value=True, key="ph_excl_lev",
            help="Short/Inverse/Leveraged/2x/3x продукти: short е залог надолу, а daily leveraged губят стойност при държане.",
        )
        uploaded_universe = render_universe_uploader(key="ph")
    _, macro_keywords = render_macro_section(key="ph", allow_autopin=False)

    if uploaded_universe:
        tickers = uploaded_universe
        st.caption(f"Универс (от качения фундаментален списък): {len(tickers)} инструмента")
    else:
        liquidity = (stock_min_cap, stock_min_turnover, etf_min_aum, etf_min_turnover)
        tickers = load_universe(
            max_instruments=None, liquidity=liquidity,
            curated_mtime=curated_file_mtime(),
        )
        st.caption(f"Универс: {len(tickers)} ликвидни инструмента")
    render_universe_refresh(key="ph")
    # филтърът за ливъриджнати е ПРЕДИ ръчния списък - ръчно добавеното винаги се сканира
    if exclude_leveraged:
        excluded = [n for n in tickers if rules.is_leveraged_or_short_etp(n)]
        tickers = {n: s for n, s in tickers.items() if n not in excluded}
        if excluded:
            st.caption(f"Изключени {len(excluded)} ливъриджнати/short ETP")
    manual_universe, scan_only_manual = render_manual_universe_editor(key="ph")
    tickers = apply_manual_universe(tickers, manual_universe, scan_only_manual)
    # ръчно добавени Gettex (.MU) акции -> основното им листване (ако е намерено)
    _, resolved_symbols = load_curated_symbol_info(curated_file_mtime())
    tickers = {n: resolved_symbols.get(s, s) for n, s in tickers.items()}
    render_universe_search(key="ph")

    if st.button("🔍 Сканирай пазара", type="primary", key="ph_scan_btn"):
        params = {
            "swing_order_weekly": swing_order_weekly, "swing_order_daily": swing_order_daily,
            "min_range_atr": min_range_atr, "closed_weeks_only": closed_weeks_only,
            "choch_max_age": choch_max_age, "poi_atr_mult": poi_atr_mult,
            "stop_atr_buffer": stop_atr_buffer, "min_rr": min_rr,
        }
        progress = st.progress(0.0, text="Търсене на Phase A/B сетъпи...")
        currencies, _ = load_curated_symbol_info(curated_file_mtime())
        results, watch_list, funnel, rejects = run_photon_scan(tickers, params, progress, currencies)
        progress.empty()
        st.session_state["photon_results"] = results
        st.session_state.pop("photon_ai_text", None)
        st.session_state["photon_watchlist"] = watch_list
        st.session_state["photon_funnel"] = (funnel, rejects, datetime.now().strftime("%d.%m %H:%M"))

    results = st.session_state.get("photon_results", [])
    watch_list = st.session_state.get("photon_watchlist", [])

    if "photon_funnel" in st.session_state:
        funnel, rejects, scanned_at = st.session_state["photon_funnel"]
        cols = st.columns(len(funnel))
        for col, (label, count) in zip(cols, funnel.items()):
            col.metric(label, count)
        if rejects:
            with st.expander(f"📉 Защо отпаднаха инструментите (скан от {scanned_at})"):
                st.dataframe(
                    pd.DataFrame(sorted(rejects.items(), key=lambda kv: -kv[1]), columns=["Причина", "Брой"]),
                    hide_index=True, width="stretch",
                )

    held = held_symbols()
    setups_by_name = {x.name: x for x in results + watch_list}

    st.divider()
    section_header("✅ Готови за вход", status="go", subtitle="Phase A (цена в POI) или Phase B (свеж 4ч CHoCH), в discount и с R/R над минимума")
    df_ready = render_setup_table(results, "ph_ready_table", held, macro_keywords, tickers)
    if df_ready.empty:
        st.info("Няма Phase A/B сетъпи с пълно потвърждение в момента.")

    st.divider()
    section_header("👀 Watchlist", status="watch", subtitle="Pro Swing потвърден - колоната 'Бележка' казва какво чакаме")
    df_watch = render_setup_table(watch_list, "ph_watch_table", held, macro_keywords, tickers)
    if df_watch.empty:
        st.info("Няма инструменти на watchlist в момента.")

    st.divider()
    section_header("🤖 AI Анализ", status="info")
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not anthropic_api_key:
        anthropic_api_key = st.text_input("Anthropic API Key", type="password", key="ph_key")
    if st.button("🚀 Генерирай Анализ и Търговски План", type="primary", width="stretch", key="ph_ai_btn"):
        if not anthropic_api_key:
            st.error("Липсва Anthropic API ключ!")
        else:
            try:
                st.session_state["photon_ai_text"] = st.write_stream(
                    generate_ai_analysis_photon(df_ready, df_watch, anthropic_api_key)
                )
            except Exception as e:
                st.error(f"Грешка: {e}")
    elif st.session_state.get("photon_ai_text"):
        # анализът остава видим и след клик по таблицата (всеки клик = rerun)
        st.markdown(st.session_state["photon_ai_text"])

    st.divider()
    section_header("📈 Преглед на графика", status="info")
    if tickers:
        selected_name = st.selectbox("Избери инструмент", list(tickers.keys()), key="ph_chart_select")
        render_photon_chart(tickers[selected_name], setups_by_name.get(selected_name), swing_order_daily, min_range_atr)


@st.cache_data(ttl=300, show_spinner=False)
def held_symbols():
    """{Yahoo символ: 'N' / 'T' / 'N+T'} - отворените позиции в двата T212
    акаунта (live, с ключовете от Streamlit Secrets). None, ако няма нито един
    конфигуриран акаунт. Кеш 5 мин. (T212 лимит: 1 заявка/5 сек. на акаунт)."""
    accounts = [a for a in T212_ACCOUNTS if st.secrets.get(a["key_secret_name"]) and st.secrets.get(a["secret_secret_name"])]
    if not accounts:
        return None
    t212_to_symbol = t212_ticker_to_symbol()
    held = {}
    for account in accounts:
        try:
            auth = t212.build_auth_header(st.secrets[account["key_secret_name"]], st.secrets[account["secret_secret_name"]])
            positions = t212.fetch_open_positions(t212.T212_ENV_TO_BASE_URL["live"], auth)
        except Exception:
            continue  # недостъпен акаунт не бива да чупи скрийнъра
        if positions.empty:
            continue
        for ticker in positions["Тикер"]:
            symbol = t212_to_symbol.get(ticker)
            if symbol is None and "_US_" in ticker:
                symbol = ticker.split("_US_")[0]  # US листване в T212 (напр. AAPL_US_EQ -> AAPL)
            if symbol:
                labels = held.setdefault(symbol, [])
                if account["label"] not in labels:
                    labels.append(account["label"])
    return {symbol: "+".join(labels) for symbol, labels in held.items()}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def t212_ticker_to_symbol() -> dict:
    """T212 тикер (напр. 'SAPd_EQ') -> Yahoo символът, който скрийнърът сканира
    (за Gettex - основното листване от curated файла)."""
    path = Path(INSTRUMENTS_FILE)
    if not path.exists():
        return {}
    _, resolved = load_curated_symbol_info(curated_file_mtime())
    mapping = {}
    for inst in json.loads(path.read_text(encoding="utf-8")).get("instruments", []):
        suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
        if suffix is None:
            continue
        yahoo_ticker = f"{inst.get('shortName', '')}{suffix}"
        mapping[inst["ticker"]] = resolved.get(yahoo_ticker, yahoo_ticker)
    return mapping


def render_setup_table(setups: list, key: str, held: dict, macro_keywords, tickers: dict) -> pd.DataFrame:
    """Таблица с PhotonSetup-и; клик по ред избира инструмента за графиката.
    held = {symbol: 'N'/'T'/'N+T'} или None (няма T212 ключове - колоната се скрива)."""
    if not setups:
        return pd.DataFrame()
    df = pd.DataFrame([x.to_row((held or {}).get(x.symbol, "")) for x in setups])
    if held is None:
        df = df.drop(columns=["💼 Държа"])
    df = flag_macro_signal(df, macro_keywords)
    st.caption("👆 Кликни върху ред, за да заредиш графиката му по-долу.")
    event = st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "📰 Медиен сигнал": st.column_config.CheckboxColumn("📰 Медиен сигнал"),
            "4ч CHoCH сега": st.column_config.CheckboxColumn("4ч CHoCH сега"),
            "💼 Държа": st.column_config.TextColumn("💼 Държа", help="Отворена позиция в T212: N / T (акаунт)"),
        },
        on_select="rerun", selection_mode="single-row", key=key,
    )
    rows = event.selection.rows if event and event.selection else []
    if rows and df.iloc[rows[0]]["Име"] in tickers:
        st.session_state["ph_chart_select"] = df.iloc[rows[0]]["Име"]
    return df


def swing_markers(fig, df_with_swings: pd.DataFrame, x_format: str = None):
    """Триъгълници на swing high (▼) и swing low (▲). x_format - за графики с
    категорийна ос (етикетите на свещите са низове, не дати)."""
    points = alternating_swings(df_with_swings)
    for kind, marker, color, label in (("H", "triangle-down", "#E85D5D", "Swing high"), ("L", "triangle-up", "#3DDC97", "Swing low")):
        pts = [pt for pt in points if pt[1] == kind]
        if pts:
            fig.add_trace(go.Scatter(
                x=[pt[0].strftime(x_format) if x_format else pt[0] for pt in pts], y=[pt[2] for pt in pts],
                mode="markers", name=label, marker=dict(symbol=marker, size=10, color=color),
            ))


def render_photon_chart(symbol: str, setup, swing_order_daily: int, min_range_atr: float):
    """Дневна графика (swing точки, подкрепа/съпротива/equilibrium, седмична
    съпротива, stop, POI) + 4ч графика (swing точки, CHoCH ниво, stop).
    Нивата на сетъпа се показват, ако инструментът е в резултатите от скана."""
    daily = fetch_ohlc_batch((symbol,), "2y", "1d").get(symbol)
    if daily is None or daily.empty:
        st.info("Няма данни за този инструмент.")
        return
    order = setup.daily_order if setup else swing_order_daily
    if setup:
        sup_lvl, res_lvl = setup.daily_support, setup.daily_resistance
        st.caption(f"{setup.phase} ({'Pro' if setup.phase == 'A' else 'Counter'} Internal) · {setup.zone} · {setup.note} · цените са в {setup.currency}")
    else:
        chart_s, found_order, _ = significant_daily_structure(daily, swing_order_daily, min_range_atr, average_true_range(daily, period=14))
        sup_lvl = chart_s["last_low"] if chart_s else None
        res_lvl = chart_s["last_high"] if chart_s else None
        order = found_order or swing_order_daily
        st.caption("Инструментът не е в резултатите от последния скан - показват се само дневните нива.")

    tab_daily, tab_4h = st.tabs(["Дневна", "4ч"])
    with tab_daily:
        fig = go.Figure(go.Candlestick(
            x=daily.index, open=daily["Open"], high=daily["High"], low=daily["Low"], close=daily["Close"], name="Цена",
        ))
        swing_markers(fig, find_swing_points(daily, order=order))
        if sup_lvl and res_lvl:
            fig.add_hline(y=res_lvl, line_dash="dot", line_color="#E85D5D", annotation_text="Дневна съпротива (цел)")
            fig.add_hline(y=sup_lvl + (res_lvl - sup_lvl) / 2, line_dash="dash", line_color="gray", annotation_text="Equilibrium (50%)")
            fig.add_hline(y=sup_lvl, line_dash="dot", line_color="#3DDC97", annotation_text="Дневна подкрепа")
        if setup:
            if setup.weekly_resistance > res_lvl:
                fig.add_hline(y=setup.weekly_resistance, line_dash="dot", line_color="#E8A23D", annotation_text="Седм. съпротива (цел 2)")
            fig.add_hline(y=setup.stop, line_color="#E85D5D", line_width=2, annotation_text="Stop")
            if setup.poi_low is not None:
                fig.add_hrect(y0=setup.poi_low, y1=setup.poi_high, fillcolor="#5B8DEF", opacity=0.18, line_width=0, annotation_text="POI")
        # по подразбиране последните ~9 месеца (цялата история е достъпна с zoom out)
        fig.update_xaxes(range=[daily.index[-1] - pd.Timedelta(days=270), daily.index[-1] + pd.Timedelta(days=5)])
        visible = daily[daily.index >= daily.index[-1] - pd.Timedelta(days=270)]
        fig.update_yaxes(range=[visible["Low"].min() * 0.97, visible["High"].max() * 1.03])
        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with tab_4h:
        intraday = fetch_ohlc_batch((symbol,), "60d", "60m").get(symbol)
        if intraday is None or intraday.empty:
            st.info("Няма 4ч данни за този инструмент.")
        else:
            h4 = resample_session_halves(intraday)
            # категорийна ос - без празни нощи/уикенди между свещите
            x_format = "%d.%m %H:%M"
            fig4 = go.Figure(go.Candlestick(
                x=[ts.strftime(x_format) for ts in h4.index],
                open=h4["Open"], high=h4["High"], low=h4["Low"], close=h4["Close"], name="4ч",
            ))
            swing_markers(fig4, find_swing_points(h4, order=1), x_format=x_format)
            if setup:
                fig4.add_hline(y=setup.choch_level, line_dash="dash", line_color="#E8A23D",
                               annotation_text="CHoCH ниво" + (" ✓" if setup.choch_now else ""))
                fig4.add_hline(y=setup.stop, line_color="#E85D5D", line_width=2, annotation_text="Stop")
                if setup.poi_low is not None:
                    fig4.add_hrect(y0=setup.poi_low, y1=setup.poi_high, fillcolor="#5B8DEF", opacity=0.18, line_width=0, annotation_text="POI")
            fig4.update_xaxes(type="category", nticks=12)
            fig4.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig4, width="stretch")
