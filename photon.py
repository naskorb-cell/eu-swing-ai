"""Photon Phases стратегията (SMC/MTF, Phase A/B, само long): анализ, скан, таблици, графики."""

import json
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import fundamentals as fund
import t212_portfolio as t212
import universe_rules as rules
from ai_client import AI_KEY_SECRETS, stream_ai
from indicators import (
    alternating_swings, average_true_range, detect_structure_events, drop_incomplete_week,
    find_swing_points, resample_ohlc, resample_session_halves, swing_structure,
)
from portfolio_ui import T212_ACCOUNTS
from ui_common import (
    ai_api_key, ai_provider, format_eur, friendly_ai_error, gemini_model, levels_html, section_header, show_ai_error,
)
from universe import (
    INSTRUMENTS_FILE, add_to_manual_universe, apply_manual_universe, curated_file_mtime, exchange_to_yahoo_suffix,
    load_curated_symbol_info, load_universe,
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
    Връща (готови, watchlist, фуния {етап: брой}, {причина за отпадане: брой},
    {symbol: статус}); готовите и watchlist-ът са списъци от PhotonSetup."""
    items = list(tickers.items())
    funnel = {"Сканирани": len(items)}
    rejects = {}

    statuses = {}

    def reject(reason, symbol):
        rejects[reason] = rejects.get(reason, 0) + 1
        statuses[symbol] = f"Отпадна: {reason}"

    progress.progress(0.0, text="Тегля дневни данни...")
    daily_data = fetch_ohlc_many(
        [s for _, s in items], "2y", "1d",
        on_progress=lambda done, total: progress.progress(0.5 * done / total, text=f"Дневни данни: {done}/{total}"),
    )

    survivors = []
    for name, symbol in items:
        ctx, why = analyze_photon_daily(daily_data.get(symbol), p)
        if ctx is None:
            reject(why, symbol)
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
            reject(why, symbol)
        else:
            setup.currency = (currencies or {}).get(symbol, "EUR")
            (results if setup.ready else watch_list).append(setup)
            statuses[symbol] = ("✅ Готов за вход" if setup.ready else "👀 Watchlist") + f" - {setup.note}"
    funnel["Watchlist"] = len(watch_list)
    funnel["Готови за вход"] = len(results)

    results.sort(key=lambda s: s.rr or 0, reverse=True)
    watch_list.sort(key=lambda s: s.range_pos)
    return results, watch_list, funnel, rejects, statuses


def generate_ai_analysis_photon(df_ready: pd.DataFrame, df_watch: pd.DataFrame, provider: str, api_key: str):

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
    - Колоните "📊 Фундамент" (консенсус на анализаторите, потенциал до целевата
      цена, ръст EPS), "📰 Новини" и "Отчет": фундаментално потвърждение.
      Давай предимство на сетъпите, потвърдени и от фундамента (✅/⭐), и
      предупреждавай при ⚠️ или отчет до дни (риск от гап).

    Използвай СТРИКТНО само данните по-долу.

    === ГОТОВИ ЗА ВХОД ===
    {ready_text}

    === WATCHLIST ===
    {watch_text}

    ЖЕЛЕЗНИ ПРАВИЛА:
    - Избирай ЕДИНСТВЕНО измежду инструментите по-долу.
    - Ако категория е празна, кажи го ясно.
    - За Watchlist ползвай колоната "Бележка" за конкретното условие, което чакаме.

    За "ГОТОВИ ЗА ВХОД": обясни фазата, входа с ЛИМИТ поръчка (колоните "Лимит вход"
    и "Вид поръчка"), Stop (колоната "Stop"), Target 1 на дневна съпротива, Target 2
    на седмична съпротива и ориентировъчната печалба/загуба (колоните с 💶).
    Бъди кратък, удобен за телефон.
    """
    yield from stream_ai(prompt, provider, api_key, max_tokens=4096, gemini_model=gemini_model())


# ============================================================================
# ИНТЕРФЕЙС
# Страницата: горе бутон „Сканирай“ + статус + фуния; отдолу табове с
# резултатите (Готови / Watchlist / Позиции / AI план) и таб „🛠 Универс и
# настройки“ с всичко служебно. Кодът попълва таба с настройките ПРЕДИ горната
# част (контейнерите в Streamlit се пълнят в произволен ред), защото сканът
# зависи от настройките и универса.
# ============================================================================

# Готови профили за сигналите; "Разширени" плъзгачите могат да ги променят ръчно
PROFILE_DEFAULTS = {
    "ph_min_rr": 2.0, "ph_choch_age": 3, "ph_poi_atr": 1.0, "ph_min_range": 3.0, "ph_stop_buf": 0.5,
    "ph_swo_w": 2, "ph_swo_d": 3,
}
PROFILES = {
    "🛡️ Консервативен": {**PROFILE_DEFAULTS, "ph_min_rr": 2.5, "ph_choch_age": 2, "ph_poi_atr": 0.75, "ph_min_range": 3.5},
    "⚖️ Стандартен": dict(PROFILE_DEFAULTS),
    "🚀 Агресивен": {**PROFILE_DEFAULTS, "ph_min_rr": 1.5, "ph_choch_age": 4, "ph_poi_atr": 1.5, "ph_min_range": 2.5,
                    "ph_stop_buf": 0.3},
}
PROFILE_HELP = {
    "🛡️ Консервативен": "по-малко, но по-чисти сетъпи: R/R ≥ 2.5, само съвсем свеж CHoCH, тесен POI",
    "⚖️ Стандартен": "балансът по подразбиране: R/R ≥ 2, CHoCH до 3 свещи, POI 1 x ATR",
    "🚀 Агресивен": "повече кандидати: R/R ≥ 1.5, CHoCH до 4 свещи, по-широк POI, по-тесни диапазони",
}

# Колони в таблиците: основните се виждат винаги, останалите - с „Още колони“
MAIN_COLUMNS = ["Име", "📊 Фундамент", "📰 Новини", "💼 Държа", "Цена", "Лимит вход", "Валута", "Зона",
                "Позиция в диапазона (%)", "R/R (до дневна съпротива)", "💶 Цел 1 / Цел 2", "Бележка"]
PHASE_BADGES = {"A": "🟦 A · Pro", "B": "🟪 B · Counter"}
ZONE_BADGES = {"Discount": "🟢 Discount", "Premium": "🟠 Premium"}
DEFAULT_INVESTMENT = 1000  # € за ориентировъчната печалба/загуба


def limit_entry(x):
    """Предложение за вход с лимит поръчка -> (цена, пояснение, лимит ли е).
    Phase A: лимит в POI зоната (горният ѝ край, или текущата цена, ако вече е вътре).
    Phase B с пробит CHoCH: лимит на ретест на пробитото ниво.
    Phase B без пробив: входът е ПОТВЪРЖДЕНИЕ над CHoCH - това е buy stop, не лимит."""
    if x.phase == "A" and x.poi_high is not None:
        entry, how, is_limit = min(x.price, x.poi_high), "лимит в POI зоната", True
    elif x.choch_now or x.price > x.choch_level:
        entry, how, is_limit = min(x.price, x.choch_level), "лимит на ретест на CHoCH", True
    else:
        entry, how, is_limit = x.choch_level, "buy stop над CHoCH (чака пробив)", False
    if entry <= x.stop:  # нивото е под stop-а - входът е на текущата цена
        entry, how, is_limit = x.price, "лимит на текущата цена", True
    return entry, how, is_limit


def investment_amount() -> float:
    return float(st.session_state.get("ph_invest", DEFAULT_INVESTMENT))


def pnl_eur(entry: float, level: float, amount: float) -> float:
    """Ориентировъчна печалба/загуба в € при amount € вход на entry (без такси и курсови разлики)."""
    return amount * (level / entry - 1) if entry else 0.0


def fmt_pnl(value: float) -> str:
    return f"{'+' if value >= 0 else '−'}{abs(value):,.0f} €".replace(",", " ")


def trade_plan(x) -> dict:
    """Лимит вход + ориентировъчната печалба до цел 1/2 и загубата до stop за сумата от настройките."""
    entry, how, is_limit = limit_entry(x)
    amount = investment_amount()
    risk = entry - x.stop
    return {
        "entry": entry, "how": how, "is_limit": is_limit, "amount": amount,
        "t1": pnl_eur(entry, x.daily_resistance, amount), "t2": pnl_eur(entry, x.weekly_resistance, amount),
        "stop": pnl_eur(entry, x.stop, amount),
        "rr": round((x.daily_resistance - entry) / risk, 2) if risk > 0 and x.daily_resistance > entry else None,
    }


def pnl_line_html(plan: dict) -> str:
    """Бледият ред „при 1000 €: цел 1 +78 € · цел 2 +120 € · stop −25 €“."""
    return (f'<div class="pnl-hint">при {plan["amount"]:,.0f} €: '.replace(",", " ")
            + f'цел 1 <b>{fmt_pnl(plan["t1"])}</b> · цел 2 <b>{fmt_pnl(plan["t2"])}</b> · stop {fmt_pnl(plan["stop"])}'
            + ' <span class="pnl-note">ориентировъчно, без такси и курсови разлики</span></div>')


def zone_badge(zone: str) -> str:
    return ZONE_BADGES.get(zone, f"🔵 {zone}")


def apply_profile():
    """Callback на избора на профил: слага стойностите му в плъзгачите."""
    profile = PROFILES.get(st.session_state.get("ph_profile"))
    if profile:
        st.session_state.update(profile)


def current_profile_label() -> str:
    selected = st.session_state.get("ph_profile", "⚖️ Стандартен")
    values = PROFILES.get(selected, {})
    changed = any(st.session_state.get(k, v) != v for k, v in values.items())
    return f"{selected} (с ръчни промени)" if changed else selected


def render_settings_tab():
    """Таб „🛠 Универс и настройки“: профил, разширени плъзгачи, филтри,
    ликвидност, CSV, ръчен списък, обновяване и търсене в универса.
    Връща (params, liquidity, filter flags, качения универс)."""
    for k, v in PROFILE_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("ph_profile", "⚖️ Стандартен")

    st.markdown("##### 🎯 Профил на сигналите")
    st.segmented_control(
        "Профил", list(PROFILES), key="ph_profile", on_change=apply_profile, label_visibility="collapsed",
    )
    st.caption(PROFILE_HELP.get(st.session_state.get("ph_profile"), "") or "Избери профил.")

    with st.expander("🔧 Разширени настройки на сигналите"):
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Минимален R/R (до дневна съпротива)", 1.0, 4.0, step=0.5, key="ph_min_rr")
            st.slider("Свежест на CHoCH (макс. 4ч свещи назад)", 1, 6, key="ph_choch_age",
                      help="Phase B е 'готов' само ако пробивът над 4ч swing high е станал до толкова свещи назад.")
            st.slider("Ширина на POI зоната (x ATR 4ч)", 0.5, 2.0, step=0.25, key="ph_poi_atr",
                      help="Phase A: цената трябва да е до толкова ATR над последния 4ч higher low.")
            st.slider("Stop буфер под reference low (x ATR 4ч)", 0.0, 1.0, step=0.1, key="ph_stop_buf")
        with c2:
            st.slider("Мин. ширина на дневния диапазон (x дневен ATR)", 1.0, 6.0, step=0.5, key="ph_min_range",
                      help="По-тесен диапазон е шум, не swing - тогава се търсят по-значими swing точки.")
            st.slider("Чувствителност на седмичните swing точки", 1, 4, key="ph_swo_w")
            st.slider("Чувствителност на дневните swing точки", 2, 6, key="ph_swo_d")
            st.checkbox("Седмичен тренд само по затворени седмици", value=True, key="ph_closed_w",
                        help="Текущата незавършена седмица не участва в седмичните swing точки.")

    st.markdown("##### 🧹 Филтри на универса")
    f1, f2, f3 = st.columns(3)
    exclude_leveraged = f1.toggle(
        "Без ливъриджнати/short ETP", value=True, key="ph_excl_lev",
        help="Short/Inverse/Leveraged/2x/3x продукти: short е залог надолу, а daily leveraged губят стойност при държане.",
    )
    exclude_cash_bond = f2.toggle(
        "Без парични/облигационни фондове", value=True, key="ph_excl_cash",
        help="Overnight/€ Cash/облигационни ETF-и почти не се движат и нямат swing структура.",
    )
    exclude_overseas = f3.toggle(
        "Само Европа + САЩ", value=True, key="ph_excl_overseas",
        help="Скрива акции с основна борса в Азия/Австралия, Канада и др. (.T, .HK, .AX, .TO...): в T212 се търгуват, "
             "когато основната им борса е затворена. Ръчно добавените се сканират винаги.",
    )

    st.number_input(
        "💶 Сума за ориентировъчната печалба (€)", min_value=100, max_value=1_000_000, value=DEFAULT_INVESTMENT,
        step=100, key="ph_invest", help="Колко би вложил в една сделка - за бледите сметки „цел 1 / цел 2 / stop“.",
    )

    with st.expander("💧 Ликвидност (месечната селекция вече е филтрирана - тук само вдигаш праговете)"):
        liq_cols = st.columns(2)
        with liq_cols[0]:
            stock_min_cap = st.select_slider(
                "Акции: мин. капитализация", options=[2_000_000_000, 5_000_000_000, 10_000_000_000, 20_000_000_000, 50_000_000_000],
                value=rules.STOCK_MIN_MARKET_CAP, format_func=format_eur, key="ph_stock_cap",
            )
            stock_min_turnover = st.select_slider(
                "Акции: мин. оборот/ден", options=[5_000_000, 10_000_000, 20_000_000, 50_000_000],
                value=rules.STOCK_MIN_TURNOVER, format_func=format_eur, key="ph_stock_turn",
                help="Само за акции на родна (ЕС/ЕИП) борса. Чуждите акции се гледат по капитализация.",
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

    with st.expander("📤 Качи списък от InvestingPro (CSV/Excel)"):
        uploaded_universe = render_universe_uploader(key="ph")

    params = {
        "swing_order_weekly": st.session_state["ph_swo_w"], "swing_order_daily": st.session_state["ph_swo_d"],
        "min_range_atr": st.session_state["ph_min_range"], "closed_weeks_only": st.session_state.get("ph_closed_w", True),
        "choch_max_age": st.session_state["ph_choch_age"], "poi_atr_mult": st.session_state["ph_poi_atr"],
        "stop_atr_buffer": st.session_state["ph_stop_buf"], "min_rr": st.session_state["ph_min_rr"],
    }
    liquidity = (stock_min_cap, stock_min_turnover, etf_min_aum, etf_min_turnover)
    return params, liquidity, (exclude_leveraged, exclude_cash_bond, exclude_overseas), uploaded_universe


def build_scan_universe(liquidity, flags, uploaded_universe):
    """Универсът за скана (в таб „Настройки“ се показват и броячите):
    curated/CSV -> филтри -> ръчен списък -> основни листвания.
    Връща (tickers, filtered_out {symbol: причина})."""
    exclude_leveraged, exclude_cash_bond, exclude_overseas = flags
    if uploaded_universe:
        curated = load_universe(max_instruments=None, liquidity=liquidity, curated_mtime=curated_file_mtime())
        curated_symbols = set(curated.values())
        new_items = {n: s for n, s in uploaded_universe.items() if s not in curated_symbols}
        st.caption(
            f"От файла {len(uploaded_universe)} са в Trading 212: {len(uploaded_universe) - len(new_items)} "
            f"вече са в универса, {len(new_items)} са нови."
        )
        csv_mode = st.radio(
            "Какво да сканирам с качения файл?", CSV_MODES, horizontal=True, key="ph_csv_mode",
            help="По подразбиране файлът само допълва месечния списък с липсващите в него инструменти.",
        )
        if csv_mode == CSV_MODES[0]:
            tickers = {**curated, **new_items}
        elif csv_mode == CSV_MODES[1]:
            _, _, curated_types = load_curated_symbol_info(curated_file_mtime())
            tickers = {**{n: s for n, s in curated.items() if curated_types.get(s) == "ETF"}, **uploaded_universe}
        else:
            tickers = dict(uploaded_universe)
        if new_items:
            with st.expander(f"Новите от файла ({len(new_items)})"):
                st.dataframe(pd.DataFrame({"Инструмент": list(new_items), "Символ": list(new_items.values())}),
                             hide_index=True, width="stretch")
                github_token = st.secrets.get("GITHUB_TOKEN", None)
                if st.button(f"💾 Запази новите {len(new_items)} трайно в ръчния списък", key="ph_csv_save",
                             help="Ще се сканират винаги, без да качваш файла отново."):
                    if not github_token:
                        st.error("Липсва GITHUB_TOKEN в Streamlit Secrets.")
                    else:
                        ok, msg = add_to_manual_universe(new_items, github_token)
                        (st.success if ok else st.error)(msg)
    else:
        tickers = load_universe(max_instruments=None, liquidity=liquidity, curated_mtime=curated_file_mtime())

    # филтрите са ПРЕДИ ръчния списък - ръчно добавеното винаги се сканира
    _, resolved_symbols, types = load_curated_symbol_info(curated_file_mtime())
    filtered_out = {}  # symbol -> причина (за "Моите позиции")
    filters = []
    if exclude_leveraged:
        filters.append((lambda n, s: rules.is_leveraged_or_short_etp(n), "Изключен: ливъриджнат/short ETP", "ливъриджнати/short"))
    if exclude_cash_bond:
        filters.append((lambda n, s: types.get(s) == "ETF" and rules.is_cash_or_bond_fund(n),
                        "Изключен: паричен/облигационен фонд", "парични/облигационни"))
    if exclude_overseas:
        filters.append((lambda n, s: rules.is_non_eu_us_listing(resolved_symbols.get(s, s)),
                        "Изключен: основна борса извън Европа/САЩ", "извън Европа/САЩ"))
    excluded_counts = []
    for check, reason, short in filters:
        excluded = {n: s for n, s in tickers.items() if check(n, s)}
        tickers = {n: s for n, s in tickers.items() if n not in excluded}
        filtered_out.update({s: reason for s in excluded.values()})
        if excluded:
            excluded_counts.append(f"{short}: {len(excluded)}")
    if excluded_counts:
        st.caption("Изключени от филтрите - " + " · ".join(excluded_counts))

    st.markdown("##### 🗂️ Универс")
    manual_universe, scan_only_manual = render_manual_universe_editor(key="ph")
    tickers = apply_manual_universe(tickers, manual_universe, scan_only_manual)
    # ръчно добавени Gettex/чужди листвания -> основното им листване (ако е намерено)
    tickers = {n: resolved_symbols.get(s, s) for n, s in tickers.items()}
    render_universe_refresh(key="ph")
    render_universe_search(key="ph")
    return tickers, filtered_out


def funnel_bar_html(funnel: dict) -> str:
    """Фунията като една лента: всеки етап с брой и % от сканираните."""
    total = max(next(iter(funnel.values()), 0), 1)
    colors = ["var(--info)", "var(--info)", "var(--watch)", "var(--go)"]
    cells = []
    for i, (label, count) in enumerate(funnel.items()):
        pct = 100 * count / total
        pct_txt = "" if i == 0 else f'<span class="pct">{pct:.1f}%</span>'
        cells.append(
            f'<div class="funnel-step" style="border-top-color:{colors[min(i, 3)]}">'
            f'<div class="n">{count}</div><div class="lbl">{label} {pct_txt}</div></div>'
        )
    arrow = '<div class="funnel-arrow">›</div>'
    return f'<div class="funnel">{arrow.join(cells)}</div>'


def render_status_box(tickers: dict, params: dict):
    """Горната част: голям бутон за скан, ред със статус и фунията."""
    currencies, _, types = load_curated_symbol_info(curated_file_mtime())
    scan_col, info_col = st.columns([1, 2], vertical_alignment="center")
    with scan_col:
        clicked = st.button("🔍 Сканирай пазара", type="primary", key="ph_scan_btn", width="stretch")
    with info_col:
        last = st.session_state.get("photon_funnel")
        last_txt = f"Последен скан: **{last[2]}**" if last else "Още няма скан в тази сесия"
        st.markdown(
            f"{last_txt} · Универс: **{len(tickers)}** инструмента · Профил: **{current_profile_label()}** · "
            f"AI: **{ai_provider()}**"
        )

    if clicked:
        progress = st.progress(0.0, text="Търсене на Phase A/B сетъпи...")
        results, watch_list, funnel, rejects, statuses = run_photon_scan(tickers, params, progress, currencies)
        progress.empty()
        st.session_state["photon_results"] = results
        st.session_state.pop("photon_ai_text", None)
        st.session_state["photon_watchlist"] = watch_list
        st.session_state["photon_funnel"] = (funnel, rejects, datetime.now().strftime("%d.%m %H:%M"))
        st.session_state["photon_statuses"] = statuses
        # ниво 1 на фундаменталното потвърждение: анализатори от Yahoo (без ETF-ите) -
        # за сетъпите от скана и за отворените позиции в T212
        held_now = [p["symbol"] for p in fetch_held_positions() or [] if p["symbol"]]
        stock_symbols = tuple(sorted({s for s in [x.symbol for x in results + watch_list] + held_now
                                      if types.get(s) != "ETF"}))
        with st.spinner(f"Тегля анализаторски данни за {len(stock_symbols)} акции..."):
            st.session_state["photon_fund"] = fund.fetch_analyst_data(stock_symbols)
        st.rerun()  # табовете горе показват броя си - прерисуваме с новите резултати

    if "photon_funnel" in st.session_state:
        funnel, rejects, scanned_at = st.session_state["photon_funnel"]
        st.markdown(funnel_bar_html(funnel), unsafe_allow_html=True)
        fund_data = st.session_state.get("photon_fund", {})
        with st.expander("📉 Детайли: защо отпаднаха инструментите, покритие на анализаторите"):
            if fund_data:
                render_fund_coverage(fund_data)
            if rejects:
                st.dataframe(
                    pd.DataFrame(sorted(rejects.items(), key=lambda kv: -kv[1]), columns=["Причина", "Брой"]),
                    hide_index=True, width="stretch",
                )


def render_photon_strategy():
    status_box = st.container()
    results = st.session_state.get("photon_results", [])
    watch_list = st.session_state.get("photon_watchlist", [])
    positions = fetch_held_positions()
    labels = [
        f"✅ Готови ({len(results)})", f"👀 Watchlist ({len(watch_list)})",
        "💼 Позиции" + (f" ({len(positions)})" if positions else ""), "🤖 AI план", "🛠 Универс и настройки",
    ]
    tab_ready, tab_watch, tab_pos, tab_ai, tab_settings = st.tabs(labels)

    with tab_settings:
        params, liquidity, flags, uploaded_universe = render_settings_tab()
        tickers, filtered_out = build_scan_universe(liquidity, flags, uploaded_universe)
        st.markdown("##### 📈 Графика на произволен инструмент")
        if tickers:
            selected_name = st.selectbox("Инструмент", list(tickers.keys()), key="ph_chart_select", index=None,
                                         placeholder="Избери или напиши име...")
            if selected_name and st.button("Отвори графиката", key="ph_chart_open"):
                open_instrument(selected_name, tickers[selected_name])

    with status_box:
        render_status_box(tickers, params)

    held = held_symbols(positions)
    fund_data = st.session_state.get("photon_fund", {})
    news = today_news()
    # по зона, после потвърдените от анализаторите/новините (виж sort_by_confirmation)
    results = sort_by_confirmation(results, fund_data, news)
    shown_order = sort_by_confirmation(watch_list, fund_data, {})
    st.session_state["ph_setups_by_name"] = {x.name: x for x in results + watch_list}
    st.session_state["ph_chart_params"] = (params["swing_order_daily"], params["min_range_atr"])
    scanned = "photon_funnel" in st.session_state

    with tab_ready:
        if not scanned:
            st.info("Натисни **🔍 Сканирай пазара** горе.")
        elif not results:
            st.info("Няма Phase A/B сетъпи с пълно потвърждение в момента - виж Watchlist.")
        else:
            st.caption("Phase A (цена в POI) или Phase B (свеж 4ч CHoCH), в discount и с R/R над минимума. "
                       "Зелено = потвърдено и от анализаторите.")
            if st.toggle("Табличен изглед", key="ph_ready_as_table"):
                render_setup_table(results, "ph_ready_table", held, fund_data, news)
            else:
                render_setup_cards(results, held, fund_data, news)
    df_ready = setups_dataframe(results, held, fund_data, news)  # за AI анализа

    with tab_watch:
        news_targets = list(results)
        if not scanned:
            st.info("Натисни **🔍 Сканирай пазара** горе.")
            df_watch = pd.DataFrame()
        else:
            st.caption("Pro Swing потвърден - колоната „Бележка“ казва какво чакаме. Кликни ред за графика и новини.")
            show_far = st.toggle(
                f"Покажи и далечните (над {FAR_ABOVE_RANGE_PCT}% от дневния диапазон)", value=False, key="ph_show_far",
                help="Над съпротивата = след пробив нагоре; до вход има нужда от нов пулбек, често дълъг.",
            )
            shown_watch = shown_order if show_far else [x for x in shown_order if x.range_pos <= FAR_ABOVE_RANGE_PCT]
            news_targets += shown_watch[:NEWS_WATCHLIST_TOP]  # ниво 2 - преди подреждането по новини
            shown_watch = sort_by_confirmation(shown_watch, fund_data, news)
            if len(shown_watch) < len(watch_list):
                st.caption(f"Скрити {len(watch_list) - len(shown_watch)} инструмента далеч над съпротивата.")
            df_watch = render_setup_table(shown_watch, "ph_watch_table", held, fund_data, news)
            if df_watch.empty:
                st.info("Няма инструменти на watchlist в момента.")
        if news_targets:
            st.divider()
            section_header("📰 Новини и анализи", status="info",
                           subtitle=f"Готовите за вход + първите {NEWS_WATCHLIST_TOP} от Watchlist")
            render_news_section([SimpleNamespace(name=x.name, symbol=x.symbol) for x in news_targets], news, key="ph_news")

    with tab_pos:
        if positions is None:
            st.info("Няма настроени T212 ключове в Secrets - позициите не могат да се заредят.")
        elif "photon_statuses" not in st.session_state:
            st.info("Пусни скан, за да видиш къде е всяка отворена позиция спрямо сигналите.")
        else:
            render_positions_status(positions, st.session_state["photon_statuses"], set(tickers.values()), filtered_out,
                                    fund_data, news)
            held_targets = [SimpleNamespace(name=p["name"], symbol=p["symbol"]) for p in positions if p["symbol"]]
            if held_targets:
                st.divider()
                section_header("📰 Новини и анализи за позициите ми", status="info")
                render_news_section(held_targets, news, key="ph_news_pos")

    with tab_ai:
        provider = ai_provider()
        api_key = ai_api_key(provider, key="ph_key")
        st.caption("Търговски план за готовите + какво да следиш в Watchlist, на база последния скан.")
        if st.button(f"🚀 Генерирай Анализ и Търговски План с {provider}", type="primary", width="stretch",
                     key="ph_ai_btn", disabled=not scanned):
            if not api_key:
                st.error(f"Липсва {AI_KEY_SECRETS[provider]} в Streamlit Secrets!")
            else:
                try:
                    st.session_state["photon_ai_text"] = st.write_stream(
                        generate_ai_analysis_photon(df_ready, df_watch, provider, api_key)
                    )
                except Exception as e:
                    show_ai_error(e, provider, key="ph_ai")
        elif st.session_state.get("photon_ai_text"):
            # анализът остава видим и след други кликове (всеки клик = rerun)
            st.markdown(st.session_state["photon_ai_text"])

    # изскачащият прозорец с графика/новини (от клик по ред, карта или новина)
    pending = st.session_state.pop("ph_dialog", None)
    if pending:
        instrument_dialog(*pending)


# ---------------------------------------------------------------- прозорец за инструмент

def open_instrument(name: str, symbol: str):
    """Отваря прозореца на инструмента при следващото прерисуване (в края на страницата)."""
    st.session_state["ph_dialog"] = (name, symbol)


def instrument_dialog(name: str, symbol: str):
    @st.dialog(name, width="large")
    def body():
        setup = st.session_state.get("ph_setups_by_name", {}).get(name)
        fund_data = st.session_state.get("photon_fund", {})
        news = today_news()
        if setup:
            plan = trade_plan(setup)
            st.markdown(levels_html([
                ("Цена сега", f"{setup.price:.2f} {setup.currency}"),
                ("Лимит вход" if plan["is_limit"] else "Buy stop", f"{plan['entry']:.2f}"), ("Stop", f"{setup.stop:.2f}"),
                ("Цел 1 (дневна)", f"{setup.daily_resistance:.2f}"), ("Цел 2 (седмична)", f"{setup.weekly_resistance:.2f}"),
                ("R/R", f"{plan['rr']:.2f}" if plan["rr"] else "—"),
            ]), unsafe_allow_html=True)
            st.markdown(pnl_line_html(plan), unsafe_allow_html=True)
            st.caption(f"Вход: {plan['how']}.")
        cols = fund.fundamental_columns(fund_data.get(symbol))
        facts = [cols["📊 Фундамент"]]
        if cols["Анализатори"]:
            facts.append(f"Анализатори: {cols['Анализатори']}")
        if cols["Потенциал до целта (%)"] is not None:
            facts.append(f"Потенциал до целта: {cols['Потенциал до целта (%)']:.1f}%")
        if cols["Отчет"]:
            facts.append(f"Отчет: {cols['Отчет']}")
        st.markdown(" · ".join(facts))
        tab_chart, tab_news = st.tabs(["📈 Графика", "📰 Новини"])
        with tab_chart:
            swing_order_daily, min_range_atr = st.session_state.get("ph_chart_params", (3, 3.0))
            render_photon_chart(symbol, setup, swing_order_daily, min_range_atr)
        with tab_news:
            r = news.get(symbol)
            if r and "error" not in r:
                render_news_item(r)
            else:
                if r:
                    st.warning(f"Предишната проверка беше неуспешна: {friendly_ai_error(r['error'], news_provider())}")
                provider = news_provider()
                if st.button(f"🔎 Провери новините с {provider}", key="ph_dialog_news"):
                    check_news([SimpleNamespace(name=name, symbol=symbol)], news)
                    r = today_news().get(symbol)
                    if r and "error" not in r:
                        render_news_item(r)
                    elif r:
                        st.error(friendly_ai_error(r["error"], news_provider()))
    body()


def render_news_item(r: dict):
    st.markdown(f"**{r['verdict']}** - {r['summary']}")
    if r.get("analyst_actions"):
        st.markdown(f"**Анализатори:** {r['analyst_actions']}")
    if r.get("next_earnings"):
        st.markdown(f"**Следващ отчет:** {r['next_earnings']}")
    if r.get("sources"):
        st.markdown("**Източници:** " + " · ".join(f"[{s.get('title') or s['url']}]({s['url']})" for s in r["sources"]))


# ---------------------------------------------------------------- карти за „Готови“

def render_setup_cards(setups: list, held: dict, fund_data: dict, news: dict):
    """Готовите за вход като карти (удобно на телефон): име, значки, нива, бутон."""
    per_row = 2
    for i in range(0, len(setups), per_row):
        cols = st.columns(per_row)
        for col, x in zip(cols, setups[i:i + per_row]):
            fcols = fund.fundamental_columns(fund_data.get(x.symbol))
            verdict = (news.get(x.symbol) or {}).get("verdict", "")
            fund_label = fcols["📊 Фундамент"]
            if fund_label == fund.FUND_CONFIRMED and verdict == fund.NEWS_POSITIVE:
                fund_label = STAR_LABEL
            confirmed = fund_label in (fund.FUND_CONFIRMED, STAR_LABEL) and verdict != fund.NEWS_NEGATIVE
            held_by = (held or {}).get(x.symbol)
            with col.container(border=True):
                st.markdown(
                    f'<div class="card-head{" confirmed" if confirmed else ""}">'
                    f'<span class="card-name">{x.name}</span><span class="card-ticker">{x.symbol}</span></div>',
                    unsafe_allow_html=True,
                )
                badges = [PHASE_BADGES.get(x.phase, x.phase), zone_badge(x.zone), fund_label]
                if verdict:
                    badges.append(verdict)
                if held_by:
                    badges.append(f"💼 {held_by}")
                if fcols["Отчет"].startswith("⚠️"):
                    badges.append(f"Отчет {fcols['Отчет']}")
                st.markdown(" ".join(f'<span class="badge">{b}</span>' for b in badges), unsafe_allow_html=True)
                plan = trade_plan(x)
                st.markdown(levels_html([
                    ("Лимит вход" if plan["is_limit"] else "Buy stop", f"{plan['entry']:.2f}"), ("Stop", f"{x.stop:.2f}"),
                    ("Цел 1", f"{x.daily_resistance:.2f}"), ("Цел 2", f"{x.weekly_resistance:.2f}"),
                    ("R/R", f"{plan['rr']:.2f}" if plan["rr"] else "—"),
                ]), unsafe_allow_html=True)
                st.markdown(pnl_line_html(plan), unsafe_allow_html=True)
                st.caption(f"{plan['how']} · сега {x.price:.2f} {x.currency} · {x.note}")
                if st.button("📈 Графика и новини", key=f"ph_card_{x.symbol}", width="stretch"):
                    open_instrument(x.name, x.symbol)


FAR_ABOVE_RANGE_PCT = 120
CSV_MODES = ["Допълни месечния списък с липсващите от файла", "Акциите от файла + ETF от месечния списък", "Само файла"]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_held_positions():
    """Отворените позиции в двата T212 акаунта (live, с ключовете от Streamlit
    Secrets): [{ticker, name, symbol (None, ако няма съвпадение), accounts}].
    None, ако няма нито един конфигуриран акаунт. Кеш 5 мин. (T212 лимит:
    1 заявка/5 сек. на акаунт)."""
    accounts = [a for a in T212_ACCOUNTS if st.secrets.get(a["key_secret_name"]) and st.secrets.get(a["secret_secret_name"])]
    if not accounts:
        return None
    info = t212_ticker_to_symbol()
    by_ticker = {}
    for account in accounts:
        try:
            auth = t212.build_auth_header(st.secrets[account["key_secret_name"]], st.secrets[account["secret_secret_name"]])
            open_positions = t212.fetch_open_positions(t212.T212_ENV_TO_BASE_URL["live"], auth)
        except Exception:
            continue  # недостъпен акаунт не бива да чупи скрийнъра
        if open_positions.empty:
            continue
        for ticker in open_positions["Тикер"]:
            entry = by_ticker.get(ticker)
            if entry is None:
                symbol, name = info.get(ticker, (None, ticker))
                if symbol is None and "_US_" in ticker:
                    symbol = ticker.split("_US_")[0]  # US листване в T212 (напр. AAPL_US_EQ -> AAPL)
                entry = by_ticker[ticker] = {"ticker": ticker, "name": name, "symbol": symbol, "accounts": []}
            if account["label"] not in entry["accounts"]:
                entry["accounts"].append(account["label"])
    return [{**e, "accounts": "+".join(e["accounts"])} for e in by_ticker.values()]


def held_symbols(positions):
    """{Yahoo символ: 'N' / 'T' / 'N+T'} за колоната "💼 Държа"; None без T212 ключове."""
    if positions is None:
        return None
    return {p["symbol"]: p["accounts"] for p in positions if p["symbol"]}


def render_positions_status(positions: list, statuses: dict, scanned_symbols: set, filtered_out: dict,
                            fund_data: dict = None, news: dict = None):
    """Таблица: всяка отворена позиция, къде е спрямо последния скан и какво
    казват анализаторите (ниво 1) и новините (ниво 2) - както за сетъпите."""
    fund_data, news = fund_data or {}, news or {}
    if not positions:
        st.info("Няма отворени позиции в T212.")
        return
    rows = []
    for p in positions:
        symbol = p["symbol"]
        if symbol is None:
            status = "⚠️ Няма съвпадение с Yahoo символ (не може да се сканира)"
        elif symbol in statuses:
            status = statuses[symbol]
        elif symbol in filtered_out:
            status = filtered_out[symbol]
        elif symbol in scanned_symbols:
            status = "Не е сканиран още - пусни скан"
        else:
            status = "Извън универса (не минава критериите за ликвидност) - добави го ръчно, за да се сканира"
        cols = fund.fundamental_columns(fund_data.get(symbol))
        rows.append({"Позиция": p["name"], "📊 Фундамент": cols["📊 Фундамент"],
                     "📰 Новини": (news.get(symbol) or {}).get("verdict", ""), "Статус": status,
                     "Анализатори": cols["Анализатори"], "Потенциал до целта (%)": cols["Потенциал до целта (%)"],
                     "Отчет": cols["Отчет"], "Акаунт": p["accounts"],
                     "T212 тикер": p["ticker"], "Сканиран символ": symbol or "-"})
    order = lambda r: (0 if r["Статус"].startswith("✅") else 1 if r["Статус"].startswith("👀") else 2, r["Позиция"])
    rows = sorted(rows, key=order)
    items = [(r["Позиция"], r["Сканиран символ"]) for r in rows]
    if any(sym != "-" for _, sym in items):
        st.caption("👆 Кликни позиция за графика, анализатори и новини.")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch", key="ph_pos_table", selection_mode="single-row",
        on_select=lambda: _open_selected("ph_pos_table", [it if it[1] != "-" else (None, None) for it in items]),
        column_config={
            "📊 Фундамент": st.column_config.TextColumn(
                help="⚠️ Против (Sell или цел под цената) при отворена позиция = повод да прегледаш stop-а"),
            "Потенциал до целта (%)": st.column_config.NumberColumn(format="%.1f"),
            "Отчет": st.column_config.TextColumn(help=f"⚠️ = до {fund.EARNINGS_WARN_DAYS} дни (риск от гап при задържане)"),
        },
    )


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def t212_ticker_to_symbol() -> dict:
    """T212 тикер (напр. 'SAPd_EQ') -> (Yahoo символът, който скрийнърът сканира
    (за Gettex - основното листване от curated файла), име)."""
    path = Path(INSTRUMENTS_FILE)
    if not path.exists():
        return {}
    _, resolved, _ = load_curated_symbol_info(curated_file_mtime())
    mapping = {}
    for inst in json.loads(path.read_text(encoding="utf-8")).get("instruments", []):
        suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
        if suffix is None:
            continue
        yahoo_ticker = f"{inst.get('shortName', '')}{suffix}"
        label = f"{inst.get('shortName', inst['ticker'])} ({inst['name']})"
        mapping[inst["ticker"]] = (resolved.get(yahoo_ticker, yahoo_ticker), label)
    return mapping


NEWS_WATCHLIST_TOP = 10
STAR_LABEL = "⭐ Структура + фундамент"


def news_provider() -> str:
    return ai_provider()  # общият превключвател „🤖 AI анализи чрез“ горе


def today_news() -> dict:
    """Резултатите от ниво 2 (новини) за днес от избрания доставчик - {symbol: резултат};
    от вчера се нулират."""
    stored = st.session_state.get("photon_news")
    if not stored or stored.get("date") != datetime.now().date().isoformat():
        return {}
    return stored["by_provider"].get(news_provider(), {})


ZONE_ORDER = {"Discount": 0, "Premium": 1}  # "Над съпротивата (BOS)" и др. - последни


def sort_by_confirmation(setups: list, fund_data: dict, news: dict) -> list:
    """Първо зоната (Discount преди Premium преди над съпротивата - техническата
    близост до вход е водеща), вътре в нея - потвърдените от фундамента/новините;
    при равенство остава техническият ред (сортирането е стабилно)."""
    return sorted(setups, key=lambda x: (
        ZONE_ORDER.get(x.zone, 2),
        -fund.confirmation_score(fund.fundamental_verdict(fund_data.get(x.symbol)),
                                 (news.get(x.symbol) or {}).get("verdict")),
    ))


def render_fund_coverage(fund_data: dict):
    """Колко акции имат анализаторски данни; бутон за дотегляне на неуспелите."""
    failed = fund.failed_symbols(fund_data)
    got = len(fund_data) - len(failed)
    covered = sum(1 for d in fund_data.values() if fund.fundamental_verdict(d) != fund.FUND_NO_DATA)
    st.caption(f"📊 Анализаторски данни от Yahoo: изтеглени за {got} от {len(fund_data)} акции, "
               f"{covered} с анализаторско покритие.")
    if not failed:
        return
    first_error = next(fund_data[s]["error"] for s in failed)
    st.warning(f"Yahoo не върна данни за {len(failed)} акции (най-често временно ограничение на заявките). "
               f"Пример: {first_error}")
    if st.button(f"🔄 Дотегли липсващите ({len(failed)})", key="ph_fund_retry"):
        with st.spinner("Дотеглям..."):
            st.session_state["photon_fund"] = {**fund_data, **fund.fetch_analyst_data(failed)}
        st.rerun()


def check_news(targets: list, news: dict) -> dict:
    """Проверява новините за targets (обекти с .name/.symbol) с избрания AI модел
    и записва резултатите за деня. Връща намереното."""
    provider = news_provider()
    api_key = st.secrets.get(AI_KEY_SECRETS[provider], None)
    if not api_key:
        st.error(f"Липсва {AI_KEY_SECRETS[provider]} в Streamlit Secrets.")
        return {}
    _, _, types = load_curated_symbol_info(curated_file_mtime())
    fund_data = st.session_state.get("photon_fund", {})

    def is_etf(symbol):
        return types.get(symbol) == "ETF" or (fund_data.get(symbol) or {}).get("quote_type") == "ETF"

    bar = st.progress(0.0, text=f"Търся новини и анализи за {len(targets)} инструмента - "
                                "обикновено 1-3 минути, всеки отнема 20-60 сек...")
    found = fund.research_news_many(
        [(x.name, x.symbol, is_etf(x.symbol)) for x in targets], provider, api_key, gemini_model=gemini_model(),
        on_done=lambda i, n: bar.progress(i / n, text=f"Проверени {i}/{n}"))
    bar.empty()
    today = datetime.now().date().isoformat()
    stored = st.session_state.get("photon_news")
    by_provider = stored["by_provider"] if stored and stored.get("date") == today else {}
    by_provider[provider] = {**news, **found}
    st.session_state["photon_news"] = {"date": today, "by_provider": by_provider}
    return found


def render_news_section(targets: list, news: dict, key: str):
    """Ниво 2: бутон за проверка с избрания AI модел + компактна таблица с
    резултатите; клик по ред отваря прозореца с подробностите и източниците.
    targets - обекти с .name и .symbol (сетъпи или позиции)."""
    provider = news_provider()
    missing = [x for x in targets if x.symbol not in news or "error" in news[x.symbol]]
    billing = "Google AI (Gemini API)" if provider == "Gemini" else "Anthropic API"
    st.caption(f"Чрез **{provider}** (сменя се с „🤖 AI анализи чрез“ горе) - всяка проверка търси в интернет "
               f"и се таксува в {billing}. Резултатите се пазят до края на деня; проверяват се само липсващите.")
    if st.button(f"🔎 Провери новини и анализи с {provider} ({len(missing)} инструмента)", key=f"{key}_btn",
                 disabled=not missing, width="stretch"):
        found = check_news(missing, news)
        if found:
            searches = sum(r.get("searches", 0) for r in found.values())
            errors = sum(1 for r in found.values() if "error" in r)
            st.toast(f"Готово: {len(found) - errors} проверени, {searches} web търсения"
                     + (f", {errors} с грешка (натисни пак)" if errors else ""))
            st.rerun()  # таблиците се подреждат наново с новите оценки
    rows, symbols = [], []
    for x in targets:
        r = news.get(x.symbol)
        if not r:
            continue
        failed = "error" in r
        rows.append({"Инструмент": x.name, "Оценка": "⚠️ Грешка" if failed else r["verdict"],
                     "Накратко": friendly_ai_error(r["error"], provider) if failed else r["summary"]})
        symbols.append((x.name, x.symbol))
    if rows:
        st.caption("👆 Кликни ред за подробностите, анализаторите и източниците.")
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch", key=f"{key}_table",
            column_config={"Накратко": st.column_config.TextColumn(width="large")},
            on_select=lambda k=f"{key}_table", s=symbols: _open_selected(k, s), selection_mode="single-row",
        )


def _open_selected(table_key: str, items: list):
    """Callback на избор на ред в таблица: отваря прозореца на инструмента."""
    event = st.session_state.get(table_key) or {}
    rows = (event.get("selection") or {}).get("rows") or []
    if rows and rows[0] < len(items) and items[rows[0]][1]:
        open_instrument(*items[rows[0]])


def setups_dataframe(setups: list, held: dict, fund_data: dict, news: dict) -> pd.DataFrame:
    """PhotonSetup-и като таблица с фундамента и новините (и за AI анализа)."""
    rows = []
    for x in setups:
        row = x.to_row((held or {}).get(x.symbol, ""))
        row["Фаза"] = PHASE_BADGES.get(x.phase, row["Фаза"])
        row["Зона"] = zone_badge(x.zone)
        cols = fund.fundamental_columns(fund_data.get(x.symbol))
        verdict = (news.get(x.symbol) or {}).get("verdict", "")
        if cols["📊 Фундамент"] == fund.FUND_CONFIRMED and verdict == fund.NEWS_POSITIVE:
            cols["📊 Фундамент"] = STAR_LABEL
        plan = trade_plan(x)
        row["Лимит вход"] = plan["entry"]
        row["Вид поръчка"] = plan["how"]
        row["R/R (до дневна съпротива)"] = plan["rr"]  # от лимит цената, не от текущата
        row["💶 Цел 1 / Цел 2"] = f"{fmt_pnl(plan['t1'])} / {fmt_pnl(plan['t2'])}"
        row["💶 Stop"] = fmt_pnl(plan["stop"])
        rows.append({**row, **cols, "📰 Новини": verdict})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if held is None:
        df = df.drop(columns=["💼 Държа"])
    front = [c for c in MAIN_COLUMNS if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


NUMBER_FORMATS = {
    "Цена": "{:.2f}", "Лимит вход": "{:.2f}", "Позиция в диапазона (%)": "{:.1f}", "Дневна подкрепа": "{:.2f}", "Дневна съпротива": "{:.2f}",
    "Stop": "{:.2f}", "Ширина (x ATR)": "{:.1f}", "R/R (до дневна съпротива)": "{:.2f}",
    "R/R (до седм. съпротива)": "{:.2f}", "Потенциал до целта (%)": "{:.1f}", "Ръст EPS (%)": "{:.1f}",
}


def render_setup_table(setups: list, key: str, held: dict, fund_data: dict, news: dict) -> pd.DataFrame:
    """Таблица с PhotonSetup-и; клик по ред отваря прозореца с графиката и новините.
    held = {symbol: 'N'/'T'/'N+T'} или None (няма T212 ключове - колоната се скрива).
    Колоните за фундамента (ниво 1) и новините (ниво 2) само подчертават - не филтрират."""
    df = setups_dataframe(setups, held, fund_data, news)
    if df.empty:
        return df
    show_all = st.toggle("Още колони", key=f"{key}_all_cols")
    order = list(df.columns) if show_all else [c for c in MAIN_COLUMNS if c in df.columns]

    def highlight(r):
        confirmed = r["📊 Фундамент"] in (fund.FUND_CONFIRMED, STAR_LABEL) and r["📰 Новини"] != fund.NEWS_NEGATIVE
        return ["background-color: rgba(61, 220, 151, 0.14)" if confirmed else ""] * len(r)

    items = [(x.name, x.symbol) for x in setups]
    st.dataframe(
        df.style.apply(highlight, axis=1).format({c: f for c, f in NUMBER_FORMATS.items() if c in df.columns}, na_rep="—"),
        width="stretch", hide_index=True, column_order=order,
        column_config={
            "Име": st.column_config.TextColumn("Име", pinned=True),
            "📊 Фундамент": st.column_config.TextColumn(
                "📊 Фундамент", help=f"✅ Buy/Strong Buy от поне {fund.MIN_ANALYSTS} анализатори и потенциал ≥ "
                                    f"{fund.MIN_UPSIDE_PCT}% до средната целева цена; ⚠️ Sell или цел под цената; "
                                    "⭐ = потвърден + положителни новини"),
            "📰 Новини": st.column_config.TextColumn("📰 Новини", help="От бутона „Провери новини и анализи“"),
            "Позиция в диапазона (%)": st.column_config.Column(
                "Позиция %", help="0% = дневна подкрепа, 50% = equilibrium, 100% = съпротива"),
            "R/R (до дневна съпротива)": st.column_config.Column("R/R", help="До дневната съпротива (цел 1)"),
            "Бележка": st.column_config.TextColumn("Бележка", width="large"),
            "Ръст EPS (%)": st.column_config.Column(help="Прогнозна спрямо последната годишна печалба на акция"),
            "Отчет": st.column_config.TextColumn(help=f"Следващ отчет; ⚠️ = до {fund.EARNINGS_WARN_DAYS} дни (риск от гап)"),
            "4ч CHoCH сега": st.column_config.CheckboxColumn("4ч CHoCH сега"),
            "💼 Държа": st.column_config.TextColumn("💼", help="Отворена позиция в T212: N / T (акаунт)"),
        },
        on_select=lambda: _open_selected(key, items), selection_mode="single-row", key=key,
    )
    return df


# ---------------------------------------------------------------- графики

CHART_UP, CHART_DOWN = "#3DDC97", "#E85D5D"
CHART_SURFACE, CHART_GRID, CHART_INK, CHART_MUTED = "#0A1628", "#16284A", "#E6EDF3", "#8497B0"
CHART_FAINT = "#5C6F8A"  # бледите бележки (ориентировъчна печалба) до нивата
LEVEL_STYLES = {  # (цвят, тип линия, дебелина)
    "Цел 2 · седм. съпротива": ("#E8A23D", "dot", 1),
    "Цел 1 · дневна съпротива": ("#E85D5D", "dot", 1),
    "Equilibrium 50%": ("#7C8B99", "dash", 1),
    "Дневна подкрепа": ("#3DDC97", "dot", 1),
    "CHoCH ниво": ("#E8A23D", "dash", 1),
    "Stop": ("#E85D5D", "solid", 1.5),
    "Лимит вход": ("#5B8DEF", "dash", 1.2),
    "Buy stop": ("#5B8DEF", "dash", 1.2),
}


def swing_markers(fig, df_with_swings: pd.DataFrame, x_values=None):
    """Малки триъгълници на swing high (▼ над свещта) и swing low (▲ под нея).
    x_values - етикетите на свещите при категорийна ос (4ч)."""
    points = alternating_swings(df_with_swings)
    pos = {ts: i for i, ts in enumerate(df_with_swings.index)}
    pad = (df_with_swings["High"].max() - df_with_swings["Low"].min()) * 0.012
    for kind, marker, color, label in (("H", "triangle-down", CHART_DOWN, "Swing high"), ("L", "triangle-up", CHART_UP, "Swing low")):
        pts = [pt for pt in points if pt[1] == kind]
        if pts:
            fig.add_trace(go.Scatter(
                x=[x_values[pos[pt[0]]] if x_values is not None else pt[0] for pt in pts],
                y=[pt[2] + pad if kind == "H" else pt[2] - pad for pt in pts],
                mode="markers", name=label, showlegend=False, hovertemplate=f"{label}: %{{customdata:.2f}}<extra></extra>",
                customdata=[pt[2] for pt in pts], marker=dict(symbol=marker, size=7, color=color, opacity=0.85),
            ))


def candle_figure(df: pd.DataFrame, levels: list, poi=None, visible_bars: int = 130, categorical: bool = False,
                  swings_order: int = None, height: int = 520):
    """Свещи + нива с етикети в дясното поле (не върху свещите), тънки линии,
    приглушена мрежа, кръстосан курсор. levels = [(етикет, цена)]; poi = (от, до).
    По подразбиране се виждат последните visible_bars свещи (zoom out = цялата история)."""
    df = df.dropna(subset=["Close"])
    x = [ts.strftime("%d.%m %H:%M") for ts in df.index] if categorical else df.index
    fig = go.Figure(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Цена", showlegend=False,
        increasing=dict(line=dict(color=CHART_UP, width=1), fillcolor=CHART_UP),
        decreasing=dict(line=dict(color=CHART_DOWN, width=1), fillcolor=CHART_DOWN),
        whiskerwidth=0,
    ))
    if swings_order:
        swing_markers(fig, find_swing_points(df, order=swings_order), x_values=x if categorical else None)

    visible = df.iloc[-visible_bars:]
    last_close = float(df["Close"].iloc[-1])
    # ниво = (име, цена) или (име, цена, бледа бележка, напр. „+78 €“)
    levels = [lv if len(lv) == 3 else (lv[0], lv[1], "") for lv in levels]
    lo = min([visible["Low"].min()] + [v for _, v, _ in levels if v] + ([poi[0]] if poi else []))
    hi = max([visible["High"].max()] + [v for _, v, _ in levels if v] + ([poi[1]] if poi else []))
    pad = (hi - lo) * 0.06
    y0, y1 = lo - pad, hi + pad

    if poi:
        fig.add_hrect(y0=poi[0], y1=poi[1], fillcolor="#5B8DEF", opacity=0.15, line_width=0)
        levels = levels + [("POI", (poi[0] + poi[1]) / 2, "")]
    # етикетите в дясното поле, разтворени по вертикала, за да не се застъпват
    labels = [lv for lv in levels if lv[1]] + [("Цена", last_close, "")]
    labels.sort(key=lambda lv: lv[1])
    min_gap = (y1 - y0) * 0.045
    placed = []
    for name, value, note in labels:
        y_label = max(value, placed[-1][2] + min_gap) if placed else value
        placed.append((name, value, y_label, note))
    overflow = placed[-1][2] - (y1 - min_gap / 2) if placed else 0
    if overflow > 0:  # най-горните излизат над графиката - сваляме всички малко надолу
        placed = [(n, v, yl - overflow, nt) for n, v, yl, nt in placed]
    for name, value, y_label, note in placed:
        color, dash, width = LEVEL_STYLES.get(name, ("#5B8DEF", "dot", 1))
        if name == "Цена":
            color = CHART_INK
        elif name != "POI":
            fig.add_hline(y=value, line_dash=dash, line_color=color, line_width=width, opacity=0.9)
        fig.add_annotation(
            xref="paper", x=1.0, xanchor="left", yref="y", y=y_label, showarrow=False, align="left",
            text=f"<b>{value:,.2f}</b> {name}" + (f" <span style='color:{CHART_FAINT}'>{note}</span>" if note else ""),
            font=dict(size=11, color=CHART_SURFACE if name == "Цена" else color),
            bgcolor=CHART_INK if name == "Цена" else CHART_SURFACE, borderpad=2,
        )

    if categorical:
        fig.update_xaxes(type="category", range=[max(len(df) - visible_bars, 0) - 0.5, len(df) + 1.5], nticks=10)
    else:
        span = df.index[-1] - df.index[-min(visible_bars, len(df))]
        fig.update_xaxes(range=[df.index[-min(visible_bars, len(df))], df.index[-1] + span * 0.03])
    fig.update_yaxes(range=[y0, y1], side="left", tickformat=",.2f")
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor=CHART_MUTED, spikedash="dot")
    fig.update_yaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor=CHART_MUTED, spikedash="dot")
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor=CHART_SURFACE, plot_bgcolor=CHART_SURFACE,
        xaxis_rangeslider_visible=False, hovermode="x", dragmode="pan",
        margin=dict(l=8, r=215, t=10, b=10), font=dict(size=11, color=CHART_MUTED),
        xaxis=dict(gridcolor=CHART_GRID, zeroline=False), yaxis=dict(gridcolor=CHART_GRID, zeroline=False),
        hoverlabel=dict(bgcolor="#11213A", font_size=12),
    )
    return fig


CHART_CONFIG = {"displaylogo": False, "scrollZoom": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"]}
CHART_HEIGHTS = {"S": 420, "M": 560, "L": 720, "XL": 900}
CHART_HINT = ("↕ влачи **края** на ценовата скала (горе/долу) = разтягане вертикално · ↔ влачи края на времевата "
              "ос = разтягане хоризонтално · влачи в графиката = местене · колелото = zoom · двоен клик = връщане · "
              "⛶ горе вдясно на графиката = цял екран")


def render_photon_chart(symbol: str, setup, swing_order_daily: int, min_range_atr: float):
    """Три графики в табове - седмична (тренд и цел 2), дневна (диапазон,
    подкрепа/съпротива, equilibrium, stop, POI) и 4ч (CHoCH ниво, stop, POI).
    Нивата на сетъпа се показват, ако инструментът е в резултатите от скана."""
    daily = fetch_ohlc_batch((symbol,), "2y", "1d").get(symbol)
    if daily is None or daily.empty:
        st.info("Няма данни за този инструмент.")
        return
    order = setup.daily_order if setup else swing_order_daily
    if setup:
        sup_lvl, res_lvl = setup.daily_support, setup.daily_resistance
        st.caption(f"{PHASE_BADGES.get(setup.phase, setup.phase)} · {zone_badge(setup.zone)} · {setup.note} · "
                   f"цените са в {setup.currency}")
    else:
        chart_s, found_order, _ = significant_daily_structure(daily, swing_order_daily, min_range_atr, average_true_range(daily, period=14))
        sup_lvl = chart_s["last_low"] if chart_s else None
        res_lvl = chart_s["last_high"] if chart_s else None
        order = found_order or swing_order_daily
        st.caption("Инструментът не е в резултатите от последния скан - показват се само дневните нива.")

    plan = trade_plan(setup) if setup else None
    stop = [("Stop", setup.stop, fmt_pnl(plan["stop"]))] if setup else []
    entry = [("Лимит вход" if plan["is_limit"] else "Buy stop", plan["entry"])] if setup else []
    poi = (setup.poi_low, setup.poi_high) if setup and setup.poi_low is not None else None
    target2 = ([("Цел 2 · седм. съпротива", setup.weekly_resistance, fmt_pnl(plan["t2"]))]
               if setup and setup.weekly_resistance > (res_lvl or 0) else [])
    target1_note = fmt_pnl(plan["t1"]) if setup else ""
    if setup:
        st.markdown(pnl_line_html(plan), unsafe_allow_html=True)

    size_col, hint_col = st.columns([1, 4], vertical_alignment="center")
    with size_col:
        st.session_state.setdefault("ph_chart_h", "M")
        st.segmented_control("Височина", list(CHART_HEIGHTS), key="ph_chart_h", label_visibility="collapsed",
                             help="Височина на графиката: S / M / L / XL")
    with hint_col:
        st.caption(CHART_HINT)
    height = CHART_HEIGHTS.get(st.session_state.get("ph_chart_h") or "M", 560)

    tab_w, tab_d, tab_4h = st.tabs(["📅 Седмична", "📆 Дневна", "⏱️ 4ч"], default="📆 Дневна")
    with tab_w:
        weekly = resample_ohlc(daily, "W")
        levels = target2 + ([("Цел 1 · дневна съпротива", res_lvl, target1_note), ("Дневна подкрепа", sup_lvl)]
                            if sup_lvl and res_lvl else []) + stop + entry
        fig = candle_figure(weekly, levels, visible_bars=104, swings_order=st.session_state.get("ph_swo_w", 2), height=height)
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=f"chart_w_{symbol}")
        st.caption("Тренд (HH + HL на седмичните swing точки) и цел 2 - седмичната съпротива.")
    with tab_d:
        levels = list(target2)
        if sup_lvl and res_lvl:
            levels += [("Цел 1 · дневна съпротива", res_lvl, target1_note),
                       ("Equilibrium 50%", sup_lvl + (res_lvl - sup_lvl) / 2), ("Дневна подкрепа", sup_lvl)]
        fig = candle_figure(daily, levels + stop + entry, poi=poi, visible_bars=130, swings_order=order, height=height)
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=f"chart_d_{symbol}")
    with tab_4h:
        intraday = fetch_ohlc_batch((symbol,), "60d", "60m").get(symbol)
        if intraday is None or intraday.empty:
            st.info("Няма 4ч данни за този инструмент.")
        else:
            h4 = resample_session_halves(intraday)
            levels = ([("CHoCH ниво", setup.choch_level)] if setup else []) + stop + entry
            fig = candle_figure(h4, levels, poi=poi, visible_bars=60, categorical=True, swings_order=1, height=height)
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=f"chart_4h_{symbol}")
            if setup:
                st.caption("Phase B: вход при затваряне над CHoCH нивото" + (" - ✓ вече пробито" if setup.choch_now else "")
                           + " · Phase A: вход в POI зоната.")
