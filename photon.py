"""Photon Phases стратегията (SMC/MTF, Phase A/B, само long): анализ, скан, таблици, графики."""

import dataclasses
import json
import threading
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
    alternating_swings, average_true_range, bos_marks, detect_structure_events, drop_incomplete_week, ema_alignment,
    find_swing_points, pullback_choch, resample_ohlc, resample_session_halves, swing_structure,
)
from portfolio_ui import T212_ACCOUNTS
from ui_common import (
    ai_api_key, ai_provider, claude_news_model, format_eur, friendly_ai_error, gemini_model, gemini_news_model,
    levels_html, section_header,
    show_ai_error,
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
REJECT_WEEKLY = "Седмичният тренд не е бичи (няма BOS нагоре / под силното дъно)"
REJECT_DAILY = "Дневният тренд не е бичи (няма BOS нагоре / под силното дъно)"
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
        if s["weak_high"] - s["strong_low"] >= min_range and s["weak_high"] > s["strong_low"]:
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
    if weekly_s is None or weekly_s["trend"] != "up":
        return None, REJECT_WEEKLY

    # --- Swing/MTF (Дневен): Pro Swing (нагоре) със значим диапазон - Phase C/D извън обхват ---
    atr_daily = average_true_range(daily_df, period=14)
    daily_s, used_order, why = significant_daily_structure(daily_df, p["swing_order_daily"], p["min_range_atr"], atr_daily)
    if daily_s is None:
        return None, why
    if daily_s["trend"] != "up":
        return None, REJECT_DAILY

    current_price = float(daily_df["Close"].iloc[-1])
    # Photon: диапазонът е от силното дъно (довело до последния BOS) до слабия връх (целта).
    # Затваряне под силното дъно = дневната структура е счупена (не е discount)
    if current_price < daily_s["strong_low"]:
        return None, REJECT_BROKEN

    return {
        "current_price": current_price, "atr_daily": atr_daily, "daily_order": used_order,
        "daily_support": daily_s["strong_low"], "daily_resistance": daily_s["weak_high"],
        "pullback_start": daily_s["weak_high_idx"],  # дневният пулбек започва от слабия връх
        "ema_aligned": (ema_alignment(daily_df["Close"]) or (None,))[0],
        "atr_pct": round(100 * atr_daily / current_price, 2) if atr_daily and current_price else None,
        "weekly_resistance": weekly_s["weak_high"],
        "leg_speed": up_leg_speeds(daily_df, used_order),
    }, None


def up_leg_speeds(daily_df: pd.DataFrame, order: int, legs: int = 8):
    """Колко бързо акцията обикновено изминава възходящите си дневни swing-ове:
    (бърз, типичен, бавен) ръст на цената за една свещ - 75-и, 50-и и 25-и
    перцентил от последните `legs` хода swing low -> swing high. None при < 2 хода."""
    points = alternating_swings(find_swing_points(daily_df, order=order))
    pos = {ts: i for i, ts in enumerate(daily_df.index)}
    speeds = [
        (hi[2] - lo[2]) / (pos[hi[0]] - pos[lo[0]])
        for lo, hi in zip(points, points[1:])
        if lo[1] == "L" and hi[1] == "H" and pos[hi[0]] > pos[lo[0]] and hi[2] > lo[2]
    ][-legs:]
    if len(speeds) < 2:
        return None
    s = pd.Series(speeds)
    return float(s.quantile(0.75)), float(s.median()), float(s.quantile(0.25))


def ema_label(aligned) -> str:
    return "✓" if aligned else ("✗" if aligned is False else "—")


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
    leg_speed: tuple | None = None  # (бърз, типичен, бавен) ръст/свещ на възходящите дневни swing-ове
    choch_tf: str = "4ч"            # рамката на CHoCH-а („1ч“ при включено по-ранно потвърждение)
    pullback_start: object = None   # датата на дневния слаб връх - началото на пулбека
    ema_aligned: bool | None = None # EMA20 > EMA50 > EMA200 на дневната графика (Full Trend Alignment)
    atr_pct: float | None = None    # дневен ATR(14) като % от цената - волатилността

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
            "📐 EMA": ema_label(getattr(self, "ema_aligned", None)), "🌊 ATR %": getattr(self, "atr_pct", None),
        }


CHOCH_1H_BARS_PER_4H = 4  # ~4 часови свещи в една „4ч“ (половин сесия)


def analyze_photon_intraday(name: str, symbol: str, ctx: dict, intraday: pd.DataFrame, p: dict):
    """Стъпка 2: 4ч Internal структура, фаза, stop и R/R.
    Връща (PhotonSetup, None) или (None, причина за отпадане)."""
    events, atr_4h, choch_1h = None, None, None
    if intraday is not None and not intraday.empty:
        h4 = resample_session_halves(intraday)
        if len(h4) >= 10:
            events = detect_structure_events(find_swing_points(h4, order=1), p["choch_max_age"],
                                             pullback_start=ctx.get("pullback_start"))
            atr_4h = average_true_range(h4, period=14)
        if p.get("choch_1h") and len(intraday) >= 30:
            # по-ранно потвърждение на по-малката рамка: 1ч CHoCH над силния 1ч lower high
            choch_1h = pullback_choch(find_swing_points(intraday, order=2), ctx.get("pullback_start"))
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
    choch_tf = "4ч"
    if phase == "B" and not events["choch_bullish_now"] and choch_1h and choch_1h["choch_bars_ago"] is not None \
            and choch_1h["choch_bars_ago"] < p["choch_max_age"] * CHOCH_1H_BARS_PER_4H:
        # 4ч CHoCH още няма, но 1ч вече е обърнал - по-ранен вход с по-ниско CHoCH ниво
        choch_tf = "1ч"
        events = {**events, "choch_bullish_now": True, "choch_level": choch_1h["choch_level"],
                  "choch_bars_ago": choch_1h["choch_bars_ago"], "reference_low": choch_1h["pullback_low"]}
        setup_ok = in_discount

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
        note = "Вход на POI" if phase == "A" else f"{choch_tf} CHoCH преди {events['choch_bars_ago']} свещи"
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
        poi_low=poi_low, poi_high=poi_high, daily_order=ctx["daily_order"], leg_speed=ctx.get("leg_speed"),
        choch_tf=choch_tf, pullback_start=ctx.get("pullback_start"),
        ema_aligned=ctx.get("ema_aligned"), atr_pct=ctx.get("atr_pct"),
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
    - Седмичен тренд трябва да е бичи (последният BOS е нагоре и цената е над
      силното дъно) - твърд филтър.
    - Дневен Swing тренд трябва също да е бичи - твърд филтър.
      (Counter Swing фази C/D са изключени - твърде агресивни за системата.)
    - Дневният диапазон е от СИЛНОТО дъно (довело до последния BOS) до СЛАБИЯ
      връх (целта, „Дневна съпротива“); вътрешните дъна/върхове в пулбека не го местят.
    - Phase A (Pro Swing + Pro Internal): 4ч структурата е HH+HL - влизаме
      на POI (зоната над силното 4ч дъно), без да чакаме CHoCH.
    - Phase B (Pro Swing + Counter Internal): 4ч е в пулбек (lower high/low) -
      влизаме на СВЕЖ CHoCH: първо затваряне над силния 4ч lower high (върхът
      преди най-ниското дъно на пулбека). По избор и по-ранен 1ч CHoCH
      (бележката тогава казва „1ч CHoCH“).
    - И двете фази: само в discount зона (под 50% от дневния диапазон),
      Stop под силното дъно/дъното на пулбека с ATR буфер, минимален R/R спрямо
      дневната съпротива.
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
MAIN_COLUMNS = ["Име", "📊 Фундамент", "📰 Новини", "💼 Държа", "Цена", "Лимит вход", "Валута", "Зона", "📐 EMA", "🌊 ATR %",
                "Позиция в диапазона (%)", "R/R (до дневна съпротива)", "💶 Цел 1 / Цел 2", "⏱ До цел 1", "Бележка"]
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


def target2(x):
    """Цел 2 = седмичната съпротива, само ако е над цел 1 (дневната съпротива); иначе None.
    Когато седмичната е под или равна на дневната, няма смислена втора цел."""
    return x.weekly_resistance if x.weekly_resistance and x.weekly_resistance > x.daily_resistance else None


def fmt_target2(x) -> str:
    t2 = target2(x)
    return f"{t2:.2f}" if t2 else "—"


def trade_plan(x) -> dict:
    """Лимит вход + ориентировъчната печалба до цел 1/2 и загубата до stop за сумата от настройките."""
    entry, how, is_limit = limit_entry(x)
    amount = investment_amount()
    risk = entry - x.stop
    return {
        "entry": entry, "how": how, "is_limit": is_limit, "amount": amount,
        "t1": pnl_eur(entry, x.daily_resistance, amount), "t2": pnl_eur(entry, target2(x), amount) if target2(x) else None,
        "stop": pnl_eur(entry, x.stop, amount),
        "rr": round((x.daily_resistance - entry) / risk, 2) if risk > 0 and x.daily_resistance > entry else None,
        "days": days_to_target(x, entry),
    }


def days_to_target(x, entry: float):
    """Ориентировъчен срок до цел 1 в търговски дни -> (бързо, типично, бавно) или None.
    По скоростта на досегашните възходящи swing-ове на акцията; ако няма достатъчно
    история - по дневния ATR (типичен тренд ~0.3 ATR на ден)."""
    distance = x.daily_resistance - entry
    if distance <= 0:
        return None
    speeds = x.leg_speed
    if not speeds and x.width_atr:
        atr = (x.daily_resistance - x.daily_support) / x.width_atr
        speeds = (0.5 * atr, 0.3 * atr, 0.2 * atr)
    if not speeds or min(speeds) <= 0:
        return None
    return tuple(max(1, round(distance / sp)) for sp in speeds)


MONTH_TRADING_DAYS = 22
HORIZON_OPTIONS = {"Без ограничение": 0, "~2 седмици": 10, "~3 седмици": 15, "~1 месец": MONTH_TRADING_DAYS,
                   "~2 месеца": 44}


def max_horizon_days() -> int:
    return HORIZON_OPTIONS.get(st.session_state.get("ph_horizon", "~1 месец"), MONTH_TRADING_DAYS)


def fmt_days(days) -> str:
    if not days:
        return "—"
    fast, typical, slow = days
    return f"~{typical} дни ({fast}-{slow})" if fast != slow else f"~{typical} дни"


def split_by_horizon(results: list, watch_list: list):
    """Филтър за хоризонта: готовите, при които типичният срок до цел 1 е над
    избрания максимум, отиват в Watchlist с бележка. Връща (готови, watchlist)."""
    limit = max_horizon_days()
    if not limit:
        return results, watch_list
    keep, moved = [], []
    for x in results:
        days = trade_plan(x)["days"]
        if days and days[1] > limit:
            moved.append(dataclasses.replace(
                x, note=f"Цел 1 твърде далеч за хоризонта (~{days[1]} търг. дни > {limit}) · {x.note}"))
        else:
            keep.append(x)
    return keep, moved + watch_list


ATR_PCT_SLIDER_MAX = 10.0  # горната граница на плъзгача = без горен лимит


def split_by_trend_volatility(results: list, watch_list: list):
    """Филтри от настройките (по подразбиране изключени): само EMA20 > EMA50 > EMA200 и
    ATR % в избрания диапазон. Готовите извън тях отиват в Watchlist с бележка."""
    need_ema = st.session_state.get("ph_need_ema", False)
    atr_lo, atr_hi = st.session_state.get("ph_atr_range", (0.0, ATR_PCT_SLIDER_MAX))
    if not need_ema and atr_lo <= 0 and atr_hi >= ATR_PCT_SLIDER_MAX:
        return results, watch_list
    keep, moved = [], []
    for x in results:
        atr = getattr(x, "atr_pct", None)
        why = None
        if need_ema and not getattr(x, "ema_aligned", None):
            why = "EMA20 > EMA50 > EMA200 не е изпълнено"
        elif atr is not None and atr < atr_lo:
            why = f"твърде спокойна (ATR {atr:.1f}% < {atr_lo:.1f}%)"
        elif atr is not None and atr_hi < ATR_PCT_SLIDER_MAX and atr > atr_hi:
            why = f"твърде волатилна (ATR {atr:.1f}% > {atr_hi:.1f}%)"
        if why:
            moved.append(dataclasses.replace(x, note=f"{why} · {x.note}"))
        else:
            keep.append(x)
    return keep, moved + watch_list


SORT_OPTIONS = {
    "Потвърждение": None,
    "📐 EMA подредени първо": lambda x: 0 if getattr(x, "ema_aligned", None) else 1,
    "🌊 ATR % ↓": lambda x: -(getattr(x, "atr_pct", None) or 0),
    "🌊 ATR % ↑": lambda x: getattr(x, "atr_pct", None) if getattr(x, "atr_pct", None) is not None else 999,
}


def sort_control(setups: list, key: str) -> list:
    """Избор „Подреди по“ над таблицата/картите; „Потвърждение“ = досегашният ред
    (зона, после фундамент/новини). Сортирането е стабилно - в рамките на равните
    остава редът по потвърждение."""
    st.session_state.setdefault(key, "Потвърждение")
    choice = st.segmented_control("Подреди по", list(SORT_OPTIONS), key=key,
                                  help="📐 EMA = EMA20 > EMA50 > EMA200 (Full Trend Alignment); "
                                       "🌊 ATR % = среден дневен ход като % от цената (волатилност)")
    sort_key = SORT_OPTIONS.get(choice)
    return sorted(setups, key=sort_key) if sort_key else setups


def pnl_line_html(plan: dict) -> str:
    """Бледият ред „при 1000 €: цел 1 +78 € · цел 2 +120 € · stop −25 €“."""
    return (f'<div class="pnl-hint">при {plan["amount"]:,.0f} €: '.replace(",", " ")
            + f'цел 1 <b>{fmt_pnl(plan["t1"])}</b> · цел 2 <b>{fmt_pnl(plan["t2"]) if plan["t2"] is not None else "—"}</b> · stop {fmt_pnl(plan["stop"])}'
            + (f' · ⏱ до цел 1 <b>{fmt_days(plan["days"])}</b>' if plan.get("days") else "")
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

    st.markdown("##### ⏱ Хоризонт на сделката")
    st.segmented_control(
        "Макс. срок до цел 1", list(HORIZON_OPTIONS), key="ph_horizon", label_visibility="collapsed",
        help="Готовите, при които типичният срок до цел 1 е по-дълъг, отиват в Watchlist с бележка. Срокът е "
             "ориентировъчен - по скоростта на досегашните възходящи swing-ове на акцията.",
    )

    st.markdown("##### 📐 Тренд и волатилност")
    t1, t2 = st.columns([1, 2])
    with t1:
        st.toggle("Само с EMA20 > EMA50 > EMA200", value=False, key="ph_need_ema",
                  help="Full Trend Alignment на дневната. Изключено по подразбиране: в дълбок Photon пулбек (Phase B) "
                       "EMA20 често пада под EMA50, а точно там са най-добрите входове. Готовите без него отиват в Watchlist.")
    with t2:
        st.slider("ATR % (дневен ход като % от цената)", 0.0, ATR_PCT_SLIDER_MAX, value=(0.0, ATR_PCT_SLIDER_MAX),
                  step=0.5, key="ph_atr_range",
                  help="Готовите извън диапазона отиват в Watchlist. 10 = без горна граница. Ориентир за swing до месец: "
                       "под ~1% е бавна, 1.5-4% е добре, над ~5% е нервна (широк stop).")

    with st.expander("🔧 Разширени настройки на сигналите"):
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Минимален R/R (до дневна съпротива)", 1.0, 4.0, step=0.5, key="ph_min_rr")
            st.slider("Свежест на CHoCH (макс. 4ч свещи назад)", 1, 6, key="ph_choch_age",
                      help="Phase B е „готов“ само ако пробивът над силния 4ч lower high (CHoCH) е станал до толкова свещи назад.")
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
        st.toggle("⚡ По-ранен вход с 1ч CHoCH (Phase B)", value=False, key="ph_choch_1h",
                  help="Ако 4ч CHoCH още няма, но на 1ч цената вече е пробила силния lower high в discount зоната, "
                       "сетъпът е готов: по-ранен вход и по-ниско CHoCH ниво (по-висок R/R), но повече фалшиви сигнали.")

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
        "choch_1h": st.session_state.get("ph_choch_1h", False),
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


FUND_WATCHLIST_TOP = 30  # за толкова от Watchlist анализаторите се теглят веднага след скана


def fetch_pending_fund_data(slot):
    """Тегли чакащите анализаторски данни (след скана или от бутоните) с прогрес в slot
    и прерисува страницата. Вика се в края - резултатите вече се виждат."""
    pending = st.session_state.get("photon_fund_pending")
    if not pending:
        return
    with slot:
        bar = st.progress(0.0, text=f"📊 Анализаторски данни от Yahoo за {len(pending)} акции "
                                    "(резултатите вече са готови отдолу)...")
    got = fund.fetch_analyst_data(pending, on_done=lambda i, n: bar.progress(
        i / n, text=f"📊 Анализаторски данни от Yahoo: {i}/{n} (резултатите вече са готови отдолу)"))
    st.session_state["photon_fund"] = {**st.session_state.get("photon_fund", {}), **got}
    st.session_state.pop("photon_fund_pending", None)
    st.rerun()


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
        # ниво 1 на фундаменталното потвърждение: анализатори от Yahoo (без ETF-ите). Тегли се СЛЕД
        # като резултатите се покажат (в края на страницата) и само за важните: готовите, отворените
        # позиции и първите FUND_WATCHLIST_TOP от Watchlist (най-близо до подкрепата); останалите - с бутон
        held_now = [p["symbol"] for p in fetch_held_positions() or [] if p["symbol"]]
        near = [x.symbol for x in sorted(watch_list, key=lambda x: x.range_pos)]
        is_stock = lambda s: s and types.get(s) != "ETF"  # noqa: E731
        priority = list(dict.fromkeys(s for s in [x.symbol for x in results] + held_now + near[:FUND_WATCHLIST_TOP]
                                      if is_stock(s)))
        st.session_state["photon_fund"] = {}
        st.session_state["photon_fund_pending"] = tuple(priority)
        st.session_state["photon_fund_rest"] = tuple(s for s in near[FUND_WATCHLIST_TOP:] if is_stock(s) and s not in priority)
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


LEGEND_MD = f"""
| Символ | Значение |
|---|---|
| **⭐ Структура + фундамент** | най-силният сигнал: ✅ от анализаторите **и** 🟢 новини |
| **✅ Потвърден** | анализаторите: Buy / Strong Buy от поне {fund.MIN_ANALYSTS}, потенциал ≥ {fund.MIN_UPSIDE_PCT}% до средната целева цена |
| **➖ Неутрален** | Hold, малко анализатори или малък потенциал |
| **⚠️ Против** | Sell или целевата цена е под текущата |
| **— няма данни** | ETF или акция без анализаторско покритие в Yahoo |
| **🟢 / ⚪ / 🔴 Новини** | положителни / неутрални / отрицателни новини от проверката с AI (празно = непроверено днес) |
| **Зелен ред** | ✅ или ⭐ и новините не са 🔴 |
| **💼 N / T / N+T** | отворена позиция в T212: твоят акаунт / на съпругата / и двата |
| **🟦 A · Pro** | 4ч структурата е възходяща - вход в POI зоната над силното 4ч дъно |
| **🟪 B · Counter** | 4ч е в пулбек - вход след пробив (CHoCH) над силния lower high - върхът преди най-ниското дъно на пулбека; „1ч CHoCH“ = по-ранното потвърждение от настройките |
| **Силно дъно / слаб връх** | силното дъно е довело до пробив нагоре (BOS) - то е дневната подкрепа и под него структурата е счупена; слабият връх още не е „защитен“ - той е целта (на графиката: BOS линии и надписи) |
| **🟢 Discount / 🟠 Premium / 🔵 над съпротивата** | долната половина на дневния диапазон / горната / над дневната съпротива |
| **Позиция %** | 0% = дневна подкрепа, 50% = equilibrium, 100% = съпротива (цел 1) |
| **Лимит вход** | предложената цена за поръчка; ако е **над** текущата цена, това е buy stop - Phase B, който чака пробив на CHoCH |
| **⏱ До цел 1** | ориентировъчен срок в търговски дни (типичен и диапазон бързо-бавно) по скоростта на досегашните възходящи swing-ове на акцията; готовите над избрания хоризонт отиват в Watchlist |
| **💶 Цел 1 / Цел 2** | ориентировъчна печалба до дневната / седмичната съпротива при сумата от настройките |
| **📐 EMA ✓** | EMA20 > EMA50 > EMA200 на дневната - пълно подреждане на тренда (Full Trend Alignment) |
| **🌊 ATR %** | среден дневен ход (ATR 14) като % от цената: 100 € и ход 4 € = 4%; под ~1% бавна, 1.5-4% добре за swing, над ~5% нервна |
| **Отчет ⚠️** | следващият отчет е до {fund.EARNINGS_WARN_DAYS} дни - риск от гап |
"""


def render_legend():
    with st.expander("ℹ️ Легенда на символите"):
        st.markdown(LEGEND_MD)


def render_photon_strategy():
    status_box = st.container()
    # хоризонтът е в таба с настройките (рисува се по-долу) - стойността е от сесията
    st.session_state.setdefault("ph_horizon", "~1 месец")
    results, watch_list = split_by_horizon(st.session_state.get("photon_results", []),
                                           st.session_state.get("photon_watchlist", []))
    results, watch_list = split_by_trend_volatility(results, watch_list)
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
        fund_slot = st.empty()

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
            render_legend()
            st.caption("Phase A (цена в POI) или Phase B (свеж 4ч CHoCH), в discount и с R/R над минимума. "
                       "Зелено = потвърдено и от анализаторите.")
            shown_ready = sort_control(results, "ph_sort_ready")
            if st.toggle("Табличен изглед", key="ph_ready_as_table"):
                render_setup_table(shown_ready, "ph_ready_table", held, fund_data, news)
            else:
                render_setup_cards(shown_ready, held, fund_data, news)
    df_ready = setups_dataframe(results, held, fund_data, news)  # за AI анализа

    with tab_watch:
        news_targets = list(results)
        if not scanned:
            st.info("Натисни **🔍 Сканирай пазара** горе.")
            df_watch = pd.DataFrame()
        else:
            render_legend()
            st.caption("Pro Swing потвърден - колоната „Бележка“ казва какво чакаме. Кликни ред за графика и новини.")
            show_far = st.toggle(
                f"Покажи и далечните (над {FAR_ABOVE_RANGE_PCT}% от дневния диапазон)", value=False, key="ph_show_far",
                help="Над съпротивата = след пробив нагоре; до вход има нужда от нов пулбек, често дълъг.",
            )
            shown_watch = shown_order if show_far else [x for x in shown_order if x.range_pos <= FAR_ABOVE_RANGE_PCT]
            news_targets += shown_watch[:NEWS_WATCHLIST_TOP]  # ниво 2 - преди подреждането по новини
            shown_watch = sort_control(sort_by_confirmation(shown_watch, fund_data, news), "ph_sort_watch")
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
            render_legend()
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

    install_chart_scripts()  # преди прозореца - за да го затваря стрелката назад
    # изскачащият прозорец с графика/новини (от клик по ред, карта или новина)
    pending = st.session_state.pop("ph_dialog", None)
    if pending:
        instrument_dialog(*pending)
    else:
        # анализаторите - след като всичко горе вече е нарисувано (прогресът е в горния статус)
        fetch_pending_fund_data(fund_slot)


# ---------------------------------------------------------------- прозорец за инструмент

def open_instrument(name: str, symbol: str):
    """Отваря прозореца на инструмента при следващото прерисуване (в края на страницата)."""
    st.session_state["ph_dialog"] = (name, symbol)


def instrument_dialog(name: str, symbol: str, actions=None):
    """actions - по избор функция без аргументи, рисувана най-горе (напр. „⭐ Добави в Селектирани“)."""
    @st.dialog(name, width="large")
    def body():
        if actions:
            actions()
        setup = st.session_state.get("ph_setups_by_name", {}).get(name)
        fund_data = st.session_state.get("photon_fund", {})
        news = today_news()
        if setup:
            plan = trade_plan(setup)
            st.markdown(levels_html([
                ("Цена сега", f"{setup.price:.2f} {setup.currency}"),
                ("Лимит вход" if plan["is_limit"] else "Buy stop", f"{plan['entry']:.2f}"), ("Stop", f"{setup.stop:.2f}"),
                ("Цел 1 (дневна)", f"{setup.daily_resistance:.2f}"), ("Цел 2 (седмична)", fmt_target2(setup)),
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
                if getattr(x, "ema_aligned", None):
                    badges.append("📐 EMA 20>50>200")
                if getattr(x, "atr_pct", None) is not None:
                    badges.append(f"🌊 ATR {x.atr_pct:.1f}%")
                st.markdown(" ".join(f'<span class="badge">{b}</span>' for b in badges), unsafe_allow_html=True)
                plan = trade_plan(x)
                st.markdown(levels_html([
                    ("Лимит вход" if plan["is_limit"] else "Buy stop", f"{plan['entry']:.2f}"), ("Stop", f"{x.stop:.2f}"),
                    ("Цел 1", f"{x.daily_resistance:.2f}"), ("Цел 2", fmt_target2(x)),
                    ("R/R", f"{plan['rr']:.2f}" if plan["rr"] else "—"),
                    ("⏱ До цел 1", f"~{plan['days'][1]} дни" if plan["days"] else "—"),
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
            # без пайовете - те са дългосрочни кошници, не swing позиции
            open_positions = t212.exclude_pies(t212.fetch_open_positions(t212.T212_ENV_TO_BASE_URL["live"], auth))
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


@st.cache_resource(show_spinner=False)
def _news_store() -> dict:
    """Общ за ВСИЧКИ сесии склад с резултатите от новините за деня: обновяване на
    страницата, друг браузър или телефон не плащат повторно за вече проверено.
    (Изчиства се само при рестарт/обновяване на самото приложение.)"""
    return {"lock": threading.Lock(), "date": None, "by_provider": {}, "checks": {}}


def _store_today() -> dict:
    store = _news_store()
    today = datetime.now().date().isoformat()
    with store["lock"]:
        if store["date"] != today:  # нов ден - новите новини се проверяват наново
            store.update(date=today, by_provider={}, checks={})
    return store


def today_news() -> dict:
    """Резултатите от ниво 2 (новини) за днес от избрания доставчик - {symbol: резултат}."""
    return dict(_store_today()["by_provider"].get(news_provider(), {}))


def news_checks_today(provider: str) -> int:
    """Колко платени проверки на новини са направени днес с този доставчик (всички сесии)."""
    return _store_today()["checks"].get(provider, 0)


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
    rest = [s for s in st.session_state.get("photon_fund_rest", ()) if s not in fund_data]
    if rest:
        st.caption(f"За останалите {len(rest)} от Watchlist (по-далеч от подкрепата) анализаторите не са изтеглени, "
                   "за да е по-бърз сканът.")
        if st.button(f"📊 Изтегли и за тях ({len(rest)})", key="ph_fund_rest"):
            st.session_state["photon_fund_pending"] = tuple(rest)
            st.rerun()
    if not failed:
        return
    first_error = next(fund_data[s]["error"] for s in failed)
    st.warning(f"Yahoo не върна данни за {len(failed)} акции (най-често временно ограничение на заявките). "
               f"Пример: {first_error}")
    if st.button(f"🔄 Дотегли липсващите ({len(failed)})", key="ph_fund_retry"):
        st.session_state["photon_fund_pending"] = tuple(failed)
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
        [(x.name, x.symbol, is_etf(x.symbol)) for x in targets], provider, api_key, gemini_model=gemini_news_model(),
        gemini_fallback_model=gemini_model(), claude_model=claude_news_model(),
        on_done=lambda i, n: bar.progress(i / n, text=f"Проверени {i}/{n}"))
    bar.empty()
    store = _store_today()
    with store["lock"]:
        store["by_provider"].setdefault(provider, {}).update(found)
        store["checks"][provider] = store["checks"].get(provider, 0) + len(found)
    return found


def render_news_section(targets: list, news: dict, key: str):
    """Ниво 2: бутон за проверка с избрания AI модел + компактна таблица с
    резултатите; клик по ред отваря прозореца с подробностите и източниците.
    targets - обекти с .name и .symbol (сетъпи или позиции)."""
    provider = news_provider()
    missing = [x for x in targets if x.symbol not in news or "error" in news[x.symbol]]
    billing = "Google AI (Gemini API)" if provider == "Gemini" else "Anthropic API"
    st.caption(f"Чрез **{provider}** (сменя се с „🤖 AI анализи чрез“ горе) - всяка проверка търси в интернет "
               f"и се таксува в {billing}. Резултатите се пазят до края на деня за всички сесии; проверяват се "
               f"само липсващите. Днес с {provider}: **{news_checks_today(provider)}** проверки.")
    confirm_key = f"{key}_confirm"
    if st.button(f"🔎 Провери новини и анализи с {provider} ({len(missing)} инструмента)", key=f"{key}_btn",
                 disabled=not missing, width="stretch"):
        st.session_state[confirm_key] = True
    run_check = False
    if st.session_state.get(confirm_key) and missing:
        with st.container(border=True):
            st.warning(f"Ще се направят **{len(missing)} платени проверки** с {provider} (всяка = търсене в "
                       f"интернет + отговор на модела). Продължаваме ли?")
            c1, c2 = st.columns(2)
            if c1.button("✅ Да, провери", key=f"{key}_yes", type="primary", width="stretch"):
                st.session_state.pop(confirm_key, None)
                run_check = True
            if c2.button("✖ Откажи", key=f"{key}_no", width="stretch"):
                st.session_state.pop(confirm_key, None)
                st.rerun()
    if run_check:
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
        row["💶 Цел 1 / Цел 2"] = f"{fmt_pnl(plan['t1'])} / {fmt_pnl(plan['t2']) if plan['t2'] is not None else '—'}"
        if target2(x) is None:
            row["R/R (до седм. съпротива)"] = None  # седмичната не е над цел 1 - няма втора цел
        row["💶 Stop"] = fmt_pnl(plan["stop"])
        row["⏱ До цел 1"] = fmt_days(plan["days"])
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
    "🌊 ATR %": "{:.1f}",
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
            "📐 EMA": st.column_config.TextColumn("📐 EMA", help="✓ = EMA20 > EMA50 > EMA200 на дневната (Full Trend Alignment)"),
            "🌊 ATR %": st.column_config.NumberColumn("🌊 ATR %", format="%.1f",
                                                     help="Среден дневен ход (ATR 14) като % от цената - волатилността"),
            "💼 Държа": st.column_config.TextColumn("💼", help="Отворена позиция в T212: N / T (акаунт)"),
        },
        on_select=lambda: _open_selected(key, items), selection_mode="single-row", key=key,
    )
    return df


# ---------------------------------------------------------------- графики

# цветовете са по образец на Наско (T212): почти черен синьо-зелен фон, видима мрежа, ярки свещи
CHART_UP, CHART_DOWN = "#4BD65E", "#F5434B"
CHART_SURFACE, CHART_GRID, CHART_INK, CHART_MUTED = "#0A141B", "#1D2B35", "#E6EDF3", "#8497B0"
CHART_SWING = "#9AA7B4"  # сивите точки на swing върховете/дъната
CHART_LABEL_BG = "rgba(10, 20, 27, 0.82)"  # полупрозрачен фон на етикетите в графиката - четими и върху свещите
LEVEL_STYLES = {  # (цвят, тип линия, дебелина) - ярки, различими цветове за тъмния фон
    "Цел 2 · седм. съпротива": ("#F5B642", "dot", 1),
    "Цел 1 · дневна съпротива": ("#FF7A7A", "dot", 1),
    "Equilibrium 50%": ("#A7B4C2", "dash", 1),
    "Дневна подкрепа": ("#3DDC97", "dot", 1),
    "CHoCH ниво": ("#C792FF", "dash", 1),
    "Stop": ("#FF4D57", "solid", 1.5),
    "Лимит вход": ("#6EA8FE", "dash", 1.2),
    "Buy stop": ("#6EA8FE", "dash", 1.2),
    "POI": ("#6EA8FE", "dot", 1),
}


def swing_markers(fig, df_with_swings: pd.DataFrame, x_values=None):
    """Малки сиви точки на swing high (над свещта) и swing low (под нея) - ненатрапчиви.
    x_values - етикетите на свещите при категорийна ос (4ч)."""
    points = alternating_swings(df_with_swings)
    pos = {ts: i for i, ts in enumerate(df_with_swings.index)}
    pad = (df_with_swings["High"].max() - df_with_swings["Low"].min()) * 0.012
    for kind, marker, color, label in (("H", "circle", CHART_SWING, "Swing high"), ("L", "circle", CHART_SWING, "Swing low")):
        pts = [pt for pt in points if pt[1] == kind]
        if pts:
            fig.add_trace(go.Scatter(
                x=[x_values[pos[pt[0]]] if x_values is not None else pt[0] for pt in pts],
                y=[pt[2] + pad if kind == "H" else pt[2] - pad for pt in pts],
                mode="markers", name=label, showlegend=False, hovertemplate=f"{label}: %{{customdata:.2f}}<extra></extra>",
                customdata=[pt[2] for pt in pts], marker=dict(symbol=marker, size=5, color=color, opacity=0.75),
            ))


def candle_figure(df: pd.DataFrame, levels: list, poi=None, visible_bars: int = 130, categorical: bool = False,
                  swings_order: int = None, height: int = 520, marks: list = None, tags: list = None):
    """Свещи + нива като в T212: ценовата скала вдясно (с етикет на текущата цена),
    празно място между последната свещ и скалата, етикетите на нивата вляво в самата
    графика (полупрозрачен фон, цвят на линията). levels = [(етикет, цена)]; poi = (от, до).
    По подразбиране се виждат последните visible_bars свещи (zoom out = цялата история).
    marks = [(от, до или None = последната свещ, цена, текст, цвят, тип линия)] - къси линии като
    BOS/CHoCH в учебниците на Photon; tags = [(момент, цена, текст, цвят, "above"/"below")] -
    надписи при swing точка („силно дъно“, „слаб връх“)."""
    df = df.dropna(subset=["Close"])
    x = [ts.strftime("%d.%m %H:%M") for ts in df.index] if categorical else df.index

    def to_x(ts):
        ts = df.index[-1] if ts is None else ts
        return ts.strftime("%d.%m %H:%M") if categorical else ts
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
        fig.add_hrect(y0=poi[0], y1=poi[1], fillcolor="#6EA8FE", opacity=0.13, line_width=0)
        levels = levels + [("POI", (poi[0] + poi[1]) / 2, "")]
    # етикетите вляво в графиката, разтворени по вертикала, за да не се застъпват
    labels = [lv for lv in levels if lv[1]]
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
        color, dash, width = LEVEL_STYLES.get(name, ("#6EA8FE", "dot", 1))
        if name != "POI":
            fig.add_hline(y=value, line_dash=dash, line_color=color, line_width=width, opacity=0.9)
        fig.add_annotation(
            xref="paper", x=0.006, xanchor="left", yref="y", y=y_label, showarrow=False, align="left",
            text=f"{name} <b>{value:,.2f}</b>" + (f" <span style='color:{CHART_MUTED}'>{note}</span>" if note else ""),
            font=dict(size=11, color=color), bgcolor=CHART_LABEL_BG, bordercolor=color, borderwidth=1, borderpad=3,
        )
    # текущата цена: пунктир през графиката + цветен етикет върху ценовата скала (както в T212)
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
    price_color = CHART_UP if last_close >= prev_close else CHART_DOWN
    fig.add_hline(y=last_close, line_dash="dot", line_color=price_color, line_width=1, opacity=0.8)
    fig.add_annotation(
        xref="paper", x=1.0, xanchor="left", yref="y", y=last_close, showarrow=False,
        text=f"<b>{last_close:,.2f}</b>", font=dict(size=11, color="#FFFFFF"), bgcolor=price_color, borderpad=3,
    )

    for x0, x1, y, text, color, dash in marks or []:
        fig.add_shape(type="line", xref="x", yref="y", x0=to_x(x0), x1=to_x(x1), y0=y, y1=y,
                      line=dict(color=color, width=1.2, dash=dash))
        fig.add_annotation(xref="x", yref="y", x=to_x(x1), y=y, text=text, showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=10, color=color))
    for ts, y, text, color, where in tags or []:
        fig.add_annotation(xref="x", yref="y", x=to_x(ts), y=y, text=text, showarrow=True, arrowhead=0,
                           arrowcolor=color, ax=0, ay=-26 if where == "above" else 26, font=dict(size=10, color=color),
                           bgcolor=CHART_LABEL_BG, borderpad=2)

    if categorical:
        shown = min(visible_bars, len(df))
        fig.update_xaxes(type="category", range=[len(df) - shown - 0.5, len(df) - 0.5 + shown * 0.08], nticks=10)
    else:
        # празно място вдясно (~8%), за да не се сливат последните свещи с ценовата скала
        span = df.index[-1] - df.index[-min(visible_bars, len(df))]
        fig.update_xaxes(range=[df.index[-min(visible_bars, len(df))], df.index[-1] + span * 0.08])
    fig.update_yaxes(range=[y0, y1], side="right", tickformat=",.2f", ticklabelstandoff=6)
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor=CHART_SURFACE, plot_bgcolor=CHART_SURFACE,
        xaxis_rangeslider_visible=False, hovermode=False, dragmode="pan",  # без каре над свещите - кръстът е от CROSSHAIR_JS
        margin=dict(l=8, r=64, t=10, b=10), font=dict(size=11, color=CHART_MUTED),
        xaxis=dict(showgrid=True, gridcolor=CHART_GRID, zeroline=False), yaxis=dict(showgrid=True, gridcolor=CHART_GRID, zeroline=False),
        hoverlabel=dict(bgcolor="#13212B", font_size=12),
    )
    return fig


MARK_BOS, MARK_CHOCH, MARK_STRONG, MARK_WEAK = "#C9D4E0", "#C792FF", "#3DDC97", "#FF7A7A"


def structure_annotations(df_with_swings: pd.DataFrame) -> dict:
    """BOS линиите (пробитите върхове) и надписите „силно дъно“ / „слаб връх“ за candle_figure."""
    marks = [(ts, brk, price, "BOS", MARK_BOS, "solid") for ts, brk, price in bos_marks(df_with_swings)]
    s = swing_structure(df_with_swings)
    tags = []
    if s is not None:
        tags = [(s["strong_low_idx"], s["strong_low"], "силно дъно", MARK_STRONG, "below"),
                (s["weak_high_idx"], s["weak_high"], "слаб връх · цел", MARK_WEAK, "above")]
    return {"marks": marks, "tags": tags}


def choch_annotations(df_with_swings: pd.DataFrame, pullback_start) -> dict:
    """4ч: линия от силния lower high до пробива му (CHoCH) или до днес, ако още чака,
    и надпис на дъното на пулбека."""
    pb = pullback_choch(df_with_swings, pullback_start)
    if pb is None:
        return {}
    done = pb["break_idx"] is not None
    return {
        "marks": [(pb["level_idx"], pb["break_idx"], pb["choch_level"], "CHoCH" if done else "CHoCH?", MARK_CHOCH,
                   "solid" if done else "dot")],
        "tags": [(pb["pullback_low_idx"], pb["pullback_low"], "дъно на пулбека", MARK_STRONG, "below")],
    }


CHART_CONFIG = {"displaylogo": False, "scrollZoom": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d", "toggleSpikelines"]}
CHART_HEIGHTS = {"S": 420, "M": 560, "L": 720, "XL": 900}
# Кръст при натискане на средния бутон (колелото): линии през цялата графика + цената на
# курсора върху ценовата скала. Plotly няма такъв режим, затова е малък скрипт върху
# страницата (st.html с JS) - слуша всички графики, инсталира се веднъж.
CROSSHAIR_JS = """
<script>
(() => {
  // само прозорецът на приложението: в Streamlit Cloud то е в iframe, а родителят е
  // обвивката на хостинга (с „Manage app“) - там графиките ги няма
  const w = window;
  if (w.__phCrosshair) return;
  w.__phCrosshair = true;
  const doc = w.document;
  const plotOf = (el) => el && el.closest ? el.closest('.js-plotly-plot') : null;
  const fmt = (v) => v.toFixed(2);
  function overlay(gd) {
    if (gd.__phX) return gd.__phX;
    const box = doc.createElement('div');
    box.style.cssText = 'position:absolute;inset:0;pointer-events:none;z-index:20;display:none';
    const line = (css) => { const d = doc.createElement('div'); d.style.cssText = 'position:absolute;' + css; box.appendChild(d); return d; };
    const v = line('width:0;border-left:1px dashed #C9D4E0;opacity:.85');
    const h = line('height:0;border-top:1px dashed #C9D4E0;opacity:.85');
    const label = 'padding:2px 6px;border-radius:3px;background:#C9D4E0;color:#0A141B;font:600 11px monospace;white-space:nowrap';
    const tag = line(label), date = line(label + ';transform:translateX(-50%)');
    if (getComputedStyle(gd).position === 'static') gd.style.position = 'relative';
    gd.appendChild(box);
    gd.__phX = {box, v, h, tag, date};
    return gd.__phX;
  }
  function draw(gd, ev) {  // по позицията на мишката
    const r = gd.getBoundingClientRect();
    drawAt(gd, ev.clientX - r.left, ev.clientY - r.top, false);
  }
  function drawAt(gd, px, py, clamp) {  // px/py - спрямо графиката; clamp = задържа кръста в полето
    const L = gd._fullLayout; if (!L || !L.yaxis) return;
    const s = L._size, o = overlay(gd);
    if (clamp) {
      px = Math.min(Math.max(px, s.l), s.l + s.w); py = Math.min(Math.max(py, s.t), s.t + s.h);
    } else if (px < s.l || px > s.l + s.w || py < s.t || py > s.t + s.h) { o.box.style.display = 'none'; return; }
    gd.__phPos = {px, py};
    o.box.style.display = 'block';
    o.v.style.left = px + 'px'; o.v.style.top = s.t + 'px'; o.v.style.height = s.h + 'px';
    o.h.style.top = py + 'px'; o.h.style.left = s.l + 'px'; o.h.style.width = s.w + 'px';
    const price = L.yaxis.p2l(py - s.t);
    o.tag.textContent = fmt(price);
    o.tag.style.left = (s.l + s.w + 2) + 'px'; o.tag.style.top = (py - 9) + 'px';
    // долу: дата (D/W) или дата и час на свещта (4h - категорийна ос)
    const xa = L.xaxis, xl = xa.p2l(px - s.l);
    let when = '';
    if (xa.type === 'category') when = (xa._categories || [])[Math.round(xl)] || '';
    else { const d = new Date(xl), p2 = (n) => String(n).padStart(2, '0');
           when = p2(d.getUTCDate()) + '.' + p2(d.getUTCMonth() + 1) + '.' + d.getUTCFullYear(); }
    o.date.textContent = when; o.date.style.display = when ? 'block' : 'none';
    o.date.style.left = px + 'px'; o.date.style.top = (s.t + s.h + 3) + 'px';
  }
  // среден бутон: включва/изключва кръста; спира и автоматичното превъртане на браузъра
  doc.addEventListener('mousedown', (ev) => {
    if (ev.button !== 1) return;
    const gd = plotOf(ev.target); if (!gd) return;
    ev.preventDefault();
    gd.__phOn = !gd.__phOn;
    if (gd.__phOn) draw(gd, ev); else if (gd.__phX) gd.__phX.box.style.display = 'none';
  }, true);
  doc.addEventListener('auxclick', (ev) => { if (ev.button === 1 && plotOf(ev.target)) ev.preventDefault(); }, true);
  doc.addEventListener('mousemove', (ev) => {
    const gd = plotOf(ev.target);
    doc.querySelectorAll('.js-plotly-plot').forEach((p) => { if (p !== gd && p.__phX) p.__phX.box.style.display = 'none'; });
    if (gd && gd.__phOn) draw(gd, ev);
  }, true);
  // разтягане по цялата скала (не само в краищата), както в T212:
  // ценовата скала - влачене нагоре/надолу свива/разтяга около средата;
  // времевата ос - влачене наляво/надясно разтяга/свива, десният край остава на място
  const AXIS_Y = ['nsdrag', 'ndrag', 'sdrag'], AXIS_X = ['ewdrag', 'wdrag', 'edrag'];
  let axisDrag = null;
  doc.addEventListener('mousedown', (ev) => {
    if (ev.button !== 0 || !w.Plotly) return;
    const rect = ev.target, gd = plotOf(rect);
    if (!gd || !rect.classList) return;
    const isY = AXIS_Y.some((c) => rect.classList.contains(c)), isX = AXIS_X.some((c) => rect.classList.contains(c));
    if (!isY && !isX) return;
    ev.preventDefault(); ev.stopPropagation();  // вместо местенето на Plotly в средата на оста
    const ax = gd._fullLayout[isY ? 'yaxis' : 'xaxis'];
    axisDrag = {gd, isY, ax, x0: ev.clientX, y0: ev.clientY, r0: ax.range.map((v) => ax.r2l(v)), frame: null};
    doc.body.style.cursor = isY ? 'ns-resize' : 'ew-resize';
  }, true);
  doc.addEventListener('mousemove', (ev) => {
    if (!axisDrag) return;
    ev.preventDefault(); ev.stopPropagation();
    const d = axisDrag, [a, b] = d.r0;
    let range;
    if (d.isY) {  // надолу = по-голям обхват (свещите се свиват), нагоре = разтягане
      const k = Math.exp((ev.clientY - d.y0) / 150), mid = (a + b) / 2, half = (b - a) / 2 * k;
      range = [mid - half, mid + half];
    } else {      // надясно = разтягане (по-малко свещи), наляво = свиване
      const k = Math.exp(-(ev.clientX - d.x0) / 200);
      range = [b - (b - a) * k, b];
    }
    if (d.frame) cancelAnimationFrame(d.frame);
    d.frame = requestAnimationFrame(() => w.Plotly.relayout(d.gd, {[(d.isY ? 'yaxis' : 'xaxis') + '.range']: range.map((v) => d.ax.l2r(v))}));
  }, true);
  doc.addEventListener('mouseup', () => { if (axisDrag) { axisDrag = null; doc.body.style.cursor = ''; } }, true);

  const hideAll = () => doc.querySelectorAll('.js-plotly-plot').forEach((p) => {
    p.__phOn = false; if (p.__phX) p.__phX.box.style.display = 'none';
  });
  doc.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') hideAll(); }, true);

  // „назад“ (стрелката на телефона / браузъра): записи в историята за кръста (телефон) и за
  // изскачащия прозорец с графиката; назад първо скрива кръста, после затваря прозореца -
  // вместо да излиза от приложението
  // Chrome на телефона ПРЕСКАЧА при „назад“ записи, добавени без истинско докосване
  // (user activation) - затова записите се добавят само при активно докосване/клик, а ако
  // в момента няма такова - при следващото (pending).
  const stack = [];
  const pending = [];
  let ignorePops = 0;
  const activeNow = () => !navigator.userActivation || navigator.userActivation.isActive;
  const pushEntry = (kind) => { stack.push(kind); w.history.pushState({ph: kind, phDepth: stack.length}, ''); };
  const requestEntry = (kind) => {
    if (stack.includes(kind) || pending.includes(kind)) return;
    if (activeNow()) pushEntry(kind); else pending.push(kind);
  };
  const flushPending = () => {
    while (pending.length && activeNow()) {
      const kind = pending.shift();
      if (kind === 'dialog' && dialogOpen()) pushEntry(kind);
    }
  };
  ['touchend', 'pointerup', 'click', 'keydown'].forEach((t) => doc.addEventListener(t, () => setTimeout(flushPending, 0), true));
  const crossOn = () => [...doc.querySelectorAll('.js-plotly-plot')].some((p) => p.__phOn);
  const dialogOpen = () => doc.querySelector('[role="dialog"]');
  const closeDialog = () => {
    const dlg = dialogOpen(); if (!dlg) return;
    const btn = dlg.querySelector('button[aria-label="Close"], button[aria-label="close"], [data-testid="stDialogCloseButton"]');
    if (btn) btn.click(); else doc.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', bubbles: true}));
  };
  let wasOpen = false;
  const watchDialog = () => {
    const open = !!dialogOpen();
    if (open && !wasOpen) requestEntry('dialog');
    if (!open && wasOpen) {
      const i = pending.indexOf('dialog'); if (i >= 0) pending.splice(i, 1);
    }
    if (!open && wasOpen && stack.includes('dialog')) {
      // затворен с X - махаме нашите записи (и на кръста вътре), за да не остане „празно“ назад
      const n = stack.length - stack.indexOf('dialog');
      stack.splice(stack.length - n, n);
      ignorePops += 1;
      w.history.go(-n);
    }
    wasOpen = open;
  };
  new MutationObserver(watchDialog).observe(doc.body, {childList: true, subtree: true});
  watchDialog();
  w.addEventListener('popstate', (ev) => {
    if (ignorePops > 0) { ignorePops -= 1; return; }
    const depth = (ev.state && ev.state.phDepth) || 0;
    const dialogDepth = stack.indexOf('dialog') + 1;  // 0 = прозорецът няма запис
    if (crossOn()) {
      // кръстът няма собствен запис (Chrome прескача записи, добавени при задържане на пръста):
      // „назад“ само го скрива и се връща напред до записа на прозореца - без нов pushState,
      // затова прозорецът остава отворен, а следващото „назад“ го затваря
      hideAll();
      if (dialogDepth && depth < dialogDepth) { ignorePops += 1; w.history.go(dialogDepth - depth); }
      return;
    }
    if (dialogDepth && depth < dialogDepth) { wasOpen = false; closeDialog(); }
    stack.splice(depth);
  });

  // телефон: задържане на пръста ~0.5 сек = кръст; после пръстът го МЕСТИ (относително, като
  // тъчпад) - не прескача там, където е пипнал; стрелката назад го скрива
  let timer = null, start = null, touchGd = null, posStart = null;
  const cancel = () => { if (timer) { clearTimeout(timer); timer = null; } };
  doc.addEventListener('touchstart', (ev) => {
    const gd = plotOf(ev.target);
    cancel();
    if (!gd || ev.touches.length !== 1) return;
    const t = ev.touches[0];
    start = {x: t.clientX, y: t.clientY}; touchGd = gd;
    if (gd.__phOn) { posStart = gd.__phPos; return; }
    timer = setTimeout(() => {
      timer = null;
      hideAll();
      const r = gd.getBoundingClientRect();
      gd.__phOn = true;
      drawAt(gd, start.x - r.left, start.y - r.top, true);
      posStart = gd.__phPos;
      if (navigator.vibrate) navigator.vibrate(15);
    }, 450);
  }, {capture: true, passive: true});
  doc.addEventListener('touchmove', (ev) => {
    const t = ev.touches[0];
    if (timer && start && Math.hypot(t.clientX - start.x, t.clientY - start.y) > 10) cancel();
    if (touchGd && touchGd.__phOn && posStart) {
      ev.preventDefault(); ev.stopPropagation();
      drawAt(touchGd, posStart.px + (t.clientX - start.x), posStart.py + (t.clientY - start.y), true);
    }
  }, {capture: true, passive: false});
  doc.addEventListener('touchend', cancel, true);
  doc.addEventListener('touchcancel', cancel, true);
  doc.addEventListener('contextmenu', (ev) => { if (plotOf(ev.target)) ev.preventDefault(); }, true);
})();
</script>
"""
def install_chart_scripts():
    """Кръстът, разтягането на скалите и „назад“ за прозореца - инсталира се веднъж на страница
    (скриптът сам пази да не се закачи два пъти)."""
    st.html(CROSSHAIR_JS, unsafe_allow_javascript=True)


CHART_HINT = ("↕ влачи ценовата скала (вдясно) = разтягане вертикално · ↔ влачи времевата "
              "ос (долу) = разтягане хоризонтално · влачи в графиката = местене · колелото = zoom · двоен клик = връщане · "
              "натисни колелото (на телефон: задръж пръста) = кръст с цената; пак колелото / Esc / стрелката назад = скрий · "
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
        sup_lvl = chart_s["strong_low"] if chart_s else None
        res_lvl = chart_s["weak_high"] if chart_s else None
        order = found_order or swing_order_daily
        st.caption("Инструментът не е в резултатите от последния скан - показват се само дневните нива.")

    plan = trade_plan(setup) if setup else None
    stop = [("Stop", setup.stop, fmt_pnl(plan["stop"]))] if setup else []
    entry = [("Лимит вход" if plan["is_limit"] else "Buy stop", plan["entry"])] if setup else []
    poi = (setup.poi_low, setup.poi_high) if setup and setup.poi_low is not None else None
    t2_levels = ([("Цел 2 · седм. съпротива", target2(setup), fmt_pnl(plan["t2"]))]
                 if setup and target2(setup) and target2(setup) > (res_lvl or 0) else [])
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
    install_chart_scripts()
    height = CHART_HEIGHTS.get(st.session_state.get("ph_chart_h") or "M", 560)

    tab_w, tab_d, tab_4h = st.tabs(["W", "D", "4h"], default="D")
    with tab_w:
        weekly = resample_ohlc(daily, "W")
        levels = t2_levels + ([("Цел 1 · дневна съпротива", res_lvl, target1_note), ("Дневна подкрепа", sup_lvl)]
                            if sup_lvl and res_lvl else []) + stop + entry
        swo_w = st.session_state.get("ph_swo_w", 2)
        fig = candle_figure(weekly, levels, visible_bars=104, swings_order=swo_w, height=height,
                            **structure_annotations(find_swing_points(weekly, order=swo_w)))
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=f"chart_w_{symbol}")
        st.caption("Тренд по седмичните BOS (бичи, докато цената е над силното дъно) и цел 2 - седмичният слаб връх.")
    with tab_d:
        levels = list(t2_levels)
        if sup_lvl and res_lvl:
            levels += [("Цел 1 · дневна съпротива", res_lvl, target1_note),
                       ("Equilibrium 50%", sup_lvl + (res_lvl - sup_lvl) / 2), ("Дневна подкрепа", sup_lvl)]
        fig = candle_figure(daily, levels + stop + entry, poi=poi, visible_bars=130, swings_order=order, height=height,
                            **structure_annotations(find_swing_points(daily, order=order)))
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=f"chart_d_{symbol}")
    with tab_4h:
        intraday = fetch_ohlc_batch((symbol,), "60d", "60m").get(symbol)
        if intraday is None or intraday.empty:
            st.info("Няма 4ч данни за този инструмент.")
        else:
            h4 = resample_session_halves(intraday)
            levels = ([(f"CHoCH ниво ({setup.choch_tf})" if setup.choch_tf != "4ч" else "CHoCH ниво", setup.choch_level)]
                      if setup else []) + stop + entry
            fig = candle_figure(h4, levels, poi=poi, visible_bars=60, categorical=True, swings_order=1, height=height,
                                **choch_annotations(find_swing_points(h4, order=1), setup.pullback_start if setup else None))
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key=f"chart_4h_{symbol}")
            if setup:
                st.caption("Phase B: вход при затваряне над CHoCH нивото" + (" - ✓ вече пробито" if setup.choch_now else "")
                           + " · Phase A: вход в POI зоната.")
