"""НЕИЗПОЛЗВАНИ стратегии (не се викат от UI): дневна (pullback/обръщане),
Multi-Timeframe и Supply & Demand. Запазени по решение на собственика - не трий
без изрично съгласие."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from ai_client import call_claude
from indicators import (
    average_true_range, compute_macd, compute_rsi, find_fresh_demand_zone, find_swing_points,
    flatten_columns, is_overextended, resample_ohlc, structure_trend,
)
from ui_common import section_header
from universe import (
    apply_manual_universe, flag_macro_signal, load_universe, render_macro_section,
    render_manual_universe_editor, render_universe_search, render_universe_uploader,
)

@st.cache_data(ttl=3600)
def fetch_market_data_daily(tickers: dict):
    summary_list = []
    charts_data = {}

    for name, symbol in tickers.items():
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50:
                continue
            df = flatten_columns(df)

            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['RSI'] = compute_rsi(df['Close'], 14)
            df['MACD_Hist'] = compute_macd(df['Close'])
            df['Vol20'] = df['Volume'].rolling(window=20).mean()

            latest, prev = df.iloc[-1], df.iloc[-2]
            close_price = float(latest['Close'])
            change_pct = ((close_price - float(prev['Close'])) / float(prev['Close'])) * 100

            ema20_val = float(latest['EMA20'])
            ema50_val = float(latest['EMA50'])
            ema200_val = float(latest['EMA200'])
            rsi_val = float(latest['RSI'])
            macd_val = float(latest['MACD_Hist'])
            vol20 = float(latest['Vol20'])
            vol_surge = float(latest['Volume']) / vol20 if vol20 > 0 else 0

            is_uptrend = (ema50_val > ema200_val) and (close_price > ema200_val)
            is_downtrend = ema50_val < ema200_val

            if is_uptrend:
                trend = "↗ Възходящ"
            elif is_downtrend:
                trend = "↘ Низходящ"
            else:
                trend = "→ Консолидация"

            diff_ema50 = ((close_price - ema50_val) / ema50_val) * 100
            diff_ema200 = ((close_price - ema200_val) / ema200_val) * 100

            # --- Потвърдено обръщане ---
            recent_15d = df.tail(15)
            had_extreme_oversold = bool((recent_15d['RSI'] < 30).any())
            recovered_from_oversold = rsi_val > 35
            reclaimed_ema20 = close_price > ema20_val

            recent_5d_macd = df['MACD_Hist'].tail(5)
            macd_bullish_cross = bool((recent_5d_macd.iloc[:-1] < 0).any()) and macd_val > 0
            volume_confirms = vol_surge >= 1.2

            weekly_close = df['Close'].resample('W').last().dropna()
            weekly_rsi_series = compute_rsi(weekly_close, window=14)
            weekly_rsi_val = (
                float(weekly_rsi_series.iloc[-1])
                if not weekly_rsi_series.empty and not pd.isna(weekly_rsi_series.iloc[-1])
                else None
            )
            weekly_sma40 = weekly_close.rolling(window=40, min_periods=30).mean()
            weekly_sma40_val = (
                float(weekly_sma40.iloc[-1])
                if not weekly_sma40.empty and not pd.isna(weekly_sma40.iloc[-1])
                else None
            )

            weekly_confirms = True
            if weekly_rsi_val is not None:
                weekly_confirms = weekly_rsi_val > 45
            if weekly_confirms and weekly_sma40_val is not None:
                weekly_confirms = weekly_confirms and (close_price >= weekly_sma40_val * 0.85)

            # --- Pullback ---
            if len(df) >= 11:
                ema50_10d_ago = float(df['EMA50'].iloc[-11])
                ema50_rising = ema50_val > ema50_10d_ago
            else:
                ema50_rising = True

            near_or_below_ema50 = -3.5 <= diff_ema50 <= 0.5
            low_volume_pullback = vol_surge <= 1.1

            recent_10d_rsi = df['RSI'].tail(10)
            rsi_recently_dropped = bool((recent_10d_rsi > 55).any()) and rsi_val < 48

            pullback_cond = (
                is_uptrend and ema50_rising and near_or_below_ema50
                and low_volume_pullback and rsi_recently_dropped
            )

            reversal_cond = (
                is_downtrend and had_extreme_oversold and recovered_from_oversold
                and reclaimed_ema20 and macd_bullish_cross and volume_confirms and weekly_confirms
            )

            in_buy_zone = pullback_cond or reversal_cond

            summary_list.append({
                "Име": name, "Тикер": symbol, "Цена (€)": round(close_price, 2),
                "Промяна (%)": round(change_pct, 2), "Тренд": trend,
                "RSI": round(rsi_val, 1),
                "От EMA50 (%)": round(diff_ema50, 2), "От EMA200 (%)": round(diff_ema200, 2),
                "MACD Hist": round(macd_val, 3), "Обем (x Средния)": round(vol_surge, 2),
                "Бай Зона": in_buy_zone,
                "Потвърдено обръщане": reversal_cond,
                "Седмичен RSI": round(weekly_rsi_val, 1) if weekly_rsi_val is not None else None,
            })
            charts_data[name] = df
        except Exception:
            continue
    return pd.DataFrame(summary_list), charts_data


def generate_ai_analysis_daily(df_data: pd.DataFrame, api_key: str) -> str:

    pullback_candidates = df_data[
        (df_data["Тренд"] == "↗ Възходящ") & (df_data["Бай Зона"] == True) & (df_data["Потвърдено обръщане"] == False)
    ].sort_values(by="RSI").head(10)

    reversal_candidates = df_data[df_data["Потвърдено обръщане"] == True].sort_values(
        by="Обем (x Средния)", ascending=False
    ).head(10)

    pullback_text = (
        pullback_candidates.to_string(index=False)
        if not pullback_candidates.empty
        else "НЯМА кандидати днес, които да отговарят на условията за pullback."
    )
    reversal_text = (
        reversal_candidates.to_string(index=False)
        if not reversal_candidates.empty
        else "НЯМА кандидати днес, които да отговарят на условията за потвърдено обръщане."
    )

    prompt = f"""
    Ти си професионален суинг търговец. Използвай СТРИКТНО само данните по-долу -
    те вече са преминали строги технически филтри в кода, ти НЕ преценяваш сам
    дали инструмент отговаря на условията, само интерпретираш готовите резултати.

    === КАТЕГОРИЯ 1 кандидати: "За бърз суинг по тренда (Trend Following Pullback)" ===
    {pullback_text}

    === КАТЕГОРИЯ 2 кандидати: "Акумулиране при ПОТВЪРДЕНО обръщане (Confirmed Reversal)" ===
    {reversal_text}

    ЖЕЛЕЗНИ ПРАВИЛА:
    - Избирай ЕДИНСТВЕНО измежду инструментите, изредени по-горе за всяка категория.
      НЕ добавяй, НЕ предполагай и НЕ включвай никакъв друг инструмент.
    - Ако за дадена категория пише "НЯМА кандидати", напиши точно това - НЕ импровизирай замяна.
    - До 5 инструмента на категория.

    За всяка избрана акция бъди ясен с булети:
    - Обясни защо техническият ѝ сетъп е добър.
    - Посочи ценови нива за влизане с 2 лимитирани транша.

    МНОГО ВАЖНО - ОБОБЩАВАЩА ТАБЛИЦА (ТЪРГОВСКИ ПЛАН):
    В самия край, генерирай Markdown таблица само за реално избраните инструменти
    (пропусни таблицата, ако и двете категории са празни).

    Изисквания за колоните:
    1. **Инструмент:** Име, тикер под него в наклонен шрифт (`Apple <br> *APC.DE*`).
    2. **Категория:** Бърз суинг ИЛИ Потвърдено обръщане.
    3. **Транш 1 (Вход):** Цена и % от капитала (`150.00 € (40%)`).
    4. **Транш 2 (Вход):** Цена и % от капитала (`142.00 € (60%)`).
    5. **Цел (Take Profit):** Цена и очакван % печалба (`165.00 € (+12%)`).
    """
    return call_claude(prompt, api_key, max_tokens=4096)


def render_daily_strategy():
    with st.expander("⚙️ Настройки на скрининга", expanded=False):
        max_instr = st.slider("Максимален брой инструменти", 20, 500, 150, step=20, key="daily_max")
        pinned_input = st.text_input(
            "Винаги включвай (имена, разделени със запетая)", value="Gold, Silver", key="daily_pinned",
            help="Тези инструменти винаги влизат в сканирането, дори извън обичайния лимит по-горе.",
        )
        uploaded_universe = render_universe_uploader(key="daily")
    manual_keywords = tuple(k.strip() for k in pinned_input.split(",") if k.strip())
    auto_pin, macro_keywords = render_macro_section(key="daily")
    pinned_keywords = tuple(dict.fromkeys(manual_keywords + tuple(macro_keywords))) if auto_pin else manual_keywords

    if uploaded_universe:
        tickers = uploaded_universe
        st.caption(f"Универс (от качения фундаментален списък): {len(tickers)} инструмента")
    else:
        tickers = load_universe(max_instruments=max_instr, pinned_keywords=pinned_keywords)
        st.caption(f"Универс: {len(tickers)} инструмента")
    manual_universe, scan_only_manual = render_manual_universe_editor(key="daily")
    tickers = apply_manual_universe(tickers, manual_universe, scan_only_manual)
    render_universe_search(key="daily")

    with st.spinner("Синхронизиране и търсене на суинг възможности..."):
        df_summary, charts_data = fetch_market_data_daily(tickers)

    if df_summary.empty:
        st.info("Няма данни - провери универса/интернет връзката.")
        return

    df_summary = flag_macro_signal(df_summary, macro_keywords)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Възможности (Бай Зона)", len(df_summary[df_summary["Бай Зона"] == True]))
    with col2:
        top_vol = df_summary.loc[df_summary["Обем (x Средния)"].idxmax()]
        st.metric("Институционален Обем", f"{top_vol['Тикер']}", f"{top_vol['Обем (x Средния)']}x")
    with col3:
        top_rsi = df_summary.loc[df_summary["RSI"].idxmin()]
        st.metric("Най-свръхпродадена (RSI)", f"{top_rsi['Тикер']}", f"{top_rsi['RSI']}")
    with col4:
        bullish = len(df_summary[df_summary["Тренд"] == "↗ Възходящ"])
        st.metric("Активи във възходящ тренд", bullish)
    with col5:
        media_buy = len(df_summary[(df_summary["Бай Зона"] == True) & (df_summary["📰 Медиен сигнал"] == True)])
        st.metric("Бай Зона + медиен сигнал", media_buy)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🤖 AI АНАЛИЗ И ПЛАН", "📊 ПЪЛНА ТАБЛИЦА", "📈 ИНТЕРАКТИВНА ГРАФИКА"])

    with tab1:
        anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if not anthropic_api_key:
            anthropic_api_key = st.text_input("Anthropic API Key", type="password", key="daily_key")
        if st.button("🚀 Генерирай Анализ и Търговски План", type="primary", width="stretch"):
            if not anthropic_api_key:
                st.error("Липсва Anthropic API ключ!")
            else:
                with st.spinner("Claude анализира данните..."):
                    try:
                        st.markdown(generate_ai_analysis_daily(df_summary, anthropic_api_key), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Грешка: {e}")

    with tab2:
        display_df = df_summary.drop(columns=["Бай Зона"])
        st.dataframe(
            display_df, width="stretch", hide_index=True,
            column_config={
                "RSI": st.column_config.ProgressColumn("RSI (Моментум)", format="%.1f", min_value=0, max_value=100),
                "Обем (x Средния)": st.column_config.NumberColumn("Обем (x Средния)", format="%.2fx"),
                "Потвърдено обръщане": st.column_config.CheckboxColumn("Потвърдено обръщане"),
                "📰 Медиен сигнал": st.column_config.CheckboxColumn("📰 Медиен сигнал"),
            },
        )

    with tab3:
        selected_name = st.selectbox("Избери инструмент за анализ:", list(tickers.keys()), key="daily_chart")
        if selected_name in charts_data:
            df_chart = charts_data[selected_name].tail(150)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Цена'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA20'], line=dict(color='white', width=1, dash='dot'), name='EMA 20'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA50'], line=dict(color='orange', width=1.5), name='EMA 50'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA200'], line=dict(color='blue', width=1.5), name='EMA 200'))
            colors = ['green' if val >= 0 else 'red' for val in df_chart['MACD_Hist']]
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['MACD_Hist'], marker_color=colors, name='MACD', yaxis='y2', opacity=0.3))
            fig.update_layout(
                yaxis_title="Цена (€)", xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10), height=550, template="plotly_dark",
                yaxis2=dict(title="MACD", overlaying='y', side='right', showgrid=False,
                            range=[df_chart['MACD_Hist'].min() * 3, df_chart['MACD_Hist'].max() * 3]),
            )
            st.plotly_chart(fig, width="stretch")


@st.cache_data(ttl=3600)
def analyze_instrument_mtf(name: str, symbol: str, swing_order_weekly: int, swing_order_daily: int):
    daily_df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if daily_df.empty or len(daily_df) < 150:
        return None
    daily_df = flatten_columns(daily_df)

    # --- 1. Седмичен тренд (твърд филтър) ---
    weekly = resample_ohlc(daily_df, "W")
    if len(weekly) < 20:
        return None
    weekly = find_swing_points(weekly, order=swing_order_weekly)
    weekly_uptrend, weekly_resistance, weekly_support = structure_trend(weekly)
    if not weekly_uptrend or weekly_support is None or weekly_resistance is None:
        return None

    weekly_range = weekly_resistance - weekly_support
    if weekly_range <= 0:
        return None

    # --- 2. Цената в долната половина на седмичния диапазон (твърд филтър) ---
    current_price = float(daily_df["Close"].iloc[-1])
    weekly_lower_half_top = weekly_support + weekly_range / 2
    in_weekly_lower_half = weekly_support <= current_price <= weekly_lower_half_top
    if not in_weekly_lower_half:
        return None

    # --- 3. Дневен тренд по структура (твърд филтър) ---
    daily_swings = find_swing_points(daily_df, order=swing_order_daily)
    daily_uptrend, daily_resistance, daily_support = structure_trend(daily_swings)
    if not daily_uptrend or daily_support is None or daily_resistance is None:
        return None

    daily_range = daily_resistance - daily_support
    if daily_range <= 0:
        return None

    # --- 4. Цената в долната третина на ДНЕВНИЯ диапазон ---
    daily_lower_third_top = daily_support + daily_range / 3
    in_daily_lower_third = daily_support <= current_price <= daily_lower_third_top

    # --- 5. 4-часова структура: потвърждава ли up-тренд? ---
    h4_uptrend = False
    try:
        intraday = yf.download(symbol, period="60d", interval="60m", progress=False, auto_adjust=True)
        if not intraday.empty:
            intraday = flatten_columns(intraday)
            h4 = resample_ohlc(intraday, "4h")
            if len(h4) >= 10:
                h4 = find_swing_points(h4, order=1)
                h4_uptrend, _, _ = structure_trend(h4)
    except Exception:
        pass

    # "Готов за вход" изисква и двете: цената в прецизната зона (долна третина
    # от дневния диапазон) И 4ч структура да потвърждава начеващ up-тренд.
    ready = in_daily_lower_third and h4_uptrend

    risk = current_price - daily_support
    reward_daily = daily_resistance - current_price
    reward_weekly = weekly_resistance - current_price
    rr_daily = (reward_daily / risk) if risk > 0 else None
    rr_weekly = (reward_weekly / risk) if risk > 0 else None

    return {
        "Име": name, "Тикер": symbol, "Цена (€)": round(current_price, 2),
        "Седм. подкрепа": round(weekly_support, 2), "Седм. съпротива": round(weekly_resistance, 2),
        "Дневна подкрепа": round(daily_support, 2), "Дневна съпротива": round(daily_resistance, 2),
        "В долна 1/3 (дневно)": in_daily_lower_third,
        "4ч up-тренд": h4_uptrend,
        "Risk (до дневна подкрепа)": round(risk, 2),
        "R/R (до дневна съпротива)": round(rr_daily, 2) if rr_daily else None,
        "R/R (до седмична съпротива)": round(rr_weekly, 2) if rr_weekly else None,
        "Готов за вход": ready,
    }


def generate_ai_analysis_mtf(df_ready: pd.DataFrame, df_watch: pd.DataFrame, api_key: str) -> str:

    ready_text = (
        df_ready.to_string(index=False) if not df_ready.empty
        else "НЯМА инструменти с пълно потвърждение на трите таймфрейма в момента."
    )
    watch_text = (
        df_watch.to_string(index=False) if not df_watch.empty
        else "НЯМА инструменти в зона на подкрепа в момента."
    )

    prompt = f"""
    Ти си професионален суинг търговец, ползващ каскадна multi-timeframe стратегия:
    1) седмичен тренд по структура (HH+HL) трябва да е възходящ, и цената да е
       в долната половина на седмичния диапазон подкрепа-съпротива;
    2) дневен тренд по структура ТРЯБВА също да е възходящ;
    3) цената трябва да е в долната третина на ДНЕВНИЯ диапазон подкрепа-съпротива;
    4) 4-часова структура потвърждава начеващ up-тренд точно в тази прецизна зона.

    Използвай СТРИКТНО само данните по-долу.

    === ГОТОВИ ЗА ВХОД ===
    {ready_text}

    === WATCHLIST (седмичен+дневен тренд ОК, но точката за вход на 4ч ОЩЕ НЕ е потвърдена) ===
    {watch_text}

    ЖЕЛЕЗНИ ПРАВИЛА:
    - Избирай ЕДИНСТВЕНО измежду инструментите по-горе.
    - Ако категория е празна, кажи го ясно - не импровизирай замяна.
    - За Watchlist обясни какво чакаме да видим на 4ч, за да минат в "готови".

    За "ГОТОВИ ЗА ВХОД": кратко обяснение, 2 транша за вход, Stop/Target от данните.
    Бъди кратък, удобен за преглед на телефон.
    """
    return call_claude(prompt, api_key, max_tokens=4096)


@st.cache_data(ttl=3600)
def analyze_instrument_sd(name: str, symbol: str, swing_order_weekly: int, swing_order_daily: int):
    daily_df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if daily_df.empty or len(daily_df) < 150:
        return None
    daily_df = flatten_columns(daily_df)

    # --- 1. Седмичен тренд (твърд филтър - "търгувай само по посока на HTF") ---
    weekly = resample_ohlc(daily_df, "W")
    if len(weekly) < 20:
        return None
    weekly = find_swing_points(weekly, order=swing_order_weekly)
    weekly_uptrend, weekly_resistance, weekly_support = structure_trend(weekly)
    if not weekly_uptrend:
        return None

    # --- 2. Fresh demand зона на дневен таймфрейм (2:1 imbalance + силна база) ---
    daily_swings_full = find_swing_points(daily_df, order=swing_order_daily)
    zone = find_fresh_demand_zone(daily_df, max_base=6)
    if zone is None:
        return None
    if zone["retests"] != 0:
        return None  # само FRESH зони - твърдо правило, не "used up"
    if not (zone["base_width"] <= 6 and zone["avg_base_body_ratio"] <= 0.5 and zone["imbalance_ratio"] >= 2.0):
        return None

    current_price = float(daily_df["Close"].iloc[-1])
    proximal, distal = zone["proximal"], zone["distal"]
    price_in_zone = distal <= current_price <= proximal
    approaching = proximal < current_price <= proximal * 1.05
    if not (price_in_zone or approaching):
        return None

    # --- 3. Пренатегнатост ("спри след 3+ CP модела в една посока") ---
    overextended, extension_count = is_overextended(daily_swings_full)

    # --- 4. Risk/Reward спрямо най-близкия swing high над зоната (мин. 3:1) ---
    swing_highs_above = daily_swings_full.loc[
        (daily_swings_full["SwingHigh"]) & (daily_swings_full["High"] > current_price), "High"
    ]
    target = float(swing_highs_above.min()) if not swing_highs_above.empty else weekly_resistance
    risk = current_price - distal
    reward = (target - current_price) if target else None
    rr = (reward / risk) if (risk and risk > 0 and reward and reward > 0) else None
    if not rr or rr < 3.0:
        return None

    # --- 5. 4-часова структура: потвърждава ли бичи реакция точно сега? ---
    h4_bullish_reaction = False
    try:
        intraday = yf.download(symbol, period="60d", interval="60m", progress=False, auto_adjust=True)
        if not intraday.empty:
            intraday = flatten_columns(intraday)
            h4 = resample_ohlc(intraday, "4h")
            if len(h4) >= 10:
                h4 = find_swing_points(h4, order=1)
                h4_bullish_reaction, _, _ = structure_trend(h4)
    except Exception:
        pass

    ready = price_in_zone and h4_bullish_reaction and not overextended

    # --- 6. Очакван хоризонт (груба ATR оценка, ориентир за ~3-седмичен суинг) ---
    atr = average_true_range(daily_df, period=14)
    est_days = round(reward / atr) if atr and atr > 0 else None
    within_3_weeks = est_days is not None and est_days <= 15

    return {
        "Име": name, "Тикер": symbol, "Цена (€)": round(current_price, 2),
        "Зона (Distal–Proximal)": f"{round(distal, 2)}–{round(proximal, 2)}",
        "Imbalance": f"{zone['imbalance_ratio']:.1f}x", "База (свещи)": zone["base_width"],
        "R/R (до цел)": round(rr, 2), "Очаквано (дни, ATR)": est_days,
        "≤3 седмици": within_3_weeks, "Пренатегнат": overextended,
        "4ч бичи реакция": h4_bullish_reaction, "В зоната сега": price_in_zone,
        "Готов за вход": ready,
    }


def generate_ai_analysis_sd(df_ready: pd.DataFrame, df_watch: pd.DataFrame, api_key: str) -> str:

    ready_text = df_ready.to_string(index=False) if not df_ready.empty else "НЯМА зони, готови за вход в момента."
    watch_text = df_watch.to_string(index=False) if not df_watch.empty else "НЯМА зони на watchlist в момента."

    prompt = f"""
    Ти си суинг търговец, ползващ Supply & Demand методологията на Alfonso Mores
    (Set & Forget), адаптирана за Седмичен → Дневен → 4ч каскада, само LONG
    (demand зони), с хоризонт на сделката до ~3 седмици:
    1) седмичен тренд по структура (HH+HL) трябва да е възходящ;
    2) на дневен таймфрейм трябва да има FRESH (0 ретеста) demand зона:
       база от макс. 6 тесни свещи (тяло <= 50% от диапазона), последвана от
       departure импулс с 2:1 imbalance (диапазон >= 2x средния на базата);
    3) минимум 3:1 Risk/Reward до най-близкия swing high над зоната;
    4) без пренатоварен тренд (спри след 3+ поредни по-високи върха);
    5) 4-часова структура потвърждава бичи реакция точно сега.

    Използвай СТРИКТНО само данните по-долу.

    === ГОТОВИ ЗА ВХОД ===
    {ready_text}

    === WATCHLIST (fresh зона има, но 4ч реакция ОЩЕ НЕ е потвърдена / цената приближава) ===
    {watch_text}

    ЖЕЛЕЗНИ ПРАВИЛА:
    - Избирай ЕДИНСТВЕНО измежду инструментите по-горе.
    - Ако категория е празна, кажи го ясно - не импровизирай замяна.
    - "Очаквано (дни, ATR)" е груба ориентировъчна оценка, не гаранция -
      представи я като такава.
    - За Watchlist обясни какво чакаме да видим на 4ч, за да минат в "готови".

    За "ГОТОВИ ЗА ВХОД": обясни зоната (proximal/distal), Stop под distal линията,
    Target от данните, защо отговаря на 3:1. Бъди кратък, удобен за телефон.
    """
    return call_claude(prompt, api_key, max_tokens=4096)


def render_sd_strategy():
    with st.expander("⚙️ Настройки на скрининга", expanded=False):
        max_instr = st.slider("Максимален брой инструменти", 20, 500, 250, step=20, key="sd_max")
        pinned_input = st.text_input(
            "Винаги включвай (имена, разделени със запетая)", value="Gold, Silver", key="sd_pinned",
            help="Тези инструменти винаги влизат в сканирането, дори извън обичайния лимит по-горе.",
        )
        swing_order_weekly = st.slider("Чувствителност на седмичните swing точки", 1, 4, 2, key="sd_swo_w")
        swing_order_daily = st.slider("Чувствителност на дневните swing точки", 2, 6, 3, key="sd_swo_d")
        uploaded_universe = render_universe_uploader(key="sd")
    manual_keywords = tuple(k.strip() for k in pinned_input.split(",") if k.strip())
    auto_pin, macro_keywords = render_macro_section(key="sd")
    pinned_keywords = tuple(dict.fromkeys(manual_keywords + tuple(macro_keywords))) if auto_pin else manual_keywords

    if uploaded_universe:
        tickers = uploaded_universe
        st.caption(f"Универс (от качения фундаментален списък): {len(tickers)} инструмента")
    else:
        tickers = load_universe(max_instruments=max_instr, pinned_keywords=pinned_keywords)
        st.caption(f"Универс: {len(tickers)} инструмента")
    manual_universe, scan_only_manual = render_manual_universe_editor(key="sd")
    tickers = apply_manual_universe(tickers, manual_universe, scan_only_manual)
    render_universe_search(key="sd")

    if st.button("🔎 Сканирай пазара", type="primary", width="stretch", key="sd_scan_btn"):
        results, watch_list = [], []
        progress = st.progress(0, text="Търсене на fresh demand зони...")
        items = list(tickers.items())
        for i, (name, symbol) in enumerate(items):
            res = analyze_instrument_sd(name, symbol, swing_order_weekly, swing_order_daily)
            if res:
                (results if res["Готов за вход"] else watch_list).append(res)
            progress.progress((i + 1) / len(items), text=f"Проверих {i + 1}/{len(items)}: {name}")
        progress.empty()
        st.session_state["sd_results"] = results
        st.session_state["sd_watch"] = watch_list

    results = st.session_state.get("sd_results", [])
    watch_list = st.session_state.get("sd_watch", [])

    st.divider()
    section_header(
        "✅ Готови за вход", status="go",
        subtitle="Fresh demand зона + 4ч бичи реакция + 3:1 R/R + без пренатоварен тренд",
    )
    if results:
        df_ready = pd.DataFrame(results).drop(columns=["Готов за вход"])
        df_ready = flag_macro_signal(df_ready, macro_keywords)
        st.dataframe(
            df_ready, width="stretch", hide_index=True,
            column_config={
                "📰 Медиен сигнал": st.column_config.CheckboxColumn("📰 Медиен сигнал"),
                "≤3 седмици": st.column_config.CheckboxColumn("≤3 седмици"),
                "Пренатегнат": st.column_config.CheckboxColumn("Пренатегнат"),
            },
        )
    else:
        df_ready = pd.DataFrame()
        st.info("Няма fresh demand зони с пълно потвърждение в момента.")

    st.divider()
    section_header("👀 Watchlist", status="watch", subtitle="Fresh зона има, чакаме 4ч потвърждение или доближаване до зоната")
    if watch_list:
        df_watch = pd.DataFrame(watch_list).drop(columns=["Готов за вход"])
        df_watch = flag_macro_signal(df_watch, macro_keywords)
        st.dataframe(
            df_watch, width="stretch", hide_index=True,
            column_config={"📰 Медиен сигнал": st.column_config.CheckboxColumn("📰 Медиен сигнал")},
        )
    else:
        df_watch = pd.DataFrame()
        st.info("Няма инструменти на watchlist в момента.")

    st.divider()
    section_header("🤖 AI Анализ", status="info")
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not anthropic_api_key:
        anthropic_api_key = st.text_input("Anthropic API Key", type="password", key="sd_key")
    if st.button("🚀 Генерирай Анализ и Търговски План", type="primary", width="stretch", key="sd_ai_btn"):
        if not anthropic_api_key:
            st.error("Липсва Anthropic API ключ!")
        else:
            with st.spinner("Анализирам зоните..."):
                try:
                    analysis = generate_ai_analysis_sd(df_ready, df_watch, anthropic_api_key)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Грешка: {e}")

    st.divider()
    section_header("📈 Преглед на графика", status="info")
    if tickers:
        selected_name = st.selectbox("Избери инструмент", list(tickers.keys()), key="sd_chart_select")
        symbol = tickers[selected_name]
        daily = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if not daily.empty:
            daily = flatten_columns(daily)
            zone = find_fresh_demand_zone(daily, max_base=6)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=daily.index, open=daily["Open"], high=daily["High"], low=daily["Low"], close=daily["Close"],
                name="Дневна цена",
            ))
            if zone:
                fig.add_hrect(
                    y0=zone["distal"], y1=zone["proximal"], fillcolor="#3DDC97", opacity=0.18, line_width=1,
                    line_color="#3DDC97",
                    annotation_text=f"Demand зона (retests: {zone['retests']}, {zone['imbalance_ratio']:.1f}x imbalance)",
                )
            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")


def render_mtf_strategy():
    with st.expander("⚙️ Настройки на скрининга", expanded=False):
        max_instr = st.slider("Максимален брой инструменти", 20, 500, 300, step=20, key="mtf_max")
        pinned_input = st.text_input(
            "Винаги включвай (имена, разделени със запетая)", value="Gold, Silver", key="mtf_pinned",
            help="Тези инструменти винаги влизат в сканирането, дори извън обичайния лимит по-горе.",
        )
        swing_order_weekly = st.slider("Чувствителност на седмичните swing точки", 1, 4, 2)
        swing_order_daily = st.slider("Чувствителност на дневните swing точки", 2, 6, 3)
        uploaded_universe = render_universe_uploader(key="mtf")
    manual_keywords = tuple(k.strip() for k in pinned_input.split(",") if k.strip())
    auto_pin, macro_keywords = render_macro_section(key="mtf")
    pinned_keywords = tuple(dict.fromkeys(manual_keywords + tuple(macro_keywords))) if auto_pin else manual_keywords

    if uploaded_universe:
        tickers = uploaded_universe
        st.caption(f"Универс (от качения фундаментален списък): {len(tickers)} инструмента")
    else:
        tickers = load_universe(max_instruments=max_instr, pinned_keywords=pinned_keywords)
        st.caption(f"Универс: {len(tickers)} инструмента")
    manual_universe, scan_only_manual = render_manual_universe_editor(key="mtf")
    tickers = apply_manual_universe(tickers, manual_universe, scan_only_manual)
    render_universe_search(key="mtf")

    if st.button("🔍 Сканирай пазара", type="primary"):
        results, watch_list = [], []
        progress = st.progress(0.0, text="Стартиране на анализа...")
        items = list(tickers.items())
        for idx, (name, symbol) in enumerate(items):
            progress.progress((idx + 1) / len(items), text=f"Анализирам {name}...")
            res = analyze_instrument_mtf(name, symbol, swing_order_weekly, swing_order_daily)
            if res is not None:
                (results if res["Готов за вход"] else watch_list).append(res)
        progress.empty()
        st.session_state["mtf_results"] = results
        st.session_state["mtf_watchlist"] = watch_list

    results = st.session_state.get("mtf_results", [])
    watch_list = st.session_state.get("mtf_watchlist", [])

    st.divider()
    section_header("✅ Готови за вход", status="go", subtitle="Пълно потвърждение на седмичен + дневен + 4ч таймфрейм")
    if results:
        df_ready = pd.DataFrame(results).drop(columns=["Готов за вход"])
        df_ready = flag_macro_signal(df_ready, macro_keywords)
        st.dataframe(
            df_ready, width="stretch", hide_index=True,
            column_config={"📰 Медиен сигнал": st.column_config.CheckboxColumn("📰 Медиен сигнал")},
        )
    else:
        df_ready = pd.DataFrame()
        st.info("Няма инструменти с пълно потвърждение на трите таймфрейма в момента.")

    st.divider()
    section_header("👀 Watchlist", status="watch", subtitle="Минали седмичен + дневен филтър, чакат 4ч потвърждение")
    if watch_list:
        df_watch = pd.DataFrame(watch_list).drop(columns=["Готов за вход", "4ч up-тренд"])
        df_watch = flag_macro_signal(df_watch, macro_keywords)
        st.dataframe(
            df_watch, width="stretch", hide_index=True,
            column_config={"📰 Медиен сигнал": st.column_config.CheckboxColumn("📰 Медиен сигнал")},
        )
    else:
        df_watch = pd.DataFrame()
        st.info("Няма инструменти в зона на подкрепа в момента.")

    st.divider()
    section_header("🤖 AI Анализ", status="info")
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not anthropic_api_key:
        anthropic_api_key = st.text_input("Anthropic API Key", type="password", key="mtf_key")

    if st.button("Генерирай AI анализ на резултатите", type="primary"):
        if not anthropic_api_key:
            st.error("Липсва Anthropic API ключ!")
        elif df_ready.empty and df_watch.empty:
            st.warning("Първо натисни 'Сканирай пазара'.")
        else:
            with st.spinner("Claude анализира резултатите..."):
                try:
                    st.markdown(generate_ai_analysis_mtf(df_ready, df_watch, anthropic_api_key))
                except Exception as e:
                    st.error(f"Грешка: {e}")

    st.divider()
    section_header("📈 Преглед на графика", status="info")
    selected_name = st.selectbox("Избери инструмент:", list(tickers.keys()), key="mtf_chart")
    if selected_name:
        symbol = tickers[selected_name]
        daily = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if not daily.empty:
            daily = flatten_columns(daily)
            weekly = resample_ohlc(daily, "W")
            weekly = find_swing_points(weekly, order=swing_order_weekly)
            _, weekly_resistance, weekly_support = structure_trend(weekly)

            daily_swings = find_swing_points(daily, order=swing_order_daily)
            _, daily_resistance, daily_support = structure_trend(daily_swings)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=daily.index, open=daily["Open"], high=daily["High"], low=daily["Low"], close=daily["Close"], name="Дневна цена"))

            if weekly_support and weekly_resistance:
                weekly_mid = weekly_support + (weekly_resistance - weekly_support) / 2
                fig.add_hrect(y0=weekly_support, y1=weekly_mid, fillcolor="green", opacity=0.12, line_width=0, annotation_text="Седм. долна половина")
                fig.add_hline(y=weekly_resistance, line_dash="dot", line_color="red", annotation_text="Седм. съпротива")
                fig.add_hline(y=weekly_support, line_dash="dot", line_color="green", annotation_text="Седм. подкрепа")

            if daily_support and daily_resistance:
                daily_lower_third = daily_support + (daily_resistance - daily_support) / 3
                fig.add_hrect(y0=daily_support, y1=daily_lower_third, fillcolor="cyan", opacity=0.18, line_width=0, annotation_text="Дневна долна 1/3")
                fig.add_hline(y=daily_resistance, line_dash="dash", line_color="orange", annotation_text="Дневна съпротива")

            fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")
