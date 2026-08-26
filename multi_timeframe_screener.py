import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from anthropic import Anthropic
import json
from pathlib import Path

st.set_page_config(
    page_title="Swing Screener AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stTabs [data-baseweb="tab-list"] {gap: 10px;}
            .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 4px 4px 0px 0px; padding: 10px; border: 1px solid #333; border-bottom: none;}
            .stTabs [aria-selected="true"] {background-color: #2E5B88; color: white;}
            .stMarkdown table { width: 100%; border-collapse: collapse; background-color: #FFFFFF !important; }
            .stMarkdown th { background-color: #2E5B88 !important; color: white !important; font-size: 15px; padding: 12px 8px !important; border-bottom: 2px solid #ccc; }
            .stMarkdown td { padding: 12px 8px !important; border-bottom: 1px solid #eee; font-size: 14px; color: #000000 !important; }
            .stMarkdown tbody tr:nth-child(even) { background-color: #F8F9FA !important; }
            .stMarkdown tbody tr:nth-child(odd) { background-color: #FFFFFF !important; }
            .stMarkdown tbody tr:hover { background-color: #E2E6EA !important; transition: background-color 0.2s ease; }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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
def load_universe(max_instruments: int = 500, pinned_keywords: tuple = ()):
    """Зарежда универса за сканиране. Приоритет:
    1) закачени (pinned_keywords) инструменти - винаги от ПЪЛНИЯ eu_instruments.json,
       за да не пропуснем нищо, дори ако не са в месечната селекция;
    2) curated_universe.json (месечна селекция по ликвидност+моментум+медиен buzz),
       ако съществува;
    3) fallback - суровият ред от eu_instruments.json, ако все още няма curated файл."""
    pinned_keywords_lower = [kw.lower() for kw in pinned_keywords if kw.strip()]
    mapped = {}

    full_path = Path(INSTRUMENTS_FILE)
    full_instruments = None
    if full_path.exists():
        full_instruments = json.loads(full_path.read_text(encoding="utf-8")).get("instruments", [])

        if pinned_keywords_lower:
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
            mapped[label] = item["symbol"]
        st.caption(
            f"📅 Универс от месечна селекция (обновена: {curated_data.get('generated_at', '?')}, "
            f"медийно трендиращи: {curated_data.get('media_trending_count', 0)})"
        )
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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ============================================================================
# СТРАТЕГИЯ 1: ДНЕВНА (Pullback по тренда / Потвърдено обръщане)
# ============================================================================

def compute_macd(data, short=12, long=26, signal=9):
    ema_short = data.ewm(span=short, adjust=False).mean()
    ema_long = data.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


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
    client = Anthropic(api_key=api_key)

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
    response = client.messages.create(
        model="claude-sonnet-5", max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ValueError(f"Claude върна празен отговор (stop_reason: {response.stop_reason}). Опитай пак.")
    return text


@st.cache_data(ttl=6 * 3600)
def load_full_universe_for_search():
    """Отделна функция само за търсене - винаги чете суровия eu_instruments.json
    в пълен размер, независимо от месечната curated_universe.json селекция, за
    да можеш да провериш дали инструмент изобщо съществува в T212, дори да не
    е попаднал в тазмесечния топ 500."""
    path = Path(INSTRUMENTS_FILE)
    if not path.exists():
        return {}
    instruments = json.loads(path.read_text(encoding="utf-8")).get("instruments", [])
    mapped = {}
    for inst in instruments:
        suffix = exchange_to_yahoo_suffix(inst.get("exchangeName", ""))
        if suffix is None:
            continue
        yahoo_ticker = f"{inst.get('shortName', '')}{suffix}"
        label = f"{inst.get('shortName', inst['ticker'])} ({inst['name']})"
        mapped[label] = yahoo_ticker
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


def render_daily_strategy():
    with st.expander("⚙️ Настройки на скрининга", expanded=False):
        max_instr = st.slider("Максимален брой инструменти", 20, 500, 150, step=20, key="daily_max")
        pinned_input = st.text_input(
            "Винаги включвай (имена, разделени със запетая)", value="Gold, Silver", key="daily_pinned",
            help="Тези инструменти винаги влизат в сканирането, дори извън обичайния лимит по-горе.",
        )
    pinned_keywords = tuple(k.strip() for k in pinned_input.split(",") if k.strip())
    tickers = load_universe(max_instruments=max_instr, pinned_keywords=pinned_keywords)
    st.caption(f"Универс: {len(tickers)} инструмента")
    render_universe_search(key="daily")

    with st.spinner("Синхронизиране и търсене на суинг възможности..."):
        df_summary, charts_data = fetch_market_data_daily(tickers)

    if df_summary.empty:
        st.info("Няма данни - провери универса/интернет връзката.")
        return

    col1, col2, col3, col4 = st.columns(4)
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

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🤖 AI АНАЛИЗ И ПЛАН", "📊 ПЪЛНА ТАБЛИЦА", "📈 ИНТЕРАКТИВНА ГРАФИКА"])

    with tab1:
        anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if not anthropic_api_key:
            anthropic_api_key = st.text_input("Anthropic API Key", type="password", key="daily_key")
        if st.button("🚀 Генерирай Анализ и Търговски План", type="primary", use_container_width=True):
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
            display_df, use_container_width=True, hide_index=True,
            column_config={
                "RSI": st.column_config.ProgressColumn("RSI (Моментум)", format="%.1f", min_value=0, max_value=100),
                "Обем (x Средния)": st.column_config.NumberColumn("Обем (x Средния)", format="%.2fx"),
                "Потвърдено обръщане": st.column_config.CheckboxColumn("Потвърдено обръщане"),
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
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# СТРАТЕГИЯ 2: MULTI-TIMEFRAME (Седмичен тренд → зона на подкрепа → 4ч)
# ============================================================================

def find_swing_points(df: pd.DataFrame, order: int = 2) -> pd.DataFrame:
    df = df.copy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        window_h = highs[i - order: i + order + 1]
        window_l = lows[i - order: i + order + 1]
        if highs[i] == window_h.max():
            swing_high[i] = True
        if lows[i] == window_l.min():
            swing_low[i] = True
    df["SwingHigh"] = swing_high
    df["SwingLow"] = swing_low
    return df


def structure_trend(df: pd.DataFrame):
    highs = df.loc[df["SwingHigh"], "High"]
    lows = df.loc[df["SwingLow"], "Low"]
    if len(highs) < 2 or len(lows) < 2:
        return False, None, None
    higher_high = highs.iloc[-1] > highs.iloc[-2]
    higher_low = lows.iloc[-1] > lows.iloc[-2]
    return (higher_high and higher_low), float(highs.iloc[-1]), float(lows.iloc[-1])


@st.cache_data(ttl=3600)
def analyze_instrument_mtf(name: str, symbol: str, swing_order_weekly: int, swing_order_daily: int):
    daily_df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if daily_df.empty or len(daily_df) < 150:
        return None
    daily_df = flatten_columns(daily_df)

    # --- 1. Седмичен тренд (твърд филтър) ---
    weekly = (
        daily_df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    )
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
            h4 = (
                intraday.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            )
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
    client = Anthropic(api_key=api_key)

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
    response = client.messages.create(
        model="claude-sonnet-5", max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ValueError(f"Claude върна празен отговор (stop_reason: {response.stop_reason}). Опитай пак.")
    return text


def render_mtf_strategy():
    with st.expander("⚙️ Настройки на скрининга", expanded=False):
        max_instr = st.slider("Максимален брой инструменти", 20, 500, 300, step=20, key="mtf_max")
        pinned_input = st.text_input(
            "Винаги включвай (имена, разделени със запетая)", value="Gold, Silver", key="mtf_pinned",
            help="Тези инструменти винаги влизат в сканирането, дори извън обичайния лимит по-горе.",
        )
        swing_order_weekly = st.slider("Чувствителност на седмичните swing точки", 1, 4, 2)
        swing_order_daily = st.slider("Чувствителност на дневните swing точки", 2, 6, 3)
    pinned_keywords = tuple(k.strip() for k in pinned_input.split(",") if k.strip())

    tickers = load_universe(max_instruments=max_instr, pinned_keywords=pinned_keywords)
    st.caption(f"Универс: {len(tickers)} инструмента")
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
    st.subheader("✅ Готови за вход")
    if results:
        df_ready = pd.DataFrame(results).drop(columns=["Готов за вход"])
        st.dataframe(df_ready, use_container_width=True, hide_index=True)
    else:
        df_ready = pd.DataFrame()
        st.info("Няма инструменти с пълно потвърждение на трите таймфрейма в момента.")

    st.divider()
    st.subheader("👀 Watchlist")
    if watch_list:
        df_watch = pd.DataFrame(watch_list).drop(columns=["Готов за вход", "4ч up-тренд"])
        st.dataframe(df_watch, use_container_width=True, hide_index=True)
    else:
        df_watch = pd.DataFrame()
        st.info("Няма инструменти в зона на подкрепа в момента.")

    st.divider()
    st.subheader("🤖 AI Анализ")
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
    st.subheader("📈 Преглед на графика")
    selected_name = st.selectbox("Избери инструмент:", list(tickers.keys()), key="mtf_chart")
    if selected_name:
        symbol = tickers[selected_name]
        daily = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        if not daily.empty:
            daily = flatten_columns(daily)
            weekly = (
                daily.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
            )
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
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# ИНТЕРФЕЙС: избор на стратегия
# ============================================================================

st.title("⚡ Swing Screener AI")

strategy = st.radio(
    "Избери стратегия за скрининг:",
    ["📅 Дневна (Pullback / Потвърдено обръщане)", "🎯 Multi-Timeframe (Седмичен → Дневен → 4ч)"],
    horizontal=True,
)

st.markdown("---")

if strategy.startswith("📅"):
    render_daily_strategy()
else:
    render_mtf_strategy()
