import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from anthropic import Anthropic

st.set_page_config(
    page_title="Multi-Timeframe Swing Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Списък с инструменти (същият, който вече ползваме) ------------------
TICKERS = {
    "SXRV (iShares Nasdaq 100)": "SXRV.DE", "SXR8 (iShares Core S&P 500)": "SXR8.DE",
    "SEC0 (iShares Semiconductor)": "SEC0.DE",
    "WGLD (WisdomTree Physical Gold)": "WGLD.DE", "PHAG (WisdomTree Physical Silver)": "PHAG.DE",
    "URNU (Global X Uranium)": "URNU.DE", "COPX (Global X Copper Miners)": "COPX.DE",
    "APC (Apple)": "APC.DE", "MSF (Microsoft)": "MSF.DE", "ABE (Alphabet/Google)": "ABE.DE",
    "AMZ (Amazon)": "AMZ.DE", "NVD (Nvidia)": "NVD.DE", "FB2A (Meta/Facebook)": "FB2A.DE",
    "TL0 (Tesla)": "TL0.DE", "AMD (AMD)": "AMD.DE", "14B (Broadcom)": "14B.DE",
    "QCI (Qualcomm)": "QCI.DE", "CRM (Salesforce)": "CRM.DE", "ORC (Oracle)": "ORC.DE",
    "ELY (Eli Lilly)": "ELY.DE", "NOVC (Novo Nordisk)": "NOVC.DE", "JNJ (Johnson & Johnson)": "JNJ.DE",
    "PFE (Pfizer)": "PFE.DE", "UNH (UnitedHealth)": "UNH.DE", "SAN (Sanofi)": "SAN.PA",
    "3V64 (Visa)": "3V64.DE", "M4I (Mastercard)": "M4I.DE", "NFC (Netflix)": "NFC.DE",
    "DIS (Walt Disney)": "DIS.DE", "WMT (Walmart)": "WMT.DE", "MCD (McDonald's)": "MDO.DE",
    "KO (Coca-Cola)": "CCC3.DE", "BNP (BNP Paribas)": "BNP.PA", "ING (ING Group)": "INGA.AS",
    "ISP (Intesa Sanpaolo)": "ISP.MI",
    "ASML (ASML Holding)": "ASML.DE", "SAP (SAP SE)": "SAP.DE", "RHM (Rheinmetall)": "RHM.DE",
    "SIE (Siemens)": "SIE.DE", "ENR (Siemens Energy)": "ENR.DE", "IFX (Infineon)": "IFX.DE",
    "ALV (Allianz)": "ALV.DE", "MBG (Mercedes-Benz)": "MBG.DE", "BMW (BMW)": "BMW.DE",
    "VOW3 (Volkswagen)": "VOW3.DE",
    "AIR (Airbus)": "AIR.PA", "MOH (LVMH)": "MC.PA", "RMS (Hermès)": "RMS.PA",
    "LOR (L'Oreal)": "OR.PA", "SU (Schneider Electric)": "SU.PA", "AI (Air Liquide)": "AI.PA",
    "SAF (Safran)": "SAF.PA", "DG (Vinci)": "DG.PA", "TOT (TotalEnergies)": "TTE.PA",
    "PRX (Prosus)": "PRX.AS", "ADYEN (Adyen)": "ADYEN.AS", "HEIA (Heineken)": "HEIA.AS",
    "RACE (Ferrari)": "RACE.MI", "ENEL (Enel)": "ENEL.MI", "ENI (Eni)": "ENI.MI",
    "PRY (Prysmian)": "PRY.MI",
}

# Защитен филтър: пазим само борси в евро - Германия, Франция, Италия, Нидерландия.
# Дори ако някой добави тикер от друга борса по-нататък в TICKERS по-горе,
# той автоматично отпада тук, вместо да влезе тихо в анализа.
ALLOWED_EXCHANGE_SUFFIXES = {
    ".DE": "Германия (Xetra)",
    ".PA": "Франция (Euronext Paris)",
    ".MI": "Италия (Borsa Italiana)",
    ".AS": "Нидерландия (Euronext Amsterdam)",
}
_dropped = {name: sym for name, sym in TICKERS.items() if not sym.endswith(tuple(ALLOWED_EXCHANGE_SUFFIXES))}
TICKERS = {name: sym for name, sym in TICKERS.items() if sym.endswith(tuple(ALLOWED_EXCHANGE_SUFFIXES))}


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def find_swing_points(df: pd.DataFrame, order: int = 2) -> pd.DataFrame:
    """Fractal swing точки: бар е SwingHigh/SwingLow, ако е екстремум спрямо
    `order` бара преди и след него. Последните `order` бара в поредицата не
    могат да бъдат потвърдени още (нужни са бъдещи бара за потвърждение)."""
    df = df.copy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(order, n - order):
        window_h = highs[i - order : i + order + 1]
        window_l = lows[i - order : i + order + 1]
        if highs[i] == window_h.max():
            swing_high[i] = True
        if lows[i] == window_l.min():
            swing_low[i] = True
    df["SwingHigh"] = swing_high
    df["SwingLow"] = swing_low
    return df


def structure_trend(df: pd.DataFrame):
    """Връща (uptrend: bool, последен swing high, последен swing low) на база
    последните 2 потвърдени swing точки от всеки тип (Higher High + Higher Low)."""
    highs = df.loc[df["SwingHigh"], "High"]
    lows = df.loc[df["SwingLow"], "Low"]
    if len(highs) < 2 or len(lows) < 2:
        return False, None, None
    higher_high = highs.iloc[-1] > highs.iloc[-2]
    higher_low = lows.iloc[-1] > lows.iloc[-2]
    return (higher_high and higher_low), float(highs.iloc[-1]), float(lows.iloc[-1])


@st.cache_data(ttl=3600)
def analyze_instrument(name: str, symbol: str, support_tolerance_pct: float, swing_order_weekly: int):
    # --- 1. Дневни данни (2 години - за достатъчно седмични барове) ---
    daily = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if daily.empty or len(daily) < 150:
        return None
    daily = flatten_columns(daily)

    # --- 2. Седмичен таймфрейм: тренд по структура + зони ---
    weekly = (
        daily.resample("W")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )
    if len(weekly) < 20:
        return None
    weekly = find_swing_points(weekly, order=swing_order_weekly)
    weekly_uptrend, resistance_level, support_level = structure_trend(weekly)

    if not weekly_uptrend or support_level is None:
        return None  # твърд филтър: само седмичен uptrend по структура (HH+HL)

    tolerance = support_level * (support_tolerance_pct / 100)
    support_zone_low = support_level - tolerance
    support_zone_high = support_level + tolerance

    # --- 3. Дневна проверка: цената в седмичната зона на подкрепа ли е? ---
    current_price = float(daily["Close"].iloc[-1])
    in_support_zone = support_zone_low <= current_price <= support_zone_high

    if not in_support_zone:
        return None  # твърд филтър: чакаме цената реално да стигне зоната

    # --- 4. 4-часова структура: започва ли HH+HL точно сега в зоната? ---
    # Yahoo няма нативни 4h данни - теглим часови бар (налични ~60 дни назад)
    # и ги ресемплираме в 4-часови свещи.
    h4_uptrend = False
    h4_last_low = None
    try:
        intraday = yf.download(symbol, period="60d", interval="60m", progress=False, auto_adjust=True)
        if not intraday.empty:
            intraday = flatten_columns(intraday)
            h4 = (
                intraday.resample("4h")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna()
            )
            if len(h4) >= 10:
                h4 = find_swing_points(h4, order=1)  # по-къс lookback за по-бърз таймфрейм
                h4_uptrend, _, h4_last_low = structure_trend(h4)
    except Exception:
        pass

    risk = current_price - support_zone_low
    reward = resistance_level - current_price
    rr_ratio = (reward / risk) if risk > 0 else None

    return {
        "Име": name,
        "Тикер": symbol,
        "Цена (€)": round(current_price, 2),
        "Седм. подкрепа (зона)": f"{support_zone_low:.2f}–{support_zone_high:.2f}",
        "Седм. съпротива (Target)": round(resistance_level, 2),
        "Risk (до дъното на зоната)": round(risk, 2),
        "Reward (до Target)": round(reward, 2),
        "Risk/Reward": round(rr_ratio, 2) if rr_ratio else None,
        "4ч ранен up-тренд (HH+HL)": h4_uptrend,
        "Готов за вход": h4_uptrend,
    }


# --- ИНТЕРФЕЙС --------------------------------------------------------------

def generate_ai_analysis(df_ready: pd.DataFrame, df_watch: pd.DataFrame, api_key: str) -> str:
    client = Anthropic(api_key=api_key)

    # ВАЖНО: подаваме на Gemini само реално изчислените резултати. Той не
    # преценява сам дали инструмент отговаря на условията - трите таймфрейма
    # (седмичен тренд, дневна зона, 4ч потвърждение) вече са проверени в кода.
    ready_text = (
        df_ready.to_string(index=False)
        if not df_ready.empty
        else "НЯМА инструменти с пълно потвърждение на трите таймфрейма в момента."
    )
    watch_text = (
        df_watch.to_string(index=False)
        if not df_watch.empty
        else "НЯМА инструменти в зона на подкрепа в момента."
    )

    prompt = f"""
    Ти си професионален суинг търговец, ползващ multi-timeframe стратегия:
    седмичен тренд по структура (higher high + higher low) определя посоката,
    зона на подкрепа около последния седмичен swing low е мястото за вход,
    а 4-часова структура (started higher high + higher low вътре в зоната)
    потвърждава момента за реално влизане.

    Използвай СТРИКТНО само данните по-долу - вече са преминали трите проверки
    в кода, ти НЕ преценяваш сам дали инструмент отговаря на условията.

    === ГОТОВИ ЗА ВХОД (и трите таймфрейма потвърдени) ===
    {ready_text}

    === WATCHLIST (седмичен тренд + в зона на подкрепа, но 4ч ОЩЕ НЕ е потвърдил) ===
    {watch_text}

    ЖЕЛЕЗНИ ПРАВИЛА:
    - Избирай ЕДИНСТВЕНО измежду инструментите по-горе. Не добавяй никой друг.
    - Ако "ГОТОВИ ЗА ВХОД" е празно, кажи го ясно - не превръщай Watchlist
      кандидати в препоръка за вход, те само чакат потвърждение.
    - За Watchlist инструментите обясни какво точно очакваме да видим на 4ч
      графиката, за да минат в "готови" (напр. свеж higher low над зоната).

    За всеки инструмент от "ГОТОВИ ЗА ВХОД" дай:
    - Кратко обяснение защо сетъпът е добър (седмична структура + защо точно тази зона).
    - Предложение за 2 транша за вход (в рамките на зоната на подкрепа, посочена в данните).
    - Stop и Target нивата вече са изчислени в данните - използвай ги directно, не измисляй нови.

    Бъди кратък, систематизиран, удобен за преглед на телефон.
    """
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ValueError(f"Claude върна празен отговор (stop_reason: {response.stop_reason}). Опитай пак.")
    return text


st.title("🎯 Multi-Timeframe Swing Screener")
st.caption(
    "Седмичен тренд по структура (HH/HL) → зона на подкрепа → дневна цена в зоната "
    "→ 4ч потвърждение на нов up-тренд вътре в зоната."
)
st.caption(f"Универс: {len(TICKERS)} инструмента (Германия, Франция, Италия, Нидерландия - в евро)")
if _dropped:
    st.caption(f"⚠️ Премахнати извън дозволените борси: {', '.join(_dropped.keys())}")

with st.sidebar:
    st.subheader("Настройки")
    support_tolerance_pct = st.slider("Толеранс на зоната на подкрепа (%)", 1.0, 8.0, 3.0, step=0.5)
    swing_order_weekly = st.slider(
        "Чувствителност на седмичните swing точки", 1, 4, 2,
        help="По-високо число = по-малко, но по-значими swing точки (по-дълъг lag за потвърждение).",
    )

if st.button("🔍 Сканирай пазара", type="primary"):
    results = []
    watch_list = []  # инструменти в зоната, но БЕЗ 4ч потвърждение все още
    progress = st.progress(0.0, text="Стартиране на анализа...")

    items = list(TICKERS.items())
    for idx, (name, symbol) in enumerate(items):
        progress.progress((idx + 1) / len(items), text=f"Анализирам {name}...")
        res = analyze_instrument(name, symbol, support_tolerance_pct, swing_order_weekly)
        if res is not None:
            if res["Готов за вход"]:
                results.append(res)
            else:
                watch_list.append(res)
    progress.empty()

    st.session_state["mtf_results"] = results
    st.session_state["mtf_watchlist"] = watch_list

results = st.session_state.get("mtf_results", [])
watch_list = st.session_state.get("mtf_watchlist", [])

st.divider()
st.subheader("✅ Готови за вход (седмичен up-тренд + в зона на подкрепа + 4ч потвърждение)")
if results:
    df_ready = pd.DataFrame(results).drop(columns=["Готов за вход"])
    st.dataframe(df_ready, use_container_width=True, hide_index=True)
else:
    df_ready = pd.DataFrame()
    st.info("Няма инструменти с пълно потвърждение на трите таймфрейма в момента.")

st.divider()
st.subheader("👀 Watchlist (в зона на подкрепа, чакаме 4ч потвърждение)")
if watch_list:
    df_watch = pd.DataFrame(watch_list).drop(columns=["Готов за вход", "4ч ранен up-тренд (HH+HL)"])
    st.dataframe(df_watch, use_container_width=True, hide_index=True)
else:
    df_watch = pd.DataFrame()
    st.info("Няма инструменти в зона на подкрепа в момента (или все още не си сканирал).")

st.divider()
st.subheader("🤖 AI Анализ")
gemini_api_key = st.secrets.get("GEMINI_API_KEY", None)
if not gemini_api_key:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

if st.button("Генерирай AI анализ на резултатите", type="primary"):
    if not gemini_api_key:
        st.error("Липсва Gemini API ключ (провери Streamlit Secrets).")
    elif df_ready.empty and df_watch.empty:
        st.warning("Първо натисни 'Сканирай пазара' горе, за да има какво да се анализира.")
    else:
        with st.spinner("Gemini анализира резултатите..."):
            try:
                st.markdown(generate_ai_analysis(df_ready, df_watch, gemini_api_key))
            except Exception as e:
                st.error(f"Грешка при AI анализа: {e}")

st.divider()
st.subheader("📈 Преглед на графика")
selected_name = st.selectbox("Избери инструмент:", list(TICKERS.keys()))
if selected_name:
    symbol = TICKERS[selected_name]
    daily = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    if not daily.empty:
        daily = flatten_columns(daily)
        weekly = (
            daily.resample("W")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )
        weekly = find_swing_points(weekly, order=swing_order_weekly)
        _, resistance_level, support_level = structure_trend(weekly)

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=daily.index, open=daily["Open"], high=daily["High"],
                low=daily["Low"], close=daily["Close"], name="Дневна цена",
            )
        )
        if support_level:
            tolerance = support_level * (support_tolerance_pct / 100)
            fig.add_hrect(
                y0=support_level - tolerance, y1=support_level + tolerance,
                fillcolor="green", opacity=0.15, line_width=0,
                annotation_text="Седмична зона на подкрепа",
            )
        if resistance_level:
            fig.add_hline(
                y=resistance_level, line_dash="dot", line_color="red",
                annotation_text="Седмична съпротива (Target)",
            )
        fig.update_layout(
            height=550, template="plotly_dark", xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
