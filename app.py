import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

# 1. КОНФИГУРАЦИЯ НА СТРАНИЦАТА
st.set_page_config(
    page_title="Swing Screener AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Променен CSS за СВЕТЛА таблица с черен текст
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Дизайн на табовете */
            .stTabs [data-baseweb="tab-list"] {gap: 10px;}
            .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 4px 4px 0px 0px; padding: 10px; border: 1px solid #333; border-bottom: none;}
            .stTabs [aria-selected="true"] {background-color: #2E5B88; color: white;}
            
            /* Дизайн на Markdown таблицата (Търговския план) - БЯЛ ФОН */
            .stMarkdown table {
                width: 100%;
                border-collapse: collapse;
                background-color: #FFFFFF !important;
            }
            .stMarkdown th {
                background-color: #2E5B88 !important;
                color: white !important;
                font-size: 15px;
                padding: 12px 8px !important;
                border-bottom: 2px solid #ccc;
            }
            .stMarkdown td {
                padding: 12px 8px !important;
                border-bottom: 1px solid #eee;
                font-size: 14px;
                color: #000000 !important; /* Форсираме черен цвят на текста */
            }
            /* Зеброва шарка: бяло и светло сиво */
            .stMarkdown tbody tr:nth-child(even) {
                background-color: #F8F9FA !important; 
            }
            .stMarkdown tbody tr:nth-child(odd) {
                background-color: #FFFFFF !important; 
            }
            /* Ховър ефект: леко потъмняване при посочване */
            .stMarkdown tbody tr:hover {
                background-color: #E2E6EA !important;
                transition: background-color 0.2s ease;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

TICKERS = {
    "SXRV (iShares Nasdaq 100)": "SXRV.DE", "SXR8 (iShares Core S&P 500)": "SXR8.DE",
    "SXRG (iShares MSCI World)": "SXRG.DE", "EXH1 (iShares STOXX Europe 600)": "EXH1.DE",
    "SEC0 (iShares Semiconductor)": "SEC0.DE", "QDVE (iShares S&P 500 Info Tech)": "QDVE.DE",
    "2B7K (iShares Cybersecurity)": "2B7K.DE", "URNU (Global X Uranium)": "URNU.DE",
    "CBRS (First Trust Nasdaq Cybersecurity)": "CBRS.DE", "WGLD (WisdomTree Gold)": "WGLD.DE",
    "PHAG (WisdomTree Silver)": "PHAG.DE",
    
    "APC (Apple)": "APC.DE", "MSF (Microsoft)": "MSF.DE", "ABE (Alphabet/Google)": "ABE.DE",
    "AMZ (Amazon)": "AMZ.DE", "NVD (Nvidia)": "NVD.DE", "FB2A (Meta/Facebook)": "FB2A.DE",
    "TL0 (Tesla)": "TL0.DE",
    
    "AMD (AMD)": "AMD.DE", "14B (Broadcom)": "14B.DE", "QCI (Qualcomm)": "QCI.DE",
    "INL (Intel)": "INL.DE", "CIS (Cisco)": "CIS.DE", "CRM (Salesforce)": "CRM.DE",
    "ORC (Oracle)": "ORC.DE",

    "ELY (Eli Lilly)": "ELY.DE", "NOVC (Novo Nordisk)": "NOVC.DE", "JNJ (Johnson & Johnson)": "JNJ.DE",
    "PFE (Pfizer)": "PFE.DE", "MRK (Merck & Co)": "MCC.DE", "UNH (UnitedHealth)": "UNH.DE",

    "3V64 (Visa)": "3V64.DE", "M4I (Mastercard)": "M4I.DE", "NFC (Netflix)": "NFC.DE",
    "DIS (Walt Disney)": "DIS.DE", "WMT (Walmart)": "WMT.DE", "SBUX (Starbucks)": "SRB.DE",
    "MCD (McDonald's)": "MDO.DE", "KO (Coca-Cola)": "CCC3.DE",

    "ASML (ASML)": "ASML.DE", "SAP (SAP SE)": "SAP.DE", "RHM (Rheinmetall)": "RHM.DE",
    "SIE (Siemens)": "SIE.DE", "ALV (Allianz)": "ALV.DE", "AIR (Airbus)": "AIR.DE",
    "MOH (LVMH)": "MOH.DE", "LOR (L'Oreal)": "LOR.DE", "TOT (TotalEnergies)": "TOT.DE",
    "MBG (Mercedes-Benz)": "MBG.DE", "BMW (BMW)": "BMW.DE", "VOW3 (Volkswagen)": "VOW3.DE"
}

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data, short=12, long=26, signal=9):
    ema_short = data.ewm(span=short, adjust=False).mean()
    ema_long = data.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

@st.cache_data(ttl=3600)
def fetch_market_data():
    summary_list = []
    charts_data = {}

    for name, symbol in TICKERS.items():
        try:
            df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 50: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
            df['RSI'] = compute_rsi(df['Close'], 14)
            df['MACD_Hist'] = compute_macd(df['Close'])
            df['Vol20'] = df['Volume'].rolling(window=20).mean()

            latest, prev = df.iloc[-1], df.iloc[-2]
            close_price = float(latest['Close'])
            change_pct = ((close_price - float(prev['Close'])) / float(prev['Close'])) * 100
            
            ema50_val = float(latest['EMA50'])
            ema200_val = float(latest['EMA200'])
            
            if ema50_val > ema200_val and close_price > ema200_val: trend = "↗ Възходящ"
            elif ema50_val < ema200_val: trend = "↘ Низходящ"
            else: trend = "→ Консолидация"
            
            diff_ema50 = ((close_price - ema50_val) / ema50_val) * 100
            diff_ema200 = ((close_price - ema200_val) / ema200_val) * 100
            
            vol20 = float(latest['Vol20'])
            vol_surge = float(latest['Volume']) / vol20 if vol20 > 0 else 0
            
            in_buy_zone = (abs(diff_ema50) <= 3.5) or (abs(diff_ema200) <= 4.0) or (float(latest['RSI']) < 42) or (vol_surge >= 1.2)

            summary_list.append({
                "Име": name, "Тикер": symbol, "Цена (€)": round(close_price, 2),
                "Промяна (%)": round(change_pct, 2), "Тренд": trend, 
                "RSI": round(float(latest['RSI']), 1),
                "От EMA50 (%)": round(diff_ema50, 2), "От EMA200 (%)": round(diff_ema200, 2),
                "MACD Hist": round(float(latest['MACD_Hist']), 3), "Обем (x Средния)": round(vol_surge, 2),
                "Бай Зона": in_buy_zone
            })
            charts_data[name] = df
        except: continue
    return pd.DataFrame(summary_list), charts_data

def generate_ai_analysis(df_data, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')

    df_filtered = df_data[df_data["Бай Зона"] == True]
    if len(df_filtered) < 15: df_filtered = df_data.sort_values(by="RSI").head(20)

    prompt = f"""
    Ти си професионален суинг търговец. Пред теб са глоба акции (търгувани в евро на Xetra / Trading 212):
    {df_filtered.to_string(index=False)}

    ИЗБЕРИ ТОП 10 НАЙ-ОБЕЩАВАЩИ ВЪЗМОЖНОСТИ, КАТО СПАЗВАШ СЛЕДНИТЕ ЖЕЛЕЗНИ ПРАВИЛА:

    КАТЕГОРИЯ 1: 5 Акции "За бърз суинг по тренда (Trend Following Swing)"
    - ЗАДЪЛЖИТЕЛНО УСЛОВИЕ: Избирай САМО акции, при които "Тренд" е "↗ Възходящ"! АБСОЛЮТНО ЗАБРАНЕНО е да включваш акции с Низходящ тренд.

    КАТЕГОРИЯ 2: 5 Акции "За дългосрочно акумулиране (Mean Reversion)"
    - Тук можеш да избереш акции в "↘ Низходящ" или "→ Консолидация". Търсим екстремна свръхпродаденост (много нисък RSI) ИЛИ огромен обем (над 1.2x).
    
    За всяка акция бъди ясен с булети:
    - Обясни защо техническият им сетъп е добър.
    - Посочи ценови нива за влизане с 2 лимитирани транша (около EMA 50 / EMA 200).

    МНОГО ВАЖНО - ОБОБЩАВАЩА ТАБЛИЦА (ТЪРГОВСКИ ПЛАН):
    В самия край на твоя отговор, задължително генерирай една обобщаваща Markdown таблица, която да съдържа всички 10 избрани инструмента.
    
    Изисквания за колоните в таблицата:
    1. **Инструмент:** Напиши Името на компанията/ETF, а под него на нов ред добави тикера с наклонен шрифт (използвай HTML тага `<br>`, например: `Apple <br> *APC.DE*`).
    2. **Категория:** Напиши дали е Бърз суинг или Акумулиране.
    3. **Транш 1 (Вход):** Посочи цената за първи вход и В СКОБИ добави процент от предвидения капитал (напр. `150.00 € (40%)`). Прецени процента спрямо риска и вероятността за отскок (по-рисков вход = по-малък процент).
    4. **Транш 2 (Вход):** Посочи цената за втори вход и В СКОБИ добави останалия процент от капитала (напр. `142.00 € (60%)`). Общо двата транша трябва да правят 100%.
    5. **Цел (Take Profit):** Посочи целевата цена за продажба и В СКОБИ добави очаквания процент печалба спрямо средната цена на влизане (напр. `165.00 € (+12%)`).
    """
    return model.generate_content(prompt).text

st.title("⚡ Swing Screener AI")

api_key = st.secrets.get("GEMINI_API_KEY", None)
if not api_key: api_key = st.sidebar.text_input("Gemini API Key", type="password")

with st.spinner("Синхронизиране на пазарни данни..."):
    df_summary, charts_data = fetch_market_data()

if not df_summary.empty:
    # 2. KPI МЕТРИКИ
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

    # 3. ТАБОВЕ
    tab1, tab2, tab3 = st.tabs(["🤖 AI АНАЛИЗ И ПЛАН", "📊 ПЪЛНА ТАБЛИЦА", "📈 ИНТЕРАКТИВНА ГРАФИКА"])

    with tab1:
        if st.button("🚀 Генерирай ТОП 10 Анализ и Търговски План", type="primary", use_container_width=True):
            if not api_key: st.error("Въведи Gemini API ключ!")
            else:
                with st.spinner("Gemini анализира данните и изготвя търговски план..."):
                    try:
                        st.markdown(generate_ai_analysis(df_summary, api_key), unsafe_allow_html=True)
                    except Exception as e: st.error(f"Грешка: {e}")

    with tab2:
        display_df = df_summary.drop(columns=["Бай Зона"])
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "RSI": st.column_config.ProgressColumn(
                    "RSI (Моментум)",
                    help="Стойности под 40 са свръхпродадени, над 70 са свръхкупени",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "Обем (x Средния)": st.column_config.NumberColumn(
                    "Обем (x Средния)",
                    help="Обем спрямо 20-дневната средна",
                    format="%.2fx",
                ),
                "Тренд": st.column_config.TextColumn(
                    "Тренд",
                )
            }
        )

    with tab3:
        selected_name = st.selectbox("Избери инструмент за анализ:", list(TICKERS.keys()))
        if selected_name in charts_data:
            df_chart = charts_data[selected_name].tail(150)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Цена'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA50'], line=dict(color='orange', width=1.5), name='EMA 50'))
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA200'], line=dict(color='blue', width=1.5), name='EMA 200'))
            
            colors = ['green' if val >= 0 else 'red' for val in df_chart['MACD_Hist']]
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['MACD_Hist'], marker_color=colors, name='MACD', yaxis='y2', opacity=0.3))
            
            fig.update_layout(
                yaxis_title="Цена (€)",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                height=550,
                template="plotly_dark",
                yaxis2=dict(title="MACD", overlaying='y', side='right', showgrid=False, range=[df_chart['MACD_Hist'].min()*3, df_chart['MACD_Hist'].max()*3])
            )
            st.plotly_chart(fig, use_container_width=True)
