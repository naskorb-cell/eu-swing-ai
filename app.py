import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai

st.set_page_config(
    page_title="Global Swing Screener (EU Tax-Free)",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TICKERS = {
    "SXRV (iShares Nasdaq 100)": "SXRV.DE",
    "SXR8 (iShares Core S&P 500)": "SXR8.DE",
    "SXRG (iShares MSCI World)": "SXRG.DE",
    "EXH1 (iShares STOXX Europe 600)": "EXH1.DE",
    "SEC0 (iShares Semiconductor)": "SEC0.DE",
    "QDVE (iShares S&P 500 Info Tech)": "QDVE.DE",
    "2B7K (iShares Cybersecurity)": "2B7K.DE",
    "URNU (Global X Uranium)": "URNU.DE",
    "CBRS (iShares Clean Energy)": "CBRS.DE",
    "WGLD (WisdomTree Gold)": "WGLD.DE",
    "PHAG (WisdomTree Silver)": "PHAG.DE",
    
    "APC (Apple)": "APC.DE",
    "MSF (Microsoft)": "MSF.DE",
    "ABE (Alphabet/Google)": "ABE.DE",
    "AMZ (Amazon)": "AMZ.DE",
    "NVD (Nvidia)": "NVD.DE",
    "FB2A (Meta/Facebook)": "FB2A.DE",
    "TL0 (Tesla)": "TL0.DE",

    "AMD (AMD)": "AMD.DE",
    "14B (Broadcom)": "14B.DE",
    "QCI (Qualcomm)": "QCI.DE",
    "INL (Intel)": "INL.DE",
    "CIS (Cisco)": "CIS.DE",
    "CRM (Salesforce)": "CRM.DE",
    "ORC (Oracle)": "ORC.DE",

    "ELY (Eli Lilly)": "ELY.DE",
    "NOVC (Novo Nordisk)": "NOVC.DE",
    "JNJ (Johnson & Johnson)": "JNJ.DE",
    "PFE (Pfizer)": "PFE.DE",
    "MRK (Merck & Co)": "MCC.DE",
    "UNH (UnitedHealth)": "UNH.DE",

    "3V64 (Visa)": "3V64.DE",
    "M4I (Mastercard)": "M4I.DE",
    "NFC (Netflix)": "NFC.DE",
    "DIS (Walt Disney)": "DIS.DE",
    "WMT (Walmart)": "WMT.DE",
    "SBUX (Starbucks)": "SRB.DE",
    "MCD (McDonald's)": "MDO.DE",
    "KO (Coca-Cola)": "CCC3.DE",

    "ASML (ASML Holding)": "ASML.DE",
    "SAP (SAP SE)": "SAP.DE",
    "RHM (Rheinmetall)": "RHM.DE",
    "SIE (Siemens)": "SIE.DE",
    "ALV (Allianz)": "ALV.DE",
    "AIR (Airbus)": "AIR.DE",
    "MOH (LVMH)": "MOH.DE",
    "LOR (L'Oreal)": "LOR.DE",
    "TOT (TotalEnergies)": "TOT.DE",
    "MBG (Mercedes-Benz)": "MBG.DE",
    "BMW (BMW Group)": "BMW.DE",
    "VOW3 (Volkswagen)": "VOW3.DE"
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
            
            diff_ema50 = ((close_price - float(latest['EMA50'])) / float(latest['EMA50'])) * 100
            diff_ema200 = ((close_price - float(latest['EMA200'])) / float(latest['EMA200'])) * 100
            
            vol20 = float(latest['Vol20'])
            vol_surge = float(latest['Volume']) / vol20 if vol20 > 0 else 0

            in_buy_zone = (abs(diff_ema50) <= 3.5) or (abs(diff_ema200) <= 4.0) or (float(latest['RSI']) < 42)

            summary_list.append({
                "Име": name, "Тикер": symbol, "Последна цена (€)": round(close_price, 2),
                "Промяна (%)": round(change_pct, 2), "RSI (14)": round(float(latest['RSI']), 1),
                "Отстояние EMA50 (%)": round(diff_ema50, 2), "Отстояние EMA200 (%)": round(diff_ema200, 2),
                "MACD Хистограма": round(float(latest['MACD_Hist']), 3), "Обем Спрямо Среден": vol_surge,
                "Бай Зона": in_buy_zone
            })
            charts_data[name] = df
        except: continue
    return pd.DataFrame(summary_list), charts_data

def generate_ai_analysis(df_data, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')

    df_filtered = df_data[df_data["Бай Зона"] == True]
    if df_filtered.empty: df_filtered = df_data.sort_values(by="RSI (14)").head(10)

    prompt = f"""
    Ти си професионален суинг търговец. Пред теб са високоликвидни глобални акции (търгувани в евро на Xetra / Trading 212), които днес са в "Бай Зона":
    {df_filtered.to_string(index=False)}

    Избери ТОП 4 НАЙ-ОБЕЩАВАЩИ ВЪЗМОЖНОСТИ. Дай категорично предимство на активи с "Обем Спрямо Среден" над 1.0 и позитивен "MACD Хистограма".
    1. Обясни защо техническият им сетъп е добър (посочи цената спрямо EMA, Обема и MACD).
    2. Посочи ценови нива за влизане с 2 лимитирани транша (около EMA 50 / EMA 200).
    3. Раздели ги на: "За бърз суинг" и "За дългосрочно акумулиране".
    Бъди ясен и систематизиран с булети.
    """
    return model.generate_content(prompt).text

st.title("🌍 Global Swing Screener AI")
st.caption("Автоматичен скринер за ETF-и & Глобални Акции (Включва MACD и Volume Surge)")

api_key = st.secrets.get("GEMINI_API_KEY", None)
if not api_key: api_key = st.sidebar.text_input("Gemini API Key", type="password")

with st.spinner("Изчисляване на индикатори... (може да отнеме 15 сек)"):
    df_summary, charts_data = fetch_market_data()

if st.button("🤖 Генерирай ТОП Суинг Анализ", type="primary"):
    if not api_key: st.error("Въведи Gemini API ключ!")
    else:
        with st.spinner("Gemini анализира..."):
            try:
                st.subheader("🎯 ТОП Суинг Възможности за Днес")
                st.markdown(generate_ai_analysis(df_summary, api_key))
            except Exception as e: st.error(f"Грешка: {e}")

st.divider()
st.subheader("📊 Данни за всички активи (1D)")

def style_dataframe(row):
    styles = [''] * len(row)
    if 0 < row['RSI (14)'] < 40: styles[row.index.get_loc('RSI (14)')] = 'background-color: #1e4620; color: white'
    if abs(row['Отстояние EMA50 (%)']) <= 3.5: styles[row.index.get_loc('Отстояние EMA50 (%)')] = 'background-color: #1e3a5f; color: white'
    if abs(row['Отстояние EMA200 (%)']) <= 4.0: styles[row.index.get_loc('Отстояние EMA200 (%)')] = 'background-color: #1e3a5f; color: white'
    if row['MACD Хистограма'] > 0: styles[row.index.get_loc('MACD Хистограма')] = 'color: #00ff00'
    elif row['MACD Хистограма'] < 0: styles[row.index.get_loc('MACD Хистограма')] = 'color: #ff4c4c'
    if row['Обем Спрямо Среден'] >= 1.2: styles[row.index.get_loc('Обем Спрямо Среден')] = 'color: #00ff00; font-weight: bold'
    return styles

if not df_summary.empty:
    display_df = df_summary.drop(columns=["Бай Зона"])
    
    # Коригирано стилизиране и форматиране
    styled_df = display_df.style.apply(style_dataframe, axis=1).format({
        'Обем Спрямо Среден': '{:.2f}x'
    })
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("🔍 Преглед на графика")
selected_name = st.selectbox("Избери инструмент:", list(TICKERS.keys()))
if selected_name in charts_data:
    df_chart = charts_data[selected_name].tail(150)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Цена'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA50'], line=dict(color='orange', width=1.5), name='EMA 50'))
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA200'], line=dict(color='blue', width=1.5), name='EMA 200'))
    fig.update_layout(title=f"{selected_name} - 1D Графика", yaxis_title="Цена (€)", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), height=450, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
