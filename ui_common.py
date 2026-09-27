"""Общи UI елементи: CSS темата, секционни заглавия, форматиране на суми."""

import streamlit as st

hide_st_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

            :root {
                --void: #0B0F14;
                --panel: #141A21;
                --ink: #E6EDF3;
                --ink-muted: #7C8B99;
                --hairline: #232B33;
                --go: #3DDC97;
                --watch: #E8A23D;
                --info: #5B8DEF;
                --risk: #E85D5D;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
            h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

            /* Заглавие */
            .app-hero { display: flex; align-items: baseline; gap: 12px; margin-bottom: 2px; }
            .app-hero h1 { font-size: 1.9rem !important; margin: 0 !important; }
            .app-hero .app-sub { color: var(--ink-muted); font-size: 0.92rem; }

            /* Радио бутони като pill-табове за избор на стратегия */
            div[data-testid="stRadio"] > div { gap: 8px; }
            div[data-testid="stRadio"] label {
                background: var(--panel); border: 1px solid var(--hairline);
                border-radius: 999px; padding: 8px 18px !important; transition: all 0.15s ease;
            }
            div[data-testid="stRadio"] label:has(input:checked) {
                border-color: var(--info); background: rgba(91,141,239,0.12);
            }

            /* Метрики като приборни табла */
            div[data-testid="stMetric"] {
                background: var(--panel); border: 1px solid var(--hairline);
                border-left: 3px solid var(--info); border-radius: 8px;
                padding: 12px 16px;
            }
            div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
            div[data-testid="stMetricLabel"] { color: var(--ink-muted); }

            /* Секционни статус-ленти (готово/watchlist/риск/инфо) */
            .section-rail { border-left: 4px solid var(--info); padding: 6px 0 6px 14px; margin: 22px 0 12px 0; }
            .section-rail.go { border-color: var(--go); }
            .section-rail.watch { border-color: var(--watch); }
            .section-rail.risk { border-color: var(--risk); }
            .section-rail .title {
                font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.08rem; color: var(--ink);
            }
            .section-rail .sub { color: var(--ink-muted); font-size: 0.85rem; margin-top: 2px; }

            /* Тънки разделители вместо дебели markdown "---" */
            hr { border: none; border-top: 1px solid var(--hairline); margin: 18px 0; }

            /* Табове */
            .stTabs [data-baseweb="tab-list"] { gap: 8px; }
            .stTabs [data-baseweb="tab"] {
                height: 46px; white-space: pre-wrap; background-color: var(--panel);
                border-radius: 8px 8px 0 0; padding: 10px 16px; border: 1px solid var(--hairline); border-bottom: none;
                font-family: 'Space Grotesk', sans-serif;
            }
            .stTabs [aria-selected="true"] { background-color: rgba(91,141,239,0.16); color: var(--info); }

            /* Expander като карта */
            div[data-testid="stExpander"] {
                border: 1px solid var(--hairline) !important; border-radius: 10px !important; background: var(--panel);
            }

            /* Вторични бутони - по-тих "ghost" вид, primary остава акцентен */
            .stButton > button[kind="secondary"] {
                background: transparent; border: 1px solid var(--hairline); color: var(--ink-muted);
            }
            .stButton > button[kind="secondary"]:hover { border-color: var(--info); color: var(--info); }

            /* Markdown таблици (напр. в AI анализа) */
            .stMarkdown table { width: 100%; border-collapse: collapse; background-color: var(--panel) !important; }
            .stMarkdown th {
                background-color: rgba(91,141,239,0.16) !important; color: var(--ink) !important;
                font-family: 'Space Grotesk', sans-serif; font-size: 14px; padding: 10px 8px !important;
                border-bottom: 1px solid var(--hairline);
            }
            .stMarkdown td {
                padding: 10px 8px !important; border-bottom: 1px solid var(--hairline);
                font-size: 14px; color: var(--ink) !important; font-family: 'JetBrains Mono', monospace;
            }
            .stMarkdown tbody tr:nth-child(even) { background-color: rgba(255,255,255,0.02) !important; }
            .stMarkdown tbody tr:hover { background-color: rgba(91,141,239,0.08) !important; transition: background-color 0.15s ease; }

            /* Текстови/парола/число полета - постоянна видима рамка, не само при hover/focus */
            div[data-testid="stTextInput"] input,
            div[data-testid="stNumberInput"] input,
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] {
                background-color: var(--panel) !important;
                border: 1px solid var(--hairline) !important;
                color: var(--ink) !important;
            }
            div[data-testid="stTextInput"] input:focus,
            div[data-testid="stNumberInput"] input:focus,
            div[data-baseweb="input"]:focus-within {
                border-color: var(--info) !important;
            }
            div[data-testid="stTextInput"] input::placeholder { color: var(--ink-muted) !important; opacity: 1; }

            /* Caption-и (напр. "Обновено: ...") в моноспейс - усещане за таймстемп на терминал */
            [data-testid="stCaptionContainer"] { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem !important; }
            </style>
            """


def section_header(title: str, status: str = "info", subtitle: str = ""):
    """Секционно заглавие с цветова статус-лента: 'go' (зелено, готово за
    вход), 'watch' (кехлибарено, наблюдавай), 'risk' (червено), 'info' (синьо,
    по подразбиране - AI/аналитични секции)."""
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-rail {status}"><div class="title">{title}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def format_eur(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:g} млрд. €"
    if value >= 1e6:
        return f"{value / 1e6:g} млн. €"
    return f"{value / 1e3:g} хил. €"
