"""Общи UI елементи: CSS темата, секционни заглавия, форматиране на суми."""

import streamlit as st

from ai_client import AI_KEY_SECRETS, AI_PROVIDERS, GEMINI_DEFAULT_MODEL

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

            /* Фунията на скана като една лента */
            .funnel { display: flex; align-items: stretch; gap: 6px; margin: 0.6rem 0 0.4rem; flex-wrap: wrap; }
            .funnel-step { flex: 1 1 120px; background: var(--panel); border: 1px solid var(--hairline);
                           border-top: 3px solid var(--info); border-radius: 10px; padding: 0.55rem 0.8rem; }
            .funnel-step .n { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600; color: var(--ink); }
            .funnel-step .lbl { font-size: 0.8rem; color: var(--ink-muted); }
            .funnel-step .pct { color: var(--ink); font-family: 'JetBrains Mono', monospace; margin-left: 4px; }
            .funnel-arrow { align-self: center; color: var(--ink-muted); font-size: 1.4rem; }

            /* Карти за „Готови за вход“ */
            .card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
                         border-left: 3px solid var(--hairline); padding-left: 8px; margin-bottom: 4px; }
            .card-head.confirmed { border-left-color: var(--go); }
            .card-name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--ink); }
            .card-ticker { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--ink-muted); }
            .badge { display: inline-block; background: #1B232C; border: 1px solid var(--hairline); border-radius: 999px;
                     padding: 1px 9px; margin: 0 4px 4px 0; font-size: 0.78rem; color: var(--ink); white-space: nowrap; }
            [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
            .lvl-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: 6px; margin: 6px 0; }
            .lvl { background: var(--void); border: 1px solid var(--hairline); border-radius: 8px; padding: 6px 10px; }
            .lvl-label { font-size: 0.72rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.04em; }
            .lvl-value { font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 600; color: var(--ink);
                         white-space: nowrap; }

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
        return f"{value / 1e9:.3g} млрд. €"
    if value >= 1e6:
        return f"{value / 1e6:.3g} млн. €"
    return f"{value / 1e3:.3g} хил. €"


def ai_provider() -> str:
    """Избраният AI модел ("Gemini" / "Claude") от превключвателя горе в приложението."""
    # segmented_control може да се "отмаркира" (None) - тогава моделът по подразбиране
    return st.session_state.get("ai_provider") or AI_PROVIDERS[0]


def ai_api_key(provider: str, key: str):
    """API ключът на доставчика от Secrets; ако липсва - поле за ръчно въвеждане."""
    api_key = st.secrets.get(AI_KEY_SECRETS[provider], None)
    if not api_key:
        api_key = st.text_input(f"{provider} API Key ({AI_KEY_SECRETS[provider]})", type="password", key=key)
    return api_key


def gemini_model() -> str:
    return st.secrets.get("GEMINI_MODEL", GEMINI_DEFAULT_MODEL)


def levels_html(items: list) -> str:
    """Компактна решетка „етикет / стойност“ (цена, stop, цели, R/R) - за карти и
    прозорци; st.metric е с твърде едър шрифт и реже числата в тесни колони."""
    cells = "".join(f'<div class="lvl"><div class="lvl-label">{label}</div><div class="lvl-value">{value}</div></div>'
                    for label, value in items)
    return f'<div class="lvl-grid">{cells}</div>'


def friendly_ai_error(error, provider: str) -> str:
    """Човешко обяснение за честите грешки на AI доставчиците (кредити, лимити, ключ)."""
    text = str(error)
    low = text.lower()
    if provider == "Gemini" and ("prepayment" in low or "credits are depleted" in low or "402" in low):
        return ("Кредитите в Gemini API са изчерпани. Зареди ги в Google AI Studio → Billing "
                "(aistudio.google.com) или превключи на Claude.")
    if provider == "Claude" and ("credit balance" in low or "billing" in low):
        return "Кредитите в Anthropic API са изчерпани. Зареди ги в console.anthropic.com → Billing или превключи на Gemini."
    if "resource_exhausted" in low or "429" in low or "rate limit" in low or "quota" in low:
        return f"{provider} върна лимит на заявките (quota/429). Опитай след минута или превключи модела."
    if "api key" in low or "401" in low or "403" in low or "permission" in low:
        return f"Ключът за {provider} е невалиден или без права ({AI_KEY_SECRETS[provider]} в Secrets)."
    return f"Грешка от {provider}: {text}"


def _switch_ai_provider(to: str):
    st.session_state["ai_provider"] = to


def show_ai_error(error, provider: str, key: str):
    """Грешка от AI + бутон за превключване към другия модел."""
    st.error(friendly_ai_error(error, provider))
    other = next(p for p in AI_PROVIDERS if p != provider)
    st.button(f"🔁 Превключи на {other}", key=f"{key}_switch_ai", on_click=_switch_ai_provider, args=(other,))
