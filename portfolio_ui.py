"""Секция 💼 Портфолио & P&L (Trading 212, само четене) + AI анализ на представянето."""

import hashlib
import threading
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

import t212_portfolio as t212
from ai_client import AI_KEY_SECRETS, stream_ai
from ui_common import ai_api_key, ai_provider, gemini_model, section_header, show_ai_error

# ============================================================================
# ПОРТФОЛИО: Trading 212 отворени позиции + P&L анализ (READ-ONLY)
# ============================================================================
# ВАЖНО - сигурност: ключът/секретът, използвани тук, трябва да имат
# ЕДИНСТВЕНО права за четене на Portfolio + History (никога Orders/write).
# Trading 212 API няма endpoint за прехвърляне на пари, така че дори при
# изтичане на този ключ, щетата е ограничена до преглед на данни.

PERIOD_PRESETS = {
    "Тази седмица": 7,
    "Последните 30 дни": 30,
    "Последните 90 дни": 90,
    "Тази година": 365,
}


def _period_bounds(preset_label: str, custom_range=None):
    today = date.today()
    if preset_label == "Персонализиран период" and custom_range:
        start, end = custom_range
        return start, end
    if preset_label == "Този месец":
        return today.replace(day=1), today
    if preset_label == "Миналия месец":
        last_end = today.replace(day=1) - timedelta(days=1)
        return last_end.replace(day=1), last_end
    days = PERIOD_PRESETS.get(preset_label, 30)
    return today - timedelta(days=days), today


# История на сделките: T212 позволява само 1 заявка на ~10 сек (50 записа на страница),
# затова цялата история се тегли бавно. Заредената се пази в общ склад (за всички
# сесии до рестарт на приложението) и следващите зареждания теглят само новите сделки.
HISTORY_LOOKBACK_DAYS = 180  # толкова преди началото на периода - за покупките (средна цена)
HISTORY_MAX_PAGES = 20


@st.cache_resource
def _history_store() -> dict:
    return {"lock": threading.Lock(), "accounts": {}}


def _account_id(env: str, api_key: str) -> str:
    return hashlib.sha256(f"{env}:{api_key}".encode()).hexdigest()[:16]


def load_order_history(base_url: str, auth_header: str, account_id: str, period_start: date, progress) -> dict:
    """Връща записа от склада {"items", "keys", "next_path", "oldest"}, допълнен с новите
    сделки и (при нужда) с по-стари страници до началото на периода - HISTORY_LOOKBACK_DAYS."""
    needed = pd.Timestamp(period_start, tz="UTC") - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS)

    def on_page(page, count, oldest):
        until = f" · назад до {oldest:%d.%m.%Y}" if oldest is not None else ""
        progress.progress(min(page / HISTORY_MAX_PAGES, 1.0),
                          text=f"История: страница {page} · {count} нови записа{until} "
                               "(T212 позволява 1 заявка на 10 сек)")

    store = _history_store()
    with store["lock"]:
        entry = store["accounts"].get(account_id)
    if entry is None:
        items, next_path, _ = t212.fetch_order_history(base_url, auth_header, max_pages=HISTORY_MAX_PAGES,
                                                       stop_before=needed, on_page=on_page)
        entry = {"items": items, "next_path": next_path}
    else:
        # само новите сделки - обикновено една заявка
        new, _, _ = t212.fetch_order_history(base_url, auth_header, max_pages=HISTORY_MAX_PAGES,
                                             known_keys=entry["keys"], on_page=on_page)
        entry = {"items": new + entry["items"], "next_path": entry["next_path"], "oldest": entry["oldest"],
                 "keys": entry["keys"] | {t212.order_key(i) for i in new}}
        if entry["next_path"] and (entry["oldest"] is None or entry["oldest"] > needed):
            older, next_path, _ = t212.fetch_order_history(base_url, auth_header, max_pages=HISTORY_MAX_PAGES,
                                                           start_path=entry["next_path"], stop_before=needed,
                                                           on_page=on_page)
            entry["items"] += [i for i in older if t212.order_key(i) not in entry["keys"]]
            entry["next_path"] = next_path
    dates = [d for d in (t212.item_date(i) for i in entry["items"]) if d is not None]
    entry["oldest"] = min(dates) if dates else None
    entry["keys"] = {t212.order_key(i) for i in entry["items"]}
    with store["lock"]:
        store["accounts"][account_id] = entry
    return entry


def generate_ai_analysis_portfolio(open_df, closed_df, summary: dict, period_label: str, provider: str, api_key: str):

    open_text = (
        open_df.to_string(index=False) if not open_df.empty
        else "НЯМА текущо отворени позиции."
    )
    closed_text = (
        closed_df.drop(columns=[]).to_string(index=False) if not closed_df.empty
        else "НЯМА затворени сделки в избрания период."
    )

    prompt = f"""
    Ти си професионален суинг търговец, който прави преглед на резултатите на
    друг търговец за периода "{period_label}". Използвай СТРИКТНО само данните
    по-долу - те идват директно от Trading 212 API (реални изпълнени сделки и
    текущи позиции), не предполагай нищо извън тях.

    === ОТВОРЕНИ ПОЗИЦИИ В МОМЕНТА ===
    {open_text}

    === ЗАТВОРЕНИ СДЕЛКИ ЗА ПЕРИОДА "{period_label}" ===
    {closed_text}

    === ОБОБЩЕНА СТАТИСТИКА ЗА ПЕРИОДА ===
    Брой сделки: {summary.get('trades')}
    Печеливши: {summary.get('wins')}, Губещи: {summary.get('losses')}
    Win rate: {summary.get('win_rate')}%
    Обща реализирана P&L: {summary.get('total_pl')} €
    Средна печалба: {summary.get('avg_win')} €, Средна загуба: {summary.get('avg_loss')} €
    Най-добра сделка: {summary.get('best')}, Най-лоша сделка: {summary.get('worst')}

    ЗАДАЧА:
    1. Кратко обобщение на представянето за периода (2-3 изречения).
    2. Какво е минало добре и какво не, на база самите данни (без да гадаеш причини,
       които не личат от данните - напр. не измисляй "лоша пазарна конюнктура",
       ако няма индикация за това).
    3. Кратък коментар за текущите отворени позиции - има ли концентрация в
       един инструмент/сектор, голяма нереализирана загуба, която да следи.
    4. 2-3 конкретни, практични препоръки за следващия период (без финансови
       съвети от рода "купи/продай точно това", а по-скоро дисциплина/риск
       мениджмънт наблюдения на база самите числа).

    Бъди кратък и конкретен, удобен за преглед на телефон. Не давай дисклеймъри
    за инвестиционни съвети по-дълги от едно изречение, ако изобщо е нужно.
    """
    yield from stream_ai(prompt, provider, api_key, max_tokens=2048, gemini_model=gemini_model())


# Именувани T212 профили - всеки със свои Secrets ключове, за да могат
# няколко човека (напр. Наско + съпругата му) да следят P&L отделно, без
# ключовете им да се смесват. Добавяш нов профил, като добавиш ред тук и
# съответните T212_API_KEY_<SUFFIX> / T212_API_SECRET_<SUFFIX> в Secrets.
T212_ACCOUNTS = [
    {"label": "N", "slug": "me", "key_secret_name": "T212_API_KEY", "secret_secret_name": "T212_API_SECRET"},
    {"label": "T", "slug": "wife", "key_secret_name": "T212_API_KEY_WIFE", "secret_secret_name": "T212_API_SECRET_WIFE"},
]


def render_open_positions(df_open, slug: str):
    """Отворените позиции с текущата P&L; по подразбиране без частта в пайовете.
    Връща показаната таблица (тя отива и в AI анализа)."""
    section_header("📌 Отворени позиции", status="watch")
    no_pies = st.toggle("Без позициите в пайовете", value=True, key=f"t212_no_pies_{slug}",
                        help="Пайовете (дългосрочните кошници) не влизат в бройката и P&L")
    if no_pies:
        df_open = t212.exclude_pies(df_open)
    if df_open.empty:
        st.info("Нямаш текущо отворени позиции" + (" извън пайовете." if no_pies else "."))
        return df_open
    total_pl = df_open["P&L (€)"].sum()
    total_invested = df_open["Инвестирано (€)"].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Брой позиции", len(df_open))
    col2.metric("Общо инвестирано (€)", round(total_invested, 2))
    col3.metric("Нереализирана P&L (€)", round(total_pl, 2),
                delta=f"{total_pl / total_invested * 100:+.2f}%" if total_invested else None)
    if no_pies and "Частично в пай" in df_open.columns and df_open["Частично в пай"].any():
        st.caption("Инструментите с „Частично в пай“ са и в пай - за тях бройката и P&L са само за частта извън пая "
                   "(пропорционално, ориентировъчно).")
    st.dataframe(df_open, width="stretch", hide_index=True)
    return df_open


def render_portfolio_section():
    section_header(
        "💼 Портфолио & P&L (Trading 212)",
        status="info",
        subtitle="Само за четене - Portfolio + History. Никога не се използват права за поръчки/прехвърляния.",
    )

    account_labels = [a["label"] for a in T212_ACCOUNTS]
    selected_label = st.radio("Сметка:", account_labels, horizontal=True, key="t212_account_select")
    account = next(a for a in T212_ACCOUNTS if a["label"] == selected_label)
    slug = account["slug"]

    with st.expander(f"⚙️ Достъп до Trading 212 (read-only) — {selected_label}", expanded=False):
        t212_env = st.radio(
            "Среда", ["live", "demo"], horizontal=True, key=f"t212_env_{slug}",
            help="'live' е реалната сметка. Ползвай 'demo', ако тестваш с demo профил.",
        )
        t212_key = st.secrets.get(account["key_secret_name"], None)
        t212_secret = st.secrets.get(account["secret_secret_name"], None)
        if not t212_key:
            t212_key = st.text_input("T212 API Key", type="password", key=f"t212_key_input_{slug}")
        if not t212_secret:
            t212_secret = st.text_input("T212 API Secret", type="password", key=f"t212_secret_input_{slug}")
        st.caption(
            "⚠️ Ключът трябва да има ЕДИНСТВЕНО права за 'Portfolio' и 'History' (read). "
            "НЕ добавяй 'Orders' (write) права на ключа, който ползваш тук - за преглед "
            "на P&L не са нужни, а ограничават риска при евентуално изтичане на ключа. "
            f"По-добре запиши ключа/секрета трайно в Streamlit Secrets ({account['key_secret_name']} / "
            f"{account['secret_secret_name']}), вместо да ги въвеждаш всеки път тук."
        )

    col_preset, col_range = st.columns([1, 1.4])
    with col_preset:
        preset = st.selectbox(
            "Период за анализ на затворените сделки:",
            ["Този месец", "Миналия месец", "Тази седмица", "Последните 30 дни", "Последните 90 дни", "Тази година", "Персонализиран период"],
            index=0, key=f"t212_period_preset_{slug}",
        )
    custom_range = None
    if preset == "Персонализиран период":
        with col_range:
            custom_range = st.date_input(
                "От - До", value=(date.today() - timedelta(days=30), date.today()),
                key=f"t212_custom_range_{slug}",
            )
            if isinstance(custom_range, (tuple, list)) and len(custom_range) != 2:
                custom_range = None

    period_start, period_end = _period_bounds(preset, custom_range)
    st.caption(f"Избран период: {period_start} → {period_end}")

    if st.button("🔄 Зареди портфолио и история", type="primary", key=f"t212_load_btn_{slug}"):
        if not t212_key or not t212_secret:
            st.error("Липсва T212 API Key/Secret!")
        else:
            base_url = t212.T212_ENV_TO_BASE_URL[t212_env]
            auth_header = t212.build_auth_header(t212_key, t212_secret)
            progress = st.progress(0.0, text="Зареждам отворените позиции...")
            try:
                df_open = t212.fetch_open_positions(base_url, auth_header)
                st.session_state[f"t212_open_{slug}"] = df_open
                progress.progress(0.0, text=f"Позиции: {len(df_open)} ✓ · зареждам историята на сделките...")
                entry = load_order_history(base_url, auth_header, _account_id(t212_env, t212_key), period_start, progress)
                raw_orders = entry["items"]
                st.session_state[f"t212_closed_all_{slug}"] = t212.orders_to_dataframe(raw_orders)
                st.session_state[f"t212_history_from_{slug}"] = (entry["oldest"], entry["next_path"] is None)
                st.session_state[f"t212_loaded_at_{slug}"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            except PermissionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Грешка при връзка с Trading 212: {e}")
            finally:
                progress.empty()

    if st.button("⚡ Само текущата P&L на отворените позиции", key=f"t212_open_btn_{slug}",
                 help="Една бърза заявка, без историята на сделките"):
        if not t212_key or not t212_secret:
            st.error("Липсва T212 API Key/Secret!")
        else:
            try:
                st.session_state[f"t212_open_{slug}"] = t212.fetch_open_positions(
                    t212.T212_ENV_TO_BASE_URL[t212_env], t212.build_auth_header(t212_key, t212_secret))
                st.session_state[f"t212_open_at_{slug}"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            except PermissionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Грешка при връзка с Trading 212: {e}")

    df_open = st.session_state.get(f"t212_open_{slug}")
    df_closed_all = st.session_state.get(f"t212_closed_all_{slug}")

    if df_open is None:
        st.info(f"Натисни 'Зареди портфолио и история' или '⚡ Само текущата P&L', за да видиш данните на {selected_label}.")
        return
    if df_closed_all is None:
        st.caption(f"Позициите са от {st.session_state.get(f't212_open_at_{slug}', '?')} · историята не е заредена.")
        render_open_positions(df_open, slug)
        return

    oldest, complete = st.session_state.get(f"t212_history_from_{slug}", (None, False))
    history_note = ("цялата история" if complete else
                    f"история от {oldest:%d.%m.%Y}" if oldest is not None else "без история")
    st.caption(f"Последно заредено ({selected_label}): {st.session_state.get(f't212_loaded_at_{slug}', '?')} · "
               f"{history_note} (следващото зареждане тегли само новите сделки)")

    df_open = render_open_positions(df_open, slug)

    df_period = t212.filter_by_period(df_closed_all, period_start, period_end)
    summary = t212.summarize_closed_trades(df_period)

    section_header("📊 Затворени сделки (избран период)", status="go", subtitle=preset)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Сделки", summary["trades"])
    col2.metric("Win rate", f"{summary['win_rate']}%" if summary["win_rate"] is not None else "—")
    col3.metric("Реализирана P&L (€)", summary["total_pl"])
    col4.metric(
        "Най-добра / Най-лоша",
        f"{summary['best'][0]}" if summary["best"] else "—",
        f"{summary['best'][1]} € / {summary['worst'][1]} €" if summary["best"] and summary["worst"] else None,
    )
    if df_period.empty:
        st.info("Няма затворени сделки в избрания период.")
    else:
        st.dataframe(df_period, width="stretch", hide_index=True)

    st.divider()
    section_header("🤖 AI Анализ на представянето", status="info")
    provider = ai_provider()
    api_key = ai_api_key(provider, key=f"t212_ai_key_{slug}")

    if st.button(f"Генерирай AI анализ на портфолиото с {provider}", type="primary", key=f"t212_ai_btn_{slug}"):
        if not api_key:
            st.error(f"Липсва {AI_KEY_SECRETS[provider]} в Streamlit Secrets!")
        else:
            try:
                st.session_state[f"t212_ai_text_{slug}"] = st.write_stream(
                    generate_ai_analysis_portfolio(df_open, df_period, summary, preset, provider, api_key)
                )
            except Exception as e:
                show_ai_error(e, provider, key=f"t212_ai_{slug}")
    elif st.session_state.get(f"t212_ai_text_{slug}"):
        st.markdown(st.session_state[f"t212_ai_text_{slug}"])
