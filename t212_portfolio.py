"""
Trading 212 Portfolio & P&L helpers.

САМО read-only endpoints (Portfolio + History) - НИКОГА Orders/write.
Trading 212 API няма endpoint за прехвърляне на пари, така че дори при
изтекъл/компрометиран ключ с тези права, щетата е ограничена до преглед
на данни, не до действия с парите.

Auth схема, идентична с fetch_eu_instruments.py:
    token = base64.b64encode(f"{key}:{secret}".encode()).decode()
    Authorization: Basic <token>
"""
import base64
import time
from datetime import datetime, timezone

import pandas as pd
import requests

T212_ENV_TO_BASE_URL = {
    "live": "https://live.trading212.com/api/v0",
    "demo": "https://demo.trading212.com/api/v0",
}


def build_auth_header(api_key: str, api_secret: str) -> str:
    token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
    return f"Basic {token}"


def _get(base_url: str, path: str, auth_header: str, params: dict = None, timeout: int = 20, max_retries: int = 4):
    """GET с автоматичен retry при 429 (T212 - History лимитът е само 6/мин).
    Уважава 'Retry-After' хедъра, ако е върнат, иначе expon. backoff."""
    url = f"{base_url}{path}"
    attempt = 0
    while True:
        resp = requests.get(url, headers={"Authorization": auth_header}, params=params or {}, timeout=timeout)
        if resp.status_code == 401:
            raise PermissionError(
                "401 - невалиден ключ/секрет, или ключът няма права за Portfolio/History."
            )
        if resp.status_code == 429:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError("429 - твърде много заявки към Trading 212 дори след изчакване. Опитай пак след минута.")
            wait_s = resp.headers.get("Retry-After")
            wait_s = float(wait_s) if wait_s else 11.0 * attempt
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        return resp.json()


def fetch_open_positions(base_url: str, auth_header: str) -> pd.DataFrame:
    """GET /equity/portfolio - текущо отворени позиции с нереализирана P&L."""
    data = _get(base_url, "/equity/portfolio", auth_header)
    if not data:
        return pd.DataFrame()

    rows = []
    for pos in data:
        qty = pos.get("quantity", 0) or 0
        avg_price = pos.get("averagePrice", 0) or 0
        current_price = pos.get("currentPrice", 0) or 0
        ppl = pos.get("ppl", 0) or 0  # profit/loss в EUR (или базовата валута на сметката)
        cost_basis = qty * avg_price
        ppl_pct = (ppl / cost_basis * 100) if cost_basis else 0
        rows.append({
            "Тикер": pos.get("ticker", "?"),
            "Кол-во": qty,
            "Ср. цена (вход)": round(avg_price, 2),
            "Текуща цена": round(current_price, 2),
            "P&L (€)": round(ppl, 2),
            "P&L (%)": round(ppl_pct, 2),
            "Инвестирано (€)": round(cost_basis, 2),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="P&L (€)", ascending=False).reset_index(drop=True)
    return df


def fetch_order_history(base_url: str, auth_header: str, max_pages: int = 20) -> list:
    """GET /equity/history/orders, страница по страница.
    T212 връща 'nextPagePath' - ГОТОВ относителен път с вече вграден в него
    cursor query-параметър (напр. '/api/v0/equity/history/orders?cursor=...&limit=50').
    Той трябва да се използва КАКТО Е - да не се увива втори път в нов 'cursor'
    параметър (това причиняваше 400 Bad Request с двойно екраниран URL)."""
    from urllib.parse import urlsplit, parse_qs

    all_items = []
    path = "/equity/history/orders"
    params = {"limit": 50}

    for _ in range(max_pages):
        data = _get(base_url, path, auth_header, params=params)
        items = data.get("items", []) if isinstance(data, dict) else data
        if not items:
            break
        all_items.extend(items)

        next_page_path = data.get("nextPagePath") if isinstance(data, dict) else None
        if not next_page_path:
            break

        # base_url вече завършва на /api/v0 - маха се, ако nextPagePath го повтаря
        if next_page_path.startswith("/api/v0"):
            next_page_path = next_page_path[len("/api/v0"):]

        split = urlsplit(next_page_path)
        path = split.path
        params = {k: v[0] for k, v in parse_qs(split.query).items()}

        time.sleep(11)  # History лимитът е 6/мин (~1 заявка на 10 сек) - пазим резерв
    return all_items


def orders_to_dataframe(raw_orders: list) -> pd.DataFrame:
    """Превръща суровите order записи в плосък DataFrame, само за FILLED
    (реално изпълнени) поръчки с реализирана P&L от walletImpact."""
    rows = []
    for item in raw_orders:
        order = item.get("order", item)
        fill = item.get("fill", {})
        wallet = item.get("walletImpact", {})

        status = order.get("status", "")
        if status and status.upper() not in ("FILLED", "EXECUTED"):
            continue

        realised_pl = wallet.get("realisedProfitLoss")
        if realised_pl is None:
            # някои по-стари order записи може да нямат това поле, пропускаме ги
            continue

        filled_at = fill.get("filledAt") or order.get("createdAt")
        rows.append({
            "Дата": filled_at,
            "Тикер": order.get("ticker", "?"),
            "Страна": order.get("side", "?"),
            "Кол-во": fill.get("quantity", order.get("filledQuantity", 0)),
            "Цена": fill.get("price"),
            "Реализирана P&L (€)": realised_pl,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce", utc=True)
    df = df.dropna(subset=["Дата"]).sort_values(by="Дата", ascending=False).reset_index(drop=True)
    return df


def filter_by_period(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """start/end - datetime.date или datetime.datetime. Филтрира df по
    колоната 'Дата' (timezone-aware UTC)."""
    if df.empty:
        return df
    start_ts = pd.Timestamp(start, tz="UTC") if not isinstance(start, pd.Timestamp) else start
    end_ts = pd.Timestamp(end, tz="UTC") if not isinstance(end, pd.Timestamp) else end
    end_ts = end_ts + pd.Timedelta(days=1)  # включва целия последен ден
    return df[(df["Дата"] >= start_ts) & (df["Дата"] < end_ts)].reset_index(drop=True)


def summarize_closed_trades(df_period: pd.DataFrame) -> dict:
    """Обобщена статистика за реализираните сделки в избрания период."""
    if df_period.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": None,
            "total_pl": 0.0, "avg_win": None, "avg_loss": None,
            "best": None, "worst": None,
        }
    pl = df_period["Реализирана P&L (€)"]
    wins = pl[pl > 0]
    losses = pl[pl < 0]
    best_row = df_period.loc[pl.idxmax()]
    worst_row = df_period.loc[pl.idxmin()]
    return {
        "trades": len(df_period),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(df_period) * 100, 1) if len(df_period) else None,
        "total_pl": round(pl.sum(), 2),
        "avg_win": round(wins.mean(), 2) if not wins.empty else None,
        "avg_loss": round(losses.mean(), 2) if not losses.empty else None,
        "best": (best_row["Тикер"], round(best_row["Реализирана P&L (€)"], 2)),
        "worst": (worst_row["Тикер"], round(worst_row["Реализирана P&L (€)"], 2)),
    }
