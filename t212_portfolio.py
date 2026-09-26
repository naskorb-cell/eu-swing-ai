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


def _pick(d: dict, *candidates, default=None):
    """Връща първата налична стойност (не-None) от списък възможни имена
    на ключове - T212's публично API не документира изрично всяко поле,
    затова се пробват няколко варианта."""
    for c in candidates:
        v = d.get(c)
        if v is not None:
            return v
    return default


def _extract_order_fields(item: dict):
    """Плоско извлича полетата от един суров запис - работи както с
    вложена форма ({'order': {...}, 'fill': {...}}), така и с плоска форма
    (полетата директно в item), защото не сме сигурни коя точно връща
    текущата версия на T212 API."""
    order = item.get("order") if isinstance(item.get("order"), dict) else item
    fill = item.get("fill") if isinstance(item.get("fill"), dict) else {}

    status = _pick(order, "status", "orderStatus", default="")
    ticker = _pick(order, "ticker", "instrumentCode", "symbol", default="?")

    qty_raw = _pick(fill, "quantity", default=None)
    if qty_raw is None:
        qty_raw = _pick(order, "filledQuantity", "quantity", default=None)

    price = _pick(fill, "price", default=None)
    if price is None:
        price = _pick(order, "fillPrice", "averagePrice", "limitPrice", "stopPrice", default=None)

    filled_value = _pick(order, "filledValue", "filledCost", "value", default=None)

    side = _pick(order, "side", default=None)  # може изобщо да липсва

    date_val = _pick(fill, "filledAt", default=None)
    if date_val is None:
        date_val = _pick(order, "dateExecuted", "dateModified", "dateCreated", "createdAt", default=None)

    return {
        "status": status, "ticker": ticker, "qty_raw": qty_raw,
        "price": price, "filled_value": filled_value, "side": side, "date": date_val,
    }


def orders_to_dataframe(raw_orders: list) -> pd.DataFrame:
    """Превръща суровите order записи в плосък DataFrame на РЕАЛИЗИРАНИ P&L
    сделки, изчислени сами по среднопретеглена себестойност (average cost),
    защото T212's публично /equity/history/orders API НЕ връща готово поле
    за реализирана печалба на поръчка (за разлика от по-ранен допуск).

    Логика: сортираме всички FILLED поръчки хронологично (най-старата
    първо). За всеки тикер пазим текущо количество + обща себестойност.
    BUY увеличава позицията. SELL реализира P&L спрямо средната цена на
    придобиване към момента на продажбата и намалява позицията."""
    parsed = []
    for item in raw_orders:
        f = _extract_order_fields(item)
        if f["status"] and str(f["status"]).upper() not in ("FILLED", "EXECUTED", "FILLED_FULLY", "COMPLETED"):
            continue
        if f["date"] is None:
            continue
        parsed.append(f)

    if not parsed:
        return pd.DataFrame()

    for f in parsed:
        f["date_ts"] = pd.to_datetime(f["date"], errors="coerce", utc=True)
    parsed = [f for f in parsed if pd.notna(f["date_ts"])]
    parsed.sort(key=lambda f: f["date_ts"])

    positions: dict[str, dict] = {}
    closed_rows = []

    for f in parsed:
        ticker = f["ticker"]
        qty_raw = f["qty_raw"]
        price = f["price"]
        filled_value = f["filled_value"]

        # Определяне на количество и цена, ако липсва някое от двете
        if qty_raw is None and filled_value is not None and price:
            qty_raw = filled_value / price if price else 0
        if price is None and filled_value is not None and qty_raw:
            price = abs(filled_value) / abs(qty_raw) if qty_raw else 0
        if qty_raw is None or price is None:
            continue

        # Определяне на страна (BUY/SELL): по explicit 'side', иначе по
        # знака на количеството/сумата (отрицателно = продажба в T212 API).
        side = (f["side"] or "").upper()
        if side not in ("BUY", "SELL"):
            signed = qty_raw if qty_raw else filled_value
            side = "SELL" if (signed is not None and signed < 0) else "BUY"

        qty = abs(qty_raw)
        price = abs(price)
        pos = positions.setdefault(ticker, {"qty": 0.0, "cost": 0.0})

        if side == "BUY":
            pos["qty"] += qty
            pos["cost"] += qty * price
        else:  # SELL
            if pos["qty"] > 0:
                avg_cost = pos["cost"] / pos["qty"]
            else:
                avg_cost = price  # нямаме предишна позиция (напр. история преди периода на API достъп) - P&L=0
            sell_qty = min(qty, pos["qty"]) if pos["qty"] > 0 else qty
            realized = (price - avg_cost) * sell_qty
            pos["qty"] = max(pos["qty"] - qty, 0.0)
            pos["cost"] = max(pos["cost"] - avg_cost * sell_qty, 0.0)
            closed_rows.append({
                "Дата": f["date_ts"],
                "Тикер": ticker,
                "Страна": "SELL",
                "Кол-во": round(qty, 4),
                "Цена": round(price, 4),
                "Реализирана P&L (€)": round(realized, 2),
            })

    df = pd.DataFrame(closed_rows)
    if df.empty:
        return df
    df = df.sort_values(by="Дата", ascending=False).reset_index(drop=True)
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
