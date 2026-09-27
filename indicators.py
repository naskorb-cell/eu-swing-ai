"""Чисти pandas/numpy изчисления: индикатори, swing точки, пазарна структура, свещи."""

from datetime import date

import numpy as np
import pandas as pd

OHLC_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLCV свещи в по-голям таймфрейм (напр. "W" седмични, "4h")."""
    return df.resample(rule).agg(OHLC_AGG).dropna()


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
        # argmax/argmin връщат първото срещане - при равни върхове/дъна в
        # прозореца се маркира само най-левият, вместо две съседни swing точки
        if window_h.argmax() == order:
            swing_high[i] = True
        if window_l.argmin() == order:
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


def average_true_range(df: pd.DataFrame, period: int = 14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None


def is_overextended(df_with_swings: pd.DataFrame, lookback_swings: int = 4):
    """Правило 'спри след 3+ CP модела в една посока': поредни по-високи
    swing highs без сериозна корекция = пренатегнат тренд, риск от изчерпване."""
    highs = df_with_swings.loc[df_with_swings["SwingHigh"], "High"]
    if len(highs) < 2:
        return False, 0
    recent = highs.tail(lookback_swings).to_numpy()
    consecutive = 1
    max_consecutive = 1
    for i in range(1, len(recent)):
        if recent[i] > recent[i - 1]:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 1
    return max_consecutive >= 3, int(max_consecutive)


def find_fresh_demand_zone(df: pd.DataFrame, max_base: int = 6):
    """Supply & Demand (Alfonso Mores / Set & Forget) - Drop-Base-Rally детекция,
    само demand (bullish) зони, тъй като приложението е long-only:
      - база от 1-6 тесни свещи (тяло <= 50% от диапазона)
      - последвана от departure импулс с диапазон >= 2x средния диапазон
        на базата (2:1 imbalance), затварящ силно бичи (горна 20%)
    Връща НАЙ-СКОРОШНАТА невалидирана ("не пробита надолу") зона, с брой
    ретестове от формирането ѝ насам, или None ако няма такава."""
    o, h, l, c = df["Open"].to_numpy(), df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    n = len(df)
    ranges = h - l
    bodies = np.abs(c - o)
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(ranges > 0, bodies / ranges, 1.0)

    for j in range(n - 1, max_base, -1):
        if ranges[j] <= 0:
            continue
        strong_bullish_close = c[j] > o[j] and (c[j] - l[j]) >= 0.8 * ranges[j]
        if not strong_bullish_close:
            continue
        for base_width in range(1, max_base + 1):
            base_start = j - base_width
            if base_start < 0:
                break
            sl = slice(base_start, j)
            base_ranges = ranges[sl]
            base_body_ratio = body_ratio[sl]
            if np.any(base_ranges <= 0) or not np.all(base_body_ratio <= 0.5):
                continue
            avg_base_range = base_ranges.mean()
            if avg_base_range <= 0:
                continue
            imbalance_ratio = ranges[j] / avg_base_range
            if imbalance_ratio < 2.0:
                continue
            proximal, distal = float(h[sl].max()), float(l[sl].min())
            if proximal <= distal:
                continue

            invalidated, retests, in_zone = False, 0, False
            for k in range(j + 1, n):
                if c[k] < distal:
                    invalidated = True
                    break
                touching = l[k] <= proximal
                if touching and not in_zone:
                    retests += 1
                    in_zone = True
                elif not touching:
                    in_zone = False
            if invalidated:
                continue

            return {
                "formed_idx": j, "base_width": base_width, "proximal": proximal, "distal": distal,
                "imbalance_ratio": float(imbalance_ratio), "avg_base_body_ratio": float(base_body_ratio.mean()),
                "retests": retests, "formed_date": df.index[j],
            }
    return None


def alternating_swings(df_with_swings: pd.DataFrame):
    """Свежда swing точките до строго редуваща се поредица high/low/high/...
    Два поредни high-а (без low между тях) се сливат в по-високия, два поредни
    low-а - в по-ниския. Без това сравнението "последни 2 highs vs последни 2
    lows" лесно сравнява точки от едно и също движение. Връща списък от
    (timestamp, "H"/"L", цена)."""
    sub = df_with_swings[df_with_swings["SwingHigh"] | df_with_swings["SwingLow"]]
    points = []
    for ts, is_h, is_l, hi, lo in zip(sub.index, sub["SwingHigh"], sub["SwingLow"], sub["High"], sub["Low"]):
        candidates = []
        if is_h:
            candidates.append(("H", float(hi)))
        if is_l:
            candidates.append(("L", float(lo)))
        if len(candidates) == 2 and points and points[-1][1] == "H":
            candidates.reverse()  # outside bar след high: първо low, после high
        for kind, price in candidates:
            if points and points[-1][1] == kind:
                prev_price = points[-1][2]
                if (kind == "H" and price > prev_price) or (kind == "L" and price < prev_price):
                    points[-1] = (ts, kind, price)
            else:
                points.append((ts, kind, price))
    return points


def swing_structure(df_with_swings: pd.DataFrame):
    """Структура по редуващи се swing точки: последните два high-а и два low-а
    и дали имаме HH+HL (възходяща структура). None при недостатъчно точки."""
    points = alternating_swings(df_with_swings)
    highs = [p for p in points if p[1] == "H"]
    lows = [p for p in points if p[1] == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    return {
        "prev_high": highs[-2][2], "last_high": highs[-1][2], "last_high_idx": highs[-1][0],
        "prev_low": lows[-2][2], "last_low": lows[-1][2], "last_low_idx": lows[-1][0],
        "uptrend": highs[-1][2] > highs[-2][2] and lows[-1][2] > lows[-2][2],
    }


def resample_session_halves(intraday: pd.DataFrame) -> pd.DataFrame:
    """'4ч' свещи, подравнени към търговската сесия: всеки ден се разделя на
    две половини по брой часови свещи (за 09:00-17:30 -> 09-14 и 14-17:30).
    resample("4h") групира от полунощ и за EU борсите дава разкъсани/непълни
    свещи (напр. 16:00-17:30)."""
    rows, stamps = [], []
    for _, day_df in intraday.groupby(intraday.index.date):
        split = (len(day_df) + 1) // 2
        for part in (day_df.iloc[:split], day_df.iloc[split:]):
            if part.empty:
                continue
            stamps.append(part.index[0])
            rows.append({
                "Open": float(part["Open"].iloc[0]), "High": float(part["High"].max()),
                "Low": float(part["Low"].min()), "Close": float(part["Close"].iloc[-1]),
                "Volume": float(part["Volume"].sum()),
            })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(stamps))


def drop_incomplete_week(weekly: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Маха текущата (незатворена) седмица. Седмицата се смята за затворена,
    ако последната дневна свещ е петък и този петък вече е минал."""
    last_date = daily_df.index[-1].date()
    week_closed = last_date.weekday() == 4 and date.today() > last_date
    return weekly if week_closed else weekly.iloc[:-1]


def detect_structure_events(df_with_swings: pd.DataFrame, choch_max_age: int = 3):
    """Photon Trading MTF Phases логика върху 4ч ('Internal'):
      - Pro Internal: HH+HL и цената държи над последния internal low;
      - Counter Internal: lower high, lower low или затваряне под последния
        internal low (пулбекът срещу дневния тренд тече);
      - бичи CHoCH: ПЪРВОТО затваряне над последния internal swing high.
        Брои се за "сега", само ако е станало в последните choch_max_age свещи -
        иначе входът вече е изпуснат.
    Връща None, ако няма достатъчно swing точки за преценка."""
    s = swing_structure(df_with_swings)
    if s is None:
        return None

    closes = df_with_swings["Close"]
    lows = df_with_swings["Low"]
    current_close = float(closes.iloc[-1])

    lower_high = s["last_high"] < s["prev_high"]
    lower_low = s["last_low"] < s["prev_low"]
    broke_last_low = current_close < s["last_low"]
    internal_state = "counter" if (lower_high or lower_low or broke_last_low) else "pro"

    # CHoCH ниво = последният internal swing high; търсим първото затваряне над него след формирането му
    choch_level = s["last_high"]
    after_high = closes.loc[closes.index > s["last_high_idx"]]
    breaks = after_high[after_high > choch_level]
    choch_bars_ago = None
    if not breaks.empty:
        choch_bars_ago = int(len(closes) - 1 - closes.index.get_loc(breaks.index[0]))
    choch_bullish_now = choch_bars_ago is not None and choch_bars_ago < choch_max_age

    if internal_state == "pro":
        reference_low = s["last_low"]  # последният internal HL
    else:
        # дъното на пулбека: най-ниското от последния swing high насам
        # (може още да не е потвърдено като swing low)
        pullback_lows = lows.loc[lows.index > s["last_high_idx"]]
        reference_low = float(pullback_lows.min()) if not pullback_lows.empty else s["last_low"]

    return {
        "internal_state": internal_state, "choch_bullish_now": choch_bullish_now,
        "choch_bars_ago": choch_bars_ago, "choch_level": choch_level,
        "reference_low": reference_low, "internal_hl": s["last_low"],
    }
