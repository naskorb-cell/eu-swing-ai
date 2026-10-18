"""Качествени показатели за ETF-ите (за тях няма Fair Value / финансово здраве - фондът не е компания):

- „🏛 ETF сила“ - с колко процентни пункта ETF-ът бие MSCI World за 3 и 6 месеца (водещите сектори по-често
  отскачат от подкрепата, изоставащите я пробиват);
- „📉 От върха“ - колко е под 52-седмичния връх (3-15% = отстъпка в тренда, зоната на подкрепата).

Чисти функции - ползват ги и скенерът за тренда, и Photon.
"""

import re

import pandas as pd

ETF_BENCHMARK = "EUNL.DE"  # iShares Core MSCI World в EUR
PULLBACK_ZONE = (-15.0, -3.0)  # % под 52-седмичния връх
ETF_NAME_HINTS = re.compile(r"\b(ETF|ETC|ETN|UCITS|iShares|Xtrackers|Vanguard|Amundi|SPDR|WisdomTree|Invesco|"
                            r"Lyxor|VanEck|Vaneck|L&G|HSBC MSCI|BNP Paribas Easy|Global X)\b|\((Acc|Dist)\)", re.I)
ETF_COLUMN_HELP = {
    "🏛 ETF сила": "С колко процентни пункта ETF-ът бие MSCI World (EUNL) за 3 / 6 месеца: 🟢 и в двата · 🔴 изостава "
                  "и в двата · ⚪ смесено. Водещите сектори по-често отскачат от подкрепата",
    "📉 От върха": f"Колко е под 52-седмичния връх: 🟢 {-PULLBACK_ZONE[1]:.0f}-{-PULLBACK_ZONE[0]:.0f}% (отстъпка в "
                  "тренда - зоната на подкрепата) · ⚪ на самия връх · 🔴 дълбок спад",
}


def is_etf(kind: str, name: str) -> bool:
    """ETF по вида от месечния подбор; без вид (ръчно добавени, позиции) - по името."""
    return kind == "ETF" if kind else bool(ETF_NAME_HINTS.search(name or ""))


def returns(daily: pd.DataFrame) -> dict:
    """{ret_3m, ret_6m, from_high} от дневните цени (поне ~6 месеца) или {}."""
    if daily is None or daily.empty:
        return {}
    close = daily["Close"].dropna()
    if len(close) < 127:
        return {}
    price = float(close.iloc[-1])
    high = float(daily["High"].iloc[-252:].max())
    return {"ret_3m": round(100 * (price / float(close.iloc[-64]) - 1), 1),
            "ret_6m": round(100 * (price / float(close.iloc[-127]) - 1), 1),
            "from_high": round(100 * (price / high - 1), 1) if high else None}


def relative(m: dict, bench: dict) -> dict:
    """Силата спрямо бенчмарка: {rs_3m, rs_6m} в процентни пункта."""
    if not m or not bench:
        return {"rs_3m": None, "rs_6m": None}
    return {"rs_3m": round(m["ret_3m"] - bench["ret_3m"], 1), "rs_6m": round(m["ret_6m"] - bench["ret_6m"], 1)}


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


def etf_strength(rs3, rs6) -> str:
    """„🟢 +5 / +9“ - бие MSCI World и за 3, и за 6 месеца; 🔴 изостава и в двата; ⚪ смесено."""
    if _missing(rs3) or _missing(rs6):
        return "—"
    dot = "🟢" if rs3 > 0 and rs6 > 0 else "🔴" if rs3 < 0 and rs6 < 0 else "⚪"
    return f"{dot} {rs3:+.0f} / {rs6:+.0f}"


def pullback_label(v) -> str:
    """„🟢 -7%“ - 3-15% под върха (отстъпка в тренда); ⚪ на върха; 🔴 над 15% под върха."""
    if _missing(v):
        return "—"
    lo, hi = PULLBACK_ZONE
    dot = "🟢" if lo <= v <= hi else "⚪" if v > hi else "🔴"
    return f"{dot} {v:.0f}%"
