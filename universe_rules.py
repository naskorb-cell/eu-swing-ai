"""
universe_rules.py

Общи правила за подбор на универса - ползват се и от месечния
select_liquid_universe.py (твърд pre-screen), и от Streamlit приложението
(допълнително стесняване със slider-ите). Без streamlit/yfinance зависимости.
"""

import re

# --- Прагове по подразбиране (месечният скрипт записва само инструменти над тях) ---
STOCK_MIN_MARKET_CAP = 2_000_000_000     # € - large/стабилни mid cap
STOCK_MIN_TURNOVER = 5_000_000           # €/ден - само за акции на "родна" (ЕС/ЕИП) борса
STOCK_FOREIGN_MIN_TURNOVER = 50_000      # €/ден - чужди акции на вторична борса (US на Xetra/Gettex):
                                         # оборотът там не отразява реалната ликвидност, само
                                         # пазим от "мъртви" листвания с редки/стари цени
ETF_MIN_AUM = 100_000_000                # € (или USD - Yahoo връща валутата на фонда)
ETF_MIN_TURNOVER = 250_000               # €/ден - при ETF спредът зависи от маркет мейкъра и
                                         # базовите активи, затова по-нисък праг от акциите
ETF_STRONG_TURNOVER = 1_000_000          # €/ден - над това ETF не отпада заради AUM: Yahoo често дава
                                         # грешен/остарял AUM за европейски листвания (напр. iShares
                                         # Bitcoin с 1.2 млрд. € реален AUM отпадаше като "малък")

# ISIN държави от ЕС/ЕИП: акция с такъв ISIN се търгува основно на европейска борса
# и оборотът ѝ в Yahoo е представителен. Иначе (US, CH, GB, CA...) листването в
# T212 е вторично и се гледа само капитализацията.
EEA_ISIN_COUNTRIES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    "IS", "LI", "NO",
}

# Ливъриджнати/short/inverse ETP-та: short = залог надолу (противоречи на long-only),
# daily leveraged губят стойност при държане повече от ден-два (volatility decay)
LEVERAGED_ETP_PATTERN = re.compile(r"\b(short|leveraged?|ultra|bear|inverse|boost)\b|\b\d+(\.\d+)?x\b", re.IGNORECASE)


# Парични (overnight/cash) и облигационни фондове - почти без движение/с друг
# характер на цената; безсмислени за swing по структура. Само за тип ETF.
CASH_BOND_FUND_PATTERN = re.compile(
    r"\b(cash|money market|overnight|€str|estr|eonia|sonia|t-bill|treasury bills?|bonds?|govt|government|"
    r"treasury|corporate|corp|aggregate|floating rate|inflation[- ]linked|gilts?|ultrashort|ultra short)\b",
    re.IGNORECASE,
)


def is_cash_or_bond_fund(name: str) -> bool:
    return bool(CASH_BOND_FUND_PATTERN.search(name or ""))


def is_leveraged_or_short_etp(name: str) -> bool:
    return bool(LEVERAGED_ETP_PATTERN.search(name or ""))


def is_home_listing(isin: str) -> bool:
    return (isin or "")[:2].upper() in EEA_ISIN_COUNTRIES


def passes_liquidity(item: dict, stock_min_cap=STOCK_MIN_MARKET_CAP, stock_min_turnover=STOCK_MIN_TURNOVER,
                     etf_min_aum=ETF_MIN_AUM, etf_min_turnover=ETF_MIN_TURNOVER):
    """Проверява инструмент (dict с type, isin, avg_dollar_volume, market_cap, aum)
    срещу критериите. Връща (True, None) или (False, причина).
    ETF без данни за AUM в Yahoo НЕ отпада - проверява се само по оборот; ETF с
    оборот >= ETF_STRONG_TURNOVER не отпада и при малък AUM по Yahoo."""
    turnover = item.get("avg_dollar_volume") or 0
    inst_type = item.get("type")
    if inst_type == "STOCK":
        cap = item.get("market_cap")
        if not cap:
            return False, "Акция: няма данни за капитализация"
        if cap < stock_min_cap:
            return False, "Акция: малка капитализация"
        min_turnover = stock_min_turnover if is_home_listing(item.get("isin")) else STOCK_FOREIGN_MIN_TURNOVER
        if turnover < min_turnover:
            return False, "Акция: нисък оборот"
        return True, None
    if inst_type == "ETF":
        aum = item.get("aum")
        if aum and aum < etf_min_aum and turnover < max(ETF_STRONG_TURNOVER, etf_min_turnover):
            return False, "ETF: малък AUM"
        if turnover < etf_min_turnover:
            return False, "ETF: нисък оборот"
        return True, None
    return False, f"Неподдържан тип: {inst_type}"
