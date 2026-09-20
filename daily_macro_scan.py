"""
daily_macro_scan.py

Дневен макро скенер: пуска се веднъж на ден (виж .github/workflows/daily_macro.yml),
преди отваряне на европейските борси. Два източника, комбинирани:

  1) FMP (Financial Modeling Prep) Economic Indicators + Treasury Rates -
     РЕАЛНИ числа (не AI преценка): текущ Fed funds rate, инфлация (CPI),
     доходност на 10-годишни US Treasury облигации. Безплатен tier, 250
     заявки/ден - тук ползваме само 3-4.
  2) Claude + вградения web search tool - прочита днешните новини/анализи
     от реномирани източници и генерира резюме + теми, като му подаваме
     числата от FMP като твърда котва (Claude не гадае текущата лихва,
     а коментира спрямо реалната стойност, която сме му дали).

Резултатът се записва в daily_macro_signal.json в корена на repo-то и се чете
от combined_screener.py / multi_timeframe_screener.py (виж render_macro_section()).
"""

import json
import os
import re
from datetime import datetime, timezone

import requests
from anthropic import Anthropic

OUTPUT_FILE = "daily_macro_signal.json"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def fetch_fmp_macro_snapshot(api_key: str) -> dict:
    """Тегли няколко ключови макро числа от FMP: Fed funds rate, CPI
    (инфлация), 10Y Treasury доходност. При грешка/липсващ ключ връща
    празен dict - скриптът продължава, само без числовата котва."""
    if not api_key:
        return {}

    snapshot = {}

    def _get_latest(name: str):
        try:
            resp = requests.get(
                f"{FMP_BASE_URL}/economic-indicators",
                params={"name": name, "apikey": api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]  # най-новата стойност е първа
        except (requests.RequestException, ValueError):
            return None
        return None

    fed = _get_latest("federalFunds")
    if fed:
        snapshot["fed_funds_rate_pct"] = fed.get("value")
        snapshot["fed_funds_rate_date"] = fed.get("date")

    cpi = _get_latest("CPI")
    if cpi:
        snapshot["cpi_index"] = cpi.get("value")
        snapshot["cpi_date"] = cpi.get("date")

    try:
        resp = requests.get(
            f"{FMP_BASE_URL}/treasury-rates", params={"apikey": api_key}, timeout=15
        )
        resp.raise_for_status()
        rates = resp.json()
        if isinstance(rates, list) and rates:
            latest = rates[0]
            snapshot["treasury_10y_pct"] = latest.get("year10")
            snapshot["treasury_2y_pct"] = latest.get("year2")
            snapshot["treasury_date"] = latest.get("date")
    except (requests.RequestException, ValueError):
        pass

    return snapshot


SYSTEM_PROMPT_TEMPLATE = """Ти си макроикономически анализатор за суинг търговец, който търгува
само EUR-деноминирани акции/ETF-и на европейски (ЕС/ЕИП) борси плюс злато/сребро.

РЕАЛНИ ЧИСЛА ОТ FMP (стабилни, проверени данни - използвай ги като котва,
НЕ гадай текущата лихва или доходност, а коментирай спрямо тези стойности):
{fmp_snapshot_text}

Използвай web search, за да прегледаш днешните новини и анализи от РЕНОМИРАНИ и
проверени източници: Reuters, Bloomberg, Financial Times, Wall Street Journal,
MarketWatch, официални съобщения на Федералния резерв и ЕЦБ. Игнорирай форуми,
блогове без редакторски контрол, и източници без ясна репутация.

Фокусирай се конкретно върху:
- Очаквания за движение на лихвения процент на Федералния резерв (и ЕЦБ) -
  повишение, понижение, пауза - и как пазарът им реагира/очаква да реагира.
  Сравни пазарните очаквания с текущата реална лихва от FMP данните по-горе.
- Посока на златото и среброто (напр. очаквано поскъпване при понижение на
  лихвите, "safe haven" търсене, коментари на анализатори).
- Други конкретни активи, сектори или теми, които реномирани анализатори
  посочват като трендиращи или очакващи движение в следващите дни/седмици.

ЖЕЛЕЗНИ ПРАВИЛА:
- Само реални, проверими новини от днес/последните 24-48 часа - не гадай.
- Ако няма ясен сигнал за дадена тема, пропусни я - не измисляй заместител.
- В "keywords" пиши имена, с които активът реално се търси (напр. "Gold",
  "Silver", "Newmont", "SAP"), не описателни фрази.
- Отговори САМО с валиден JSON блок, обграден в ```json ... ``` - без друг
  текст преди или след него.

Формат:
```json
{{
  "summary_bg": "2-4 изречения на български, обобщаващи макро картината за деня",
  "themes": [
    {{
      "theme": "кратко заглавие на темата",
      "keywords": ["Gold", "Silver"],
      "reasoning_bg": "едно изречение защо, с позоваване на източника (по име, не линк)"
    }}
  ]
}}
```
Ако не намериш нищо съществено, върни themes: [] и обясни защо в summary_bg."""


def extract_json_block(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError(f"Не намерих JSON в отговора на Claude:\n{text[:500]}")
    return json.loads(match.group(1))


def format_fmp_snapshot(snapshot: dict) -> str:
    if not snapshot:
        return "(FMP данни не бяха достъпни за днес - разчитай само на web search.)"
    lines = []
    if "fed_funds_rate_pct" in snapshot:
        lines.append(f"- Fed Funds Rate: {snapshot['fed_funds_rate_pct']}% (към {snapshot.get('fed_funds_rate_date', '?')})")
    if "cpi_index" in snapshot:
        lines.append(f"- US CPI индекс: {snapshot['cpi_index']} (към {snapshot.get('cpi_date', '?')})")
    if "treasury_10y_pct" in snapshot:
        lines.append(f"- 10Y US Treasury доходност: {snapshot['treasury_10y_pct']}% (към {snapshot.get('treasury_date', '?')})")
    if "treasury_2y_pct" in snapshot:
        lines.append(f"- 2Y US Treasury доходност: {snapshot['treasury_2y_pct']}%")
    return "\n".join(lines) if lines else "(FMP не върна стойности - разчитай само на web search.)"


def run_daily_macro_scan(anthropic_api_key: str, fmp_api_key: str = None) -> dict:
    fmp_snapshot = fetch_fmp_macro_snapshot(fmp_api_key)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(fmp_snapshot_text=format_fmp_snapshot(fmp_snapshot))

    client = Anthropic(api_key=anthropic_api_key)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=8192,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ValueError(
            f"Claude върна празен текстов отговор (stop_reason: {response.stop_reason}). "
            "Може да е изразходвал max_tokens само за web search - опитай пак."
        )

    parsed = extract_json_block(text)
    parsed["generated_at"] = datetime.now(timezone.utc).isoformat()
    if fmp_snapshot:
        parsed["fmp_snapshot"] = fmp_snapshot
    return parsed


def main():
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise SystemExit("Липсва ANTHROPIC_API_KEY в environment.")
    fmp_api_key = os.environ.get("FMP_API_KEY")  # опционален - скриптът работи и без него

    result = run_daily_macro_scan(anthropic_api_key, fmp_api_key)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_themes = len(result.get("themes", []))
    fmp_status = "с FMP числа" if result.get("fmp_snapshot") else "БЕЗ FMP числа (провери FMP_API_KEY)"
    print(f"Записах {OUTPUT_FILE}: {n_themes} теми, {fmp_status}.")


if __name__ == "__main__":
    main()
