"""
daily_macro_scan.py

Дневен макро скенер: пуска се веднъж на ден (виж .github/workflows/daily_macro.yml),
преди отваряне на европейските борси. Вика Claude с вградения web search tool да
прегледа реномирани икономически източници (Reuters, Bloomberg, WSJ, FT,
MarketWatch, официални изявления на Фед/ЕЦБ) и да извади:

  1) кратко резюме на макро картината за деня (лихвени очаквания, злато/сребро,
     други трендиращи теми по експертни анализи),
  2) конкретен списък от активи/теми (ключови думи), които се очаква да се
     възползват - за автоматично "закачане" в дневния скрининг.

Резултатът се записва в daily_macro_signal.json в корена на repo-то и се чете
от combined_screener.py (виж render_macro_section() / auto-pin логиката).
"""

import json
import os
import re
from datetime import datetime, timezone

from anthropic import Anthropic

OUTPUT_FILE = "daily_macro_signal.json"

SYSTEM_PROMPT = """Ти си макроикономически анализатор за суинг търговец, който търгува
само EUR-деноминирани акции/ETF-и на европейски (ЕС/ЕИП) борси плюс злато/сребро.

Използвай web search, за да прегледаш днешните новини и анализи от РЕНОМИРАНИ и
проверени източници: Reuters, Bloomberg, Financial Times, Wall Street Journal,
MarketWatch, официални съобщения на Федералния резерв и ЕЦБ. Игнорирай форуми,
блогове без редакторски контрол, и източници без ясна репутация.

Фокусирай се конкретно върху:
- Очаквания за движение на лихвения процент на Федералния резерв (и ЕЦБ) -
  повишение, понижение, пауза - и как пазарът им реагира/очаква да реагира.
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
{
  "summary_bg": "2-4 изречения на български, обобщаващи макро картината за деня",
  "themes": [
    {
      "theme": "кратко заглавие на темата",
      "keywords": ["Gold", "Silver"],
      "reasoning_bg": "едно изречение защо, с позоваване на източника (по име, не линк)"
    }
  ]
}
```
Ако не намериш нищо съществено, върни themes: [] и обясни защо в summary_bg."""


def extract_json_block(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        # fallback: опитай да намериш първата { ... } двойка в текста
        match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError(f"Не намерих JSON в отговора на Claude:\n{text[:500]}")
    return json.loads(match.group(1))


def run_daily_macro_scan(api_key: str) -> dict:
    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=8192,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": SYSTEM_PROMPT}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise ValueError(
            f"Claude върна празен текстов отговор (stop_reason: {response.stop_reason}). "
            "Може да е изразходвал max_tokens само за web search - опитай пак."
        )

    parsed = extract_json_block(text)
    parsed["generated_at"] = datetime.now(timezone.utc).isoformat()
    return parsed


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Липсва ANTHROPIC_API_KEY в environment.")

    result = run_daily_macro_scan(api_key)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_themes = len(result.get("themes", []))
    print(f"Записах {OUTPUT_FILE}: {n_themes} теми.")


if __name__ == "__main__":
    main()
