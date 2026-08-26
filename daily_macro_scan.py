"""
daily_macro_scan.py

Дневен макро скенер: пуска се веднъж на ден (виж .github/workflows/daily_macro.yml),
преди отваряне на европейските борси. Вика Gemini с "Grounding with Google Search"
(директен достъп до индекса на Google, включително Google News) да прегледа
реномирани икономически източници (Reuters, Bloomberg, WSJ, FT, MarketWatch,
официални изявления на Фед/ЕЦБ) и да извади:

  1) кратко резюме на макро картината за деня (лихвени очаквания, злато/сребро,
     други трендиращи теми по експертни анализи),
  2) конкретен списък от активи/теми (ключови думи), които се очаква да се
     възползват - за автоматично "закачане" в дневния скрининг.

Резултатът се записва в daily_macro_signal.json в корена на repo-то, в СЪЩИЯ
формат, който combined_screener.py вече очаква (виж render_macro_section()) -
затова смяната на доставчика от Claude на Gemini не изисква промени в приложението.

ЗАБЕЛЕЖКА за пакета: използваме новия унифициран SDK `google-genai`
(`from google import genai`) - НЕ по-стария, вече deprecated `google-generativeai`.
requirements.txt трябва да съдържа `google-genai`, не `google-generativeai`.
"""

import json
import os
import re
from datetime import datetime, timezone

from google import genai
from google.genai import types

OUTPUT_FILE = "daily_macro_signal.json"

# gemini-3.6-flash е текущият стабилен (GA) модел към средата на 2026 г.
# Провери https://ai.google.dev/gemini-api/docs/models за по-нова стабилна версия,
# ако този модел бъде маркиран за deprecation.
MODEL_NAME = "gemini-3.6-flash"

PROMPT = """Ти си макроикономически анализатор за суинг търговец, който търгува
само EUR-деноминирани акции/ETF-и на европейски (ЕС/ЕИП) борси плюс злато/сребро.

Използвай Google Search, за да прегледаш днешните новини и анализи от РЕНОМИРАНИ
и проверени източници: Reuters, Bloomberg, Financial Times, Wall Street Journal,
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
        raise ValueError(f"Не намерих JSON в отговора на Gemini:\n{text[:500]}")
    return json.loads(match.group(1))


def run_daily_macro_scan(api_key: str) -> dict:
    client = genai.Client(api_key=api_key)

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT,
        config=config,
    )

    text = response.text or ""
    if not text.strip():
        raise ValueError("Gemini върна празен текстов отговор. Опитай пак.")

    parsed = extract_json_block(text)
    parsed["generated_at"] = datetime.now(timezone.utc).isoformat()
    return parsed


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Липсва GEMINI_API_KEY в environment.")

    result = run_daily_macro_scan(api_key)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_themes = len(result.get("themes", []))
    print(f"Записах {OUTPUT_FILE}: {n_themes} теми.")


if __name__ == "__main__":
    main()
