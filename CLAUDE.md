# EU Swing AI — контекст за Claude Code

Автоматизиран скрийнър за swing trading на европейски акции и ETF-и (EUR, борси в ЕС/ЕИП), интегриран с Trading 212.
Потребителят (Наско) пише на български; коментарите и UI текстовете в кода са на български, техническите термини на английски.

## Структура на репото
| Файл | Роля |
|---|---|
| `multi_timeframe_screener.py` | **Входна точка** на Streamlit приложението (page config, CSS, избор на секция): `streamlit run multi_timeframe_screener.py` |
| `photon.py` | Photon Phases стратегията: анализ (`PhotonSetup`), пакетен скан, таблици, графики в табове W / D / 4h (`candle_figure` като в T212: ценовата скала вдясно с етикет на текущата цена, празно място след последната свещ, етикетите на нивата вляво в графиката върху полупрозрачен фон, без застъпване), „💼 Държа“ и „Моите позиции в скана“ (T212; със същото фундаментално потвърждение + отделен бутон за новините по позициите) |
| `fundamentals.py` | Фундаментално потвърждение на сетъпите (само подчертава/подрежда, не филтрира): ниво 1 - анализатори от Yahoo `.info` (консенсус, потенциал до целта, ръст EPS, дата на отчет); ниво 2 - бутон „Провери новини и анализи“ (Gemini + Google Search или Claude + `web_search` - по общия превключвател, резултатите се пазят отделно за сравнение; готовите + първите 10 от Watchlist, линкове само от реално намерени резултати; резултатите за деня са в общ за всички сесии склад `_news_store` (`st.cache_resource`) - обновяване на страницата не плаща повторно; потвърждение „Ще се направят N платени проверки“ + брояч за деня; Gemini за новините: Flash Lite + `thinking_level=MINIMAL` (резервно LOW / основният модел), за плана и портфолиото - LOW; Claude за новините: Haiku 4.5 без мислене (резервно Sonnet с `thinking: disabled` + `effort: low`); до 2 търсения на инструмент) |
| `universe.py` | Универсът: curated/ръчен списък, CSV upload, търсене, макро сигнал, GitHub (dispatch/запис, статус на обновяването) |
| `indicators.py` | Чисти изчисления: RSI/MACD/ATR, swing точки, структура, свещи (weekly/4ч по сесия) |
| `portfolio_ui.py` | Секция Портфолио & P&L (`T212_ACCOUNTS`) |
| `ai_client.py` | `call_claude` / `stream_claude` (модел `CLAUDE_MODEL`), `stream_gemini`, `stream_ai(prompt, provider, ...)`; `AI_PROVIDERS` (Gemini първи = по подразбиране) |
| `ui_common.py` | CSS тема, `section_header`, `format_eur` |
| `legacy_strategies.py` | Мъртъв код: старите стратегии (не се импортират от UI) |
| `t212_portfolio.py` | Trading 212 API клиент (Portfolio / History, Basic auth от key+secret) |
| `fetch_eu_instruments.py` | Тегли всички T212 инструменти, филтрира EUR + ЕС/ЕИП борси → `eu_instruments.json` (без филтър по ликвидност - него прави месечният подбор) |
| `select_liquid_universe.py` | Месечен pre-screen (+ ръчно от бутона „Обнови универса сега“ в UI): акции по капитализация + оборот, ETF по AUM + оборот, без ливъриджнати/short → `curated_universe.json` (приложението сканира всички от него) |
| `universe_rules.py` | Общи правила: прагове за ликвидност, ISIN логика за родна/вторична борса, филтри за ливъриджнати/парични ETP и за листвания извън Европа/САЩ (`.T .HK .AX .TO`...), борса → Yahoo суфикс (`.DE .PA .AS .MI .BR .MC .VI .LS .MU`) — ползва се от скриптовете и от приложението |
| `daily_macro_scan.py` | (Спрян) дневен макро скенер: FMP числа (Fed rate, CPI, 10y) + Claude с web search → `daily_macro_signal.json` |
| `manual_universe.json` | Ръчно include/exclude на тикери (редактира се и от UI през GitHub API) |
| `.github/workflows/` | `daily.yml` (03:00 UTC, instruments), `daily_macro.yml` (само ръчно - графикът е спрян, макро секцията е махната от UI), `monthly_curate.yml` (1-во число, 04:00 UTC). Всички commit-ват JSON резултата обратно в `main` |
| `.streamlit/config.toml` | Тъмна тема |

## Текущо състояние на UI
`st.radio` в края на `multi_timeframe_screener.py` показва **само две секции**, а под него е общият превключвател **„🤖 AI анализи чрез: Gemini / Claude“** (`ai_provider` в session_state; ползва се от новините, търговския план и анализа на портфолиото):
1. **🧭 Photon Phases** → `photon.render_photon_strategy()` — SMC/MTF рамка на Photon Trading (BOS/CHoCH, Phase A/B, само long), каскада Weekly → Daily → 4h; колони „📊 Фундамент“ / „📰 Новини“ / „Отчет“ (зелен ред = потвърден, ⭐ = + положителни новини)
   - Подредба: горе голям бутон „🔍 Сканирай пазара“ + ред със статус + фунията като лента; отдолу табове **✅ Готови (карти, по желание таблица) / 👀 Watchlist (+ новините) / 💼 Позиции / 🤖 AI план / 🛠 Универс и настройки** (профил Консервативен/Стандартен/Агресивен + „Разширени“ плъзгачи, филтри, ликвидност, CSV, ръчен списък, обновяване, търсене). Табът с настройките се попълва първи в кода (контейнерите се пълнят в произволен ред)
   - Клик по ред/карта/новина → изскачащ прозорец (`st.dialog`) с нивата, анализаторите, графиката и новините на инструмента (`open_instrument` → `ph_dialog` в session_state → отваря се в края на страницата)
   - Таблиците показват основните колони (`MAIN_COLUMNS`), останалите с „Още колони“
   - Предложенията са за **лимит поръчки** (`limit_entry`: Phase A - лимит в POI; Phase B с пробит CHoCH - лимит на ретест; Phase B без пробив - buy stop над CHoCH); графиките са в цветове по образец на T212 (`CHART_*`: почти черен фон, видима мрежа, пунктир на текущата цена); R/R е от лимит цената; Цел 2 (седмичната съпротива) се показва само ако е над Цел 1, иначе „—“ (`target2`); натискане на колелото върху графиката (на телефон: задържане на пръста ~0.5 сек, пръстът мести кръста) = кръст с цената на скалата вдясно и датата/часа долу (`CROSSHAIR_JS` чрез `st.html`, слуша само `window` на приложението - в Cloud то е в iframe; Plotly hover/spikes са изключени - `hovermode=False`; Esc / колелото / стрелката назад на телефона скрива - `history.pushState` + `popstate`); влачене по цялата ценова скала = разтягане около средата, по цялата времева ос = разтягане с фиксиран десен край (същият скрипт, `window.Plotly.relayout`); бледа ориентировъчна печалба/загуба до цел 1/2 и stop при сумата от настройките (`ph_invest`, по подразбиране 1000 €) - в картите, таблиците, прозореца и етикетите на графиката
   - **Хоризонт:** ⏱ ориентировъчен срок до цел 1 (`days_to_target` по скоростта на досегашните възходящи дневни swing-ове - `up_leg_speeds`, резервно по ATR); филтър в настройките (`ph_horizon`, по подразбиране ~1 месец = 22 търг. дни) - готовите с по-дълъг типичен срок отиват в Watchlist с бележка (`split_by_horizon`)
   - Тема: тъмносиня (`.streamlit/config.toml` + CSS променливите в `ui_common.py` + `CHART_*` в `photon.py`)
2. **💼 Портфолио & P&L** → `portfolio_ui.render_portfolio_section()` — отворени/затворени сделки, P&L за избираем период (вкл. „Миналия месец“), AI анализ; отворените позиции по подразбиране са без частта в пайовете (`exclude_pies` по `pieQuantity`/`quantityInPies`, частично в пай - пропорционално) + бутон „⚡ Само текущата P&L“ (само позициите, без историята); два акаунта в `T212_ACCOUNTS` (собствен + на съпругата). Историята на сделките (T212: 1 заявка/10 сек, 50 записа/стр.) се тегли само до началото на периода - `HISTORY_LOOKBACK_DAYS` (180), с прогрес по страници, пази се в общ склад `_history_store` (`st.cache_resource`) и следващите зареждания теглят само новите сделки (`load_order_history`); реализираната P&L се взима от `fill.walletImpact.realisedProfitLoss`, ако T212 я връща, иначе по средна цена

**Мъртъв код:** `legacy_strategies.py` — `render_daily_strategy`, `render_mtf_strategy`, `render_sd_strategy` (Supply & Demand) и свързаните им `analyze_*` / `generate_ai_analysis_*`. Не се вика от UI. Не го трий без изрично съгласие.

## Филтри и данни
- Универсум: `curated_universe.json` (+ `manual_universe.json`), или **CSV/Excel upload** (експорт от InvestingPro screener/Watchlist — планът е Pro, не Pro+); съпоставяне по ISIN → тикер → име (цели думи); режими: допълва месечния списък само с липсващите (по подразбиране) / акциите от файла + ETF / само файла; новите могат да се запишат трайно в `manual_universe.json` (бутон „💾 Запази“)
- Индикатори: EMA50, SMA200, RSI, ATR, MACD, swing points; твърди трендови филтри
- Цени: yfinance. AI интерпретация: Gemini (`google-genai`, по подразбиране) или Claude (Anthropic API) - избира се в приложението за всички AI анализи (решение на Наско)
- FMP free tier връща 402 за EU тикери → fundamentals enrichment е премахнат; FMP се ползва само за макро
- Yahoo понякога връща празни данни при много заявки → месечният скрипт прави повторни опити (теглене + `.info`/`fast_info`) и пази паметта от миналия месец (липсваща капитализация/AUM се допълва; инструмент без данни сега остава с `carried_over`)
- Gettex (`.MU`) листванията и чуждите акции на европейски борси (не-ЕИП ISIN, напр. US акция на Xetra) → месечният скрипт намира по ISIN основното листване (US/`.ST`/`.HE`...) и сканира него (`t212_symbol` пази оригинала, `resolved_aliases` - всички T212 листвания; дубликатите Xetra/Gettex се сливат, като остава Xetra; **само-Gettex инструментите (няма друго листване в T212) се изключват** - маркет мейкър/широк спред; не се съпоставят и от CSV); оборот/капитализация/AUM се превръщат в € преди праговете, в резултатите има колона „Валута“

## Secrets (никога в кода)
- Streamlit secrets: `ANTHROPIC_API_KEY`, `GITHUB_TOKEN` (за запис на `manual_universe.json` и workflow dispatch), `T212_API_KEY` / `T212_API_SECRET`, `T212_API_KEY_WIFE` / `T212_API_SECRET_WIFE`, `GEMINI_API_KEY` (по избор, за новините чрез Gemini), `GEMINI_MODEL` (по избор, по подразбиране `gemini-3.5-flash`), `CLAUDE_NEWS_MODEL` (по избор, по-евтиният Claude за новините, по подразбиране `claude-haiku-4-5`; при недостъпен - `CLAUDE_MODEL`), `GEMINI_NEWS_MODEL` (по избор, по-евтиният модел само за новините, по подразбиране `gemini-3.5-flash-lite`; при недостъпен модел се пада на `GEMINI_MODEL`)
- GitHub Actions secrets: `ANTHROPIC_API_KEY`, `FMP_API_KEY`, `T212_API_KEY`, `T212_API_SECRET`
- Няма `.gitignore` — при добавяне на `.env` или `.streamlit/secrets.toml` локално, първо добави `.gitignore`

## Отворени идеи / roadmap
- Категория сигнали **mean reversion**
- **Alpha Vantage NEWS_SENTIMENT** в месечния curation скрипт
- Подобрения по UI
- Евентуално: изтриване на `legacy_strategies.py` (само с изрично съгласие)

## Правила при работа
- Workflows push-ват в `main` автоматично → преди push винаги `git fetch origin main` и rebase
- Малки commits с ясни съобщения; по-големи промени в отделен branch + PR
- Проверка преди commit: `python -m py_compile *.py` (и `python -m pyflakes *.py` за недефинирани имена)
- Streamlit: `width="stretch"` вместо остарелия `use_container_width=True`
