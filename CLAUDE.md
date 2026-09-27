# EU Swing AI — контекст за Claude Code

Автоматизиран скрийнър за swing trading на европейски акции и ETF-и (EUR, борси в ЕС/ЕИП), интегриран с Trading 212.
Потребителят (Наско) пише на български; коментарите и UI текстовете в кода са на български, техническите термини на английски.

## Структура на репото
| Файл | Роля |
|---|---|
| `multi_timeframe_screener.py` | **Streamlit приложението** (един голям файл, ~2150 реда). Входна точка: `streamlit run multi_timeframe_screener.py` |
| `t212_portfolio.py` | Trading 212 API клиент (Portfolio / History, Basic auth от key+secret) |
| `fetch_eu_instruments.py` | Тегли всички T212 инструменти, филтрира EUR + ЕС/ЕИП борси → `eu_instruments.json` |
| `select_liquid_universe.py` | Месечен pre-screen: ликвидност + 3м моментум → топ ~500 в `curated_universe.json` (това сканира приложението) |
| `daily_macro_scan.py` | Дневен макро скенер: FMP числа (Fed rate, CPI, 10y) + Claude с web search → `daily_macro_signal.json` |
| `manual_universe.json` | Ръчно include/exclude на тикери (редактира се и от UI през GitHub API) |
| `.github/workflows/` | `daily.yml` (03:00 UTC, instruments), `daily_macro.yml` (04:30 UTC), `monthly_curate.yml` (1-во число, 04:00 UTC). Всички commit-ват JSON резултата обратно в `main` |
| `.streamlit/config.toml` | Тъмна тема |

## Текущо състояние на UI
`st.radio` в края на `multi_timeframe_screener.py` показва **само две секции**:
1. **🧭 Photon Phases** → `render_photon_strategy()` — SMC/MTF рамка на Photon Trading (BOS/CHoCH, Phase A/B, само long), каскада Weekly → Daily → 4h
2. **💼 Портфолио & P&L** → `render_portfolio_section()` — отворени/затворени сделки, P&L за избираем период, AI анализ; два акаунта в `T212_ACCOUNTS` (собствен + на съпругата)

**Мъртъв код (все още във файла, не се вика от UI):** `render_daily_strategy`, `render_mtf_strategy`, `render_sd_strategy` (Supply & Demand) и свързаните им `analyze_*` / `generate_ai_analysis_*`. Не ги трий без изрично съгласие.

## Филтри и данни
- Универсум: `curated_universe.json` (+ `manual_universe.json`), или **CSV/Excel upload** (експорт от InvestingPro Watchlist — планът е Pro, не Pro+) който замества сканирания универсум
- Индикатори: EMA50, SMA200, RSI, ATR, MACD, swing points; твърди трендови филтри
- Цени: yfinance. AI интерпретация: Anthropic API (проектът остава само на Claude)
- FMP free tier връща 402 за EU тикери → fundamentals enrichment е премахнат; FMP се ползва само за макро

## Secrets (никога в кода)
- Streamlit secrets: `ANTHROPIC_API_KEY`, `GITHUB_TOKEN` (за запис на `manual_universe.json` и workflow dispatch), `T212_API_KEY` / `T212_API_SECRET`, `T212_API_KEY_WIFE` / `T212_API_SECRET_WIFE`
- GitHub Actions secrets: `ANTHROPIC_API_KEY`, `FMP_API_KEY`, `T212_API_KEY`, `T212_API_SECRET`
- Няма `.gitignore` — при добавяне на `.env` или `.streamlit/secrets.toml` локално, първо добави `.gitignore`

## Отворени идеи / roadmap
- Категория сигнали **mean reversion**
- **Alpha Vantage NEWS_SENTIMENT** в месечния curation скрипт
- Подобрения по UI
- Евентуално: разделяне на `multi_timeframe_screener.py` на модули и изчистване на мъртвия код

## Правила при работа
- Workflows push-ват в `main` автоматично → преди push винаги `git fetch origin main` и rebase
- Малки commits с ясни съобщения; по-големи промени в отделен branch + PR
- Проверка преди commit: `python -m py_compile *.py`
