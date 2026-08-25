"""
fetch_eu_instruments.py

Изтегля пълния списък инструменти от Trading 212 Public API,
маха US тикери и филтрира само европейски акции/ETF-и.

Изисква следните environment променливи:
    T212_API_KEY
    T212_API_SECRET

Използване:
    export T212_API_KEY="..."
    export T212_API_SECRET="..."
    python fetch_eu_instruments.py

Резултат:
    eu_instruments.json  -- филтриран списък, готов за screening скрипта
"""

import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

# --- Настройки -------------------------------------------------------------

# "live" за реалната ти сметка, "demo" за paper trading
ENVIRONMENT = os.environ.get("T212_ENV", "live")
BASE_URL = f"https://{ENVIRONMENT}.trading212.com/api/v0"

# Кои валути да пазим. Само EUR - сметката в Trading 212 е в евро, а търговия
# в друга валута (SEK, NOK, DKK, PLN, GBP, CHF) минава през конвертиране с
# такса от Trading 212, което не е желателно.
ALLOWED_CURRENCIES = {"EUR"}

# Суфикси на тикери, които T212 използва за US борси -- изключваме ги
US_SUFFIXES = ("_US_EQ",)

CACHE_FILE = "t212_instruments_raw.json"
OUTPUT_FILE = "eu_instruments.json"


def get_auth_header() -> str:
    key = os.environ.get("T212_API_KEY")
    secret = os.environ.get("T212_API_SECRET")
    if not key or not secret:
        sys.exit("Липсват T212_API_KEY / T212_API_SECRET в environment.")
    token = base64.b64encode(f"{key}:{secret}".encode()).decode()
    return f"Basic {token}"


def fetch_json(path: str, auth_header: str):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"Authorization": auth_header})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        sys.exit(f"HTTP {e.code} при заявка към {path}: {body}")
    except urllib.error.URLError as e:
        sys.exit(f"Мрежова грешка при {path}: {e}")


def load_instruments(auth_header: str):
    # Кешираме локално, защото endpoint-ът е rate-limited (~1 заявка / 50s)
    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        if age < 6 * 3600:  # ползвай кеша ако е по-нов от 6 часа
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

    print("Тегля инструменти от Trading 212 API ...")
    data = fetch_json("/equity/metadata/instruments", auth_header)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


# --- Ликвиден филтър за АКЦИИ: само членове на STOXX Europe 600 (Европа) ---
# или S&P 500 (САЩ, за акции търгувани в EUR на европейски борси).
# ETF-ите НЕ се филтрират по този списък - там AUM/оборот филтрите са достатъчни.
#
# STOXX 600 списъкът е вграден статично (обновява се на ръка на тримесечие,
# когато STOXX прави ребалансиране - март/юни/септември/декември).
# S&P 500 списъкът се тегли на живо при всяко пускане (винаги актуален).

STOXX600_NAMES = [
    "3i",
    "ABB Ltd",
    "ABN AMRO",
    "ACS Group",
    "AIXTRON",
    "ASM International",
    "ASML Holding",
    "AXA",
    "Acciona",
    "Acciona Energía",
    "Accor",
    "Acerinox",
    "Adecco",
    "Adevinta",
    "Adidas",
    "Admiral Group",
    "Adyen",
    "Aena",
    "Ageas",
    "Air France-KLM",
    "Air Liquide",
    "Airbus",
    "Aker BP",
    "AkzoNobel",
    "Alcon",
    "Alfa Laval",
    "Allfunds Group",
    "Allianz",
    "Alstom",
    "Ambu",
    "Anglo American plc",
    "Anheuser-Busch InBev",
    "Antofagasta plc",
    "ArcelorMittal",
    "Argenx",
    "Arkema",
    "Aroundtown",
    "Ashtead Group",
    "Assa Abloy",
    "Associated British Foods",
    "Aston Martin",
    "AstraZeneca",
    "Atlas Copco",
    "Atos",
    "Auto Trader Group",
    "Aviva",
    "Azelis Group",
    "B&M European Value Retail",
    "BAE Systems",
    "BASF",
    "BMW",
    "BNP Paribas",
    "BP",
    "BT Group",
    "Balder",
    "Balfour Beatty",
    "Baloise",
    "Banco Bilbao Vizcaya Argentaria",
    "Banco Sabadell",
    "Banco Santander",
    "Bankinter",
    "Barclays",
    "Barratt Developments",
    "Barry Callebaut",
    "Bayer",
    "Beazley",
    "Beiersdorf",
    "Belimo",
    "Berkeley Group Holdings",
    "Big Yellow Group",
    "Boliden AB",
    "Bolloré",
    "Bouygues",
    "Bpost",
    "Brenntag",
    "British American Tobacco",
    "Brunello Cucinelli",
    "Bunzl",
    "Burberry",
    "Bureau Veritas",
    "CD Projekt",
    "CSG A",
    "CTS Eventim",
    "CaixaBank",
    "Capgemini",
    "Carl Zeiss Meditec",
    "Carrefour",
    "Castellum",
    "Centrica",
    "Clariant",
    "Coca-Cola HBC",
    "Colruyt",
    "Commerzbank",
    "Compass Group",
    "Computer Center",
    "Continental",
    "ConvaTec",
    "Covestro",
    "Croda International",
    "Crédit Agricole",
    "DHL Group",
    "DIETEREN Group",
    "DS Smith",
    "DSM-Firmenich",
    "Daimler Truck",
    "Danone",
    "Davide Campari-Milano",
    "Delivery Hero",
    "Demant",
    "Derwent London",
    "Deutsche Börse",
    "Deutsche Pfandbriefbank",
    "Deutsche Telekom",
    "Dia",
    "Diageo",
    "Diploma",
    "Direct Line",
    "DocMorris",
    "E.ON",
    "EDP - Energias de Portugal",
    "EDP Renováveis",
    "EasyJet",
    "Edenred",
    "Eiffage",
    "Electrolux",
    "Elisa",
    "Ems-Chemie",
    "Enagás",
    "Endeavour Mining",
    "Enel",
    "Engie",
    "Eni",
    "Entain",
    "Epiroc",
    "Ericsson",
    "Erste Group",
    "EssilorLuxottica",
    "Euronext",
    "Evonik Industries",
    "Evotec",
    "Evraz",
    "Exor",
    "Experian",
    "FCC",
    "Falck",
    "Ferrari",
    "Ferrovial",
    "Flow Traders",
    "Flutter Entertainment",
    "Fortum",
    "Forvia",
    "Freenet",
    "Fresenius Medical Care",
    "GEA Group",
    "GN Store Nord",
    "GSK plc",
    "Galenica",
    "Galp Energia",
    "Games Workshop",
    "Geberit",
    "Gecina",
    "Georg Fischer",
    "Getlink",
    "Givaudan",
    "Glanbia",
    "Glencore",
    "Grafton Group",
    "Greggs",
    "Grifols",
    "Gruppo Campari",
    "H&M",
    "HSBC",
    "Haleon",
    "Halma",
    "Halma plc",
    "Hannover Re",
    "Hays",
    "Heidelberg Materials",
    "Helvetia Insurance",
    "Henkel",
    "Hermès",
    "Hikma Pharmaceuticals",
    "Hiscox",
    "Holcim",
    "Howden Joinery",
    "Hugo Boss",
    "Husqvarna",
    "IG Group",
    "IMI",
    "ING Group",
    "ITV",
    "Iberdrola",
    "Imperial Brands",
    "InPost",
    "Inchcape",
    "Inditex",
    "Indivior",
    "Indutrade",
    "Infineon Technologies",
    "Informa",
    "InterContinental Hotels Group",
    "Intermediate Capital Group",
    "International Airlines Group",
    "Interpump Group",
    "Intertek",
    "Intesa Sanpaolo",
    "Investec",
    "Ipsen",
    "Ipsos",
    "JD Sports",
    "Jerónimo Martins",
    "Johnson Matthey",
    "Julius Baer",
    "Just Eat Takeaway",
    "KBC Group",
    "KPN",
    "Kering",
    "Kingfisher plc",
    "Kone",
    "Kuehne + Nagel",
    "L'Oréal",
    "LEG Immobilien",
    "LPP",
    "LVMH",
    "Land Securities",
    "Lanvin Group",
    "Lanxess",
    "Legal & General",
    "Leonardo",
    "Lerøy Seafood",
    "Liberty Global",
    "LifeCo",
    "Linde plc",
    "Lindt & Sprüngli",
    "Lloyds Banking Group",
    "Logitech",
    "London Stock Exchange Group",
    "Lonza Group",
    "Lufthansa",
    "Lumibird",
    "Lundin Energy",
    "M&G",
    "MAN SE",
    "MTU Aero Engines",
    "Maersk",
    "Marks & Spencer",
    "Mediobanca",
    "Melexis",
    "Meliá Hotels",
    "Mercedes-Benz Group",
    "Merlin Properties",
    "Metso",
    "Michelin",
    "Millicom",
    "Mobico Group",
    "Moncler",
    "Mondi",
    "Moneysupermarket.com",
    "MorphoSys",
    "Morrisons",
    "Munich Re",
    "NN Group",
    "NatWest Group",
    "National Bank of Greece",
    "National Grid",
    "Naturgy",
    "Nemetschek",
    "Neoen",
    "Neste",
    "Nestlé",
    "Nexans",
    "Nexi",
    "Next plc",
    "Nibe Industrier",
    "Nokia",
    "Nordea",
    "Norsk Hydro",
    "Novartis",
    "Novo Nordisk",
    "OCI",
    "OMV",
    "Ocado",
    "Oerlikon",
    "Ontex",
    "Orange S.A.",
    "Orkla ASA",
    "Pan African Resources",
    "Pandora",
    "Partners Group",
    "Pearson",
    "Pennon Group",
    "Pernod Ricard",
    "Persimmon plc",
    "Phoenix Group",
    "Pirelli",
    "PolyPeptide",
    "Porsche",
    "Porsche SE",
    "PostNL",
    "Poste Italiane",
    "Proximus",
    "Prudential plc",
    "Prysmian",
    "Publicis",
    "Puma",
    "Qiagen",
    "Quilter",
    "RELX",
    "RWE",
    "Raiffeisen Bank International",
    "Rational",
    "Reckitt",
    "Recordati",
    "Redeia",
    "Renault",
    "Rentokil Initial",
    "Repsol",
    "Rexel",
    "Rheinmetall",
    "Richemont",
    "Rightmove",
    "Rio Tinto Group",
    "Roche Holding",
    "Rolls-Royce Holdings",
    "Royal Mail",
    "Rémy Cointreau",
    "SAP",
    "SCOR SE",
    "SGS S.A.",
    "SIG Group",
    "SKF",
    "SSE plc",
    "STMicroelectronics",
    "Safran",
    "Sage Group",
    "Sainsbury's",
    "Saint-Gobain",
    "SalMar",
    "Sampo Group",
    "Sandoz",
    "Sandvik AB",
    "Sanofi",
    "Sanoma",
    "Sartorius",
    "Schibsted",
    "Schindler Group",
    "Schneider Electric",
    "Schroders",
    "Scottish Mortgage Investment Trust",
    "Scout24",
    "Securitas AB",
    "Segro",
    "Severn Trent",
    "Shell plc",
    "Siemens",
    "Signify",
    "Sika",
    "Sixt",
    "Skandinaviska Enskilda Banken",
    "Skanska",
    "Sky Group",
    "Smith & Nephew",
    "Smiths Group",
    "Smurfit Kappa",
    "Snam",
    "Société Générale",
    "Sodexo",
    "Softcat",
    "Software",
    "Solvay",
    "Sonova",
    "Spectris",
    "Spirax-Sarco Engineering",
    "St. James's Place",
    "Standard Chartered",
    "Stellantis",
    "Stora Enso",
    "Storebrand",
    "Straumann",
    "Svenska Handelsbanken",
    "Swatch Group",
    "Sweco",
    "Swedish Match",
    "Swiss Life",
    "Swiss Prime Site",
    "Swiss Re",
    "Swisscom",
    "Symrise",
    "TGS",
    "TUI Group",
    "Talanx",
    "Tate & Lyle",
    "Taylor Wimpey",
    "TeamViewer",
    "Tecan",
    "TechnipFMC",
    "Technoprobe",
    "Tele2",
    "Telecom Italia",
    "Telecom Plus",
    "Telefónica",
    "Telenor",
    "Teleperformance",
    "Telia Company",
    "Temenos Group",
    "Tenaris",
    "Terna",
    "Tesco",
    "Teva Pharmaceutical Industries",
    "Thales",
    "ThyssenKrupp",
    "TomTom",
    "Tomra",
    "TotalEnergies",
    "Transocean",
    "Traton",
    "Travis Perkins",
    "Trelleborg",
    "Tryg",
    "UBS Group",
    "UPM-Kymmene",
    "Ubisoft",
    "Umicore",
    "UniCredit",
    "Unibail-Rodamco-Westfield",
    "Unilever",
    "Uniper",
    "Unite Group",
    "United Utilities",
    "Universal Music Group",
    "VAT Group",
    "Valeo",
    "Vallourec",
    "Valmet",
    "Varta",
    "Veolia",
    "Verbund",
    "Vestas",
    "Viaplay",
    "Vicat",
    "Vienna Insurance Group",
    "Vinci",
    "Vitrolife",
    "Vivendi",
    "Vodafone Group",
    "Voestalpine",
    "Volvo Group",
    "Vonovia",
    "Vopak",
    "WPP",
    "Warehouses De Pauw",
    "Weir Group",
    "Wendel",
    "Whitbread",
    "Wienerberger",
    "Wihlborgs Fastigheter",
    "Wise",
    "Wolters Kluwer",
    "Worldline",
    "Yara International",
    "Zalando",
    "Zurich Insurance Group",
    "ams OSRAM",
    "Ørsted"
]

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
SP500_CACHE_FILE = "sp500_cache.json"


def normalize_company_name(name: str) -> str:
    """Нормализира име на компания за по-надеждно съпоставяне (различни борси
    форматират имената по различен начин - 'SE', 'PLC', 'AG', интервали и т.н.)."""
    name = name.upper()
    for suffix in [
        " PLC", " SE", " AG", " SA", " NV", " AB", " ASA", " A/S", " AS",
        " GROUP", " HOLDING", " HOLDINGS", " CORPORATION", " CORP",
        " INC.", " INC", " LTD", " CO.", " CO", " N.V.", " S.A.", " S.P.A.",
        " SPA", " GMBH", " (THE)",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    for ch in [".", ",", "'", "-", "&"]:
        name = name.replace(ch, " ")
    return " ".join(name.split()).strip()


def fetch_sp500_names() -> set:
    """Тегли текущия списък S&P 500 имена от поддържан GitHub datasets източник."""
    if os.path.exists(SP500_CACHE_FILE):
        age = time.time() - os.path.getmtime(SP500_CACHE_FILE)
        if age < 24 * 3600:
            with open(SP500_CACHE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))

    try:
        req = urllib.request.Request(SP500_URL)
        with urllib.request.urlopen(req, timeout=30) as resp:
            csv_text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"Предупреждение: неуспешно теглене на S&P 500 списък ({e}), продължавам без него.")
        return set()

    lines = csv_text.strip().split("\n")
    names = set()
    for line in lines[1:]:
        # Второто поле е "Security" (име на компанията); внимаваме за запетаи в кавички
        parts = []
        current = ""
        in_quotes = False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == "," and not in_quotes:
                parts.append(current)
                current = ""
            else:
                current += ch
        parts.append(current)
        if len(parts) > 1:
            names.add(parts[1])

    with open(SP500_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(names), f, ensure_ascii=False, indent=2)
    return names


def build_liquid_stock_whitelist() -> set:
    """Комбиниран, нормализиран набор от имена STOXX600 + S&P 500."""
    sp500 = fetch_sp500_names()
    combined = set(STOXX600_NAMES) | sp500
    return {normalize_company_name(n) for n in combined}


def is_liquid_stock(instrument_name: str, whitelist_normalized: set) -> bool:
    """Проверява дали инструментът съвпада (точно или като под-низ) с индексен член."""
    norm = normalize_company_name(instrument_name)
    if norm in whitelist_normalized:
        return True
    # По-мека проверка за случаи с леко различно форматиране на името
    for wl_name in whitelist_normalized:
        if len(wl_name) > 4 and (wl_name in norm or norm in wl_name):
            return True
    return False


EXCLUDED_EXCHANGES = {
    "London Stock Exchange",
    "London Stock Exchange AIM",
    "SIX Swiss Exchange",
}


def is_european(instrument: dict) -> bool:
    ticker = instrument.get("ticker", "")
    currency = instrument.get("currencyCode", "")
    exchange_name = instrument.get("exchangeName", "")

    if ticker.endswith(US_SUFFIXES):
        return False
    if currency not in ALLOWED_CURRENCIES:
        return False
    if exchange_name in EXCLUDED_EXCHANGES:
        return False
    return True


EXCHANGES_CACHE_FILE = "t212_exchanges_raw.json"


def load_exchanges(auth_header: str):
    """Връща речник workingScheduleId -> име на борсата (реалната, не гадана)."""
    if os.path.exists(EXCHANGES_CACHE_FILE):
        age = time.time() - os.path.getmtime(EXCHANGES_CACHE_FILE)
        if age < 24 * 3600:
            with open(EXCHANGES_CACHE_FILE, "r", encoding="utf-8") as f:
                exchanges = json.load(f)
        else:
            exchanges = None
    else:
        exchanges = None

    if exchanges is None:
        print("Тегля списък с борси от Trading 212 API ...")
        exchanges = fetch_json("/equity/metadata/exchanges", auth_header)
        with open(EXCHANGES_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(exchanges, f, ensure_ascii=False, indent=2)

    schedule_to_exchange = {}
    for exch in exchanges:
        exch_name = exch.get("name", "UNKNOWN")
        for schedule in exch.get("workingSchedules", []):
            schedule_to_exchange[schedule["id"]] = exch_name
    return schedule_to_exchange


def main():
    auth_header = get_auth_header()
    instruments = load_instruments(auth_header)
    schedule_to_exchange = load_exchanges(auth_header)

    print(f"Общо инструменти от T212: {len(instruments)}")
    print(f"Намерени борси: {len(set(schedule_to_exchange.values()))}")

    # Прикачваме истинското име на борсата към всеки инструмент
    for inst in instruments:
        inst["exchangeName"] = schedule_to_exchange.get(
            inst.get("workingScheduleId"), "UNKNOWN"
        )

    filtered = [i for i in instruments if is_european(i)]
    print(f"След филтър за европейски борси/валути: {len(filtered)}")

    # --- Ликвиден филтър само за АКЦИИ: членство в STOXX 600 / S&P 500 ---
    # ETF-ите минават през без промяна - за тях AUM/оборот филтрите в screener-а
    # вече дават достатъчна гаранция за качество/ликвидност.
    print("Изграждам whitelist от STOXX 600 + S&P 500 имена (S&P 500 - на живо) ...")
    whitelist = build_liquid_stock_whitelist()
    print(f"Общо имена в whitelist-а: {len(whitelist)}")

    before_stock_count = len([i for i in filtered if i.get("type") == "STOCK"])
    result_list = []
    for inst in filtered:
        if inst.get("type") == "STOCK":
            if is_liquid_stock(inst.get("name", ""), whitelist):
                result_list.append(inst)
        else:
            result_list.append(inst)
    filtered = result_list

    after_stock_count = len([i for i in filtered if i.get("type") == "STOCK"])
    print(
        f"Акции преди/след STOXX600+S&P500 филтър: {before_stock_count} -> {after_stock_count}"
    )

    distinct_exchanges = sorted(set(i["exchangeName"] for i in filtered))
    print("Борси в резултата:", ", ".join(distinct_exchanges))

    # Разделяме на акции и ETF-и за по-лесна обработка по-нататък
    etfs = [i for i in filtered if i.get("type") == "ETF"]
    stocks = [i for i in filtered if i.get("type") == "STOCK"]
    print(f"  -> ETF-и: {len(etfs)}")
    print(f"  -> Акции: {len(stocks)}")

    result = {
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(filtered),
        "instruments": filtered,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Записано в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
