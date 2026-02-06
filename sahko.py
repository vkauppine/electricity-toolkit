#!/usr/bin/env python3
"""
Minimalistic CLI dashboard for Finnish electricity market.

Usage:
    python sahko.py              # Full dashboard
    python sahko.py today        # Same as above
    python sahko.py now          # Current status only
    python sahko.py tomorrow     # Tomorrow's forecast only
    python sahko.py week         # Weekly summary only
    python sahko.py reservoir    # Nordic hydro reservoir levels
    python sahko.py fuel         # European fuel & carbon prices
    python sahko.py table        # Hourly table view
    python sahko.py dash         # Rich dashboard
    python sahko.py --vat        # Include VAT (24%) in prices
"""

import sys
import os
import io
import time
import requests

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.system("")  # Enable ANSI/UTF-8 sequences in Windows terminal
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    FINGRID_API_KEY, FINGRID_API_URL, ENTSOE_API_KEY, OILPRICE_API_KEY,
    BALANCING_UP_REGULATION_PRICE, BALANCING_DOWN_REGULATION_PRICE, VAT_RATE,
)

# ── Language / i18n ───────────────────────────────────────────────

_LANG = "fi"  # default; set to "en" via --lang en
_INCLUDE_VAT = False  # default; set to True via --vat flag

_STRINGS = {
    "fi": {
        "weekdays_short": ["MA", "TI", "KE", "TO", "PE", "LA", "SU"],
        "weekdays_long": ["maanantai", "tiistai", "keskiviikko", "torstai",
                          "perjantai", "lauantai", "sunnuntai"],
        "header_title": "SUOMEN SAHKOMARKKINAT",
        "right_now": "JUURI NYT",
        "spot_price": "Spot-hinta",
        "consumption": "Kulutus",
        "wind_power": "Tuulivoima",
        "nuclear": "Ydinvoima",
        "temperature": "Lampotila",
        "today": "TANAAN",
        "today_avg": "TANAAN (keskiarvo)",
        "price": "Hinta",
        "cheapest_hour": "Halvin tunti",
        "most_expensive": "Kallein tunti",
        "no_prices": "Ei hintatietoja saatavilla",
        "vs_yesterday": "vs. eilen",
        "now_marker": "NYT!",
        "tomorrow": "HUOMENNA",
        "prices_published": "Hinnat julkaistaan n. klo 13:45",
        "avg_price": "Keskihinta",
        "warming": "lampenee",
        "cooling": "kylmenee",
        "windy": "tuulinen!",
        "recommendations": "SUOSITUKSET",
        "charge_now": "Lataa auto / pyykki NYT",
        "cheap_hour_today": "Halpa tunti tanaan klo",
        "avoid_consumption": "Valta kulutusta klo",
        "heat_sauna": "Lammita sauna huomenna klo",
        "shift_tomorrow": "Siirra kulutusta huomiseen (hinta",
        "normal_prices": "Normaali hintataso, ei erityisia suosituksia",
        "week": "VIIKKO",
        "no_week_data": "Ei viikkodataa",
        "avg_price_label": "Hinta keskiarvo",
        "volatility": "Volatiliteetti",
        "trend": "Trendi",
        "vol_low": "Matala",
        "vol_mid": "Keskitaso",
        "vol_high": "Korkea",
        "trend_up": "Nouseva",
        "trend_down": "Laskeva",
        "trend_flat": "Tasainen",
        # Rich dashboard
        "loading": "Haetaan sähkömarkkinadataa...",
        "header_rich": "SÄHKÖMARKKINAT",
        "spot_price_panel": "SPOT-HINTA",
        "now_label": "NYT",
        "day_label": "Vrk",
        "tomorrow_label": "Huom",
        "published_at": "julk. ~13:45",
        "trend_label": "Trendi",
        "production_panel": "TUOTANTO",
        "wind": "Tuuli",
        "nuclear_short": "Ydin",
        "hydro": "Vesi",
        "solar": "Aurinko",
        "thermal": "Lämpö",
        "import": "Tuonti",
        "export": "Vienti",
        "total": "Yht",
        "hourly_chart": "TUNTIHINTA TÄNÄÄN",
        "no_price_data": "Ei hintatietoja",
        "now_bar_marker": "NYT",
        "alerts_panel": "HÄLYTYKSET",
        "cheap_hour_now": "Halpa tunti NYT!",
        "cheap_hour_at": "Halpa tunti klo",
        "expensive_coming": "Kallis ilta tulossa",
        "tomorrow_cheaper": "Huomenna",
        "cheaper": "halvempaa",
        "more_expensive": "kalliimpaa",
        "no_alerts": "Ei erityisiä hälytyksiä",
        "weather_panel": "SÄÄ & OLOSUHTEET",
        "calm": "tyyntä",
        "breezy": "tuulista",
        "stormy": "myrskyistä",
        "clear": "Selkeää",
        "partly_cloudy": "Puolipilvistä",
        "mostly_cloudy": "Melko pilvistä",
        "cloudy": "Pilvistä",
        "tomorrow_short": "Huom",
        "footer_keys": "[q] Lopeta  [t] Tänään  [w] Viikko  [r] Päivitä",
        "goodbye": "Näkemiin!",
        "press_enter": "Paina Enter palataksesi...",
        "waiting_key": "Odotetaan näppäinpainallusta...",
        # Table view
        "table_title": "TUNTIHINTA SEURAAVAT",
        "col_hour": "Tunti",
        "col_price": "Hinta",
        "col_change": "Muutos",
        "col_wind": "Tuuli",
        "col_consumption": "Kulutus",
        "col_bal_up": "Sään.↑",
        "col_bal_down": "Sään.↓",
        "col_recommendation": "Suositus",
        "table_loading": "Haetaan dataa...",
        "table_no_prices": "Ei hintatietoja saatavilla.",
        "table_no_range": "Ei hintatietoja pyydetylle aikavälille.",
        "rec_excellent": "LOISTAVA",
        "rec_good": "HYVA",
        "rec_ok": "OK",
        "rec_expensive": "KALLIS",
        "summary_avg": "Keskihinta",
        # Reservoir view
        "reservoir_title": "VESIVARANNOT",
        "reservoir_week": "vko",
        "finland": "Suomi",
        "sweden": "Ruotsi",
        "norway": "Norja",
        "nordic_avg": "Pohjoismainen ka",
        "normal": "normaali",
        "above_normal": "yli normaalin",
        "below_normal": "alle normaalin",
        "year_comparison": "Vuosivertailu",
        "reservoir_panel": "VESIVARANNOT",
        "no_entsoe_key": "ENTSO-E API-avain puuttuu",
        # Fuel prices
        "fuel_title": "POLTTOAINEHINNAT",
        "fuel_panel": "POLTTOAINEET",
        "gas_ttf": "Maakaasu (TTF)",
        "coal_api2": "Hiili (API2)",
        "carbon_ets": "Päästöoikeus (EU)",
        "gas_marginal": "Kaasun rajakust.",
        "week_trend": "Viikkotrendi",
        "no_fuel_key": "OilPrice API-avain puuttuu",
        "electricity_equiv": "sähkö",
    },
    "en": {
        "weekdays_short": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
        "weekdays_long": ["Monday", "Tuesday", "Wednesday", "Thursday",
                          "Friday", "Saturday", "Sunday"],
        "header_title": "FINLAND ELECTRICITY MARKET",
        "right_now": "RIGHT NOW",
        "spot_price": "Spot price",
        "consumption": "Consumption",
        "wind_power": "Wind power",
        "nuclear": "Nuclear",
        "temperature": "Temperature",
        "today": "TODAY",
        "today_avg": "TODAY (average)",
        "price": "Price",
        "cheapest_hour": "Cheapest hour",
        "most_expensive": "Most expensive",
        "no_prices": "No price data available",
        "vs_yesterday": "vs. yesterday",
        "now_marker": "NOW!",
        "tomorrow": "TOMORROW",
        "prices_published": "Prices published around 13:45",
        "avg_price": "Avg price",
        "warming": "warming",
        "cooling": "cooling",
        "windy": "windy!",
        "recommendations": "RECOMMENDATIONS",
        "charge_now": "Charge EV / do laundry NOW",
        "cheap_hour_today": "Cheap hour today at",
        "avoid_consumption": "Avoid consumption at",
        "heat_sauna": "Heat sauna tomorrow at",
        "shift_tomorrow": "Shift consumption to tomorrow (price",
        "normal_prices": "Normal price level, no special recommendations",
        "week": "WEEK",
        "no_week_data": "No weekly data",
        "avg_price_label": "Avg price",
        "volatility": "Volatility",
        "trend": "Trend",
        "vol_low": "Low",
        "vol_mid": "Medium",
        "vol_high": "High",
        "trend_up": "Rising",
        "trend_down": "Falling",
        "trend_flat": "Stable",
        # Rich dashboard
        "loading": "Fetching electricity market data...",
        "header_rich": "ELECTRICITY MARKET",
        "spot_price_panel": "SPOT PRICE",
        "now_label": "NOW",
        "day_label": "Day",
        "tomorrow_label": "Tmrw",
        "published_at": "publ. ~13:45",
        "trend_label": "Trend",
        "production_panel": "PRODUCTION",
        "wind": "Wind",
        "nuclear_short": "Nucl",
        "hydro": "Hydro",
        "solar": "Solar",
        "thermal": "CHP",
        "import": "Import",
        "export": "Export",
        "total": "Total",
        "hourly_chart": "HOURLY PRICE TODAY",
        "no_price_data": "No price data",
        "now_bar_marker": "NOW",
        "alerts_panel": "ALERTS",
        "cheap_hour_now": "Cheap hour NOW!",
        "cheap_hour_at": "Cheap hour at",
        "expensive_coming": "Expensive evening ahead",
        "tomorrow_cheaper": "Tomorrow",
        "cheaper": "cheaper",
        "more_expensive": "more expensive",
        "no_alerts": "No special alerts",
        "weather_panel": "WEATHER & CONDITIONS",
        "calm": "calm",
        "breezy": "breezy",
        "stormy": "stormy",
        "clear": "Clear",
        "partly_cloudy": "Partly cloudy",
        "mostly_cloudy": "Mostly cloudy",
        "cloudy": "Cloudy",
        "tomorrow_short": "Tmrw",
        "footer_keys": "[q] Quit  [t] Today  [w] Week  [r] Refresh",
        "goodbye": "Goodbye!",
        "press_enter": "Press Enter to go back...",
        "waiting_key": "Waiting for keypress...",
        # Table view
        "table_title": "HOURLY PRICE NEXT",
        "col_hour": "Hour",
        "col_price": "Price",
        "col_change": "Change",
        "col_wind": "Wind",
        "col_consumption": "Demand",
        "col_bal_up": "Bal.↑",
        "col_bal_down": "Bal.↓",
        "col_recommendation": "Rating",
        "table_loading": "Fetching data...",
        "table_no_prices": "No price data available.",
        "table_no_range": "No price data for requested range.",
        "rec_excellent": "EXCELLENT",
        "rec_good": "GOOD",
        "rec_ok": "OK",
        "rec_expensive": "EXPENSIVE",
        "summary_avg": "Average",
        # Reservoir view
        "reservoir_title": "WATER RESERVOIRS",
        "reservoir_week": "wk",
        "finland": "Finland",
        "sweden": "Sweden",
        "norway": "Norway",
        "nordic_avg": "Nordic average",
        "normal": "normal",
        "above_normal": "above normal",
        "below_normal": "below normal",
        "year_comparison": "Year comparison",
        "reservoir_panel": "WATER RESERVOIRS",
        "no_entsoe_key": "ENTSO-E API key missing",
        # Fuel prices
        "fuel_title": "FUEL PRICES",
        "fuel_panel": "FUELS",
        "gas_ttf": "Natural gas (TTF)",
        "coal_api2": "Coal (API2)",
        "carbon_ets": "Carbon (EU ETS)",
        "gas_marginal": "Gas SRMC",
        "week_trend": "Week trend",
        "no_fuel_key": "OilPrice API key missing",
        "electricity_equiv": "elec",
    },
}


def _t(key: str) -> str:
    """Get translated string for current language."""
    return _STRINGS.get(_LANG, _STRINGS["fi"]).get(key, _STRINGS["fi"].get(key, key))


# ── Timezone & locale helpers ─────────────────────────────────────

_FI_TZ = timezone(timedelta(hours=2))  # EET (simplified, ignores DST)


def _fi_now():
    """Current time in Finnish timezone."""
    return datetime.now(timezone.utc).astimezone(_FI_TZ)


def _weekday_short(dt):
    return _t("weekdays_short")[dt.weekday()]


def _date_str(dt):
    return f"{_weekday_short(dt)} {dt.day}.{dt.month}.{dt.year}"


# ── Quiet data fetching (no print noise) ───────────────────────────

def _quiet_fingrid(dataset_id: int, start: str, end: str) -> pd.DataFrame:
    """Fetch from Fingrid API without printing."""
    if not FINGRID_API_KEY:
        return pd.DataFrame(columns=["timestamp", "value"])
    headers = {"x-api-key": FINGRID_API_KEY}
    all_rows = []
    page = 1
    while True:
        params = {
            "startTime": start, "endTime": end, "format": "json",
            "page": page, "pageSize": 20000, "locale": "en",
            "sortBy": "startTime", "sortOrder": "asc",
        }
        url = FINGRID_API_URL.format(dataset_id=dataset_id)
        for attempt in range(5):
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            break
        if resp.status_code != 200:
            return pd.DataFrame(columns=["timestamp", "value"])
        payload = resp.json()
        data = payload.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        if len(all_rows) >= payload.get("pagination", {}).get("total", 0):
            break
        page += 1
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "value"])
    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"startTime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["timestamp", "value"]].dropna()


def _quiet_spot(start: str, end: str) -> pd.DataFrame:
    """Fetch spot prices without printing."""
    resp = requests.get("https://sahkotin.fi/prices",
                        params={"start": start, "end": end}, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame(columns=["timestamp", "price"])
    rows = resp.json().get("prices", [])
    if not rows:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = pd.DataFrame(rows)
    df = df.rename(columns={"date": "timestamp", "value": "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["timestamp", "price"]].dropna().sort_values("timestamp").reset_index(drop=True)


def _quiet_fmi_temp(place: str = "Helsinki") -> float | None:
    """Fetch latest temperature observation for a city."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "service": "WFS", "version": "2.0.0", "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::hourly::simple",
        "place": place, "starttime": start, "endtime": end,
        "parameters": "TA_PT1H_AVG",
    }
    try:
        resp = requests.get("https://opendata.fmi.fi/wfs", params=params, timeout=15)
        if resp.status_code != 200:
            return None
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
        vals = []
        for el in root.findall(".//BsWfs:BsWfsElement", ns):
            v = el.find("BsWfs:ParameterValue", ns)
            if v is not None:
                try:
                    vals.append(float(v.text))
                except (ValueError, TypeError):
                    pass
        return vals[-1] if vals else None
    except Exception:
        return None


def _quiet_fmi_forecast(place: str = "Helsinki") -> pd.DataFrame:
    """Fetch temperature forecast."""
    params = {
        "service": "WFS", "version": "2.0.0", "request": "getFeature",
        "storedquery_id": "fmi::forecast::harmonie::surface::point::simple",
        "place": place, "parameters": "Temperature", "timestep": 60,
    }
    try:
        resp = requests.get("https://opendata.fmi.fi/wfs", params=params, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
        rows = []
        for el in root.findall(".//BsWfs:BsWfsElement", ns):
            t = el.find("BsWfs:Time", ns)
            v = el.find("BsWfs:ParameterValue", ns)
            if t is not None and v is not None:
                try:
                    rows.append({"timestamp": pd.to_datetime(t.text, utc=True),
                                 "temperature": float(v.text)})
                except (ValueError, TypeError):
                    pass
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Formatting helpers ─────────────────────────────────────────────

def _arrow(val, ref=0):
    if val is None or ref is None:
        return ""
    diff = val - ref
    if abs(diff) < 0.05:
        return "\u2192"  # →
    return "\u25b2" if diff > 0 else "\u25bc"  # ▲ ▼


def _signed(val, unit="", decimals=1):
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.{decimals}f}{unit}"


def _fmt_price(eur_mwh):
    """EUR/MWh → snt/kWh string."""
    if eur_mwh is None:
        return "N/A"
    price = eur_mwh / 10
    if _INCLUDE_VAT:
        price = price * (1 + VAT_RATE)
    return f"{price:.1f}"


def _fmt_mw(mw):
    if mw is None:
        return "N/A"
    return f"{mw:,.0f}"


def _latest_value(df):
    """Get the most recent value from a timestamp/value DataFrame."""
    if df.empty:
        return None, None
    row = df.sort_values("timestamp").iloc[-1]
    return row.get("value", row.get("price", None)), row["timestamp"]


def _hour_ago_value(df):
    """Get value from ~1 hour ago."""
    if df.empty:
        return None
    now = datetime.now(timezone.utc)
    target = now - timedelta(hours=1)
    df = df.copy()
    df["_diff"] = (df["timestamp"] - target).abs()
    closest = df.loc[df["_diff"].idxmin()]
    return closest.get("value", closest.get("price", None))


# ── Dashboard sections ─────────────────────────────────────────────

def _print_header():
    now = _fi_now()
    date_str = _date_str(now).upper()
    title = f"{_t('header_title')} - {date_str}"
    w = 62
    print()
    print("\u2554" + "\u2550" * w + "\u2557")
    print("\u2551" + title.center(w) + "\u2551")
    print("\u255a" + "\u2550" * w + "\u255d")


def section_now(prices_today, wind_df, cons_df, nuc_df, temp_now, temp_1h_ago):
    now = _fi_now()
    price_now, _ = _latest_value(prices_today)
    price_1h = _hour_ago_value(prices_today)

    wind_now, _ = _latest_value(wind_df)
    wind_1h = _hour_ago_value(wind_df)

    cons_now, _ = _latest_value(cons_df)
    cons_1h = _hour_ago_value(cons_df)

    nuc_now, _ = _latest_value(nuc_df)

    price_snt = float(_fmt_price(price_now)) if price_now is not None else None
    price_snt_1h = float(_fmt_price(price_1h)) if price_1h is not None else None
    price_diff = (price_snt - price_snt_1h) if price_snt is not None and price_snt_1h is not None else None

    wind_diff = (wind_now - wind_1h) if wind_now is not None and wind_1h is not None else None
    cons_diff = (cons_now - cons_1h) if cons_now is not None and cons_1h is not None else None
    temp_diff = (temp_now - temp_1h_ago) if temp_now is not None and temp_1h_ago is not None else None

    nuc_pct = f"({nuc_now / 4400 * 100:.0f}%)" if nuc_now is not None else ""

    vat_suffix = " sis.ALV" if _INCLUDE_VAT else ""
    print(f"\n\U0001f4ca {_t('right_now')} ({now.strftime('%H:%M')})")
    print(f"\u251c\u2500 {_t('spot_price')+':':<16}{price_snt if price_snt is not None else 'N/A':>8} snt/kWh{vat_suffix}  "
          f"{_arrow(price_snt, price_snt_1h)} {_signed(price_diff) if price_diff is not None else ''}")
    print(f"\u251c\u2500 {_t('consumption')+':':<16}{_fmt_mw(cons_now):>8} MW     "
          f"{_arrow(cons_now, cons_1h)} {_signed(cons_diff, ' MW', 0) if cons_diff is not None else ''}")
    print(f"\u251c\u2500 {_t('wind_power')+':':<16}{_fmt_mw(wind_now):>8} MW     "
          f"{_arrow(wind_now, wind_1h)} {_signed(wind_diff, ' MW', 0) if wind_diff is not None else ''}")
    print(f"\u251c\u2500 {_t('nuclear')+':':<16}{_fmt_mw(nuc_now):>8} MW     \u2192 {nuc_pct}")
    print(f"\u2514\u2500 {_t('temperature')+':':<16}{temp_now:>8.1f}\u00b0C     "
          f"{_arrow(temp_now, temp_1h_ago)} {_signed(temp_diff, '\u00b0C') if temp_diff is not None else ''}"
          if temp_now is not None else
          f"\u2514\u2500 {_t('temperature')+':':<16}    N/A")


def section_today(prices_today, prices_yesterday):
    if prices_today.empty:
        print(f"\n\u26a1 {_t('today')}")
        print(f"\u2514\u2500 {_t('no_prices')}")
        return

    now = _fi_now()
    avg_today = prices_today["price"].mean()
    avg_yesterday = prices_yesterday["price"].mean() if not prices_yesterday.empty else None
    pct_change = ((avg_today - avg_yesterday) / avg_yesterday * 100) if avg_yesterday else None

    cheapest_idx = prices_today["price"].idxmin()
    cheapest = prices_today.loc[cheapest_idx]
    cheapest_hour = cheapest["timestamp"].astimezone(_FI_TZ).hour

    most_exp_idx = prices_today["price"].idxmax()
    most_exp = prices_today.loc[most_exp_idx]
    most_exp_hour = most_exp["timestamp"].astimezone(_FI_TZ).hour

    is_cheapest_now = (cheapest_hour == now.hour)

    pct_str = f"({_signed(pct_change, '%', 0)} {_t('vs_yesterday')})" if pct_change is not None else ""
    vat_suffix = " sis.ALV" if _INCLUDE_VAT else ""

    print(f"\n\u26a1 {_t('today_avg')}")
    print(f"\u251c\u2500 {_t('price')+':':<16}{_fmt_price(avg_today):>8} snt/kWh{vat_suffix}  {pct_str}")
    print(f"\u251c\u2500 {_t('cheapest_hour')+':':<16}{_fmt_price(cheapest['price']):>8} snt @ {cheapest_hour:02d}:00"
          f"{'  \u2713 ' + _t('now_marker') if is_cheapest_now else ''}")
    print(f"\u2514\u2500 {_t('most_expensive')+':':<16}{_fmt_price(most_exp['price']):>8} snt @ {most_exp_hour:02d}:00")


def section_tomorrow(prices_tomorrow, temp_forecast, wind_forecast, prices_today):
    tomorrow = _fi_now() + timedelta(days=1)
    tomorrow_date = tomorrow.date()

    print(f"\n\U0001f52e {_t('tomorrow')} ({_weekday_short(tomorrow)} {tomorrow.day}.{tomorrow.month}.)")

    if prices_tomorrow.empty:
        print(f"\u2514\u2500 {_t('prices_published')}")
        return

    avg_tomorrow = prices_tomorrow["price"].mean()
    avg_today = prices_today["price"].mean() if not prices_today.empty else None
    pct_change = ((avg_tomorrow - avg_today) / avg_today * 100) if avg_today else None

    cheapest_idx = prices_tomorrow["price"].idxmin()
    cheapest = prices_tomorrow.loc[cheapest_idx]
    cheapest_hour = cheapest["timestamp"].astimezone(_FI_TZ).hour

    pct_str = f"{_arrow(avg_tomorrow, avg_today)} {_signed(pct_change, '%', 0)}" if pct_change is not None else ""
    vat_suffix = " sis.ALV" if _INCLUDE_VAT else ""

    print(f"\u251c\u2500 {_t('avg_price')+':':<16}{_fmt_price(avg_tomorrow):>8} snt/kWh{vat_suffix}  {pct_str}")
    print(f"\u251c\u2500 {_t('cheapest_hour')+':':<16}{_fmt_price(cheapest['price']):>8} snt @ {cheapest_hour:02d}:00")

    if not wind_forecast.empty:
        wf_tomorrow = wind_forecast[
            wind_forecast["timestamp"].dt.date == tomorrow_date
        ]
        if not wf_tomorrow.empty:
            avg_wind = wf_tomorrow["value"].mean()
            wind_str = f"{avg_wind:,.0f}+ MW"
            if avg_wind > 3000:
                wind_str += f" ({_t('windy')})"
            print(f"\u251c\u2500 {_t('wind_power')+':':<16}{wind_str:>8}")

    if not temp_forecast.empty:
        tf_tomorrow = temp_forecast[
            temp_forecast["timestamp"].dt.tz_convert(_FI_TZ).dt.date == tomorrow_date
        ]
        if not tf_tomorrow.empty:
            avg_temp = tf_tomorrow["temperature"].mean()
            temp_now = _quiet_fmi_temp("Helsinki")
            temp_comment = ""
            if temp_now is not None:
                diff = avg_temp - temp_now
                if diff > 2:
                    temp_comment = f" ({_t('warming')})"
                elif diff < -2:
                    temp_comment = f" ({_t('cooling')})"
            print(f"\u2514\u2500 {_t('temperature')+':':<16}{avg_temp:>+8.1f}\u00b0C{temp_comment}")
            return

    print(f"\u2514\u2500 {_t('temperature')+':':<16}     N/A")


def section_recommendations(prices_today, prices_tomorrow):
    now = _fi_now()
    vat_multiplier = (1 + VAT_RATE) if _INCLUDE_VAT else 1.0
    print(f"\n\U0001f4a1 {_t('recommendations')}")

    recs = []

    if not prices_today.empty:
        remaining = prices_today[
            prices_today["timestamp"].dt.tz_convert(_FI_TZ).dt.hour >= now.hour
        ]
        if not remaining.empty:
            cheapest = remaining.loc[remaining["price"].idxmin()]
            cheapest_hour = cheapest["timestamp"].astimezone(_FI_TZ).hour
            cheap_price = cheapest["price"] / 10 * vat_multiplier

            threshold = 5 * vat_multiplier
            if cheap_price < threshold:
                if cheapest_hour == now.hour:
                    recs.append(f"\u2705 {_t('charge_now')} ({cheap_price:.1f} snt)")
                else:
                    recs.append(f"\u2705 {_t('cheap_hour_today')} {cheapest_hour:02d}:00 ({cheap_price:.1f} snt)")

            expensive = remaining.nlargest(3, "price")
            exp_hours = sorted(expensive["timestamp"].dt.tz_convert(_FI_TZ).dt.hour.tolist())
            if len(exp_hours) >= 2:
                exp_price = expensive["price"].mean() / 10 * vat_multiplier
                threshold_high = 8 * vat_multiplier
                if exp_price > threshold_high:
                    recs.append(f"\u26a0\ufe0f  {_t('avoid_consumption')} {exp_hours[0]:02d}-{exp_hours[-1]+1:02d} ({exp_price:.0f} snt)")

    if not prices_tomorrow.empty:
        cheapest_tm = prices_tomorrow.loc[prices_tomorrow["price"].idxmin()]
        cheapest_hour_tm = cheapest_tm["timestamp"].astimezone(_FI_TZ).hour
        cheap_price_tm = cheapest_tm["price"] / 10 * vat_multiplier
        threshold = 5 * vat_multiplier
        if cheap_price_tm < threshold:
            recs.append(f"\u2705 {_t('heat_sauna')} {cheapest_hour_tm:02d}-{cheapest_hour_tm+2:02d} ({cheap_price_tm:.1f} snt)")

        avg_tm = prices_tomorrow["price"].mean() / 10
        avg_td = prices_today["price"].mean() / 10 if not prices_today.empty else None
        if avg_td and avg_tm < avg_td * 0.8:
            recs.append(f"\u2705 {_t('shift_tomorrow')} -{ (1 - avg_tm/avg_td)*100:.0f}%)")

    if not recs:
        recs.append(f"\u2705 {_t('normal_prices')}")

    for r in recs:
        print(f"  {r}")


def section_week(prices_week):
    now = _fi_now()
    wds = _t("weekdays_short")
    week_label = f"{wds[0]}-{wds[min(now.weekday(), 4)]}"

    print(f"\n\U0001f4c8 {_t('week')} ({week_label})")

    if prices_week.empty:
        print(f"\u2514\u2500 {_t('no_week_data')}")
        return

    avg = prices_week["price"].mean() / 10
    std = prices_week["price"].std() / 10
    daily_avgs = prices_week.set_index("timestamp").resample("D")["price"].mean().dropna()

    if len(daily_avgs) >= 3:
        mid = len(daily_avgs) // 2
        first_half = daily_avgs.iloc[:mid].mean()
        second_half = daily_avgs.iloc[mid:].mean()
        if second_half > first_half * 1.1:
            trend = f"\u25b2 {_t('trend_up')}"
        elif second_half < first_half * 0.9:
            trend = f"\u25bc {_t('trend_down')}"
        else:
            trend = f"\u2192 {_t('trend_flat')}"
    else:
        trend = "N/A"

    if std < 2:
        vol = _t("vol_low")
    elif std < 5:
        vol = _t("vol_mid")
    else:
        vol = _t("vol_high")

    vat_suffix = " sis.ALV" if _INCLUDE_VAT else ""
    print(f"\u251c\u2500 {_t('avg_price_label')+':':<17}{avg:>7.1f} snt/kWh{vat_suffix}")
    print(f"\u251c\u2500 {_t('volatility')+':':<17}{vol:>7}")
    print(f"\u2514\u2500 {_t('trend')+':':<17}{trend:>7}")

    if len(daily_avgs) >= 2:
        _print_sparkline(daily_avgs)


def _print_sparkline(series):
    """Print a simple ASCII sparkline of daily prices."""
    vals = series.values / 10  # snt/kWh
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx > mn else 1
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    spark = ""
    for v in vals:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        spark += blocks[idx]
    days = [s.strftime("%a")[:2] for s in series.index]
    print(f"    {'  '.join(days)}")
    print(f"    {'  '.join(spark)}")
    print(f"    {mn:.0f}{' ' * (len(days) * 3 - 4)}{mx:.0f} snt")


# ── Main ───────────────────────────────────────────────────────────

def fetch_all_data():
    """Fetch all data needed for the dashboard."""
    now = datetime.now(timezone.utc)
    fi_now = now.astimezone(_FI_TZ)
    today_start = fi_now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start.astimezone(timezone.utc)
    tomorrow_start = today_start + timedelta(days=1)
    tomorrow_end = tomorrow_start + timedelta(days=1)
    yesterday_start = today_start - timedelta(days=1)
    week_start = today_start - timedelta(days=today_start.weekday())

    def ts(dt):
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch in parallel-ish (sequential but quiet)
    data = {}

    # Prices
    data["prices_today"] = _quiet_spot(ts(today_start), ts(tomorrow_start))
    data["prices_yesterday"] = _quiet_spot(ts(yesterday_start), ts(today_start))
    data["prices_tomorrow"] = _quiet_spot(ts(tomorrow_start), ts(tomorrow_end))
    data["prices_week"] = _quiet_spot(ts(week_start), ts(tomorrow_start))

    # Real-time Fingrid (last 2 hours for current + 1h ago comparison)
    rt_start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rt_end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    data["wind"] = _quiet_fingrid(181, rt_start, rt_end)
    data["consumption"] = _quiet_fingrid(124, rt_start, rt_end)
    time.sleep(0.5)
    data["nuclear"] = _quiet_fingrid(188, rt_start, rt_end)

    # Temperature
    data["temp_now"] = _quiet_fmi_temp("Helsinki")
    data["temp_1h_ago"] = data["temp_now"]  # Approximate

    # Forecasts for tomorrow
    fc_start = ts(tomorrow_start)
    fc_end = ts(tomorrow_end)
    time.sleep(0.5)
    data["wind_forecast"] = _quiet_fingrid(245, fc_start, fc_end)
    data["temp_forecast"] = _quiet_fmi_forecast("Helsinki")

    return data


def dashboard(mode="today"):
    """Run the dashboard."""
    _print_header()
    data = fetch_all_data()

    if mode in ("today", "now"):
        section_now(
            data["prices_today"], data["wind"], data["consumption"],
            data["nuclear"], data["temp_now"], data["temp_1h_ago"],
        )

    if mode in ("today", "tomorrow"):
        section_today(data["prices_today"], data["prices_yesterday"])

    if mode in ("today", "tomorrow"):
        section_tomorrow(
            data["prices_tomorrow"], data["temp_forecast"],
            data["wind_forecast"], data["prices_today"],
        )

    if mode in ("today",):
        section_recommendations(data["prices_today"], data["prices_tomorrow"])

    if mode in ("today", "week"):
        section_week(data["prices_week"])

    print()


def _quiet_fmi_wind_speed(place: str = "Helsinki") -> float | None:
    """Fetch latest wind speed observation."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "service": "WFS", "version": "2.0.0", "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::hourly::simple",
        "place": place, "starttime": start, "endtime": end,
        "parameters": "WS_PT1H_AVG",
    }
    try:
        resp = requests.get("https://opendata.fmi.fi/wfs", params=params, timeout=15)
        if resp.status_code != 200:
            return None
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
        vals = []
        for el in root.findall(".//BsWfs:BsWfsElement", ns):
            v = el.find("BsWfs:ParameterValue", ns)
            if v is not None:
                try:
                    vals.append(float(v.text))
                except (ValueError, TypeError):
                    pass
        return vals[-1] if vals else None
    except Exception:
        return None


def _quiet_fmi_cloud(place: str = "Helsinki") -> float | None:
    """Fetch latest cloud cover observation (oktas 0-8)."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "service": "WFS", "version": "2.0.0", "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::hourly::simple",
        "place": place, "starttime": start, "endtime": end,
        "parameters": "N_PT1H_AVG",
    }
    try:
        resp = requests.get("https://opendata.fmi.fi/wfs", params=params, timeout=15)
        if resp.status_code != 200:
            return None
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
        vals = []
        for el in root.findall(".//BsWfs:BsWfsElement", ns):
            v = el.find("BsWfs:ParameterValue", ns)
            if v is not None:
                try:
                    vals.append(float(v.text))
                except (ValueError, TypeError):
                    pass
        return vals[-1] if vals else None
    except Exception:
        return None


# ── Reservoir helpers ──────────────────────────────────────────────

def _quiet_entsoe_reservoir() -> pd.DataFrame | None:
    """Fetch current Nordic reservoir data. Returns None if no API key."""
    if not ENTSOE_API_KEY:
        return None
    try:
        from fetch_entsoe import fetch_nordic_reservoirs
        now = datetime.now(timezone.utc)
        # Fetch last 2 weeks to get latest data point
        start = (now - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        return fetch_nordic_reservoirs(start, end)
    except Exception:
        return None


def _quiet_entsoe_reservoir_history(years: int = 2) -> dict[int, pd.DataFrame]:
    """Fetch reservoir history for current + previous years. Returns {year: df}."""
    if not ENTSOE_API_KEY:
        return {}
    try:
        from fetch_entsoe import fetch_nordic_reservoirs
        now = datetime.now(timezone.utc)
        current_year = now.year
        result = {}
        for y in range(current_year - years, current_year + 1):
            start = f"{y}-01-01T00:00:00Z"
            end = f"{y}-12-31T23:59:59Z" if y < current_year else now.strftime("%Y-%m-%dT%H:%M:%SZ")
            df = fetch_nordic_reservoirs(start, end)
            if not df.empty:
                result[y] = df
        return result
    except Exception:
        return {}


def _filling_bar(pct: float, width: int = 20) -> str:
    """Create a text bar like ██████████████░░░░░░"""
    if pct is None:
        return "N/A"
    filled = int(pct / 100 * width)
    filled = max(0, min(width, filled))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _week_change(df: pd.DataFrame, col: str) -> float | None:
    """Get week-over-week change for a column."""
    if df is None or df.empty or len(df) < 2 or col not in df.columns:
        return None
    vals = df[col].dropna()
    if len(vals) < 2:
        return None
    return vals.iloc[-1] - vals.iloc[-2]


def _sparkline_year(df: pd.DataFrame, col: str = "nordic_avg") -> str:
    """Build a sparkline for a year of reservoir data."""
    if df is None or df.empty or col not in df.columns:
        return ""
    vals = df[col].dropna().values
    if len(vals) < 2:
        return ""
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx > mn else 1
    blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    spark = ""
    for v in vals:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        spark += blocks[idx]
    return spark


def reservoir_view():
    """Standalone plain text reservoir view."""
    now = _fi_now()
    week_num = now.isocalendar()[1]

    print(f"\n{_t('reservoir_title')} ({_t('reservoir_week')} {week_num}/{now.year})")
    print("\u2500" * 50)

    if not ENTSOE_API_KEY:
        print(f"  {_t('no_entsoe_key')}")
        print(f"  Set ENTSOE_API_KEY in .env")
        return

    reservoir = _quiet_entsoe_reservoir()
    if reservoir is None or reservoir.empty:
        print("  N/A")
        return

    latest = reservoir.iloc[-1]

    countries = [
        ("finland", "fi_pct"),
        ("sweden", "se_pct"),
        ("norway", "no_pct"),
    ]

    for name_key, col in countries:
        val = latest.get(col)
        if val is not None and not pd.isna(val):
            change = _week_change(reservoir, col)
            arrow = ""
            if change is not None:
                arrow = f"\u25b2 +{change:.1f}%" if change > 0 else f"\u25bc {change:.1f}%"
            print(f"{_t(name_key)+':':<10}{val:5.1f}%  {_filling_bar(val)}  {arrow}")
        else:
            print(f"{_t(name_key)+':':<10}  N/A")

    print("\u2500" * 50)

    avg = latest.get("nordic_avg")
    if avg is not None and not pd.isna(avg):
        # Rough seasonal normal (~70% is typical average)
        normal = 70
        if avg > normal + 5:
            status = _t("above_normal")
        elif avg < normal - 5:
            status = _t("below_normal")
        else:
            status = _t("normal")
        print(f"{_t('nordic_avg')}: {avg:.1f}%  ({status})")

    # Year comparison sparklines
    history = _quiet_entsoe_reservoir_history(years=2)
    if history:
        print(f"\n{_t('year_comparison')}:")
        current_year = now.year
        for y in sorted(history.keys(), reverse=True):
            spark = _sparkline_year(history[y])
            marker = f"  \u2190 {_t('reservoir_week')} {week_num}" if y == current_year else ""
            print(f"{y}: {spark}{marker}")

    print()


# ── Fuel price helpers ─────────────────────────────────────────────

def _quiet_fuel_prices() -> dict:
    """Fetch fuel prices without printing. Returns dict or empty."""
    try:
        from fetch_fuel_prices import fetch_all_fuel_prices
        return fetch_all_fuel_prices()
    except Exception:
        return {}


def _quiet_fuel_history(code: str, period: str = "past_week") -> pd.DataFrame:
    """Fetch fuel price history without printing."""
    try:
        from fetch_fuel_prices import fetch_fuel_history
        return fetch_fuel_history(code, period)
    except Exception:
        return pd.DataFrame()


def fuel_view():
    """Standalone plain text fuel price view."""
    from config import OILPRICE_CODE_GAS, OILPRICE_CODE_CARBON

    now = _fi_now()
    print(f"\n{_t('fuel_title')} ({now.day}.{now.month}.{now.year})")
    print("\u2500" * 44)

    if not OILPRICE_API_KEY:
        print(f"  {_t('no_fuel_key')}")
        print("  Set OILPRICE_API_KEY in .env")
        return

    prices = _quiet_fuel_prices()
    if not prices:
        print("  N/A")
        return

    # Fetch week histories for trend sparklines
    gas_hist = _quiet_fuel_history(OILPRICE_CODE_GAS, "past_week")
    carbon_hist = _quiet_fuel_history(OILPRICE_CODE_CARBON, "past_week")

    def _pct_change(hist_df):
        if hist_df.empty or len(hist_df) < 2:
            return None
        first = hist_df["price"].iloc[0]
        last = hist_df["price"].iloc[-1]
        if first == 0:
            return None
        return (last - first) / first * 100

    gas = prices.get("gas")
    coal = prices.get("coal")
    carbon = prices.get("carbon")

    gas_pct = _pct_change(gas_hist)
    carbon_pct = _pct_change(carbon_hist)

    def _fmt_line(label, data, unit, pct=None):
        if data is None:
            print(f"{label:<22}{'N/A':>10}")
            return
        price = data["price"]
        currency = data["currency"]
        arrow = ""
        pct_str = ""
        if pct is not None:
            arrow = "\u25b2" if pct > 0 else "\u25bc" if pct < 0 else "\u2192"
            pct_str = f"{arrow} {pct:+.1f}%"
        print(f"{label:<22}{price:>8.2f} {currency}/{unit:<6} {pct_str}")

    _fmt_line(_t("gas_ttf") + ":", gas, "MWh", gas_pct)
    _fmt_line(_t("coal_api2") + ":", coal, "t")
    _fmt_line(_t("carbon_ets") + ":", carbon, "tCO2", carbon_pct)

    print("\u2500" * 44)

    # SRMC calculation
    if gas is not None and carbon is not None:
        from fetch_fuel_prices import compute_gas_srmc
        srmc = compute_gas_srmc(gas["price"], carbon["price"])
        print(f"{_t('gas_marginal')}: ~{srmc:.1f} EUR/MWh ({_t('electricity_equiv')})")

    # Week trend sparklines
    def _fuel_sparkline(hist_df):
        if hist_df.empty or len(hist_df) < 2:
            return ""
        vals = hist_df["price"].values
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx > mn else 1
        blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        spark = ""
        for v in vals:
            idx = int((v - mn) / rng * (len(blocks) - 1))
            spark += blocks[idx]
        return spark

    gas_spark = _fuel_sparkline(gas_hist)
    carbon_spark = _fuel_sparkline(carbon_hist)

    if gas_spark or carbon_spark:
        print(f"\n{_t('week_trend')}:")
        if gas_spark:
            print(f"  TTF:  {gas_spark}")
        if carbon_spark:
            print(f"  CO2:  {carbon_spark}")

    print()


# ── Rich dashboard ─────────────────────────────────────────────────

def _price_color(snt: float) -> str:
    """Return rich color name based on price level."""
    if snt < 3:
        return "green"
    if snt < 6:
        return "bright_green"
    if snt < 10:
        return "yellow"
    if snt < 15:
        return "bright_red"
    return "red bold"


def _price_bar_chart(prices_df, width: int = 52) -> str:
    """Build a vertical bar chart of hourly prices as rich-formatted text."""
    if prices_df.empty:
        return f"  {_t('no_price_data')}"

    vat_multiplier = (1 + VAT_RATE) if _INCLUDE_VAT else 1.0
    now_hour = _fi_now().hour
    df = prices_df.copy()
    df["hour"] = df["timestamp"].dt.tz_convert(_FI_TZ).dt.hour
    df["snt"] = df["price"] / 10 * vat_multiplier
    df = df.sort_values("hour")

    max_price = max(df["snt"].max(), 1)
    chart_height = 8
    scale = chart_height / max_price

    # Y-axis labels + bars
    lines = []
    for row_idx in range(chart_height, 0, -1):
        threshold = row_idx / scale
        label = f"{threshold:5.0f} │" if row_idx % 2 == 0 else "      │"
        bar_chars = []
        for _, r in df.iterrows():
            h = r["hour"]
            val = r["snt"]
            bar_height = int(val * scale)
            if bar_height >= row_idx:
                color = _price_color(val)
                if h == now_hour:
                    bar_chars.append(f"[{color}]██[/{color}]")
                else:
                    bar_chars.append(f"[{color}]██[/{color}]")
            else:
                bar_chars.append("  ")
        lines.append(label + "".join(bar_chars))

    # X-axis
    lines.append("      └" + "──" * len(df))
    # Hour labels (every 2 hours)
    hour_labels = "       "
    for _, r in df.iterrows():
        h = int(r["hour"])
        if h % 3 == 0:
            hour_labels += f"{h:02d}"
        else:
            hour_labels += "  "
    lines.append(hour_labels)

    # Now marker
    now_pos = 7 + now_hour * 2
    marker_line = " " * now_pos + f"[bold cyan]▲ {_t('now_bar_marker')}[/bold cyan]"
    lines.append(marker_line)

    return "\n".join(lines)


def _build_alerts(prices_today, prices_tomorrow, temp_forecast) -> list[str]:
    """Build alert/recommendation strings."""
    now = _fi_now()
    vat_multiplier = (1 + VAT_RATE) if _INCLUDE_VAT else 1.0
    alerts = []

    if not prices_today.empty:
        remaining = prices_today[
            prices_today["timestamp"].dt.tz_convert(_FI_TZ).dt.hour >= now.hour
        ]
        if not remaining.empty:
            cheapest = remaining.loc[remaining["price"].idxmin()]
            cheapest_hour = cheapest["timestamp"].astimezone(_FI_TZ).hour
            cheap_snt = cheapest["price"] / 10 * vat_multiplier

            threshold = 5 * vat_multiplier
            if cheap_snt < threshold and cheapest_hour == now.hour:
                alerts.append(f"[green]✅ {_t('cheap_hour_now')} ({cheapest_hour:02d}-{cheapest_hour+1:02d})[/green]")
            elif cheap_snt < threshold:
                alerts.append(f"[green]✅ {_t('cheap_hour_at')} {cheapest_hour:02d}:00 ({cheap_snt:.1f} snt)[/green]")

            expensive = remaining.nlargest(3, "price")
            exp_price = expensive["price"].mean() / 10 * vat_multiplier
            exp_hours = sorted(expensive["timestamp"].dt.tz_convert(_FI_TZ).dt.hour.tolist())
            threshold_high = 8 * vat_multiplier
            if exp_price > threshold_high and len(exp_hours) >= 2:
                alerts.append(f"[yellow]⚠️  {_t('expensive_coming')} ({exp_hours[0]:02d}:00)[/yellow]")

    if not prices_tomorrow.empty and not prices_today.empty:
        avg_tm = prices_tomorrow["price"].mean() / 10
        avg_td = prices_today["price"].mean() / 10
        if avg_td > 0:
            pct = (avg_tm - avg_td) / avg_td * 100
            if pct < -15:
                alerts.append(f"[green]📊 {_t('tomorrow_cheaper')} {pct:+.0f}% {_t('cheaper')}[/green]")
            elif pct > 20:
                alerts.append(f"[red]📊 {_t('tomorrow_cheaper')} {pct:+.0f}% {_t('more_expensive')}[/red]")

    if not alerts:
        alerts.append(f"[dim]{_t('no_alerts')}[/dim]")

    return alerts


def rich_dashboard():
    """Rich panel-based dashboard with side-by-side layout."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich import box

    console = Console()

    with console.status(f"[bold cyan]{_t('loading')}", spinner="dots"):
        data = fetch_all_data()
        # Extra data for rich dashboard
        hydro = _quiet_fingrid(191, *_rt_range())
        solar = _quiet_fingrid(248, *_rt_range())
        chp_dh = _quiet_fingrid(201, *_rt_range())
        chp_ind = _quiet_fingrid(202, *_rt_range())
        net_import = _quiet_fingrid(194, *_rt_range())
        wind_speed = _quiet_fmi_wind_speed("Helsinki")
        cloud_cover = _quiet_fmi_cloud("Helsinki")
        reservoir = _quiet_entsoe_reservoir()
        fuel_prices = _quiet_fuel_prices()

    fi_now = _fi_now()
    date_str = _date_str(fi_now)

    header = Panel(
        Text(f"{date_str}  {fi_now.strftime('%H:%M')}", justify="center", style="bold white"),
        title=f"[bold bright_white]⚡ {_t('header_rich')} ⚡[/bold bright_white]",
        box=box.HEAVY,
        style="bright_cyan",
        padding=(0, 2),
    )

    # ── Left panel: SPOT-HINTA ──
    vat_multiplier = (1 + VAT_RATE) if _INCLUDE_VAT else 1.0
    price_now, _ = _latest_value(data["prices_today"])
    price_1h = _hour_ago_value(data["prices_today"])
    snt_now = (price_now / 10 * vat_multiplier) if price_now is not None else None
    snt_1h = (price_1h / 10 * vat_multiplier) if price_1h is not None else None
    price_diff = (snt_now - snt_1h) if snt_now is not None and snt_1h is not None else None

    avg_today = (data["prices_today"]["price"].mean() / 10 * vat_multiplier) if not data["prices_today"].empty else None
    avg_yesterday = (data["prices_yesterday"]["price"].mean() / 10 * vat_multiplier) if not data["prices_yesterday"].empty else None
    today_pct = ((avg_today - avg_yesterday) / avg_yesterday * 100) if avg_today and avg_yesterday else None

    avg_tomorrow = (data["prices_tomorrow"]["price"].mean() / 10 * vat_multiplier) if not data["prices_tomorrow"].empty else None
    tomorrow_pct = ((avg_tomorrow - avg_today) / avg_today * 100) if avg_tomorrow and avg_today else None

    def _arrow_rich(val, ref):
        if val is None or ref is None:
            return ""
        d = val - ref
        if abs(d) < 0.05:
            return "→"
        return f"[green]▼[/green]" if d < 0 else f"[red]▲[/red]"

    def _change_str(diff, unit="", dec=1):
        if diff is None:
            return ""
        color = "green" if diff <= 0 else "red"
        sign = "+" if diff >= 0 else ""
        return f"[{color}]{sign}{diff:.{dec}f}{unit}[/{color}]"

    now_color = _price_color(snt_now) if snt_now is not None else "white"

    nl = _t("now_label")
    dl = _t("day_label")
    tl = _t("tomorrow_label")
    price_lines = []
    if snt_now is not None:
        price_lines.append(f"  {nl+':':<6}[{now_color}]{snt_now:>6.1f} snt/kWh[/{now_color}]  "
                           f"{_arrow_rich(snt_now, snt_1h)} {_change_str(price_diff)}")
    else:
        price_lines.append(f"  {nl+':':<6}   N/A")
    if avg_today is not None:
        td_str = f"{_change_str(today_pct, '%', 0)}" if today_pct is not None else ""
        price_lines.append(f"  {dl+':':<6}{avg_today:>6.1f} snt/kWh  {td_str}")
    if avg_tomorrow is not None:
        price_lines.append(f"  {tl+':':<6}{avg_tomorrow:>6.1f} snt/kWh  {_change_str(tomorrow_pct, '%', 0)}")
    else:
        price_lines.append(f"  {tl+':':<6} [dim]{_t('published_at')}[/dim]")

    # Sparkline for 24h trend
    if not data["prices_today"].empty:
        df_sp = data["prices_today"].copy()
        df_sp["snt"] = df_sp["price"] / 10 * vat_multiplier
        vals = df_sp["snt"].values
        mn, mx = vals.min(), vals.max()
        rng = mx - mn if mx > mn else 1
        blocks = " ▁▂▃▄▅▆▇█"
        spark = ""
        for v in vals:
            idx = int((v - mn) / rng * (len(blocks) - 1))
            spark += blocks[idx]
        price_lines.append("")
        price_lines.append(f"  {_t('trend_label')}: [cyan]{spark}[/cyan]")
        price_lines.append(f"          [dim]{mn:.0f}{'─' * (len(spark) - 4)}{mx:.0f} snt[/dim]")

    # Pad price lines to match production panel height
    while len(price_lines) < 7:
        price_lines.append("")

    vat_label = " (sis.ALV)" if _INCLUDE_VAT else ""
    price_panel = Panel(
        "\n".join(price_lines),
        title=f"[bold]{_t('spot_price_panel')}{vat_label}[/bold]",
        box=box.ROUNDED,
        width=40,
        padding=(0, 1),
    )

    # ── Right panel: TUOTANTO ──
    wind_now, _ = _latest_value(data["wind"])
    nuc_now, _ = _latest_value(data["nuclear"])
    hydro_now, _ = _latest_value(hydro)
    solar_now, _ = _latest_value(solar)
    chp_dh_now, _ = _latest_value(chp_dh)
    chp_ind_now, _ = _latest_value(chp_ind)
    cons_now, _ = _latest_value(data["consumption"])
    import_now, _ = _latest_value(net_import)

    # Calculate thermal (CHP) production
    thermal_now = sum(filter(None, [chp_dh_now, chp_ind_now]))
    thermal_now = thermal_now if thermal_now > 0 else None

    # Compute percentages of consumption
    total_domestic = sum(filter(None, [wind_now, nuc_now, hydro_now, solar_now, thermal_now]))
    import_val = max(0, import_now) if import_now is not None else 0
    total = total_domestic + import_val

    def _pct(val):
        if val is None or total == 0:
            return ""
        return f"[dim]{val / total * 100:3.0f}%[/dim]"

    prod_lines = []
    prod_lines.append(f"  [green]{_t('wind')}[/green]:{' '*(6-len(_t('wind')))}{_fmt_mw(wind_now):>7} MW {_pct(wind_now)}")
    prod_lines.append(f"  [yellow]{_t('nuclear_short')}[/yellow]:{' '*(6-len(_t('nuclear_short')))}{_fmt_mw(nuc_now):>7} MW {_pct(nuc_now)}")
    prod_lines.append(f"  [cyan]{_t('hydro')}[/cyan]:{' '*(6-len(_t('hydro')))}{_fmt_mw(hydro_now):>7} MW {_pct(hydro_now)}")
    prod_lines.append(f"  [bright_yellow]{_t('solar')}[/bright_yellow]:{' '*(6-len(_t('solar')))}{_fmt_mw(solar_now):>7} MW {_pct(solar_now)}")
    prod_lines.append(f"  [red]{_t('thermal')}[/red]:{' '*(6-len(_t('thermal')))}{_fmt_mw(thermal_now):>7} MW {_pct(thermal_now)}")
    if import_now is not None and import_now > 0:
        prod_lines.append(f"  [magenta]{_t('import')}[/magenta]:{' '*(6-len(_t('import')))}{_fmt_mw(import_now):>7} MW {_pct(import_now)}")
    elif import_now is not None:
        prod_lines.append(f"  [magenta]{_t('export')}[/magenta]:{' '*(6-len(_t('export')))}{_fmt_mw(abs(import_now)):>7} MW")
    prod_lines.append(f"  {'─' * 22}")
    prod_lines.append(f"  {_t('total')+':':<7}{_fmt_mw(cons_now):>7} MW")

    # Pad production lines to match price panel height
    while len(prod_lines) < 7:
        prod_lines.append("")

    prod_panel = Panel(
        "\n".join(prod_lines),
        title=f"[bold]{_t('production_panel')}[/bold]",
        box=box.ROUNDED,
        width=30,
        padding=(0, 1),
    )

    # ── Hourly price bar chart ──
    chart_text = _price_bar_chart(data["prices_today"])
    chart_panel = Panel(
        chart_text,
        title=f"[bold]{_t('hourly_chart')}[/bold]",
        box=box.ROUNDED,
        padding=(0, 1),
    )

    # ── Alerts panel ──
    alerts = _build_alerts(data["prices_today"], data["prices_tomorrow"], data.get("temp_forecast"))
    # Pad alerts to match weather height
    alert_lines = [f"  {a}" for a in alerts]
    while len(alert_lines) < 4:
        alert_lines.append("")

    alerts_panel = Panel(
        "\n".join(alert_lines),
        title=f"[bold]{_t('alerts_panel')}[/bold]",
        box=box.ROUNDED,
        width=40,
        padding=(0, 1),
    )

    # ── Weather panel ──
    temp = data["temp_now"]
    weather_lines = []
    if temp is not None:
        weather_lines.append(f"  🌡️  {temp:+.1f}°C (Helsinki)")
    else:
        weather_lines.append("  🌡️  N/A")

    if wind_speed is not None:
        ws_desc = ""
        if wind_speed < 3:
            ws_desc = _t("calm")
        elif wind_speed < 8:
            ws_desc = _t("breezy")
        else:
            ws_desc = _t("stormy")
        weather_lines.append(f"  💨  {wind_speed:.1f} m/s ({ws_desc})")

    if cloud_cover is not None:
        if cloud_cover <= 1:
            cc_desc = f"{_t('clear')} ☀️"
        elif cloud_cover <= 4:
            cc_desc = f"{_t('partly_cloudy')} ⛅"
        elif cloud_cover <= 6:
            cc_desc = f"{_t('mostly_cloudy')} 🌥️"
        else:
            cc_desc = f"{_t('cloudy')} ☁️"
        weather_lines.append(f"  {cc_desc}")

    # Tomorrow forecast
    if not data["temp_forecast"].empty:
        tomorrow_date = (fi_now + timedelta(days=1)).date()
        tf_tm = data["temp_forecast"][
            data["temp_forecast"]["timestamp"].dt.tz_convert(_FI_TZ).dt.date == tomorrow_date
        ]
        if not tf_tm.empty:
            avg_tmp = tf_tm["temperature"].mean()
            weather_lines.append(f"  {_t('tomorrow_short')}: {avg_tmp:+.1f}°C")

    weather_panel = Panel(
        "\n".join(weather_lines),
        title=f"[bold]{_t('weather_panel')}[/bold]",
        box=box.ROUNDED,
        width=30,
        padding=(0, 1),
    )

    # ── Reservoir panel ──
    res_lines = []
    if reservoir is not None and not reservoir.empty:
        latest_res = reservoir.iloc[-1]
        for flag, name_key, col in [("🇫🇮", "finland", "fi_pct"),
                                     ("🇸🇪", "sweden", "se_pct"),
                                     ("🇳🇴", "norway", "no_pct")]:
            val = latest_res.get(col)
            if val is not None and not pd.isna(val):
                change = _week_change(reservoir, col)
                ch_str = ""
                if change is not None:
                    ch_color = "green" if change >= 0 else "red"
                    ch_arrow = "▲" if change > 0 else "▼" if change < 0 else "→"
                    ch_str = f"[{ch_color}]{ch_arrow} {change:+.1f}%[/{ch_color}]"
                res_lines.append(f"  {flag} {val:5.1f}%  {ch_str}")
            else:
                res_lines.append(f"  {flag}   N/A")
        avg_res = latest_res.get("nordic_avg")
        if avg_res is not None and not pd.isna(avg_res):
            status = _t("normal")
            if avg_res > 75:
                status = _t("above_normal")
            elif avg_res < 65:
                status = _t("below_normal")
            res_lines.append(f"  [dim]{_t('nordic_avg')[:8]}: {avg_res:.1f}% ({status})[/dim]")
    elif not ENTSOE_API_KEY:
        res_lines.append(f"  [dim]{_t('no_entsoe_key')}[/dim]")
    else:
        res_lines.append("  [dim]N/A[/dim]")

    reservoir_panel = Panel(
        "\n".join(res_lines),
        title=f"[bold]{_t('reservoir_panel')}[/bold]",
        box=box.ROUNDED,
        width=30,
        padding=(0, 1),
    )

    # ── Fuel panel ──
    fuel_lines = []
    if fuel_prices:
        gas = fuel_prices.get("gas")
        coal = fuel_prices.get("coal")
        carbon = fuel_prices.get("carbon")

        if gas:
            fuel_lines.append(f"  ⛽ TTF:  {gas['price']:.2f} €/MWh")
        if coal:
            fuel_lines.append(f"  🪨 Hiili: {coal['price']:.2f} $/t")
        if carbon:
            fuel_lines.append(f"  🏭 CO2:  {carbon['price']:.2f} €/tCO2")

        if gas and carbon:
            from fetch_fuel_prices import compute_gas_srmc
            srmc = compute_gas_srmc(gas["price"], carbon["price"])
            fuel_lines.append(f"  SRMC: [bold]~{srmc:.1f} €/MWh[/bold]")
    elif not OILPRICE_API_KEY:
        fuel_lines.append(f"  [dim]{_t('no_fuel_key')}[/dim]")
    else:
        fuel_lines.append("  [dim]N/A[/dim]")

    fuel_panel = Panel(
        "\n".join(fuel_lines),
        title=f"[bold]{_t('fuel_panel')}[/bold]",
        box=box.ROUNDED,
        width=28,
        padding=(0, 1),
    )

    # ── Render layout ──
    console.print()
    console.print(header)
    console.print()

    # Top row: price + production side by side
    top_table = Table.grid(padding=(0, 1))
    top_table.add_row(price_panel, prod_panel)
    console.print(top_table)

    # Chart
    console.print(chart_panel)

    # Bottom row: alerts + weather + reservoir + fuel
    bottom_table = Table.grid(padding=(0, 1))
    bottom_table.add_row(alerts_panel, weather_panel, reservoir_panel, fuel_panel)
    console.print(bottom_table)

    # Footer
    console.print()
    console.print(
        f"[dim]  {_t('footer_keys')}[/dim]",
        highlight=False,
    )
    console.print()


def _rt_range():
    """Return (start, end) ISO strings for the last 2 hours."""
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return start, end


def _recommendation_label(snt: float, avg: float) -> tuple[str, str]:
    """Return (label, color) recommendation based on price vs average."""
    if snt <= avg * 0.5:
        return _t("rec_excellent"), "bold green"
    if snt <= avg * 0.75:
        return _t("rec_good"), "green"
    if snt <= avg * 1.25:
        return _t("rec_ok"), "yellow"
    return _t("rec_expensive"), "red"


def table_view(hours: int = 24):
    """Rich table showing hourly price, wind, consumption and recommendations."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    now_utc = datetime.now(timezone.utc)
    fi_now = now_utc.astimezone(_FI_TZ)

    # Current hour start (aligned)
    current_hour_utc = now_utc.replace(minute=0, second=0, microsecond=0)

    # We need prices from current hour to +hours
    # Prices come from sahkotin: today + tomorrow
    fi_today_start = fi_now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Fetch today + tomorrow + day after (to cover full range)
    fetch_start = fi_today_start - timedelta(days=1)
    fetch_end = fi_today_start + timedelta(days=3)

    def ts(dt):
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    vat_multiplier = (1 + VAT_RATE) if _INCLUDE_VAT else 1.0

    with console.status(f"[bold cyan]{_t('table_loading')}", spinner="dots"):
        prices = _quiet_spot(ts(fetch_start), ts(fetch_end))

        # Fingrid data: historical (last few hours) + forecasts (next day)
        hist_start = (now_utc - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hist_end = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        fc_start = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        fc_end = (now_utc + timedelta(hours=hours + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Historical wind & consumption
        wind_hist = _quiet_fingrid(181, hist_start, hist_end)
        cons_hist = _quiet_fingrid(124, hist_start, hist_end)
        time.sleep(0.5)

        # Forecasted wind & consumption
        wind_fc = _quiet_fingrid(245, fc_start, fc_end)
        cons_fc = _quiet_fingrid(166, fc_start, fc_end)

        # Balancing power prices (historical + current)
        bal_up = _quiet_fingrid(BALANCING_UP_REGULATION_PRICE, hist_start, fc_end)
        bal_down = _quiet_fingrid(BALANCING_DOWN_REGULATION_PRICE, hist_start, fc_end)

    if prices.empty:
        console.print(f"[red]{_t('table_no_prices')}[/red]")
        return

    # Merge historical + forecast for wind and consumption
    def _merge_series(hist_df, fc_df):
        """Merge actual + forecast, preferring actuals."""
        frames = []
        if not hist_df.empty:
            frames.append(hist_df)
        if not fc_df.empty:
            frames.append(fc_df)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "value"])
        merged = pd.concat(frames).sort_values("timestamp")
        merged["hour_key"] = merged["timestamp"].dt.floor("h")
        # Keep last value per hour (actuals added first, so forecast fills gaps)
        merged = merged.drop_duplicates(subset="hour_key", keep="first")
        return merged

    wind_all = _merge_series(wind_hist, wind_fc)
    cons_all = _merge_series(cons_hist, cons_fc)

    # Build hour-keyed lookup
    def _hour_lookup(df, val_col="value"):
        if df.empty:
            return {}
        d = df.copy()
        d["hk"] = d["timestamp"].dt.floor("h")
        return dict(zip(d["hk"], d[val_col]))

    wind_lookup = _hour_lookup(wind_all)
    cons_lookup = _hour_lookup(cons_all)
    bal_up_lookup = _hour_lookup(bal_up)
    bal_down_lookup = _hour_lookup(bal_down)

    # Filter prices to the requested window
    prices["hour_utc"] = prices["timestamp"].dt.floor("h")
    prices = prices.drop_duplicates(subset="hour_utc", keep="last")
    prices = prices[
        (prices["hour_utc"] >= current_hour_utc) &
        (prices["hour_utc"] < current_hour_utc + timedelta(hours=hours))
    ].sort_values("hour_utc").reset_index(drop=True)

    if prices.empty:
        console.print(f"[red]{_t('table_no_range')}[/red]")
        return

    avg_price = (prices["price"].mean() / 10) * vat_multiplier
    vat_label = " (sis.ALV)" if _INCLUDE_VAT else ""

    table = Table(
        title=f"{_t('table_title')} {hours}H{vat_label}",
        box=box.SIMPLE_HEAVY,
        title_style="bold bright_white",
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column(_t("col_hour"), style="bold", min_width=11)
    table.add_column(_t("col_price"), justify="right", min_width=9)
    table.add_column(_t("col_change"), justify="right", min_width=8)
    table.add_column(_t("col_wind"), justify="right", min_width=6)
    table.add_column(_t("col_consumption"), justify="right", min_width=7)
    table.add_column(_t("col_bal_up"), justify="right", min_width=7)
    table.add_column(_t("col_bal_down"), justify="right", min_width=7)
    table.add_column(_t("col_recommendation"), justify="center", min_width=12)

    prev_price = None
    for _, row in prices.iterrows():
        hour_fi = row["timestamp"].astimezone(_FI_TZ)
        snt = (row["price"] / 10) * vat_multiplier
        color = _price_color(snt)

        # Hour label: "1.2 15:00"
        hour_label = f"{hour_fi.day}.{hour_fi.month} {hour_fi.strftime('%H:%M')}"
        is_now = (row["hour_utc"] == current_hour_utc)
        if is_now:
            hour_label += " ◀"

        # Price
        price_str = f"[{color}]{snt:5.1f} snt[/{color}]"

        # Change vs previous hour
        if prev_price is not None and prev_price != 0:
            pct = (snt - prev_price) / abs(prev_price) * 100
            arr = "▲" if pct > 0 else "▼" if pct < 0 else "→"
            ch_color = "red" if pct > 0 else "green" if pct < 0 else "dim"
            change_str = f"[{ch_color}]{arr} {pct:+.0f}%[/{ch_color}]"
        else:
            change_str = "[dim]–[/dim]"
        prev_price = snt

        # Wind
        wind_val = wind_lookup.get(row["hour_utc"])
        wind_str = f"{wind_val:,.0f}" if wind_val is not None else "[dim]–[/dim]"

        # Consumption
        cons_val = cons_lookup.get(row["hour_utc"])
        cons_str = f"{cons_val:,.0f}" if cons_val is not None else "[dim]–[/dim]"

        # Balancing power prices
        bal_up_val = bal_up_lookup.get(row["hour_utc"])
        bal_down_val = bal_down_lookup.get(row["hour_utc"])

        # Color code balancing prices relative to spot price
        if bal_up_val is not None:
            bal_up_snt = (bal_up_val / 10) * vat_multiplier
            base_snt = snt
            if bal_up_snt > base_snt * 1.5:
                bal_up_str = f"[red]{bal_up_snt:.1f}[/red]"
            elif bal_up_snt > base_snt * 1.2:
                bal_up_str = f"[yellow]{bal_up_snt:.1f}[/yellow]"
            else:
                bal_up_str = f"{bal_up_snt:.1f}"
        else:
            bal_up_str = "[dim]–[/dim]"

        if bal_down_val is not None:
            bal_down_snt = (bal_down_val / 10) * vat_multiplier
            base_snt = snt
            if bal_down_snt < base_snt * 0.5:
                bal_down_str = f"[green]{bal_down_snt:.1f}[/green]"
            elif bal_down_snt < base_snt * 0.8:
                bal_down_str = f"[cyan]{bal_down_snt:.1f}[/cyan]"
            else:
                bal_down_str = f"{bal_down_snt:.1f}"
        else:
            bal_down_str = "[dim]–[/dim]"

        label, label_color = _recommendation_label(snt, avg_price)
        if label == _t("rec_excellent"):
            rec_str = f"[{label_color}]✅✅ {label}[/{label_color}]"
        elif label == _t("rec_good"):
            rec_str = f"[{label_color}]✅ {label}[/{label_color}]"
        elif label == _t("rec_ok"):
            rec_str = f"[{label_color}]⚠️  {label}[/{label_color}]"
        else:
            rec_str = f"[{label_color}]❌ {label}[/{label_color}]"

        table.add_row(hour_label, price_str, change_str, wind_str, cons_str,
                     bal_up_str, bal_down_str, rec_str)

    console.print()
    console.print(table)

    # Summary line
    min_row = prices.loc[prices["price"].idxmin()]
    max_row = prices.loc[prices["price"].idxmax()]
    min_hour = min_row["timestamp"].astimezone(_FI_TZ).strftime("%H:%M")
    max_hour = max_row["timestamp"].astimezone(_FI_TZ).strftime("%H:%M")
    console.print(
        f"  {_t('summary_avg')}: [bold]{avg_price:.1f} snt/kWh[/bold]  │  "
        f"Min: [green]{(min_row['price']/10)*vat_multiplier:.1f}[/green] @ {min_hour}  │  "
        f"Max: [red]{(max_row['price']/10)*vat_multiplier:.1f}[/red] @ {max_hour}"
    )
    console.print()


def interactive_dashboard():
    """Run the rich dashboard with keyboard interaction."""
    import msvcrt  # Windows-specific
    from rich.console import Console

    console = Console()

    while True:
        console.clear()
        rich_dashboard()

        # Wait for keypress
        console.print(f"[dim]  {_t('waiting_key')}[/dim]", highlight=False)
        try:
            if sys.platform == "win32":
                key = msvcrt.getwch()
            else:
                import tty, termios
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (ImportError, Exception):
            # Fallback: just show once
            return

        if key in ("q", "Q", "\x1b"):  # q or Escape
            console.clear()
            console.print(f"[dim]{_t('goodbye')}[/dim]")
            break
        elif key in ("r", "R"):
            continue  # Refresh
        elif key in ("t", "T"):
            console.clear()
            dashboard("today")
            input(f"\n  {_t('press_enter')}")
        elif key in ("w", "W"):
            console.clear()
            dashboard("week")
            input(f"\n  {_t('press_enter')}")
        elif key in ("h", "H"):
            console.clear()
            dashboard("tomorrow")
            input(f"\n  {_t('press_enter')}")


if __name__ == "__main__":
    # Start with all arguments
    _argv = sys.argv[1:]

    # Parse --vat flag first
    if "--vat" in _argv:
        _INCLUDE_VAT = True
        _argv = [a for a in _argv if a != "--vat"]

    # Parse --lang flag (remove from argv so it doesn't interfere with mode)
    _lang_val = None
    for i, arg in enumerate(_argv):
        if arg == "--lang" and i + 1 < len(_argv):
            _lang_val = _argv[i + 1]
            _argv = [a for j, a in enumerate(_argv) if j not in (i, i + 1)]
            break

    # Also support shorthand: sahko --en ...
    if "--en" in _argv:
        _LANG = "en"
        _argv = [a for a in _argv if a != "--en"]
    elif _lang_val:
        _LANG = _lang_val

    mode = _argv[0] if _argv else "today"
    if mode == "dash":
        try:
            rich_dashboard()
        except KeyboardInterrupt:
            pass
    elif mode == "live":
        try:
            interactive_dashboard()
        except KeyboardInterrupt:
            pass
    elif mode == "table":
        hours = 24
        for i, arg in enumerate(_argv[1:], start=1):
            if arg == "--hours" and i + 1 < len(_argv):
                try:
                    hours = int(_argv[i + 1])
                except ValueError:
                    pass
        try:
            table_view(hours=hours)
        except KeyboardInterrupt:
            pass
    elif mode == "reservoir":
        try:
            reservoir_view()
        except KeyboardInterrupt:
            pass
    elif mode == "fuel":
        try:
            fuel_view()
        except KeyboardInterrupt:
            pass
    elif mode in ("today", "now", "tomorrow", "week"):
        dashboard(mode)
    else:
        print(f"Tuntematon komento: {mode}")
        print("Usage: sahko [today|now|tomorrow|week|dash|live|table|reservoir|fuel] [--lang en] [--en] [--vat]")
        sys.exit(1)
