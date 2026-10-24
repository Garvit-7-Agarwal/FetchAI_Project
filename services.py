"""
🇮🇳 India Intel Agent — Real-Time Services & External APIs Engine
Provides live, timeout-protected, fault-tolerant integrations:
1. Real-Time Forex Exchange Rates (USD, EUR, GBP, AED, JPY, CAD, AUD to INR)
2. Real-Time Meteorological & Weather Advisories (Open-Meteo WMO API for Indian Cities)
3. Smart Dynamic Deep-Link Generators (IRCTC, Google Flights, Google Maps, Google Calendar)
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Real-Time Forex & Currency Conversion Engine
# ---------------------------------------------------------------------------

# Fallback exchange rates (updated benchmark values against 1 USD)
FALLBACK_RATES_BASE_USD: Dict[str, float] = {
    "USD": 1.0,
    "INR": 86.85,
    "EUR": 0.92,
    "GBP": 0.77,
    "AED": 3.67,
    "JPY": 152.40,
    "CAD": 1.38,
    "AUD": 1.52,
    "SGD": 1.32,
    "CHF": 0.88,
}

_FOREX_CACHE: Dict[str, Any] = {
    "timestamp": 0,
    "rates": FALLBACK_RATES_BASE_USD.copy(),
}


def fetch_live_forex_rates() -> Dict[str, float]:
    """
    Fetches real-time exchange rates against USD with a strict 2.5s timeout.
    Falls back gracefully to reliable benchmark rates if offline.
    """
    now = datetime.now().timestamp()
    # Cache for 1 hour (3600 seconds) to avoid redundant network overhead
    if now - _FOREX_CACHE["timestamp"] < 3600 and _FOREX_CACHE["rates"]:
        return _FOREX_CACHE["rates"]

    try:
        url = "https://open.er-api.com/v6/latest/USD"
        req = urllib.request.Request(url, headers={"User-Agent": "IndiaIntelAgent/2.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rates = data.get("rates", {})
            if rates and "INR" in rates:
                _FOREX_CACHE["rates"] = rates
                _FOREX_CACHE["timestamp"] = now
                return rates
    except Exception as e:
        print(f"⚠️ [Forex Service] Network call failed: {e}. Using resilient fallback rates.")

    return _FOREX_CACHE["rates"] or FALLBACK_RATES_BASE_USD


def convert_currency_to_inr(amount: float, from_curr: str) -> Dict[str, Any]:
    """
    Converts any supported foreign currency into Indian Rupees (INR)
    and provides real-world purchasing power equivalents.
    """
    rates = fetch_live_forex_rates()
    curr_clean = from_curr.strip().upper()
    usd_per_unit = 1.0 / rates.get(curr_clean, 1.0) if curr_clean != "USD" else 1.0
    inr_rate = rates.get("INR", 86.85)

    amount_in_usd = amount * usd_per_unit
    amount_in_inr = round(amount_in_usd * inr_rate, 2)
    rate_to_inr = round(usd_per_unit * inr_rate, 2)

    # Real-world Indian purchasing power benchmarks
    purchasing_power: List[str] = []
    if amount_in_inr < 1000:
        purchasing_power.append("🍛 2-3 hearty street food meals or authentic Thalis")
        purchasing_power.append("🛺 15-20 km of Auto-Rickshaw or Metro transit")
    elif amount_in_inr < 5000:
        purchasing_power.append("🏨 1 night stay at a comfortable 2-3★ boutique hotel or heritage hostel")
        purchasing_power.append("🚆 1-2 AC Chair Car / 3AC Vande Bharat Express inter-city tickets")
        purchasing_power.append("🎟️ Entry tickets + audio guides for 3-4 UNESCO Heritage monuments")
    elif amount_in_inr < 15000:
        purchasing_power.append("🏰 2-3 nights at a 3-4★ heritage hotel or luxury homestay")
        purchasing_power.append("🚕 2 full days of dedicated chauffeur-driven private cab")
        purchasing_power.append("🍽️ Fine dining at iconic culinary destinations (e.g., Peshawri, Bukhara, Karavalli)")
    else:
        purchasing_power.append("👑 1-2 nights at a world-class 5★ Palace Hotel (Taj Rambagh / Oberoi Udaivilas)")
        purchasing_power.append("✈️ Multiple domestic express flight connections across India")
        purchasing_power.append("🛍️ Premium authentic handicraft shopping (Pashmina, Blue Pottery, Silk Sarees)")

    return {
        "amount": amount,
        "from_currency": curr_clean,
        "amount_inr": amount_in_inr,
        "exchange_rate": rate_to_inr,
        "purchasing_power": purchasing_power,
    }


# ---------------------------------------------------------------------------
# 2. Real-Time Meteorological & Weather Advisories Engine
# ---------------------------------------------------------------------------

DESTINATION_COORDINATES: Dict[str, Tuple[float, float, str]] = {
    "jaipur": (26.9124, 75.7873, "Rajasthan"),
    "delhi": (28.6139, 77.2090, "Delhi"),
    "new delhi": (28.6139, 77.2090, "Delhi"),
    "agra": (27.1767, 78.0081, "Uttar Pradesh"),
    "mumbai": (19.0760, 72.8777, "Maharashtra"),
    "srinagar": (34.0837, 74.7973, "Jammu & Kashmir"),
    "varanasi": (25.3176, 82.9739, "Uttar Pradesh"),
    "udaipur": (24.5854, 73.7125, "Rajasthan"),
    "goa": (15.2993, 74.1240, "Goa"),
    "kochi": (9.9312, 76.2673, "Kerala"),
    "amritsar": (31.6340, 74.8723, "Punjab"),
    "rishikesh": (30.0869, 78.2676, "Uttarakhand"),
    "darjeeling": (27.0410, 88.2663, "West Bengal"),
    "manali": (32.2432, 77.1892, "Himachal Pradesh"),
    "shimla": (31.1048, 77.1734, "Himachal Pradesh"),
    "leh": (34.1526, 77.5771, "Ladakh"),
    "bengaluru": (12.9716, 77.5946, "Karnataka"),
    "hyderabad": (17.3850, 78.4867, "Telangana"),
    "chennai": (13.0827, 80.2707, "Tamil Nadu"),
    "kolkata": (22.5726, 88.3639, "West Bengal"),
}

WMO_WEATHER_CODES: Dict[int, Tuple[str, str]] = {
    0: ("Clear Sky", "☀️"),
    1: ("Mainly Clear", "🌤️"),
    2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Foggy", "🌫️"),
    48: ("Depositing Rime Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"),
    53: ("Moderate Drizzle", "🌦️"),
    55: ("Dense Drizzle", "🌧️"),
    61: ("Slight Rain", "🌧️"),
    63: ("Moderate Rain", "🌧️"),
    65: ("Heavy Rain", "⛈️"),
    71: ("Slight Snow", "🌨️"),
    73: ("Moderate Snow", "❄️"),
    75: ("Heavy Snow", "❄️"),
    80: ("Rain Showers", "🌦️"),
    85: ("Snow Showers", "🌨️"),
    95: ("Thunderstorm", "⛈️"),
}


def fetch_live_weather(city_name: str) -> Dict[str, Any]:
    """
    Fetches real-time weather conditions from Open-Meteo for any Indian city.
    Uses strict 2.5s timeout with offline climatology fallback.
    """
    clean_city = city_name.strip().lower()
    coords = DESTINATION_COORDINATES.get(clean_city)
    if not coords:
        for known_city, c_tuple in DESTINATION_COORDINATES.items():
            if known_city in clean_city or clean_city in known_city:
                coords = c_tuple
                break

    if not coords:
        # Default to Delhi coordinates if unknown
        coords = (28.6139, 77.2090, "India")

    lat, lon, state = coords

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "IndiaIntelAgent/2.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            curr = data.get("current", {})
            temp = curr.get("temperature_2m", 26.0)
            humidity = curr.get("relative_humidity_2m", 45)
            wcode = curr.get("weather_code", 0)
            wind = curr.get("wind_speed_10m", 8.0)
            desc, icon = WMO_WEATHER_CODES.get(wcode, ("Pleasant", "☀️"))

            # Tailored seasonal advisory
            advisory = "Ideal weather for outdoor sightseeing and heritage exploration."
            if temp > 36:
                advisory = "⚠️ Warm weather: Carry hydration, sunglasses, and schedule fort walks for mornings/evenings."
            elif temp < 10:
                advisory = "🧥 Chilly weather: Warm woolens or thermal wear recommended, especially for evening walks."
            elif "Rain" in desc or "Drizzle" in desc or "Thunderstorm" in desc:
                advisory = "☔ Wet conditions: Keep an umbrella handy and verify hill road transit advisories."

            return {
                "city": city_name.title(),
                "state": state,
                "temperature_c": round(temp, 1),
                "condition": desc,
                "icon": icon,
                "humidity_pct": humidity,
                "wind_kmh": round(wind, 1),
                "advisory": advisory,
                "is_live": True,
            }
    except Exception as e:
        print(f"⚠️ [Weather Service] Live weather fetch failed for {city_name}: {e}. Using seasonal climatology.")

    # Resilient fallback climatology
    return {
        "city": city_name.title(),
        "state": state,
        "temperature_c": 26.5,
        "condition": "Clear Sky",
        "icon": "☀️",
        "humidity_pct": 42,
        "wind_kmh": 9.5,
        "advisory": "Comfortable seasonal conditions: Ideal for monument visits and photography.",
        "is_live": False,
    }


# ---------------------------------------------------------------------------
# 3. Dynamic Deep-Link Booking & Navigation Engine
# ---------------------------------------------------------------------------

STATION_CODES: Dict[str, str] = {
    "delhi": "NDLS",
    "new delhi": "NDLS",
    "jaipur": "JP",
    "agra": "AGC",
    "mumbai": "MMCT",
    "varanasi": "BSB",
    "udaipur": "UDZ",
    "goa": "MAO",
    "kochi": "ERS",
    "amritsar": "ASR",
    "rishikesh": "RKSH",
    "haridwar": "HW",
    "darjeeling": "NJP",
    "srinagar": "JAT",  # Jammu Tawi connects to Srinagar Valley / Vande Bharat
    "manali": "CDG",    # Chandigarh railhead for Manali
    "shimla": "SML",
    "bengaluru": "SBC",
    "hyderabad": "SC",
    "chennai": "MAS",
    "kolkata": "HWH",
}


def generate_irctc_train_search_url(destination: str, origin: str = "Delhi") -> str:
    """Generates direct Indian Railways (IRCTC) verified booking search deep-link."""
    dest_clean = destination.strip().lower()
    orig_clean = origin.strip().lower()

    dest_code = STATION_CODES.get(dest_clean, "JP")
    orig_code = STATION_CODES.get(orig_clean, "NDLS")

    # If origin and destination map to same code, default origin to New Delhi
    if dest_code == orig_code:
        orig_code = "NDLS" if dest_code != "NDLS" else "MMCT"

    date_str = (datetime.now()).strftime("%Y%m%d")
    return f"https://www.irctc.co.in/nget/train-search?src={orig_code}&dst={dest_code}&date={date_str}"


def generate_google_flights_url(destination: str, origin: str = "Delhi") -> str:
    """Generates direct Google Flights live pricing search URL."""
    dest_encoded = urllib.parse.quote(f"{destination}, India")
    orig_encoded = urllib.parse.quote(f"{origin}, India")
    return f"https://www.google.com/travel/flights?q=flights+from+{orig_encoded}+to+{dest_encoded}"


def generate_booking_com_url(destination: str) -> str:
    """Generates verified Booking.com live hotel rates search URL."""
    dest_encoded = urllib.parse.quote(f"{destination}, India")
    return f"https://www.booking.com/searchresults.html?ss={dest_encoded}"


def generate_google_calendar_url(title: str, details: str, location: str, days: int = 3) -> str:
    """Generates 1-click Google Calendar itinerary synchronization URL."""
    start_date = datetime.now()
    end_date = start_date + datetime.timedelta(days=max(1, days))
    start_str = start_date.strftime("%Y%m%dT090000Z")
    end_str = end_date.strftime("%Y%m%dT180000Z")

    params = {
        "action": "TEMPLATE",
        "text": title,
        "details": details,
        "location": location,
        "dates": f"{start_str}/{end_str}",
    }
    return f"https://calendar.google.com/calendar/render?{urllib.parse.urlencode(params)}"


def generate_google_maps_route(places: List[str]) -> str:
    """Generates verified Google Maps multi-point navigation route."""
    if not places:
        return "https://www.google.com/maps"
    encoded_places = [urllib.parse.quote(p) for p in places]
    return f"https://www.google.com/maps/dir/{'/'.join(encoded_places)}"
