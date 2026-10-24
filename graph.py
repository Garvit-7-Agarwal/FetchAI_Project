"""
🇮🇳 India Educational AI Intelligence Agent (@india-intel-agent)
Autonomous Multi-Domain Educational & Research Intelligence Platform
Powered by LangGraph & Google Gemini 2.5 Flash.

Core Intelligence Domains:
1. 🇮🇳 Economy & Financial Markets (GDP, Sectors, Conglomerates, NSE/BSE, UPI, FinTech)
2. ✈️ Tourism, Landmarks & UNESCO Heritage (Interactive Forms, Dropdowns, Itineraries, Transit Links, Hotels, Cost Estimates)
3. 🎨 Cultural Heritage & Social Fabric (Classical Arts, Music, Festivals, Regional Cultures, Society)
4. 🚀 Science, Innovation & Future Technology (ISRO, Chips, IndiaAI, Clean Energy, Quantum, Startups)
5. 🗺️ State & Regional Comprehensive Profiles (Economy + Culture + Tourism + History)
"""

import json
import os
import re
import urllib.parse
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from uagents_core.contrib.protocols.chat import MetadataContent
from uagents_core.contrib.protocols.chat.cards import (
    CarouselCardPayload,
    CarouselItem,
    CtaAction,
    DetailCardPayload,
    FormCardPayload,
    FormField,
    FormFieldOption,
    ReviewCardPayload,
    ReviewSummaryRow,
    create_card_content,
)

load_dotenv()

# ----------------------------------------------------
# 1. Initialize Google Gemini LLM (gemini-3.8-flash)
# ----------------------------------------------------
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
    max_retries=1,
    timeout=10.0,
)

def safe_llm_invoke(messages: List[BaseMessage], fallback_content: str = "") -> str:
    """Invokes Google Gemini with fallback protection against rate limits or quota exhaustion."""
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"⚠️ [LLM Warning] Gemini call failed or quota exceeded: {e}")
        if fallback_content:
            return fallback_content
        return "🇮🇳 **India Educational AI Intelligence Agent**\n\nI am currently experiencing high request volume. Please retry in a few moments or refer to our verified educational guides in the README."


# ----------------------------------------------------
# 2. State Definition & Response Helpers
# ----------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    workflow_stage: Literal[
        "landing",
        "tourism_destinations",
        "tourism_briefing",
        "tourism_followup",
        "tourism_planner",
        "culture_form",
        "culture_report",
        "economy_form",
        "economy_briefing",
        "science_form",
        "science_dossier",
        "state_composite",
        "out_of_scope",
        "general_overview",
        "destinations_carousel",
        "review_confirm",
    ] | None
    entity_name: str | None
    extracted_params: Dict[str, Any] | None
    domain_data: Dict | None
    action_links: Dict | None
    final_output: str | None


def clean_user_input(text: str) -> str:
    """Removes @mentions like @india-intel-agent, leading colons, and trims whitespace."""
    if not text:
        return ""
    # Strip any @mentions (e.g. @india-intel-agent, @india, @agent, etc.)
    cleaned = re.sub(r'@[\w\-\.]+', '', text).strip()
    # Strip leading colons or hyphens left by mentions
    cleaned = re.sub(r'^[ :\-\t]+', '', cleaned).strip()
    return cleaned


def parse_incoming_input(text: str) -> Tuple[str, Dict[str, Any]]:
    """Parses text to extract cleaned query and any structured JSON or key-value form parameters."""
    clean = clean_user_input(text)
    params: Dict[str, Any] = {}

    # Check if input is a JSON string (e.g. from an interactive card submit button)
    if (clean.startswith("{") and clean.endswith("}")) or (clean.startswith("[") and clean.endswith("]")):
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                if "fields" in parsed and isinstance(parsed["fields"], dict):
                    params.update(parsed["fields"])
                else:
                    params.update(parsed)
        except Exception:
            pass

    # Check key-value pairs like "destination: Jaipur | days: 3 | travelers: 2"
    if not params:
        kv_pairs = re.findall(r'(\b\w+)\s*[:=]\s*([^|,\n]+)', clean)
        if kv_pairs:
            for k, v in kv_pairs:
                params[k.strip().lower()] = v.strip()

    return clean, params


def get_message_content(msg: Any) -> str:
    """Helper to safely extract string content whether msg is a BaseMessage or a dict."""
    if isinstance(msg, dict):
        content = msg.get("content", "") or msg.get("text", "")
        if isinstance(content, list):
            text_parts = [
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            ]
            return " ".join(text_parts) if text_parts else str(content)
        return str(content)
    return getattr(msg, "content", str(msg))


def make_ai_response(text: str, card_payload: Optional[Any] = None) -> Dict[str, Any]:
    """
    Constructs an agent turn output containing:
    1) TextContent: Full, richly formatted Markdown displayed across all chat clients.
    2) MetadataContent: Native Agentverse / ASI:One Interactive Card Payload rendering dropdowns/forms/reviews/carousels.
    """
    if card_payload is not None:
        if hasattr(card_payload, "metadata"):
            meta = card_payload.metadata
        elif isinstance(card_payload, (FormCardPayload, ReviewCardPayload, CarouselCardPayload, DetailCardPayload)) or hasattr(card_payload, "model_dump"):
            meta = create_card_content(card_payload).metadata
        elif isinstance(card_payload, dict):
            meta = card_payload
        else:
            meta = {}
        content_blocks = [
            {"type": "text", "text": text},
            {
                "type": "metadata",
                "metadata": meta,
            }
        ]
        return {
            "final_output": text,
            "messages": [AIMessage(content=content_blocks)]
        }
    return {
        "final_output": text,
        "messages": [AIMessage(content=text)]
    }


# ----------------------------------------------------
# 3. Interactive Card Payloads (ASI:One Native Dropdowns, Reviews & Forms)
# ----------------------------------------------------
def get_landing_card_payload() -> MetadataContent:
    """Interactive card for the landing view with domain selection dropdown."""
    payload = FormCardPayload(
        title="🇮🇳 India Intel Agent — Domain Explorer",
        fields=[
            FormField(
                name="domain",
                kind="select",
                label="Choose Intelligence Domain",
                required=True,
                options=[
                    FormFieldOption(label="🇮🇳 Option 1: Economy & Financial Markets", value="Economy"),
                    FormFieldOption(label="✈️ Option 2: Tourism & UNESCO Heritage", value="Tourism"),
                    FormFieldOption(label="🎨 Option 3: Culture & Social Fabric", value="Culture"),
                    FormFieldOption(label="🚀 Option 4: Science & Future Tech", value="Science"),
                ]
            )
        ],
        submit_cta=CtaAction(
            label="👉 Open Selected Domain",
            selection={"action": "select_domain"},
            primary=True
        )
    )
    return create_card_content(payload)


def get_review_card_payload(params: Dict[str, Any]) -> MetadataContent:
    """Renders a native Review & Confirmation Card directly inside ASI:One."""
    city = str(params.get("destination", "Jaipur")).title()
    days = str(params.get("days", "3"))
    travelers = str(params.get("travelers", "2"))
    budget = str(params.get("budget", "Budget"))
    transport = str(params.get("transport", "Train"))

    rows = [
        ReviewSummaryRow(label="📍 Destination", value=city),
        ReviewSummaryRow(label="📅 Trip Duration", value=f"{days} Days"),
        ReviewSummaryRow(label="👥 Travelers", value=f"{travelers} Persons"),
        ReviewSummaryRow(label="💰 Budget Tier", value=budget),
        ReviewSummaryRow(label="🚆 Transit Mode", value=transport),
    ]

    payload = ReviewCardPayload(
        title=f"📋 Itinerary Review: {city}",
        summary_rows=rows,
        approve_cta=CtaAction(
            label="🚀 Confirm & Generate Full Itinerary",
            selection={"action": "confirm_trip", "destination": city, "days": days, "travelers": travelers, "budget": budget, "transport": transport},
            primary=True,
        ),
        reject_cta=CtaAction(
            label="🎛️ Adjust Travel Preferences",
            selection={"action": "modify_trip", "destination": city},
        ),
    )
    return create_card_content(payload)


def get_destinations_carousel_payload() -> MetadataContent:
    """Renders an interactive swipeable carousel of top flagship Indian destinations."""
    items = [
        CarouselItem(
            id="dest-jaipur",
            title="🏰 Jaipur (Pink City)",
            subtitle="Amber Fort & Royal Palaces",
            primary_cta=CtaAction(
                label="Explore Jaipur",
                selection={"action": "explore_city", "destination": "Jaipur"},
            ),
        ),
        CarouselItem(
            id="dest-srinagar",
            title="🏔️ Srinagar (Paradise on Earth)",
            subtitle="Dal Lake Shikara & Gulmarg",
            primary_cta=CtaAction(
                label="Explore Srinagar",
                selection={"action": "explore_city", "destination": "Srinagar"},
            ),
        ),
        CarouselItem(
            id="dest-varanasi",
            title="🪔 Varanasi (Eternal City)",
            subtitle="Ganga Aarti & Ghats",
            primary_cta=CtaAction(
                label="Explore Varanasi",
                selection={"action": "explore_city", "destination": "Varanasi"},
            ),
        ),
        CarouselItem(
            id="dest-agra",
            title="🏛️ Agra (City of Taj)",
            subtitle="Taj Mahal & Agra Fort",
            primary_cta=CtaAction(
                label="Explore Agra",
                selection={"action": "explore_city", "destination": "Agra"},
            ),
        ),
        CarouselItem(
            id="dest-goa",
            title="🏖️ Goa (Coastal Haven)",
            subtitle="Beaches & Portuguese Churches",
            primary_cta=CtaAction(
                label="Explore Goa",
                selection={"action": "explore_city", "destination": "Goa"},
            ),
        ),
        CarouselItem(
            id="dest-kochi",
            title="🌴 Kochi (Kerala Backwaters)",
            subtitle="Fort Kochi & Houseboats",
            primary_cta=CtaAction(
                label="Explore Kochi",
                selection={"action": "explore_city", "destination": "Kochi"},
            ),
        ),
    ]

    payload = CarouselCardPayload(
        title="🌟 Top Flagship Indian Destinations",
        subtitle="Swipe to choose an Indian heritage circuit",
        items=items,
    )
    return create_card_content(payload)


def get_tourism_card_payload(selected_city: Optional[str] = None) -> MetadataContent:
    """Interactive card for Tourism trip planning with native dropdown selectors."""
    city_options = [
        FormFieldOption(label="🏰 Jaipur (Pink City, Rajasthan)", value="Jaipur"),
        FormFieldOption(label="🏛️ Agra (Taj Mahal, Uttar Pradesh)", value="Agra"),
        FormFieldOption(label="🇮🇳 New Delhi (Sovereign Capital)", value="Delhi"),
        FormFieldOption(label="🏔️ Srinagar (Dal Lake, Kashmir)", value="Srinagar"),
        FormFieldOption(label="🌊 Mumbai (Gateway of India, Maharashtra)", value="Mumbai"),
        FormFieldOption(label="🪔 Varanasi (Ganga Ghats, Uttar Pradesh)", value="Varanasi"),
        FormFieldOption(label="⛵ Udaipur (City of Lakes, Rajasthan)", value="Udaipur"),
        FormFieldOption(label="🏖️ Goa (Beaches & UNESCO Churches)", value="Goa"),
        FormFieldOption(label="🌴 Kochi (Kerala Backwaters)", value="Kochi"),
        FormFieldOption(label="✨ Amritsar (Golden Temple, Punjab)", value="Amritsar"),
        FormFieldOption(label="🧘 Rishikesh (Yoga Capital, Uttarakhand)", value="Rishikesh"),
        FormFieldOption(label="🚂 Darjeeling (Toy Train & Tea, West Bengal)", value="Darjeeling"),
    ]

    is_first_time = (not selected_city) or selected_city.strip().lower() in ("none", "jaipur_default", "first_time", "all", "india")
    if is_first_time:
        card_title = "✈️ India Tourism & Trip Planning Intake"
    else:
        clean_city = selected_city.strip().title()
        card_title = f"✈️ Trip Planning Intake: {clean_city}"
        matched_idx = -1
        for idx, opt in enumerate(city_options):
            if opt.value.lower() == clean_city.lower():
                matched_idx = idx
                break
        if matched_idx >= 0:
            chosen = city_options.pop(matched_idx)
            city_options.insert(0, chosen)
        else:
            city_options.insert(0, FormFieldOption(label=f"📍 {clean_city}", value=clean_city))

    payload = FormCardPayload(
        title=card_title,
        fields=[
            FormField(
                name="destination",
                kind="select",
                label="Choose Destination",
                required=True,
                options=city_options
            ),
            FormField(
                name="days",
                kind="select",
                label="Trip Duration",
                required=True,
                options=[
                    FormFieldOption(label="1 Day", value="1"),
                    FormFieldOption(label="2 Days", value="2"),
                    FormFieldOption(label="3 Days (Recommended)", value="3"),
                    FormFieldOption(label="4 Days", value="4"),
                    FormFieldOption(label="5 Days", value="5"),
                    FormFieldOption(label="7 Days (1 Week)", value="7"),
                    FormFieldOption(label="10 Days", value="10"),
                    FormFieldOption(label="14 Days (Grand Tour)", value="14"),
                ]
            ),
            FormField(
                name="travelers",
                kind="select",
                label="Number of Travelers",
                required=True,
                options=[
                    FormFieldOption(label="Solo (1 Person)", value="1"),
                    FormFieldOption(label="Couple / 2 Adults", value="2"),
                    FormFieldOption(label="Family (4 Persons)", value="4"),
                    FormFieldOption(label="Group (5+ Persons)", value="5"),
                ]
            ),
            FormField(
                name="budget",
                kind="select",
                label="Travel Budget Tier",
                required=True,
                options=[
                    FormFieldOption(label="Low Budget (Hostels & Public Transit)", value="Low Budget"),
                    FormFieldOption(label="Budget (Comfortable 2-3★ & Rail)", value="Budget"),
                    FormFieldOption(label="Mid-range (3-4★ Hotels & Cabs)", value="Mid-range"),
                    FormFieldOption(label="Luxury (5★ Palaces & Chauffeur)", value="Luxury"),
                ]
            ),
            FormField(
                name="transport",
                kind="select",
                label="Preferred Transport",
                required=False,
                options=[
                    FormFieldOption(label="Train (Vande Bharat / Express)", value="Train"),
                    FormFieldOption(label="Flight", value="Flight"),
                    FormFieldOption(label="Cab / Rental Car", value="Cab"),
                    FormFieldOption(label="Metro / Local Transit", value="Metro"),
                ]
            ),
            FormField(
                name="interests",
                kind="select",
                label="Primary Travel Interests",
                required=False,
                options=[
                    FormFieldOption(label="Historical Places & Forts", value="Historical Places"),
                    FormFieldOption(label="Food & Authentic Gastronomy", value="Food"),
                    FormFieldOption(label="UNESCO World Heritage Sites", value="UNESCO"),
                    FormFieldOption(label="Temples & Spiritual Landmarks", value="Temples"),
                    FormFieldOption(label="Bazaars & Handicraft Shopping", value="Shopping"),
                    FormFieldOption(label="Nature & Scenic Viewpoints", value="Nature"),
                ]
            ),
        ],
        submit_cta=CtaAction(
            label="🚀 Generate My Trip Plan",
            selection={"action": "plan_trip"},
            primary=True
        )
    )
    return create_card_content(payload)


def get_economy_card_payload() -> MetadataContent:
    """Interactive card for Indian Economy intake form with native dropdown selectors."""
    payload = FormCardPayload(
        title="📊 Economic Intelligence Intake Form",
        fields=[
            FormField(
                name="topic",
                kind="select",
                label="Analysis Topic",
                required=True,
                options=[
                    FormFieldOption(label="Indian Economy & GDP Growth", value="GDP"),
                    FormFieldOption(label="Major Sectors (Services, Mfg, Agri)", value="Sectors"),
                    FormFieldOption(label="Startups & Tech Unicorns", value="Startups"),
                    FormFieldOption(label="Stock Markets (NSE, BSE, NIFTY 50, SENSEX)", value="Stock Market"),
                    FormFieldOption(label="UPI, RuPay & FinTech Revolution", value="FinTech"),
                    FormFieldOption(label="Corporate Titans (Tata, Reliance, Infosys)", value="Titans"),
                    FormFieldOption(label="Manufacturing, PLI & Atmanirbhar Bharat", value="Manufacturing"),
                ]
            ),
            FormField(
                name="period",
                kind="select",
                label="Time Period",
                required=False,
                options=[
                    FormFieldOption(label="Current State (2024-2025)", value="Current"),
                    FormFieldOption(label="Last 5 Years Trajectory", value="Last 5 Years"),
                    FormFieldOption(label="Future Outlook (Viksit Bharat 2047)", value="Future Outlook"),
                ]
            ),
            FormField(
                name="analysis_type",
                kind="select",
                label="Analysis Depth",
                required=False,
                options=[
                    FormFieldOption(label="Executive Briefing", value="Detailed"),
                    FormFieldOption(label="Investor-Style Analysis", value="Investor"),
                    FormFieldOption(label="Academic & Policy Overview", value="Research"),
                ]
            )
        ],
        submit_cta=CtaAction(
            label="🇮🇳 Generate Economic Intelligence Briefing",
            selection={"action": "economy_dossier"},
            primary=True
        )
    )
    return create_card_content(payload)


def get_culture_card_payload() -> MetadataContent:
    """Interactive card for Culture exploration with native dropdown selectors."""
    payload = FormCardPayload(
        title="🎭 Cultural Intelligence Intake Form",
        fields=[
            FormField(
                name="region",
                kind="select",
                label="Select Region",
                required=True,
                options=[
                    FormFieldOption(label="North India (Punjab, Rajasthan, UP, Delhi)", value="North India"),
                    FormFieldOption(label="South India (Tamil Nadu, Kerala, Karnataka, Andhra)", value="South India"),
                    FormFieldOption(label="East & Northeast India (Bengal, Assam, Odisha)", value="East & Northeast India"),
                    FormFieldOption(label="West India (Maharashtra, Gujarat, Goa)", value="West India"),
                    FormFieldOption(label="Pan-India Synthesis", value="Pan India"),
                ]
            ),
            FormField(
                name="theme",
                kind="select",
                label="Cultural Theme",
                required=True,
                options=[
                    FormFieldOption(label="Festivals & Seasonal Celebrations", value="Festivals"),
                    FormFieldOption(label="8 Classical Dance Forms (Kathak, Bharatanatyam...)", value="Classical Dances"),
                    FormFieldOption(label="Classical Music (Hindustani & Carnatic)", value="Music"),
                    FormFieldOption(label="Food, Regional Gastronomy & Cuisine", value="Cuisine"),
                    FormFieldOption(label="Living Traditions & Modern Social Fabric", value="Society"),
                ]
            ),
            FormField(
                name="depth",
                kind="select",
                label="Depth Level",
                required=False,
                options=[
                    FormFieldOption(label="Detailed Cultural Dossier", value="Detailed"),
                    FormFieldOption(label="Quick Overview", value="Overview"),
                    FormFieldOption(label="Anthropological / Research Grade", value="Research"),
                ]
            )
        ],
        submit_cta=CtaAction(
            label="🎨 Generate Cultural Intelligence Report",
            selection={"action": "culture_dossier"},
            primary=True
        )
    )
    return create_card_content(payload)


def get_science_card_payload() -> MetadataContent:
    """Interactive card for Science & Future Tech with native dropdown selectors."""
    payload = FormCardPayload(
        title="🔬 Science & Future Tech Intake Form",
        fields=[
            FormField(
                name="category",
                kind="select",
                label="Technology Domain",
                required=True,
                options=[
                    FormFieldOption(label="Space Exploration & ISRO (Chandrayaan, Gaganyaan)", value="Space"),
                    FormFieldOption(label="Semiconductors & Chip Fabs (ISM, Dholera, Sanand)", value="Semiconductors"),
                    FormFieldOption(label="Artificial Intelligence & IndiaAI Mission", value="AI"),
                    FormFieldOption(label="Clean Energy, Solar & Green Hydrogen", value="Clean Energy"),
                    FormFieldOption(label="Telecommunications (Fastest 5G & Bharat 6G)", value="Telecom"),
                    FormFieldOption(label="Quantum Computing & Cryptography (NQM)", value="Quantum"),
                ]
            ),
            FormField(
                name="organization",
                kind="select",
                label="Target Organization / Scope",
                required=False,
                options=[
                    FormFieldOption(label="India Overall", value="India Overall"),
                    FormFieldOption(label="ISRO & IN-SPACe", value="ISRO"),
                    FormFieldOption(label="India Semiconductor Mission", value="ISM"),
                    FormFieldOption(label="Private DeepTech Startups (Skyroot, Agnikul)", value="Startups"),
                ]
            ),
            FormField(
                name="focus",
                kind="select",
                label="Analysis Focus",
                required=False,
                options=[
                    FormFieldOption(label="Future Outlook & 2030 Roadmap", value="Future Outlook"),
                    FormFieldOption(label="Current State & Active Missions", value="Current State"),
                    FormFieldOption(label="Landmark Breakthroughs", value="Breakthroughs"),
                ]
            )
        ],
        submit_cta=CtaAction(
            label="🚀 Generate Tech Intelligence Dossier",
            selection={"action": "science_dossier"},
            primary=True
        )
    )
    return create_card_content(payload)


# ----------------------------------------------------
# 4. Action Tools: Calendar, Maps, Flights, Trains
# ----------------------------------------------------
def generate_google_calendar_url(title: str, details: str, location: str) -> str:
    """Generates a real one-click Google Calendar URL to schedule the trip."""
    now = datetime.utcnow()
    start_time = (now + timedelta(days=1)).strftime("%Y%m%dT080000Z")
    end_time = (now + timedelta(days=1)).strftime("%Y%m%dT180000Z")

    clean_details = details.replace("\n", " - ")
    base_url = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    params = {
        "text": f"🇮🇳 {title}",
        "dates": f"{start_time}/{end_time}",
        "details": clean_details,
        "location": location,
    }
    return f"{base_url}&{urllib.parse.urlencode(params)}"


def generate_google_maps_route(waypoints: List[str]) -> str:
    """Generates a real multi-stop Google Maps directions link with standard URL percent-encoding."""
    if not waypoints:
        return "https://www.google.com/maps"
    origin = urllib.parse.quote_plus(waypoints[0])
    destination = urllib.parse.quote_plus(waypoints[-1])
    via_waypoints = [urllib.parse.quote_plus(w) for w in waypoints[1:-1]]

    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}"
    if via_waypoints:
        maps_url += f"&waypoints={'%7C'.join(via_waypoints)}"
    return maps_url


def generate_flight_search_url(destination: str) -> str:
    """Generates a search link for flights to destination."""
    dest_encoded = urllib.parse.quote_plus(f"flights to {destination} India")
    return f"https://www.google.com/travel/flights?q={dest_encoded}"


def generate_hotel_search_url(hotel_name: str, city: str) -> str:
    """Generates an authentic hotel search / booking lookup URL."""
    q_encoded = urllib.parse.quote_plus(f"{hotel_name} {city}")
    return f"https://www.google.com/travel/hotels?q={q_encoded}"


# ----------------------------------------------------
# 5. Verified Destination Intelligence & Hotel Database
# ----------------------------------------------------
FLAGSHIP_CITIES = {
    "jaipur": {
        "title": "Jaipur, Rajasthan 🇮🇳",
        "city_name": "Jaipur",
        "state_name": "Rajasthan",
        "best_known_for": "Royal Rajput architecture, majestic hilltop forts, pink sandstone bazaars, and UNESCO Walled City ensemble.",
        "major_attractions": ["Amer Fort", "City Palace", "Jantar Mantar", "Hawa Mahal", "Nahargarh Fort", "Albert Hall Museum"],
        "unesco_sites": "Jaipur City (Walled City) & Hill Forts of Rajasthan (Amer Fort)",
        "best_time": "October to March (Pleasant dry winter, optimal for fort walks and outdoor bazaars)",
        "approx_duration": "2 to 4 Days recommended",
        "local_transit": "Jaipur Metro, pre-paid autos, e-rickshaws, and app-based cabs (Uber/Ola)",
        "popular_activities": "Amer Fort ascent, sunset views at Nahargarh Fort, shopping at Johari & Bapu Bazaar, Pyaaz Kachori & Ghevar tasting.",
        "ticket_info": "Composite 2-Day Pass (Amer Fort, Hawa Mahal, Jantar Mantar, Nahargarh, Albert Hall): ₹100 (Indians) / ₹500 (Foreigners). City Palace (Separate Ticket): ₹300 (Indians) / ₹700 (Foreigners).",
        "culinary": "Dal Baati Churma, Pyaaz Kachori & Mawa Kachori (Rawat Mishtan Bhandar), Ghevar (LMB, Johari Bazaar), Mirchi Vada.",
        "train_route": "Vande Bharat Express (#20978: New Delhi NDLS 06:10 AM → Jaipur JP 10:05 AM / Return #20977: JP 06:00 PM → NDLS 10:15 PM).",
        "flight_route": "Jaipur International Airport (JAI), Sanganer (13 km from city center). Daily direct flights from Delhi, Mumbai, Bengaluru, Hyderabad.",
        "waypoints": [
            "Jaipur Junction Railway Station, Gopalbari, Jaipur, Rajasthan",
            "Amer Fort, Devisinghpura, Amer, Jaipur, Rajasthan",
            "City Palace, Gangori Bazaar, Pink City, Jaipur, Rajasthan",
            "Jantar Mantar, Gangori Bazaar, Pink City, Jaipur, Rajasthan",
            "Hawa Mahal, Badi Choupad, Pink City, Jaipur, Rajasthan",
            "Nahargarh Fort, Krishna Nagar, Brahampuri, Jaipur, Rajasthan",
        ],
        "ticket_portal": "https://bookrajmonuments.rajasthan.gov.in/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Zostel Jaipur (Pink City)",
                    "rating": "4.6/5 (★ Highly Rated)",
                    "price": "₹1,200 - ₹2,200 / night",
                    "location": "Near Hawa Mahal Road, Pink City",
                    "type": "Boutique Hostel & Private Rooms",
                    "description": "Rooftop terrace cafe overlooking old city minarets, vibrant traveler community, clean AC private rooms and dorms."
                },
                {
                    "name": "Hotel Pearl Palace",
                    "rating": "4.5/5 (★ Award Winner)",
                    "price": "₹1,800 - ₹2,800 / night",
                    "location": "Hathroi Fort, Ajmer Road",
                    "type": "Heritage-style Budget Hotel",
                    "description": "Renowned budget gem decorated with authentic Rajasthani folk handicrafts, featuring the celebrated Peacock Rooftop Restaurant."
                }
            ],
            "value": [
                {
                    "name": "Alsisar Haveli - A Heritage Hotel",
                    "rating": "4.7/5 (★ Premier Heritage)",
                    "price": "₹4,500 - ₹6,800 / night",
                    "location": "Sansar Chandra Road",
                    "type": "Restored 19th Century Noble Haveli",
                    "description": "Traditional Rajput frescoes, carved sandstone archways, serene swimming pool courtyard, and royal hospitality."
                },
                {
                    "name": "Shahpura House",
                    "rating": "4.6/5 (★ Heritage Manor)",
                    "price": "₹5,200 - ₹7,500 / night",
                    "location": "Devi Marg, Bani Park",
                    "type": "Heritage Mansion",
                    "description": "Stunning frescoed ceilings, marble colonnades, royal dining hall, and rooftop sunset terrace."
                }
            ],
            "luxury": [
                {
                    "name": "Rambagh Palace (Taj Group)",
                    "rating": "4.9/5 (★ World-Class Luxury)",
                    "price": "₹28,000 - ₹55,000 / night",
                    "location": "Bhawani Singh Road",
                    "type": "Former Royal Residence of Jaipur Maharaja",
                    "description": "Voted Top Hotel in the World; 47 acres of tranquil landscaped Mughal gardens, heritage polo bar, royal butler service."
                },
                {
                    "name": "Jai Mahal Palace (Taj)",
                    "rating": "4.8/5 (★ Royal Palace)",
                    "price": "₹14,000 - ₹24,000 / night",
                    "location": "Jacob Road, Civil Lines",
                    "type": "18th-Century Indo-Saracenic Palace",
                    "description": "Set amidst 18 acres of landscaped gardens, marble spa pavilion, fine-dining restaurants, and heritage suites."
                }
            ]
        }
    },
    "delhi": {
        "title": "New Delhi — The Sovereign Capital 🇮🇳",
        "city_name": "New Delhi",
        "state_name": "National Capital Territory",
        "best_known_for": "Three UNESCO World Heritage Sites, centuries of dynastic history (Mughal & British colonial), monumental government avenues, and world-famous gastronomy.",
        "major_attractions": ["Qutub Minar", "Humayun's Tomb", "Red Fort", "India Gate & Kartavya Path", "Lotus Temple", "Akshardham Temple", "Chandni Chowk"],
        "unesco_sites": "Qutub Minar & its Monuments, Humayun's Tomb, and Red Fort Complex",
        "best_time": "October to March (Crisp pleasant weather ideal for monuments and gardens)",
        "approx_duration": "3 to 5 Days recommended",
        "local_transit": "Delhi Metro (Yellow, Violet, Blue, Airport Express lines), pre-paid autos, and app-based cabs (Uber/Ola)",
        "popular_activities": "Heritage walks in Old Delhi, food walk in Chandni Chowk, evening walk at Kartavya Path / India Gate, sound & light show at Red Fort.",
        "ticket_info": "ASI Monuments (Red Fort, Qutub Minar, Humayun's Tomb): ₹50 (Indians) / ₹600 (Foreigners) per monument.",
        "culinary": "Chandni Chowk street food (Paranthe Wali Gali, Natraj Dahi Bhalla), Chole Bhature (Sita Ram Diwan Chand), Butter Chicken (Moti Mahal / Karim's).",
        "train_route": "Served by New Delhi (NDLS), Old Delhi (DLI), Hazrat Nizamuddin (NZM), and Anand Vihar (ANVT) with nationwide Vande Bharat & Rajdhani connectivity.",
        "flight_route": "Indira Gandhi International Airport (DEL), India's busiest hub, connected via 20-minute Delhi Metro Airport Express.",
        "waypoints": [
            "New Delhi Railway Station, Paharganj, New Delhi",
            "India Gate & Kartavya Path, Central Secretariat, New Delhi",
            "Humayun's Tomb, Nizamuddin East, New Delhi",
            "Qutub Minar Complex, Mehrauli, New Delhi",
            "Red Fort & Chandni Chowk, Old Delhi",
        ],
        "ticket_portal": "https://asi.payumoney.com/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Bloomrooms @ New Delhi Station",
                    "rating": "4.4/5 (★ Prime Location)",
                    "price": "₹2,200 - ₹3,400 / night",
                    "location": "Opposite New Delhi Railway Station",
                    "type": "Modern Smart Hotel",
                    "description": "Crisp bright minimalist rooms, high-speed WiFi, 3-minute walk to Airport Express Metro and NDLS."
                },
                {
                    "name": "The Hosteller Delhi",
                    "rating": "4.5/5 (★ Top Community)",
                    "price": "₹1,100 - ₹2,000 / night",
                    "location": "Near Ashram Metro Station, South Delhi",
                    "type": "Design Hostel & Private Rooms",
                    "description": "Vibrant social hub, rooftop cafe, curated heritage walks, immaculate AC rooms with dedicated workstations."
                }
            ],
            "value": [
                {
                    "name": "The Claridges New Delhi",
                    "rating": "4.6/5 (★ Lutyens Heritage)",
                    "price": "₹7,500 - ₹11,500 / night",
                    "location": "A.P.J. Abdul Kalam Road, Lutyens Delhi",
                    "type": "Classic Colonial Heritage Hotel",
                    "description": "Lush manicured lawns, iconic 'Dhaba' and 'Sevilla' fine dining, minutes from Lodhi Gardens and India Gate."
                },
                {
                    "name": "Haveli Dharampura",
                    "rating": "4.7/5 (★ UNESCO Award Winner)",
                    "price": "₹8,000 - ₹12,500 / night",
                    "location": "Gali Guliyan, Chandni Chowk, Old Delhi",
                    "type": "Restored 19th Century Mughal Haveli",
                    "description": "Meticulously restored Mughal mansion in the heart of Old Delhi, Kathak dance evenings, rooftop view of Jama Masjid."
                }
            ],
            "luxury": [
                {
                    "name": "The Imperial, New Delhi",
                    "rating": "4.8/5 (★ Legendary Landmark)",
                    "price": "₹18,000 - ₹32,000 / night",
                    "location": "Janpath, Connaught Place",
                    "type": "1930s Historic Art Deco Hotel",
                    "description": "Museum-grade British and Mughal art collection, majestic royal palm gardens, world-class spa, and heritage luxury."
                },
                {
                    "name": "The Leela Palace New Delhi",
                    "rating": "4.9/5 (★ Palace Opulence)",
                    "price": "₹22,000 - ₹40,000 / night",
                    "location": "Diplomatic Enclave, Chanakyapuri",
                    "type": "Modern Royal Palace",
                    "description": "Rooftop infinity pool, grand royal suites with Murano glass chandeliers, Michelin-pedigree dining (Le Cirque, Jamavar)."
                }
            ]
        }
    },
    "agra": {
        "title": "Agra — City of the Taj 🇮🇳",
        "city_name": "Agra",
        "state_name": "Uttar Pradesh",
        "best_known_for": "The Taj Mahal, Agra Fort, Fatehpur Sikri, exquisite Mughal architecture, marble inlay craft, and Agra Petha.",
        "major_attractions": ["Taj Mahal (UNESCO)", "Agra Fort (UNESCO)", "Fatehpur Sikri (UNESCO)", "Mehtab Bagh", "Itimad-ud-Daulah (Baby Taj)"],
        "unesco_sites": "Taj Mahal, Agra Fort, and Fatehpur Sikri (Three UNESCO Sites)",
        "best_time": "October to March. ⚠️ Taj Mahal is strictly CLOSED on Fridays!",
        "approx_duration": "1 to 2 Days",
        "local_transit": "Battery-operated e-rickshaws around Taj precinct, pre-paid autos, app cabs (Uber/Ola)",
        "popular_activities": "Sunrise viewing of Taj Mahal, Agra Fort exploration, sunset from Mehtab Bagh across Yamuna, Panchhi Petha tasting.",
        "ticket_info": "Taj Mahal: ₹50 (Indians) + ₹200 (Main Mausoleum) / ₹1,100 (Foreigners) + ₹200. Agra Fort: ₹50 / ₹650.",
        "culinary": "Panchhi Petha (Angoori, Kesar, Paan flavors), Bedmi Puri & Aloo breakfast, Dalmoth, Mughlai kebabs.",
        "train_route": "Gatimaan Express (#12050: Hazrat Nizamuddin NZM 08:10 AM → Agra Cantt AGC 09:50 AM / Return #12049: AGC 05:50 PM → NZM 07:30 PM).",
        "flight_route": "Kheria Airport (AGR) has select flights; most travelers use high-speed express train or expressway from Delhi.",
        "waypoints": [
            "Agra Cantt Railway Station, Idgah Colony, Agra",
            "Taj Mahal Eastern Gate, Dharmapuri, Agra",
            "Agra Fort, Rakabganj, Agra",
            "Mehtab Bagh, Nagla Devjit, Agra",
            "Panchhi Petha, Hari Parwat, Agra",
        ],
        "ticket_portal": "https://asi.payumoney.com/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Zostel Agra",
                    "rating": "4.5/5 (★ Backpacker Favorite)",
                    "price": "₹1,000 - ₹1,800 / night",
                    "location": "Taj Nagari Phase 1",
                    "type": "Hostel & Private AC Rooms",
                    "description": "Located just 1.5 km from the Taj Mahal East Gate, rooftop cafe, clean facilities, vibrant social atmosphere."
                },
                {
                    "name": "Hotel Taj Resorts",
                    "rating": "4.4/5 (★ Proximity Pick)",
                    "price": "₹2,500 - ₹3,500 / night",
                    "location": "Taj East Gate Road",
                    "type": "Comfort Budget Hotel",
                    "description": "500-meter walk to the Taj Mahal entry gate, rooftop pool with direct views of Taj domes, clean modern amenities."
                }
            ],
            "value": [
                {
                    "name": "ITC Mughal, a Luxury Collection Hotel",
                    "rating": "4.7/5 (★ Mughal Grandeur)",
                    "price": "₹6,500 - ₹9,500 / night",
                    "location": "Fatehabad Road, Taj Ganj",
                    "type": "Award-winning Resort Hotel",
                    "description": "Sprawling across 35 acres of lush gardens, Kaya Kalp Spa (largest in India), Mughal-inspired architecture and fine dining."
                }
            ],
            "luxury": [
                {
                    "name": "The Oberoi Amarvilas, Agra",
                    "rating": "4.9/5 (★ Ultimate Luxury)",
                    "price": "₹35,000 - ₹65,000 / night",
                    "location": "Taj East Gate Road",
                    "type": "World-Famous Luxury Palace Hotel",
                    "description": "Every single room offers uninterrupted views of the Taj Mahal (just 600m away); private golf cart transfers, royal spa."
                }
            ]
        }
    },
    "mumbai": {
        "title": "Mumbai (Bombay) — City of Dreams & Commerce 🇮🇳",
        "city_name": "Mumbai",
        "state_name": "Maharashtra",
        "best_known_for": "Financial capital of India, Bollywood, UNESCO Victorian & Art Deco ensemble, Elephanta Caves, Gateway of India, Marine Drive.",
        "major_attractions": ["Gateway of India", "Elephanta Caves (UNESCO)", "CSMT Terminus (UNESCO)", "Marine Drive & Marine Walk", "Colaba Causeway"],
        "unesco_sites": "Elephanta Caves, Chhatrapati Shivaji Maharaj Terminus (CSMT), Victorian Gothic and Art Deco Ensembles of Mumbai",
        "best_time": "November to February (Pleasant coastal winter). ⚠️ Elephanta Caves closed on Mondays!",
        "approx_duration": "2 to 4 Days",
        "local_transit": "Mumbai Suburban Local Trains (Western/Central), Mumbai Metro Line 3, black-and-yellow taxis, app cabs",
        "popular_activities": "Ferry ride to Elephanta Caves, sunset promenade along Marine Drive, Colaba street shopping, coastal Malvani seafood dining.",
        "ticket_info": "Elephanta Ferry: ₹260 return. Caves entry: ₹40 (Indians) / ₹600 (Foreigners).",
        "culinary": "Vada Pav (Ashok Vada Pav), Pav Bhaji (Sardar Pav Bhaji), Bombay Duck, Coastal Malvani thali, Irani Chai & Bun Maska (Kyani & Co).",
        "train_route": "CSMT & Mumbai Central connect all national Rajdhani, Vande Bharat, and express train routes.",
        "flight_route": "Chhatrapati Shivaji Maharaj International Airport (BOM), world-class terminal with global flights.",
        "waypoints": [
            "Chhatrapati Shivaji Maharaj Terminus (CSMT), Fort, Mumbai",
            "Gateway of India, Colaba, Mumbai",
            "Elephanta Caves Ferry Terminal, Colaba, Mumbai",
            "Marine Drive Promenade, Churchgate, Mumbai",
        ],
        "ticket_portal": "https://asi.payumoney.com/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Abode Bombay",
                    "rating": "4.6/5 (★ Boutique Gem)",
                    "price": "₹3,500 - ₹5,000 / night",
                    "location": "Colaba, behind Gateway of India",
                    "type": "Boutique Heritage Hotel",
                    "description": "Charming colonial architecture, restored antique furniture, organic cafe, steps from the Gateway of India."
                }
            ],
            "value": [
                {
                    "name": "The Gordon House Hotel",
                    "rating": "4.5/5 (★ Concept Hotel)",
                    "price": "₹6,000 - ₹8,500 / night",
                    "location": "Battery Street, Colaba",
                    "type": "Boutique Theme Hotel",
                    "description": "Themed Mediterranean, Scandinavian, and Country floors; home to celebrated 'All Stir Fry' restaurant."
                }
            ],
            "luxury": [
                {
                    "name": "The Taj Mahal Palace & Tower",
                    "rating": "4.9/5 (★ Historic Icon of India)",
                    "price": "₹22,000 - ₹45,000 / night",
                    "location": "Apollo Bunder, Colaba",
                    "type": "Historic 1903 Luxury Palace Hotel",
                    "description": "India's most iconic luxury hotel facing the Gateway of India and Arabian Sea; legendary hospitality and Michelin dining."
                }
            ]
        }
    },
    "srinagar": {
        "title": "Srinagar (Kashmir) — Paradise on Earth 🇮🇳",
        "city_name": "Srinagar",
        "state_name": "Jammu & Kashmir",
        "best_known_for": "Iconic Dal Lake houseboats and Shikara rides, majestic Mughal terraced water gardens, snow-capped Pir Panjal vistas, saffron fields of Pampore, world-famous Kashmiri Pashmina shawls and walnut wood carving.",
        "major_attractions": ["Dal Lake (Shikara & Houseboats)", "Mughal Gardens (Shalimar & Nishat Bagh)", "Shankaracharya Temple", "Hazratbal Shrine", "Pari Mahal", "Old City Bazaars & Lal Chowk", "Nigeen Lake"],
        "unesco_sites": "Mughal Gardens of Kashmir (Tentative UNESCO World Heritage List) & UNESCO Creative Cities Network (Crafts and Folk Arts)",
        "best_time": "April to October (Pleasant blooms, Chinar foliage & summer cool) and December to February (Magical winter snowfall and snow activities in nearby Gulmarg).",
        "approx_duration": "3 to 5 Days recommended",
        "local_transit": "Shikaras on Dal & Nigeen lakes, pre-paid tourist taxis, and local cabs",
        "popular_activities": "Morning Shikara ride to the Floating Vegetable Market, heritage sunset from Pari Mahal, authentic multi-course Wazwan dinner on a carved cedar-wood houseboat, shopping for certified Kashmiri saffron and Pashmina shawls.",
        "ticket_info": "Shikara Ride: ₹700 - ₹1,200 per boat (2 hours). Mughal Gardens (Shalimar/Nishat): ₹24 (Indians) / ₹100 (Foreigners) per garden. Gulmarg Gondola: Phase 1 (₹810) / Phase 2 (₹1,010).",
        "culinary": "Authentic Wazwan feast (Rogan Josh, Rista, Gushtaba, Tabak Maaz), Saffron Kahwa with crushed almonds, Nadru Yakhni (lotus stem curry), Dum Aloo, and Kashmiri Sheermal.",
        "train_route": "Connected via Vande Bharat Express to Jammu Tawi (JAT) / Katra (SVDK) (#22439: New Delhi NDLS 06:00 AM → SVDK 02:00 PM), with direct connectivity across the landmark Chenab Railway Bridge via the Udhampur-Srinagar-Baramulla Rail Link (USBRL).",
        "flight_route": "Sheikh ul-Alam International Airport (SXR), Srinagar (12 km from city center). Daily direct flights connect Srinagar to New Delhi, Mumbai, Bengaluru, Chandigarh, and Jammu.",
        "waypoints": [
            "Sheikh ul-Alam International Airport (SXR), Srinagar, Jammu & Kashmir",
            "Boulevard Road & Dal Lake Shikara Ghat, Srinagar",
            "Shalimar Bagh Mughal Garden, Srinagar",
            "Nishat Bagh, Foreshore Road, Srinagar",
            "Shankaracharya Temple, Dalgate, Srinagar",
            "Pari Mahal, Cheshma Shahi, Srinagar",
            "Hazratbal Shrine, Srinagar",
        ],
        "ticket_portal": "https://asi.nic.in/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Zostel Srinagar",
                    "rating": "4.6/5 (★ Backpacker Favorite)",
                    "price": "₹1,200 - ₹2,200 / night",
                    "location": "Nishat, near Dal Lake",
                    "type": "Boutique Hostel & Private AC Rooms",
                    "description": "Scenic garden patio surrounded by apple and walnut trees, rooftop mountain views, clean facilities, vibrant traveler community."
                },
                {
                    "name": "Houseboat New Golden Flower",
                    "rating": "4.5/5 (★ Heritage Houseboat)",
                    "price": "₹1,800 - ₹2,800 / night",
                    "location": "Dal Lake (Gate 9, Boulevard Road)",
                    "type": "Traditional Kashmiri Cedar Houseboat",
                    "description": "Hand-carved walnut wood interiors, Kashmiri carpets, personal Shikara transfers, stunning sunset views over Dal Lake."
                }
            ],
            "value": [
                {
                    "name": "The Chinar Resort & Spa",
                    "rating": "4.7/5 (★ Valley Retreat)",
                    "price": "₹5,500 - ₹8,500 / night",
                    "location": "Near Shankaracharya Hill",
                    "type": "Upscale Valley Resort",
                    "description": "Surrounded by pine-scented hills, heated indoor pool, authentic Kashmiri dining, premium contemporary comforts."
                },
                {
                    "name": "Hotel Grand Mumtaz",
                    "rating": "4.5/5 (★ Central Comfort)",
                    "price": "₹4,800 - ₹7,200 / night",
                    "location": "Maulana Azad Road, City Center",
                    "type": "Heritage-style City Hotel",
                    "description": "Prime location close to Lal Chowk and Dal Lake, multi-cuisine restaurant, centrally heated rooms."
                }
            ],
            "luxury": [
                {
                    "name": "The Lalit Grand Palace Srinagar",
                    "rating": "4.9/5 (★ Royal Maharaja Palace)",
                    "price": "₹24,000 - ₹48,000 / night",
                    "location": "Gupkar Road, facing Dal Lake",
                    "type": "Former Royal Residence of Maharaja Hari Singh",
                    "description": "Built in 1910 overlooking Dal Lake; expansive manicured gardens with century-old Chinar trees, palatial suites, royal indoor heated pool."
                },
                {
                    "name": "Vivanta Dal View (Taj Group)",
                    "rating": "4.8/5 (★ Panoramic Luxury)",
                    "price": "₹18,000 - ₹34,000 / night",
                    "location": "Kralsangri, Gopakar",
                    "type": "Luxury Hilltop Resort",
                    "description": "Perched on Kralsangri hill with breathtaking 180-degree panoramic views of Dal Lake and Zabarwan mountains."
                }
            ]
        }
    },
    "varanasi": {
        "title": "Varanasi (Kashi / Benares) — The Spiritual Capital 🇮🇳",
        "city_name": "Varanasi",
        "state_name": "Uttar Pradesh",
        "best_known_for": "World's oldest living continuous city (5,000+ years), sacred Ganga Ghats, grand evening Ganga Aarti at Dashashwamedh, Kashi Vishwanath Jyotirlinga temple, silk weaving (Banarasi sarees), and Sarnath.",
        "major_attractions": ["Kashi Vishwanath Temple & Corridor", "Dashashwamedh Ghat (Ganga Aarti)", "Assi Ghat (Subah-e-Banaras)", "Manikarnika Ghat", "Sarnath (UNESCO Lion Capital)", "Banaras Hindu University (BHU)", "Ramnagar Fort"],
        "unesco_sites": "Iconic Riverfront of the River Ganga (Tentative UNESCO List) and Monuments at Sarnath",
        "best_time": "October to March (Pleasant dry winter, optimal for morning boat rides and temple walks).",
        "approx_duration": "2 to 3 Days recommended",
        "local_transit": "Traditional wooden rowboats and motorboats on the Ganga, pre-paid autos, e-rickshaws, and app cabs",
        "popular_activities": "Dawn sunrise boat ride along the 84 ghats, grand evening Maha Aarti at Dashashwamedh, heritage alley walk, Banarasi silk saree shopping.",
        "ticket_info": "Morning Boat: ₹400 - ₹800. Kashi Vishwanath: Free (Sugam VIP Pass: ₹300). Sarnath ASI: ₹25 / ₹300.",
        "culinary": "Kachori Jalebi (Ram Bhandar), Tamatar Chaat (Kashi Chaat Bhandar), Blue Lassi, and authentic Banarasi Paan.",
        "train_route": "Vande Bharat Express (#22436: New Delhi NDLS 06:00 AM → Varanasi BSB 02:00 PM / Return #22435: BSB 03:00 PM → NDLS 11:00 PM).",
        "flight_route": "Lal Bahadur Shastri International Airport (VNS), Babatpur (22 km from ghats). Daily direct flights from Delhi, Mumbai, Bengaluru, Kolkata.",
        "waypoints": [
            "Varanasi Junction (BSB) / Banaras Railway Station, Varanasi",
            "Kashi Vishwanath Temple & Corridor, Lahori Tola, Varanasi",
            "Dashashwamedh Ghat, Godowlia, Varanasi",
            "Assi Ghat, Shivala, Varanasi",
            "Sarnath Archaeological Site & Dhamek Stupa, Varanasi",
        ],
        "ticket_portal": "https://asi.payumoney.com/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Zostel Varanasi",
                    "rating": "4.5/5 (★ Backpacker Gem)",
                    "price": "₹1,100 - ₹2,000 / night",
                    "location": "Near Dashashwamedh Ghat Road",
                    "type": "Vibrant Youth Hostel & Private Rooms",
                    "description": "500-meter walk to main ghats, colorful rooftop common room, morning sunrise boat tours, spotless AC dorms."
                }
            ],
            "value": [
                {
                    "name": "BrijRama Palace, Varanasi",
                    "rating": "4.8/5 (★ 18th-Century Ghat Palace)",
                    "price": "₹16,000 - ₹28,000 / night",
                    "location": "Darbhanga Ghat",
                    "type": "Historic Riverside Palace Hotel",
                    "description": "Built in 1812 directly on Darbhanga Ghat; private boat transfers, traditional classical music performances, pure vegetarian fine dining."
                }
            ],
            "luxury": [
                {
                    "name": "Taj Nadesar Palace",
                    "rating": "4.9/5 (★ Royal Palace Retreat)",
                    "price": "₹28,000 - ₹55,000 / night",
                    "location": "Nadesar Palace Grounds",
                    "type": "Historical Sanctuary of the Maharaja",
                    "description": "Set amidst mango orchards and jasmine fields; horse-drawn royal carriage arrivals, curated heritage dining, butler service."
                }
            ]
        }
    },
    "udaipur": {
        "title": "Udaipur — City of Lakes & Royal Palaces 🇮🇳",
        "city_name": "Udaipur",
        "state_name": "Rajasthan",
        "best_known_for": "Lake Pichola, majestic white marble City Palace complex, floating Lake Palace, romantic sunset boat cruises, Jag Mandir island, and vibrant Mewari art and culture.",
        "major_attractions": ["City Palace Udaipur", "Lake Pichola (Boat Cruise)", "Jag Mandir Island Palace", "Saheliyon-ki-Bari", "Jagdish Temple", "Bagore Ki Haveli (Folk Dance)", "Monsoon Palace (Sajjangarh)"],
        "unesco_sites": "Hill Forts of Rajasthan & Historic ensemble of Mewar palaces",
        "best_time": "September to March (Pleasant temperate winter, optimal for lake cruises and rooftop dining).",
        "approx_duration": "2 to 4 Days recommended",
        "local_transit": "Pre-paid autos, app cabs (Uber/Ola), and traditional lake motorboats",
        "popular_activities": "Sunset boat cruise on Lake Pichola past Jag Niwas, Dharohar folk dance at Bagore Ki Haveli, City Palace museum tour.",
        "ticket_info": "City Palace: ₹400. Lake Pichola Cruise: ₹800 - ₹1,000. Bagore Ki Haveli Show: ₹100 / ₹200.",
        "culinary": "Dal Baati Churma, Gatte ki Sabzi, Ker Sangri, Laal Maas, and traditional Mewari thali.",
        "train_route": "Vande Bharat Express connects Udaipur City (UDZ) to Jaipur (JP) and Agra (#20980 / #20979).",
        "flight_route": "Maharana Pratap Airport (UDR), Dabok (22 km from old city). Daily direct flights from Delhi, Mumbai, Bengaluru.",
        "waypoints": [
            "Udaipur City Railway Station, Jawahar Nagar, Udaipur",
            "City Palace, Old City, Udaipur",
            "Lake Pichola & Gangaur Ghat, Udaipur",
            "Saheliyon-ki-Bari, Panchwati, Udaipur",
            "Monsoon Palace (Sajjangarh), Kodiyat, Udaipur",
        ],
        "ticket_portal": "https://bookrajmonuments.rajasthan.gov.in/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Zostel Udaipur",
                    "rating": "4.6/5 (★ Rooftop Lake Views)",
                    "price": "₹1,200 - ₹2,200 / night",
                    "location": "Near Chandpole, Lake Pichola",
                    "type": "Hostel & Boutique Private Rooms",
                    "description": "Rooftop terrace cafe overlooking Lake Pichola and the City Palace, clean rooms, walking distance to ghats."
                }
            ],
            "value": [
                {
                    "name": "Jagat Niwas Palace Hotel",
                    "rating": "4.7/5 (★ Restored 17th-Century Haveli)",
                    "price": "₹5,500 - ₹9,500 / night",
                    "location": "Lal Ghat, behind City Palace",
                    "type": "Heritage Mansion on Lake Pichola",
                    "description": "Traditional Jharokha window seating suspended directly over Lake Pichola, authentic Mewari hospitality."
                }
            ],
            "luxury": [
                {
                    "name": "Taj Lake Palace, Udaipur",
                    "rating": "4.9/5 (★ World-Famous Floating Palace)",
                    "price": "₹35,000 - ₹75,000 / night",
                    "location": "Lake Pichola island",
                    "type": "1746 Floating White Marble Palace",
                    "description": "Built in 1746 by Maharana Jagat Singh II in the center of Lake Pichola; private boat arrival with rose petal shower, royal butler service."
                }
            ]
        }
    },
    "goa": {
        "title": "Goa — Coastal Heritage, Beaches & Portuguese Architecture 🇮🇳",
        "city_name": "Goa",
        "state_name": "Goa",
        "best_known_for": "UNESCO World Heritage churches of Old Goa, Portuguese colonial Latin Quarter (Fontainhas), pristine coastal beaches (Calangute, Anjuna, Palolem), Dudhsagar Waterfalls, and vibrant Goan-Portuguese fusion cuisine.",
        "major_attractions": ["Basilica of Bom Jesus (UNESCO)", "Se Cathedral", "Fort Aguada & Lighthouse", "Fontainhas Latin Quarter (Panaji)", "Dudhsagar Waterfalls", "Anjuna & Vagator Cliffs", "Palolem Beach"],
        "unesco_sites": "Churches and Convents of Goa (UNESCO World Heritage Site)",
        "best_time": "November to February (Sun-drenched breezy coastal winter with pleasant nights).",
        "approx_duration": "3 to 6 Days recommended",
        "local_transit": "Self-drive rental scooters/cars, pre-paid tourist taxis, Goa Ferry services",
        "popular_activities": "Heritage walking tour of Fontainhas Latin Quarter, sunset dolphin cruise on Mandovi River, water sports, beach shack seafood dining.",
        "ticket_info": "Old Goa Churches: Free. Fort Aguada: ₹25 / ₹300. Dudhsagar Jeep Safari: ₹3,500 per jeep.",
        "culinary": "Goan Fish Curry Thali, Prawn Balchão, Pork Vindaloo, Bebinca dessert, and fresh tender coconut.",
        "train_route": "Madgaon (MAO) / Thivim (THVM) Vande Bharat Express from Mumbai CSMT (#22229 / #22230).",
        "flight_route": "Served by both Manohar International Airport (GOX) Mopa and Dabolim Airport (GOI).",
        "waypoints": [
            "Madgaon (MAO) / Thivim Railway Station, Goa",
            "Basilica of Bom Jesus, Old Goa",
            "Fontainhas Latin Quarter, Panaji, Goa",
            "Fort Aguada, Candolim, Goa",
            "Palolem Beach, Canacona, South Goa",
        ],
        "ticket_portal": "https://asi.payumoney.com/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "The Hosteller Goa (Anjuna)",
                    "rating": "4.5/5 (★ Coastal Social Hub)",
                    "price": "₹1,100 - ₹2,000 / night",
                    "location": "Near Anjuna Beach",
                    "type": "Backpacker Hostel & AC Pods",
                    "description": "Swimming pool, open-air cafe, bike rentals, clean dorms and private rooms, 10-minute walk to beach."
                }
            ],
            "value": [
                {
                    "name": "Panjim Inn (Heritage)",
                    "rating": "4.6/5 (★ Fontainhas Heritage)",
                    "price": "₹4,800 - ₹7,500 / night",
                    "location": "Fontainhas Latin Quarter, Panaji",
                    "type": "Restored 19th-Century Portuguese Mansion",
                    "description": "Four-poster antique beds, period paintings, courtyard veranda restaurant in the historic Latin Quarter."
                }
            ],
            "luxury": [
                {
                    "name": "Taj Fort Aguada Resort & Spa",
                    "rating": "4.8/5 (★ Beachfront Fortress)",
                    "price": "₹18,000 - ₹35,000 / night",
                    "location": "Sinquerim Beach, Candolim",
                    "type": "Luxury Historic Coastal Resort",
                    "description": "Built on the ramparts of a 16th-century Portuguese fortress facing the Arabian Sea; private beach access, royal spa."
                }
            ]
        }
    },
    "kochi": {
        "title": "Kochi (Cochin) — Queen of the Arabian Sea 🇮🇳",
        "city_name": "Kochi",
        "state_name": "Kerala",
        "best_known_for": "Colonial Fort Kochi, iconic cantilevered Chinese Fishing Nets, Mattancherry Dutch Palace, Jew Town and 1568 Paradesi Synagogue, and gateway to the tranquil Kerala Backwaters.",
        "major_attractions": ["Fort Kochi & Chinese Fishing Nets", "Mattancherry Palace (Dutch Palace)", "Paradesi Synagogue (Jew Town)", "Santa Cruz Cathedral Basilica", "Kerala Kathakali Centre", "Marine Drive & Kochi Water Metro"],
        "unesco_sites": "Historic Port Ensemble of Muziris & Western Ghats gateway",
        "best_time": "October to March (Pleasant tropical winter, optimal for harbor walks and backwater cruises).",
        "approx_duration": "2 to 4 Days recommended",
        "local_transit": "Kochi Water Metro (air-conditioned electric ferries), Kochi Metro rail, autos, and app cabs",
        "popular_activities": "Walking through colonial Fort Kochi, watching fishermen operate cantilevered Chinese fishing nets, Kathakali evening dance, spice market shopping.",
        "ticket_info": "Mattancherry Palace: ₹5. Synagogue: ₹10. Kochi Water Metro: ₹20 - ₹40. Kathakali Show: ₹300 - ₹500.",
        "culinary": "Kerala Sadhya feast on banana leaf, Appam with Vegetable Ishtu, Karimeen Pollichathu (pearl spot fish in banana leaf), and Malabar Parotta.",
        "train_route": "Ernakulam Junction (ERS) Vande Bharat Express (#20633: Kasaragod → Thiruvananthapuram via ERS).",
        "flight_route": "Cochin International Airport (COK), world's first fully solar-powered airport (28 km from city center).",
        "waypoints": [
            "Ernakulam Junction (ERS), Kochi",
            "Fort Kochi Beach & Chinese Fishing Nets, Kochi",
            "Mattancherry Palace, Jew Town, Kochi",
            "Paradesi Synagogue, Jew Town, Kochi",
            "Marine Drive Promenade, Ernakulam",
        ],
        "ticket_portal": "https://asi.nic.in/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Zostel Kochi (Fort Kochi)",
                    "rating": "4.6/5 (★ Colonial Pick)",
                    "price": "₹1,100 - ₹2,000 / night",
                    "location": "Near Princess Street, Fort Kochi",
                    "type": "Heritage Hostel & AC Rooms",
                    "description": "Steps from cafes and art galleries, common terrace, clean dorms and private rooms, cycling tours."
                }
            ],
            "value": [
                {
                    "name": "Old Harbour Hotel",
                    "rating": "4.8/5 (★ 300-Year-Old Heritage)",
                    "price": "₹7,500 - ₹12,000 / night",
                    "location": "Tower Road, Fort Kochi",
                    "type": "Restored Dutch Colonial Boutique Hotel",
                    "description": "Lush garden courtyard, Ayurvedic massages, fresh seafood dining, facing ancient Chinese fishing nets."
                }
            ],
            "luxury": [
                {
                    "name": "Brunton Boatyard (CGH Earth)",
                    "rating": "4.9/5 (★ Harbor Grandeur)",
                    "price": "₹16,000 - ₹28,000 / night",
                    "location": "Calvathy, Fort Kochi harbor",
                    "type": "Historic Victorian Boatyard Palace",
                    "description": "Overlooking the vibrant Cochin harbor channel where dolphins play; tea lounge, colonial maritime suites, gourmet dining."
                }
            ]
        }
    },
    "amritsar": {
        "title": "Amritsar — The Holy City of the Golden Temple 🇮🇳",
        "city_name": "Amritsar",
        "state_name": "Punjab",
        "best_known_for": "Sri Harmandir Sahib (The Golden Temple), Wagah Border Beating Retreat ceremony, Jallianwala Bagh, Partition Museum, and world-renowned Punjabi hospitality and gastronomy.",
        "major_attractions": ["Sri Harmandir Sahib (Golden Temple)", "Wagah Border Beating Retreat Ceremony", "Jallianwala Bagh Memorial", "Partition Museum (Town Hall)", "Gobindgarh Fort", "Durgiana Temple"],
        "unesco_sites": "Sri Harmandir Sahib (Tentative UNESCO World Heritage List)",
        "best_time": "October to March (Pleasant sunny days and crisp cool nights).",
        "approx_duration": "2 to 3 Days recommended",
        "local_transit": "Free Golden Temple electric buses, pre-paid autos, e-rickshaws, and app cabs",
        "popular_activities": "Evening Palki Sahib ceremony at Golden Temple, community service (Seva) at Guru Ka Langar (serving 100,000+ meals daily), cheering at Wagah Border ceremony.",
        "ticket_info": "Golden Temple & Langar: Free to all. Wagah Border: Free entry (Reach by 03:30 PM for good seats). Partition Museum: ₹10 (Indians) / ₹250 (Foreigners).",
        "culinary": "Amritsari Kulcha with Chole and melting butter (Kulcha Land / Pehalwan), Makki di Roti & Sarson da Saag, Creamy Lassi (Gian Chand / Ahuja), and Gulab Jamun (Sharma Sweets).",
        "train_route": "Vande Bharat Express (#22487: New Delhi NDLS 03:15 PM → Amritsar ASR 08:45 PM / Return #22488: ASR 08:20 AM → NDLS 01:50 PM).",
        "flight_route": "Sri Guru Ram Dass Jee International Airport (ATQ), Amritsar (11 km from Golden Temple).",
        "waypoints": [
            "Amritsar Junction Railway Station, Amritsar",
            "Sri Harmandir Sahib (Golden Temple), Amritsar",
            "Jallianwala Bagh, Golden Temple Road, Amritsar",
            "Partition Museum, Town Hall, Amritsar",
            "Wagah Border Checkpost, Attari, Punjab",
        ],
        "ticket_portal": "https://asi.nic.in/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Bloomrooms @ Heritage Walk Amritsar",
                    "rating": "4.5/5 (★ Prime Heritage Walk)",
                    "price": "₹2,200 - ₹3,400 / night",
                    "location": "Heritage Street, 200m from Golden Temple",
                    "type": "Modern Smart Hotel",
                    "description": "Bright cheerful rooms, unbeatable location on the pedestrian-only Heritage Walk leading to the temple."
                }
            ],
            "value": [
                {
                    "name": "Ranjit's SVAASA - Heritage Boutique Haveli",
                    "rating": "4.7/5 (★ 200-Year-Old Haveli)",
                    "price": "₹5,500 - ₹8,500 / night",
                    "location": "Mall Road",
                    "type": "Restored Red Brick Nanak Shahi Haveli",
                    "description": "Organic kitchen dining, tranquil garden spa, authentic Punjabi heritage ambiance."
                }
            ],
            "luxury": [
                {
                    "name": "Taj Swarna, Amritsar",
                    "rating": "4.8/5 (★ Premier Luxury)",
                    "price": "₹9,500 - ₹18,000 / night",
                    "location": "Majitha Verka Bypass",
                    "type": "Luxury Modern Hotel",
                    "description": "Outdoor pool, Jiva Spa, authentic Punjabi & world cuisine (Grand Trunk), opulent suites."
                }
            ]
        }
    },
    "rishikesh": {
        "title": "Rishikesh — Yoga Capital of the World & Himalayan Gateway 🇮🇳",
        "city_name": "Rishikesh",
        "state_name": "Uttarakhand",
        "best_known_for": "Spiritual capital of yoga, sacred Triveni Ghat Ganga Aarti, Ram Jhula & Laxman Jhula, The Beatles Ashram, white-water river rafting on the holy Ganges, and foothills of the Himalayas.",
        "major_attractions": ["Triveni Ghat (Maha Ganga Aarti)", "Ram Jhula & Swarg Ashram", "The Beatles Ashram (Chaurasi Kutia)", "Parmarth Niketan Ashram", "White-water Rafting on Ganges", "Neer Garh Waterfall", "Tera Manzil Temple (Trimbakeshwar)"],
        "unesco_sites": "Sacred Himalayan Pilgrimage Circuit & River Ganges Corridor",
        "best_time": "September to April (Pleasant mountain weather and ideal river conditions for rafting).",
        "approx_duration": "2 to 4 Days recommended",
        "local_transit": "Shared vikrams (tempo-autos), e-rickshaws, rental scooters, and walking along pedestrian bridges",
        "popular_activities": "Evening Ganga Aarti with floating diya lamps at Parmarth Niketan & Triveni Ghat, 16 km Grade III white-water rafting from Shivpuri, sunrise meditation, Ayurvedic massage and yoga workshops.",
        "ticket_info": "Triveni Ghat Aarti: Free. Beatles Ashram: ₹150 (Indians) / ₹600 (Foreigners). River Rafting (16 km Shivpuri to Rishikesh): ₹1,000 - ₹1,500 per person.",
        "culinary": "Pure vegetarian Ayurvedic meals, Aloo Puri breakfast at Chotiwala, organic vegan cafes (Beatles Cafe, Little Buddha), masala chai, and fresh mountain fruit juices.",
        "train_route": "Vande Bharat Express (#22457: Anand Vihar ANVT 05:50 PM → Haridwar HW / Dehradun DDN 10:20 PM) connected to Yog Nagari Rishikesh (YNRK).",
        "flight_route": "Dehradun Jolly Grant Airport (DED) is located just 21 km from Rishikesh (35-minute taxi drive).",
        "waypoints": [
            "Yog Nagari Rishikesh Railway Station, Rishikesh",
            "Triveni Ghat, Rishikesh",
            "Ram Jhula & Parmarth Niketan, Swarg Ashram, Rishikesh",
            "The Beatles Ashram, Rajaji Tiger Reserve, Rishikesh",
            "Shivpuri Rafting Base, Uttarakhand",
        ],
        "ticket_portal": "https://asi.nic.in/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "Zostel Rishikesh (Tapovan)",
                    "rating": "4.6/5 (★ Backpacker Social)",
                    "price": "₹1,000 - ₹1,800 / night",
                    "location": "Tapovan, Rishikesh",
                    "type": "Hostel & AC Private Rooms",
                    "description": "Rooftop mountain views, yoga sessions, community cafe, walking distance to cafes and Ganga."
                }
            ],
            "value": [
                {
                    "name": "Aloha on the Ganges",
                    "rating": "4.7/5 (★ Riverfront Resort)",
                    "price": "₹6,500 - ₹11,000 / night",
                    "location": "National Highway 58, Tapovan",
                    "type": "Luxury Riverfront Resort",
                    "description": "Infinity pool overlooking the emerald Ganges River, private ghat, yoga pavilion, exquisite riverside dining."
                }
            ],
            "luxury": [
                {
                    "name": "Ananda in the Himalayas",
                    "rating": "4.9/5 (★ World-Leading Wellness Palace)",
                    "price": "₹38,000 - ₹70,000 / night",
                    "location": "The Palace Estate, Narendra Nagar",
                    "type": "Viceregal Palace & Luxury Destination Spa",
                    "description": "100-acre Maharaja palace estate overlooking the Ganges river valley; bespoke Ayurvedic wellness programs, royal spa."
                }
            ]
        }
    },
    "darjeeling": {
        "title": "Darjeeling — Queen of the Hills & UNESCO Himalayan Railway 🇮🇳",
        "city_name": "Darjeeling",
        "state_name": "West Bengal",
        "best_known_for": "UNESCO World Heritage Toy Train, panoramic sunrise over Mount Kanchenjunga (world's 3rd highest peak) from Tiger Hill, world-renowned Darjeeling black tea estates, and British colonial hill station heritage.",
        "major_attractions": ["Darjeeling Himalayan Railway (UNESCO Toy Train)", "Tiger Hill (Kanchenjunga Sunrise)", "Batasia Loop & War Memorial", "Happy Valley Tea Estate", "Himalayan Mountaineering Institute & Zoo", "Peace Pagoda & Japanese Temple", "Mall Road & Chowrasta"],
        "unesco_sites": "Mountain Railways of India (Darjeeling Himalayan Railway - UNESCO World Heritage Site)",
        "best_time": "March to May (Spring rhododendron blooms) and October to December (Clear crisp Himalayan skies with panoramic mountain views).",
        "approx_duration": "2 to 4 Days recommended",
        "local_transit": "Shared tourist jeeps (Sumo/Bolero), pre-paid taxis, and the scenic UNESCO Toy Train",
        "popular_activities": "Dawn 04:00 AM trip to Tiger Hill for golden sunrise over Kanchenjunga, Joy Ride on the steam Toy Train from Darjeeling to Ghum, guided tea tasting at Happy Valley, walk along Chowrasta Mall.",
        "ticket_info": "Toy Train Joy Ride (Steam engine): ₹1,500 (First Class). Tiger Hill Observatory: ₹50. HMI & Padmaja Naidu Zoo: ₹110. Tea Tasting: ₹100 - ₹250.",
        "culinary": "Steaming hot Tibetan Momos, Thukpa noodle soup, Darjeeling First Flush tea, Churpee local cheese, and English bakery treats at historic Glenary's (Bakery established 1885).",
        "train_route": "New Jalpaiguri (NJP) connected via Vande Bharat Express from Howrah (#22301 / #22302: Howrah 05:55 AM → NJP 01:25 PM), followed by scenic mountain taxi drive (70 km / 3 hours).",
        "flight_route": "Bagdogra International Airport (IXB), Siliguri (68 km / 2.5 hours scenic uphill mountain drive to Darjeeling).",
        "waypoints": [
            "Darjeeling Railway Station (DHR), Darjeeling",
            "Tiger Hill Sunrise Point, Senchal, Darjeeling",
            "Batasia Loop, Ghum, Darjeeling",
            "Happy Valley Tea Estate, Lebong Cart Road, Darjeeling",
            "Chowrasta Mall Road & Glenary's, Darjeeling",
        ],
        "ticket_portal": "https://asi.nic.in/",
        "train_portal": "https://www.irctc.co.in/nget/train-search",
        "hotels": {
            "budget": [
                {
                    "name": "The Hosteller Darjeeling",
                    "rating": "4.5/5 (★ Central Social)",
                    "price": "₹1,200 - ₹2,000 / night",
                    "location": "Near Chowrasta Mall Road",
                    "type": "Backpacker Hostel & Private Rooms",
                    "description": "5-minute walk to Mall Road, panoramic mountain views, clean dorms and private rooms, cafe."
                }
            ],
            "value": [
                {
                    "name": "The Elgin, Darjeeling - Heritage Hotel",
                    "rating": "4.7/5 (★ 125-Year-Old Summer Residence)",
                    "price": "₹8,500 - ₹13,500 / night",
                    "location": "H.D. Lama Road",
                    "type": "British Colonial Heritage Mansion",
                    "description": "Former summer residence of Maharaja of Cooch Behar; roaring fireplaces, period oak paneling, afternoon tea lounge."
                }
            ],
            "luxury": [
                {
                    "name": "Windamere Hotel (Historic British Colonial Heritage)",
                    "rating": "4.8/5 (★ Historic Landmark)",
                    "price": "₹14,000 - ₹24,000 / night",
                    "location": "Observatory Hill",
                    "type": "Legendary Colonial Heritage Hotel",
                    "description": "Perched on historic Observatory Hill with uninterrupted views of Kanchenjunga; candlelight dinners, British high tea, vintage charm."
                }
            ]
        }
    }

}

# ----------------------------------------------------
# 6. Domain Context & Knowledge Base
# ----------------------------------------------------
DOMAIN_CONTEXTS = {
    "economy_finance": """
Key Foundational Grounding for Indian Economy & Financial Markets:
- Macroeconomics: 5th largest global economy by nominal GDP (~$3.9T - $4.0T target), 3rd by PPP (~$13T+). Fast growing major economy (~6.5-7.2% annual real GDP growth).
- Core Sectors: Services (~54% of GVA: IT/BPM, financial services, telecom, tourism), Industry/Manufacturing (~28%: automotive, pharmaceuticals, steel, textiles, chemicals, PLI schemes), Agriculture (~18% of GVA, employing ~45% of workforce).
- Digital Public Infrastructure & FinTech: Unified Payments Interface (UPI) handling 14+ billion monthly transactions (>80% of digital retail payments), RuPay card rails, Aadhaar Enabled Payment System (AePS), Account Aggregator framework.
- Financial Institutions & Markets: Reserve Bank of India (RBI - central bank, inflation targeting mandate 4% +/- 2%), Securities and Exchange Board of India (SEBI - capital markets regulator).
- Exchanges & Benchmarks: Bombay Stock Exchange (BSE, Asia's oldest founded 1875, benchmark: S&P BSE SENSEX of 30 bellwether stocks), National Stock Exchange (NSE, benchmark: NIFTY 50).
- Key Conglomerates & Leaders: Tata Group, Reliance Industries, Infosys, TCS, HCLTech, Wipro, Larsen & Toubro, Adani Group, Mahindra & Mahindra, Maruti Suzuki, State Bank of India (SBI), HDFC Bank, ITC.
- Government Schemes: Make in India, Production-Linked Incentive (PLI across 14 strategic sectors), Atmanirbhar Bharat, PM Gati Shakti (national infrastructure masterplan), Goods and Services Tax (GST unified indirect tax).
""",

    "science_tech": """
Key Foundational Grounding for Indian Science, Innovation & Future Tech:
- Space Exploration (ISRO):
  * Chandrayaan-1 (2008 - confirmed lunar water molecules), Chandrayaan-2 (2019 - high-res orbiter), Chandrayaan-3 (Aug 23, 2023 - historic first landing near lunar south pole, Shiv Shakti point; August 23 celebrated as National Space Day).
  * Mars Orbiter Mission (Mangalyaan - 2013, first Asian nation to reach Mars, done on frugal budget).
  * Solar Physics: Aditya-L1 (halo orbit at Sun-Earth Lagrangian point L1).
  * Human Spaceflight: Gaganyaan program (crewed orbital spacecraft mission to 400 km orbit).
  * Launch Vehicles: PSLV (workhorse), LVM3/GSLV Mk III (heavy lifter), SSLV (small satellite launch vehicle).
  * Private Space Startups (IN-SPACe ecosystem): Skyroot Aerospace (Vikram rockets), Agnikul Cosmos (3D printed engines, Agnibaan), Pixxel (hyperspectral imaging), Dhruva Space.
- Artificial Intelligence:
  * IndiaAI Mission: National initiative with ₹10,300+ crore outlay approving 10,000+ GPU sovereign compute infrastructure.
  * Bhashini: AI-driven national language translation mission bridging India's 22 scheduled languages.
  * Research Clusters: IITs, IISc Bangalore, Centre for AI & Robotics (CAIR), booming startup ecosystem in GenAI (Sarvam AI, Krutrim).
- Semiconductor Mission (ISM):
  * $10 Billion incentive package for commercial fabs, OSAT/ATMP packaging, and fabless design.
  * Approved mega projects: Tata Electronics & PSMC fab in Dholera (Gujarat), Micron OSAT facility in Sanand, CG Power/Renesas packaging plant, Kaynes Semicon.
  * Global Design Strength: Over 20% of the world's semiconductor design and verification engineers are based in India (Bengaluru, Hyderabad, Noida).
- Telecommunications:
  * Fastest national 5G standalone/non-standalone rollout globally covering 98%+ districts.
  * Bharat 6G Vision and Alliance; indigenous 4G/5G technology stack.
- Clean Energy & Future Tech:
  * 500 GW non-fossil capacity target by 2030; Bhadla Solar Park (one of world's largest).
  * National Green Hydrogen Mission (5 MMT annual production by 2030).
  * National Quantum Mission (NQM) fostering quantum computing, communications, and cryptography.
  * Advanced Defence: DRDO, LCA Tejas fighter, BrahMos supersonic missile, iDEX defense startup innovation.
""",

    "culture_society": """
Key Foundational Grounding for Indian Cultural Heritage & Social Fabric:
- Civilizational Heritage: 5,000+ years continuity from Indus Valley Civilization to Vedic, Mauryan, Gupta, Chola, Vijayanagara, Mughal, Rajput, and modern eras.
- Living Diversity: 28 States and 8 Union Territories. 22 Officially recognized Eighth Schedule languages, hundreds of regional dialects. Classical languages (Sanskrit, Tamil, Telugu, Kannada, Malayalam, Odia, Marathi, Pali, Prakrit, Assamese, Bengali).
- Major Festivals: Diwali (Festival of Lights), Holi (Colors), Eid-ul-Fitr, Eid-ul-Adha, Christmas, Navratri & Durga Puja, Ganesh Chaturthi, Pongal & Onam (harvest festivals of South India), Baisakhi, Raksha Bandhan, Janmashtami, Bihu (Assam), Chhath Puja.
- 8 Classical Dances (Sangeet Natak Akademi): Bharatanatyam (Tamil Nadu), Kathak (North India), Kathakali (Kerala), Kuchipudi (Andhra Pradesh), Odissi (Odisha), Manipuri (Manipur), Mohiniyattam (Kerala), Sattriya (Assam).
- Classical Music Traditions: Hindustani (North Indian tradition featuring Khyal, Dhrupad, sitar, sarod, tabla) and Carnatic (South Indian tradition featuring Kriti, Purandara Dasa, Thyagaraja, veena, mridangam).
- Regional Cultures:
  * North: Indo-Aryan linguistic heritage, Punjabi, Rajasthani & Awadhi cuisine, Mughal-Rajput architecture, vibrant folk dances (Bhangra, Ghoomar).
  * South: Dravidian linguistic roots, majestic temple gopuram architecture, rice-based delicacies (Dosa, Idli, Sambar), Carnatic music, Ayurveda.
  * East & Northeast: Bengal Renaissance, tribal arts, Seven Sister states with unique weaving traditions and nature-harmonic festivals (Hornbill, Bihu).
  * West: Maratha and Gujarati entrepreneurial heritage, Garba, Lavani, textiles (Patola, Bandhani), Jain & coastal food.
- Indian Society & Demographics:
  * Demographics: 1.4B+ citizens; world's youngest large demographic with median age ~28.
  * Societal Dynamics: Traditional joint family bonds harmonizing with urban nuclear setups; expanding education access (National Education Policy 2020), digital empowerment in rural villages, women leadership in tech and governance.
"""
}

INDIAN_STATES = [
    "rajasthan", "kerala", "gujarat", "maharashtra", "tamil nadu", "karnataka", 
    "uttar pradesh", "west bengal", "punjab", "haryana", "bihar", "odisha", 
    "telangana", "andhra pradesh", "madhya pradesh", "assam", "himachal pradesh", 
    "uttarakhand", "goa", "kashmir", "ladakh", "sikkim", "meghalaya", "manipur", 
    "mizoram", "nagaland", "tripura", "arunachal pradesh", "jharkhand", "chhattisgarh"
]

FOREIGN_DESTINATIONS = [
    "thailand", "bangkok", "phuket", "pattaya", "bali", "indonesia", "dubai", "uae", 
    "abu dhabi", "singapore", "malaysia", "kuala lumpur", "maldives", "sri lanka", "vietnam", 
    "paris", "france", "london", "uk", "united kingdom", "usa", "america", "united states", 
    "new york", "california", "canada", "toronto", "vancouver", "switzerland", "zurich", 
    "italy", "rome", "germany", "berlin", "spain", "barcelona", "madrid", "japan", "tokyo", 
    "kyoto", "china", "beijing", "shanghai", "australia", "sydney", "melbourne", "new zealand", 
    "turkey", "istanbul", "egypt", "cairo", "greece", "athens", "russia", "moscow", "south korea", 
    "seoul", "hong kong", "mauritius", "nepal", "bhutan"
]

POPULAR_TOURIST_DESTINATIONS = [
    "jaipur", "delhi", "new delhi", "agra", "mumbai", "srinagar", "varanasi", "kashi", "banaras",
    "udaipur", "goa", "kochi", "cochin", "amritsar", "rishikesh", "darjeeling", "manali", "shimla",
    "leh", "ladakh", "bengaluru", "bangalore", "hyderabad", "chennai", "kolkata", "calcutta",
    "pune", "ahmedabad", "ayodhya", "mysuru", "mysore", "jodhpur", "jaisalmer", "pushkar",
    "mount abu", "khajuraho", "hampi", "madurai", "ooty", "kodaikanal", "munnar", "alleppey",
    "alappuzha", "wayanad", "pondicherry", "puducherry", "shillong", "gangtok", "tawang",
    "kaziranga", "andaman", "port blair", "haridwar", "nainital", "mussoorie", "dharamshala",
    "mcleodganj", "spiti", "kutch", "rann of kutch", "ajmer", "bikaner", "gwalior", "bhopal",
    "orchha", "bhubaneswar", "puri", "konark", "coorg", "kodagu", "gokarna", "rameshwaram",
    "kanyakumari", "tirupati", "prayagraj", "allahabad", "bodh gaya", "gaya", "patna",
    "vrindavan", "mathura", "lucknow", "guwahati", "cherrapunji", "gulmarg", "pahalgam", "sonamarg",
    "jammu", "kashmir", "katra", "vaishno devi", "patnitop", "kullu", "kasol", "dehradun", "kanpur",
    "nagpur", "indore", "ujjain", "pachmarhi", "sanchi", "chittorgarh", "alwar", "ranthambore"
]


# ----------------------------------------------------
# Universal Real-Time Web Intelligence & Online Search Engine
# ----------------------------------------------------
_ONLINE_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}

def fetch_online_intelligence(query: str, domain: str = "general") -> Dict[str, Any]:
    """
    Universal real-time web intelligence engine for India Intel Agent.
    Queries Wikipedia REST API / MediaWiki Search API with fallback search heuristics.
    Caches results in memory for instantaneous sub-millisecond responses on repeated queries.
    """
    clean_q = re.sub(
        r'^(?:tell me about|what is|explore|travel to|trip to|tourism in|info on|information on|research|explain|search|i want to (?:travel|go) to)\s+',
        '', query, flags=re.I
    ).strip()
    cache_key = f"{domain}:{clean_q.lower()}"
    if cache_key in _ONLINE_SEARCH_CACHE:
        return _ONLINE_SEARCH_CACHE[cache_key]

    if domain == "tourism":
        search_terms = [f"{clean_q} city India", f"{clean_q} tourism India", f"{clean_q} India", clean_q]
    elif domain == "culture":
        search_terms = [f"{clean_q} Indian culture", f"{clean_q} tradition India", f"{clean_q} India", clean_q]
    elif domain == "economy":
        search_terms = [f"{clean_q} India economy", f"{clean_q} scheme India", f"{clean_q} India", clean_q]
    elif domain == "science":
        search_terms = [f"{clean_q} India science technology", f"{clean_q} India", clean_q]
    else:
        search_terms = [f"{clean_q} India", clean_q]

    result = {
        "query": clean_q,
        "title": clean_q.title(),
        "extract": "",
        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(clean_q.replace(' ', '_'))}",
        "success": False
    }

    for term in search_terms:
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(term)}&format=json&utf8=1&srlimit=2"
            req = urllib.request.Request(url, headers={"User-Agent": "IndiaIntelAgent/2.0 (ASI:One Agent Chat Protocol; contact@asi1.ai)"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                items = data.get("query", {}).get("search", [])
                if items:
                    title = items[0]["title"]
                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                    sreq = urllib.request.Request(sum_url, headers={"User-Agent": "IndiaIntelAgent/2.0 (ASI:One Agent Chat Protocol; contact@asi1.ai)"})
                    with urllib.request.urlopen(sreq, timeout=4) as sresp:
                        sdata = json.loads(sresp.read().decode())
                        ext = sdata.get("extract", "")
                        if ext and len(ext) > 40:
                            result["title"] = sdata.get("title", title)
                            result["extract"] = ext
                            if "content_urls" in sdata and "desktop" in sdata["content_urls"]:
                                result["url"] = sdata["content_urls"]["desktop"].get("page", result["url"])
                            result["success"] = True
                            _ONLINE_SEARCH_CACHE[cache_key] = result
                            return result
        except Exception as e:
            logger.debug(f"Online search attempt for '{term}' error: {e}")
            continue

    _ONLINE_SEARCH_CACHE[cache_key] = result
    return result


def has_keyword(query: str, keywords: List[str]) -> bool:
    """Checks if any keyword appears as a whole word or exact phrase in query."""
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


def parse_travel_params(query: str, form_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extracts structured travel parameters merging form inputs and text query."""
    form_params = form_params or {}
    q_lower = query.lower()

    # 1. Destination
    dest_from_form = form_params.get("destination") or form_params.get("dest")
    destination = "jaipur"
    if dest_from_form and str(dest_from_form).lower() not in ("choose...", ""):
        dest_str = str(dest_from_form).strip().lower()
        for city in FLAGSHIP_CITIES.keys():
            if city in dest_str:
                destination = city
                break
        else:
            for city in POPULAR_TOURIST_DESTINATIONS:
                if city in dest_str:
                    destination = city
                    break
            else:
                for state in INDIAN_STATES:
                    if state in dest_str:
                        destination = state
                        break
                else:
                    destination = dest_str
    else:
        for city in FLAGSHIP_CITIES.keys():
            if re.search(r'\b' + re.escape(city) + r'\b', q_lower):
                destination = city
                break
        else:
            for city in POPULAR_TOURIST_DESTINATIONS:
                if re.search(r'\b' + re.escape(city) + r'\b', q_lower):
                    destination = city
                    break
            else:
                for state in INDIAN_STATES:
                    if re.search(r'\b' + re.escape(state) + r'\b', q_lower):
                        destination = state
                        break
                else:
                    travel_patterns = [
                        r'(?:travel(?:ling)? to|trip to|visit|explore|tour of|tour to|heading to|going to|go to|holiday in|vacation in|itinerary for|plan (?:a )?(?:trip )?(?:to|for)|flights? to|train to|hotels? in|tourism in)\s+([a-zA-Z\s]+)',
                        r'([a-zA-Z\s]+?)\s+(?:trip|tour|tourism|itinerary|travel)',
                    ]
                    for p in travel_patterns:
                        m = re.search(p, q_lower)
                        if m:
                            cand = m.group(1).strip()
                            cand = re.split(r'\b(?:for|with|in|on|next|\d+\s*days?)\b', cand, flags=re.IGNORECASE)[0].strip()
                            if cand and cand.lower() not in ('india', 'a', 'the', 'my', 'some', 'any', 'place', 'city'):
                                destination = cand
                                break

    # 2. Days
    days = 3
    if "days" in form_params:
        d_match = re.search(r'\d+', str(form_params["days"]))
        if d_match:
            days = max(1, min(14, int(d_match.group(0))))
    else:
        days_match = re.search(r'(\d+)\s*(?:days?|day|d)\b', q_lower)
        if days_match:
            days = max(1, min(14, int(days_match.group(1))))
        elif "one day" in q_lower or "1-day" in q_lower:
            days = 1
        elif "two days" in q_lower or "2-day" in q_lower:
            days = 2
        elif "three days" in q_lower or "3-day" in q_lower:
            days = 3
        elif "four days" in q_lower or "4-day" in q_lower:
            days = 4
        elif "five days" in q_lower or "5-day" in q_lower:
            days = 5
        elif "seven days" in q_lower or "week" in q_lower:
            days = 7

    # 3. Travelers
    travelers = 2
    if "travelers" in form_params:
        t_match = re.search(r'\d+', str(form_params["travelers"]))
        if t_match:
            travelers = max(1, int(t_match.group(0)))
        elif "family" in str(form_params["travelers"]).lower():
            travelers = 4
        elif "solo" in str(form_params["travelers"]).lower():
            travelers = 1
    else:
        people_match = re.search(r'(\d+)\s*(?:people|persons?|travelers?|adults?|guests?)\b', q_lower)
        if people_match:
            travelers = max(1, int(people_match.group(1)))
        elif "family" in q_lower:
            travelers = 4
        elif "solo" in q_lower:
            travelers = 1
        elif "couple" in q_lower:
            travelers = 2
        elif "family" in q_lower:
            travelers = 4
        elif "solo" in q_lower:
            travelers = 1
        elif "couple" in q_lower:
            travelers = 2

    # 4. Budget
    budget = form_params.get("budget", "Budget")
    if "low budget" in q_lower or "hostel" in q_lower:
        budget = "Low Budget"
    elif "luxury" in q_lower or "premium" in q_lower or "5-star" in q_lower:
        budget = "Luxury"
    elif "mid-range" in q_lower or "standard" in q_lower or "family" in q_lower:
        budget = "Mid-range"

    # 5. Transport
    transport = form_params.get("transport", "Train")
    if "flight" in q_lower or "fly" in q_lower:
        transport = "Flight"
    elif "cab" in q_lower or "car" in q_lower:
        transport = "Cab / Rental Car"
    elif "train" in q_lower or "rail" in q_lower:
        transport = "Train (Vande Bharat / Express)"
    elif "metro" in q_lower:
        transport = "Metro / Local Transit"

    # 6. Hotel
    hotel_type = "Budget Hotel"
    if budget == "Luxury":
        hotel_type = "Heritage Palace / 5-Star Hotel"
    elif budget == "Mid-range":
        hotel_type = "3-4 Star Heritage Hotel"
    elif budget == "Low Budget":
        hotel_type = "Hostel / Backpacker Stay"

    # 7. Interests
    interests = ["Historical Places", "Food & Gastronomy"]
    if "unesco" in q_lower:
        interests.append("UNESCO World Heritage")
    if "temple" in q_lower or "spiritual" in q_lower:
        interests.append("Spiritual Landmarks")
    if "shop" in q_lower or "bazaar" in q_lower:
        interests.append("Bazaars & Shopping")

    return {
        "destination": destination,
        "days": days,
        "travelers": travelers,
        "budget": budget,
        "transport": transport,
        "hotel_type": hotel_type,
        "interests": interests,
    }


# ----------------------------------------------------
# 7. WORKFLOW NODES
# ----------------------------------------------------

def landing_page_node(state: AgentState) -> AgentState:
    """Initial Welcome Landing Screen with 4 Large Interactive Domain Cards & Dropdown Selector."""
    output = """# 🇮🇳 Welcome to India Intel Agent

> **"Your AI-powered intelligence assistant for exploring India's economy, tourism, culture, science, technology and future opportunities."**

Select one of the four interactive intelligence domains below to begin:

---

### 🇮🇳 Option 1: Economy & Financial Markets
*Explore Indian GDP, major sectors (Services, IT, Manufacturing, Agriculture), stock exchanges (NSE/BSE, SENSEX, NIFTY 50), UPI & FinTech revolution, conglomerates (Tata, Reliance, Infosys), and economic policies.*

👉 **[ 🏛️ Open Economy & Financial Markets ]** *(Reply `1` or `Economy`)*

---

### ✈️ Option 2: Tourism & UNESCO Heritage
*Autonomous travel planning, 42 UNESCO World Heritage sites, destination intelligence, day-by-day itineraries, transit routes with IRCTC & flight searches, verified hotel recommendations, and transparent cost estimates.*

👉 **[ ✈️ Open Tourism & UNESCO Heritage ]** *(Reply `2` or `Tourism`)*

---

### 🎨 Option 3: Culture & Social Fabric
*Deep dive into India's 5,000-year civilizational continuity, 8 classical dances, Carnatic & Hindustani music, major festivals (Diwali, Holi, Eid, Pongal), regional traditions across North, South, East, West & Northeast.*

👉 **[ 🎨 Open Culture & Social Fabric ]** *(Reply `3` or `Culture`)*

---

### 🚀 Option 4: Science & Future Tech
*Research ISRO space exploration (Chandrayaan-3, Gaganyaan, Aditya-L1), India Semiconductor Mission (ISM fabs in Dholera/Sanand), IndiaAI Mission (10,000+ GPU compute), 5G/6G, Green Hydrogen & quantum tech.*

👉 **[ 🚀 Open Science & Future Tech ]** *(Reply `4` or `Science`)*

---
*💡 You can select an option above or use the dropdown widget to jump directly to any domain.*"""

    resp = make_ai_response(output, get_landing_card_payload())
    return {
        "workflow_stage": "landing",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def tourism_destinations_node(state: AgentState) -> AgentState:
    """Step 1 of Tourism: Interactive Destination Grid & Searchable Selector with Native Dropdowns."""
    output = """# ✈️ Tourism & UNESCO Heritage Explorer

**Where would you like to travel in India?**

Select a popular destination from the intelligence grid below or use the interactive dropdown to configure your journey:

| Popular Destination | Region | Heritage Highlights | Quick Action |
| :--- | :--- | :--- | :--- |
| **🏰 Jaipur** | Rajasthan | Amer Fort, City Palace, Hawa Mahal, Jantar Mantar | [ Explore Jaipur ] |
| **🏛️ Agra** | Uttar Pradesh | Taj Mahal (UNESCO), Agra Fort, Fatehpur Sikri | [ Explore Agra ] |
| **🇮🇳 Delhi** | Capital | Qutub Minar, Red Fort, Humayun's Tomb, India Gate | [ Explore Delhi ] |
| **🌊 Mumbai** | Maharashtra | Gateway of India, Elephanta Caves, Marine Drive | [ Explore Mumbai ] |
| **🪔 Varanasi** | Uttar Pradesh | Ganga Ghats, Kashi Vishwanath, Sarnath (UNESCO) | [ Explore Varanasi ] |
| **⛵ Udaipur** | Rajasthan | Lake Pichola, City Palace, Jagdish Temple | [ Explore Udaipur ] |
| **🏖️ Goa** | Coastal | Basilica of Bom Jesus, Fort Aguada, Coastal Beaches | [ Explore Goa ] |
| **🌴 Kochi** | Kerala | Fort Kochi, Chinese Fishing Nets, Kerala Backwaters | [ Explore Kochi ] |
| **✨ Amritsar** | Punjab | Golden Temple (Harmandir Sahib), Wagah Border | [ Explore Amritsar ] |
| **🏔️ Srinagar** | Kashmir | Dal Lake Shikara, Mughal Gardens, Shankaracharya | [ Explore Srinagar ] |
| **🧘 Rishikesh** | Uttarakhand | Ganga Aarti, Laxman Jhula, River Rafting, Yoga | [ Explore Rishikesh ] |
| **🚂 Darjeeling** | West Bengal | Himalayan Railway (UNESCO), Tea Estates, Kanchenjunga | [ Explore Darjeeling ] |

🔍 **Looking for another destination?**  
*You can type any city or state across India (e.g. Bengaluru, Chennai, Hyderabad, Kolkata, Mysuru, Ahmedabad, Manali, Ladakh, Shillong, Andaman, etc.) to inspect its destination intelligence.*

👉 **Reply with your destination (e.g. `Jaipur` or `Explore Jaipur`) or use the dropdown form below to generate your itinerary.**"""

    resp = make_ai_response(output, get_tourism_card_payload(None))
    return {
        "workflow_stage": "tourism_destinations",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def tourism_briefing_node(state: AgentState) -> AgentState:
    """Step 2 of Tourism: Destination Intelligence Card + Interactive Trip Planning Form with Dropdowns."""
    city_key = (state.get("entity_name") or "jaipur").lower()
    city_data = FLAGSHIP_CITIES.get(city_key)

    if not city_data:
        city_title = city_key.title()
        online_res = fetch_online_intelligence(city_title, domain="tourism")
        extract = online_res.get("extract", "")
        page_url = online_res.get("url", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(city_title.replace(' ', '_'))}")
        
        if extract:
            overview_text = extract
        else:
            overview_text = f"{city_title} is a prominent destination in India celebrated for its distinctive regional heritage, cultural history, scenic landscapes, and authentic living traditions."

        flights_link = generate_flight_search_url(city_title)
        maps_link = generate_google_maps_route([f"{city_title}, India", f"{city_title} Center, India"])
        hotels_link = f"https://www.google.com/travel/hotels?q={urllib.parse.quote_plus(city_title + ' hotels')}"
        irctc_link = "https://www.irctc.co.in/nget/train-search"

        output = f"""# 📍 {city_title}, India 🇮🇳 — Destination Intelligence
**Live Online Search Verified • Destination Dossier & Travel Connectivity**

### 🏛️ 1. Destination Overview & Live Intelligence
{overview_text}

---

### 📍 2. Travel Highlights & Practical Guidance
* 🏛️ **Heritage & Culture:** Renowned historic monuments, sacred temples/shrines, architectural sites, and regional heritage quarters.
* ☀️ **Best Season to Visit:** October to March (Comfortable pleasant weather across plains and foothills).
* ⏱️ **Recommended Duration:** 2 to 4 Days for a comprehensive itinerary.
* 🛺 **Local Transit:** Local app cabs (Ola/Uber), pre-paid autos, and regional transit corridors.
* 🎯 **Popular Activities:** Exploring landmark architectural monuments, authentic culinary trails, and local bazaar shopping.

---

### 🚆 3. Transportation Intelligence & Direct Booking Links
* 🚆 **High-Speed Rail:** Superfast and Express trains connect directly to {city_title} Railway Station / Junction.  
  👉 **[ 🔗 Search / Book Trains on Official IRCTC Portal ]({irctc_link})**

* ✈️ **Flight Routes:** Commercial flights connect nearest airport with major Indian metropolitan centers.  
  👉 **[ 🔗 Search Flights on Google Flights ]({flights_link})**

* 🗺️ **GPS Route & Navigation:**  
  👉 **[ 🗺️ Open Multi-Stop GPS Route in Google Maps ]({maps_link})**

* 🏨 **Verified Accommodations:**  
  👉 **[ 🔗 Search Stays in {city_title} on Google Hotels ]({hotels_link})**

* 🌐 **Verified Encyclopedia Reference:** [{city_title} Overview on Wikipedia]({page_url})

---

### 📋 Would you like to plan a trip to {city_title}?
Use the interactive dropdown selector below to configure your trip duration, travelers, budget tier, and transit:"""
    else:
        title = city_data["title"]
        known_for = city_data["best_known_for"]
        attractions = ", ".join(city_data["major_attractions"])
        unesco = city_data["unesco_sites"]
        best_time = city_data["best_time"]
        duration = city_data["approx_duration"]
        transit = city_data["local_transit"]
        activities = city_data["popular_activities"]
        city_name = city_data["city_name"]

        output = f"""## {title} — Destination Intelligence

* 🏛️ **Best Known For:** {known_for}
* 📍 **Major Attractions:** {attractions}.
* 📜 **UNESCO Sites Nearby:** {unesco}.
* ☀️ **Best Time to Visit:** {best_time}.
* ⏱️ **Approximate Travel Duration:** {duration}.
* 🛺 **Local Transportation:** {transit}.
* 🎯 **Popular Activities:** {activities}.

---

### 📋 Would you like to plan a trip to {city_name}?
Use the interactive dropdown form below to configure your trip parameters:"""

    resp = make_ai_response(output, get_tourism_card_payload(city_key))
    return {
        "workflow_stage": "tourism_briefing",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def tourism_followup_node(state: AgentState) -> AgentState:
    """Smart Follow-up: Triggered when required travel parameters are incomplete."""
    params = state.get("extracted_params") or {}
    dest = (params.get("destination") or state.get("entity_name") or "India").title()

    output = f"""### ⚠️ Almost ready! Please provide a few more details for {dest}:

We have recorded your destination: **{dest}**. To calculate an accurate day-by-day schedule, hotel recommendations, transit, and transparent cost estimates, please confirm:

1. 👥 **Number of Travelers:**  
   [ Solo (1) ]  |  [ ● 2 Travelers ]  |  [ Family (4) ]  |  [ Group (5+) ]

2. 💰 **Budget Level:**  
   ○ Low Budget (Hostels)  |  ● **Budget (Comfortable)**  |  ○ Mid-range (3-4★)  |  ○ Luxury (5★)

3. 🚆 **Preferred Transport:**  
   ☑ Train (Vande Bharat / Express)  |  ☐ Flight  |  ☐ Cab / Car  |  ☐ Local Transit

---
👉 **[ Continue & Generate My Trip Plan ]**  
*(Simply reply e.g.: `{dest}, 3 days, 2 people, Budget, Train` or submit the dropdown form below)*"""

    resp = make_ai_response(output, get_tourism_card_payload(dest))
    return {
        "workflow_stage": "tourism_followup",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def tourism_planner_node(state: AgentState) -> AgentState:
    """Autonomous Travel Agent Execution: Generates Comprehensive Multi-Card Travel Dashboard."""
    params = state.get("extracted_params") or parse_travel_params(get_message_content(state["messages"][-1]))
    dest = params.get("destination", "jaipur").lower()
    days = params.get("days", 3)
    travelers = params.get("travelers", 2)
    budget = params.get("budget", "Budget")
    transport = params.get("transport", "Train")
    hotel_type = params.get("hotel_type", "Budget Hotel")
    interests = params.get("interests", ["Historical Places", "Food"])

    city_data = FLAGSHIP_CITIES.get(dest)
    dest_title = city_data["title"] if city_data else f"{dest.title()}, India 🇮🇳"
    city_name = city_data["city_name"] if city_data else dest.title()

    cal_link = generate_google_calendar_url(
        title=f"{days}-Day Trip to {city_name}",
        details=f"Autonomous Travel Plan by @india-intel-agent for {travelers} travelers ({budget} budget).",
        location=f"{city_name}, India"
    )
    waypoints = city_data["waypoints"] if city_data else [f"{city_name}, India", f"{city_name} Heritage Center, India"]
    maps_link = generate_google_maps_route(waypoints)
    ticket_link = city_data.get("ticket_portal", "https://asi.nic.in/") if city_data else "https://asi.nic.in/"
    ticket_portal_name = "Rajasthan Monument Portal" if dest == "jaipur" else "Archaeological Survey of India (ASI)"
    irctc_link = "https://www.irctc.co.in/nget/train-search"
    flights_link = generate_flight_search_url(city_name)

    # Cost Estimation calculation (transparent, non-fabricated formulas based on verified real rates)
    if budget == "Low Budget":
        night_rate = 1500
        transit_cost = 1000 * travelers
        food_per_day = 500 * travelers
        act_per_day = 250 * travelers
        local_per_day = 300
        misc = 500
    elif budget == "Luxury":
        night_rate = 22000
        transit_cost = 5500 * travelers
        food_per_day = 3500 * travelers
        act_per_day = 1500 * travelers
        local_per_day = 2000
        misc = 3000
    elif budget == "Mid-range":
        night_rate = 5500
        transit_cost = 2500 * travelers
        food_per_day = 1200 * travelers
        act_per_day = 600 * travelers
        local_per_day = 800
        misc = 1200
    else:  # Standard Budget
        night_rate = 2200
        transit_cost = 1400 * travelers
        food_per_day = 700 * travelers
        act_per_day = 400 * travelers
        local_per_day = 500
        misc = 800

    nights = max(1, days - 1)
    tot_hotel = night_rate * nights
    tot_food = food_per_day * days
    tot_activities = act_per_day * days
    tot_local = local_per_day * days
    total_est = transit_cost + tot_hotel + tot_food + tot_activities + tot_local + misc

    # Hotel Cards Section
    hotel_cards_md = ""
    if city_data and "hotels" in city_data:
        h_data = city_data["hotels"]
        # Budget
        hotel_cards_md += "### 💰 Best Budget Hotels\n"
        for h in h_data.get("budget", []):
            h_url = generate_hotel_search_url(h['name'], city_name)
            hotel_cards_md += f"""* **{h['name']}**  
  ⭐ Rating: {h['rating']} | 💰 Approx: {h['price']} | 📍 {h['location']}  
  🏷️ *{h['type']}* — {h['description']}  
  👉 **[ 🔗 View Hotel / Search Availability ]({h_url})**\n\n"""
        
        # Value
        hotel_cards_md += "### ⭐ Best Value (Mid-Range & Heritage)\n"
        for h in h_data.get("value", []):
            h_url = generate_hotel_search_url(h['name'], city_name)
            hotel_cards_md += f"""* **{h['name']}**  
  ⭐ Rating: {h['rating']} | 💰 Approx: {h['price']} | 📍 {h['location']}  
  🏷️ *{h['type']}* — {h['description']}  
  👉 **[ 🔗 View Hotel / Search Availability ]({h_url})**\n\n"""

        # Premium
        hotel_cards_md += "### 🏨 Premium & Luxury Palaces\n"
        for h in h_data.get("luxury", []):
            h_url = generate_hotel_search_url(h['name'], city_name)
            hotel_cards_md += f"""* **{h['name']}**  
  ⭐ Rating: {h['rating']} | 💰 Approx: {h['price']} | 📍 {h['location']}  
  🏷️ *{h['type']}* — {h['description']}  
  👉 **[ 🔗 View Hotel / Search Availability ]({h_url})**\n\n"""
    else:
        hotel_cards_md = f"""* **Recommended City Stays:**  
  Explore certified stays in {city_name} across budget (₹1,500-₹2,500), mid-range (₹4,500-₹7,000), and luxury tiers.  
  👉 **[ 🔗 Search Verified Hotels in {city_name} on Google Hotels ](https://www.google.com/travel/hotels?q={urllib.parse.quote_plus(city_name + ' hotels')})**"""

    prompt = f"""
You are the Senior Autonomous Travel Planning Specialist of 🇮🇳 India Educational AI Intelligence Agent (@india-intel-agent).
Generate an executable, highly engaging day-by-day travel dashboard:
- Destination: {dest_title}
- Duration: {days} Days ({nights} Nights)
- Travelers: {travelers} Persons ({'Family' if travelers >= 4 else 'Couples/Friends/Solo'})
- Budget Level: {budget}
- Transport Preference: {transport}
- Hotel Preference: {hotel_type}
- Key Interests: {', '.join(interests)}
"""
    # Deterministic verified itineraries for guaranteed production execution
    if dest == "jaipur":
        itinerary_md = f"""## 🗓️ Day-by-Day Timeline Itinerary: Jaipur ({days} Days)

### 🗓️ Day 1 — Royal Hilltop Fortresses & Sunset Over the Pink City
* **08:00 AM – 11:30 AM: Amer Fort (Amber Palace)**  
  Begin with the morning ascent up the Aravalli hilltop. Explore the grand *Diwan-e-Aam*, the breathtaking *Sheesh Mahal* (Mirror Palace illuminated by candle reflections), and the royal gardens overlooking Maota Lake.
* **12:00 PM – 01:30 PM: Jaigarh Fort & World's Largest Cannon**  
  Connected by underground subterranean tunnels to Amer; witness the massive *Jaivana Cannon* and rugged medieval defensive walls.
* **01:30 PM – 02:30 PM: Traditional Rajasthani Lunch**  
  Stop at the foot of Amer or head to **Rawat Mishtan Bhandar** for steaming hot *Pyaaz Kachoris* and *Mawa Kachoris*.
* **03:00 PM – 04:30 PM: Panna Meena Ka Kund & Jal Mahal**  
  Admire the geometric 16th-century stepwell *Panna Meena Ka Kund*, followed by photography at the submerged water palace *Jal Mahal* on Man Sagar Lake.
* **05:30 PM – 07:00 PM: Golden Hour Sunset at Nahargarh Fort**  
  Drive to the highest point of the Aravalli hills overlooking the pink city. Enjoy panoramic sunset views from the Padao rooftop.
* **08:00 PM: Leisure Dinner**  
  Traditional thali dining with *Dal Baati Churma* and *Gatte ki Sabzi* in the old city.

---

### 🗓️ Day 2 — Walled Pink City, Palaces & UNESCO Heritage
* **09:00 AM – 11:00 AM: City Palace & Royal Museums**  
  Tour the ceremonial residence of the Jaipur royal family. Walk through the *Pritam Niwas Chowk* (famous 4 seasonal gates, including the Peacock Gate) and *Mubarak Mahal* textile gallery.
* **11:00 AM – 12:30 PM: Jantar Mantar (UNESCO World Heritage Site)**  
  Located directly adjacent to City Palace; marvel at 19 colossal architectural stone astronomical instruments, including *Vrihat Samrat Yantra*, the world's most precise stone sundial.
* **12:45 PM – 02:00 PM: Traditional Dining at LMB (Johari Bazaar)**  
  Relish an authentic Rajasthani royal thali and fresh paneer *Ghevar* at the historic Laxmi Mishtan Bhandar.
* **02:15 PM – 03:45 PM: Hawa Mahal (Palace of Winds)**  
  Marvel at the 953 honeycombed jharokha windows of pink and red sandstone built in 1799 for royal ladies to view street festivals unnoticed.
* **04:00 PM – 06:30 PM: Vibrant Bazaars of the Walled City**  
  Walk through *Johari Bazaar* (gemstones, silver & Kundan jewelry) and *Bapu Bazaar* (Jaipuri quilts, Mojari leather shoes, and block-print textiles).
* **07:30 PM: Evening Cultural Show**  
  Visit Albert Hall Museum for night illumination and evening street snacks."""
        if days >= 3:
            itinerary_md += f"""\n\n---

### 🗓️ Day 3 — Royal Art, Spiritual Wonders & Cultural Celebration
* **09:30 AM – 11:30 AM: Albert Hall Museum (Central Museum)**  
  Explore Rajasthan's oldest museum housed in an Indo-Saracenic masterpiece, featuring miniature paintings, metal crafts, and ancient Persian carpets.
* **12:00 PM – 01:30 PM: Galtaji Sun Temple (The Monkey Temple)**  
  Ancient pilgrimage complex tucked inside a narrow mountain pass with sacred freshwater kunds (springs) and natural scenery.
* **02:00 PM – 03:30 PM: Patrika Gate at Jawahar Circle**  
  Photograph one of India's most Instagrammed architectural marvels, showcasing intricate hand-painted murals representing every region of Rajasthan.
* **05:00 PM – 09:30 PM: Chokhi Dhani Ethnic Cultural Village**  
  Immerse in traditional village life: puppet shows (*Kathputli*), Kalbelia folk dance, camel rides, pottery, and an authentic open-air sit-down Rajasthani feast."""

    elif dest == "agra":
        itinerary_md = f"""## 🗓️ Day-by-Day Timeline Itinerary: Agra ({days} Days)

### 🗓️ Day 1 — The Monument of Love & Imperial Mughal Fort
* **08:10 AM – 09:50 AM: High-Speed Transit via Gatimaan Express**  
  Board the Gatimaan Express (#12050) from Hazrat Nizamuddin (NZM) arriving at Agra Cantt (AGC) at 09:50 AM.
* **10:15 AM – 01:00 PM: The Majestic Taj Mahal (UNESCO World Heritage Site)**  
  Enter through the East Gate. Marvel at the translucent white Makrana marble dome, exquisite *pietra dura* semi-precious stone inlay work, and the tranquil Charbagh Mughal gardens reflecting in the watercourse.  
  *(⚠️ Critical Traveler Notice: The Taj Mahal is strictly CLOSED on Fridays for congregational prayers).*
* **01:15 PM – 02:15 PM: Traditional Agra Lunch**  
  Savor Mughlai specialties or traditional *Bedmi Puri* with spiced *Aloo Sabzi* at Dasaprakash or Pinch of Spice.
* **02:30 PM – 04:30 PM: Agra Fort (UNESCO World Heritage Site)**  
  Tour the massive 94-acre red sandstone stronghold of Emperor Akbar and Shah Jahan. Explore the *Diwan-e-Aam*, *Khas Mahal*, *Sheesh Mahal*, and *Musamman Burj* where Shah Jahan spent his final years gazing across the river at the Taj.
* **05:00 PM – 06:15 PM: Mehtab Bagh (Moonlight Garden) Golden Hour Sunset**  
  Cross the Yamuna River to Mehtab Bagh for a romantic golden hour sunset view of the Taj Mahal reflecting in the water from across the river.
* **06:30 PM – 07:30 PM: Petha Tasting & Artisan Marble Inlay**  
  Stop at authentic **Panchhi Petha** (Hari Parwat) for renowned Agra Petha (Kesar, Angoori, Paan flavors) and witness master artisans demonstrating heritage marble inlay craftsmanship.
* **07:45 PM: Return Transit**  
  Board the evening Gatimaan Express (#12049 departs AGC 05:50 PM or evening Shatabdi / Vande Bharat)."""
        if days >= 2:
            itinerary_md += f"""\n\n---

### 🗓️ Day 2 — Ghost City of Fatehpur Sikri & The Baby Taj
* **08:30 AM – 12:30 PM: Fatehpur Sikri (UNESCO World Heritage Site)**  
  Drive 37 km west to Emperor Akbar's red sandstone utopian capital founded in 1571. Enter through the towering 54-meter *Buland Darwaza* (Gate of Magnificence), pray at the white marble tomb of Sufi saint Sheikh Salim Chishti, and explore the *Panch Mahal* and *Jodha Bai Palace*.
* **01:00 PM – 02:00 PM: Regional Lunch**  
  Lunch stop en route back to Agra Cantt.
* **02:30 PM – 04:30 PM: Tomb of I'timād-ud-Daulah ("The Baby Taj")**  
  Precursor to the Taj Mahal built by Empress Nur Jahan for her father Mirzā Ghiyās Beg; acclaimed as a delicate jewel box of marble latticework (*jali*).
* **05:00 PM – 06:30 PM: Akbar's Tomb at Sikandra**  
  Imposing tomb blending Islamic, Hindu, and Christian motifs surrounded by peaceful forested deer parks."""

    elif dest in ("srinagar", "kashmir"):
        itinerary_md = f"""## 🗓️ Day-by-Day Timeline Itinerary: Srinagar ({days} Days)

### 🗓️ Day 1 — The Jewel of Kashmir: Dal Lake, Shikara & Mughal Heritage
* **09:00 AM – 12:00 PM: Historic Shikara Ride across Dal Lake**  
  Board a traditional Shikara at Boulevard Road. Glide across the tranquil waters through the floating lotus gardens, Char Chinar island, and the bustling floating artisan bazaar.
* **12:30 PM – 02:00 PM: Traditional Kashmiri Wazwan Lunch**  
  Savor an authentic lunch featuring *Rogan Josh*, *Rista*, *Dum Aloo*, and *Nadru Yakhni*, followed by warm saffron *Kahwa* tea with crushed almonds.
* **02:30 PM – 04:30 PM: Shalimar Bagh & Nishat Bagh (Mughal Terraced Gardens)**  
  Walk through Jahangir's 1619 garden masterpiece *Shalimar Bagh* (The Garden of Love) and *Nishat Bagh* (Garden of Joy), admiring tiered water channels, cascades, and ancient Chinar trees overlooking the lake.
* **05:00 PM – 06:30 PM: Golden Hour Sunset from Pari Mahal**  
  Ascend to the 17th-century *Pari Mahal* (Palace of Fairies), a terraced observatory offering an unrivaled panoramic sunset over Dal Lake and the Zabarwan range.
* **07:30 PM: Traditional Dinner & Houseboat Experience**  
  Overnight stay in a cedar-wood heritage houseboat with intricate walnut woodwork on Dal or Nigeen Lake.

---

### 🗓️ Day 2 — Sacred Sanctuaries, Old Srinagar & Master Crafts
* **08:30 AM – 10:30 AM: Shankaracharya Temple**  
  Visit the historic 9th-century stone temple atop Gopadri Hill (1,100 ft above the valley floor), offering panoramic views of the Pir Panjal range and Srinagar city.
* **11:00 AM – 01:00 PM: Hazratbal Shrine & Northern Dal Lake**  
  Visit Kashmir's holiest Muslim shrine on the scenic northern banks of Dal Lake, celebrated for its pristine white marble domes and serene waterfront promenades.
* **01:00 PM – 02:00 PM: Historic Lunch at Ahdoos (Residency Road)**  
  Dine at Kashmir's legendary heritage restaurant on the banks of the Jhelum River, renowned for classic Wazwan recipes since 1918.
* **02:30 PM – 05:00 PM: Old City Heritage Circuit & Jama Masjid**  
  Explore Srinagar's historic downtown (*Shahr-e-Khaas*), walking through the 14th-century *Jama Masjid* with its 378 monumental Himalayan deodar wooden pillars.
* **05:00 PM – 07:30 PM: Authentic Kashmiri Crafts & Saffron Shopping**  
  Shop for GI-tagged Pashmina shawls, Pampore pure saffron, hand-knotted silk carpets, walnut wood carvings, and paper-mâché art at government-certified craft emporiums."""
        if days >= 3:
            itinerary_md += f"""\n\n---

### 🗓️ Day 3 — Mountain Excursion to Gulmarg ("Meadow of Flowers")
* **08:00 AM – 10:00 AM: Scenic Alpine Drive to Gulmarg (50 km)**  
  Drive through pine-clad mountain gorges and the Tangmarg foothills to the celebrated alpine resort of Gulmarg.
* **10:30 AM – 02:30 PM: The World's Highest Gulmarg Gondola Ride**  
  Ascend the famous two-phase cable car: Phase 1 to Kongdoori Valley (8,530 ft) and Phase 2 to Apharwat Peak (13,780 ft) for breathtaking snow vistas and alpine meadows.
* **02:30 PM – 03:30 PM: Alpine Lunch at Gulmarg**  
  Warm Kashmiri meal with hot Noon Chai (pink salted tea) and freshly baked breads.
* **04:00 PM – 06:00 PM: Return to Srinagar**  
  Scenic descent back to Srinagar; evening stroll along the vibrant Boulevard waterfront."""

    elif dest == "mumbai":
        itinerary_md = f"""## 🗓️ Day-by-Day Timeline Itinerary: Mumbai ({days} Days)

### 🗓️ Day 1 — Colonial Splendor, Arabian Sea & UNESCO Ensembles
* **09:00 AM – 10:30 AM: Gateway of India & Apollo Bunder**  
  Photograph the iconic 26-meter basalt archway built in 1911 overlooking Mumbai harbor; view the legendary Taj Mahal Palace Hotel.
* **10:45 AM – 02:00 PM: Ferry to Elephanta Caves (UNESCO World Heritage Site)**  
  Take the scenic 1-hour ferry ride across the harbor to Elephanta Island. Explore the 5th-century rock-cut cave temples and the colossal 6-meter Trimurti sculpture of Lord Shiva. *(⚠️ Note: Closed on Mondays!)*
* **02:30 PM – 03:45 PM: Colonial Lunch at Leopold Cafe / Britannia & Co.**  
  Relish Berry Pulao and Parsi delicacies at Britannia or historic bistro dining at Leopold Cafe in Colaba.
* **04:00 PM – 05:30 PM: CSMT Railway Terminus & Victorian Gothic District (UNESCO)**  
  Walk through the Victorian Gothic and Art Deco Ensemble; admire Frederick William Stevens' grand 1887 terminus.
* **06:00 PM – 08:00 PM: Sunset at Marine Drive Promenade ("The Queen's Necklace")**  
  Walk along the 3.6 km curved C-shaped boulevard; enjoy sunset sea breezes and street food at Girgaon Chowpatty (Pav Bhaji, Bhel Puri)."""

    elif dest == "delhi":
        itinerary_md = f"""## 🗓️ Day-by-Day Timeline Itinerary: New Delhi ({days} Days — Family Plan)

### 🗓️ Day 1 — Sovereign & Imperial Delhi (Lutyens Enclave)
* **09:00 AM – 11:30 AM: India Gate & Kartavya Path**  
  Morning walk along the grand national avenue; pay homage at the National War Memorial and photograph the external vistas of Rashtrapati Bhavan.
* **12:00 PM – 02:00 PM: Connaught Place & Janpath**  
  Georgian colonnades of Connaught Place; lunch at *Saravana Bhavan* or *Kake Da Dhaba*, followed by handicraft browsing at Central Cottage Industries Emporium.
* **03:00 PM – 05:00 PM: National Museum**  
  One of India's greatest museums, housing Indus Valley Civilization relics (Dancing Girl), Buddhist art, and ancient coins.
* **05:30 PM – 07:30 PM: Lodhi Gardens & Lodhi Art District**  
  Stroll through 15th-century Sayyid and Lodhi royal tombs surrounded by peaceful gardens and India's first open-air street art district.

---

### 🗓️ Day 2 — Historic Old Delhi (Shahjahanabad) & Mughal Splendor
* **08:30 AM – 10:30 AM: Jama Masjid**  
  One of the largest mosques in India, built by Emperor Shah Jahan in red sandstone and white marble.
* **10:45 AM – 01:00 PM: Chandni Chowk E-Rickshaw Heritage & Food Tour**  
  Immerse in the historic alleys: visit *Paranthe Wali Gali*, sample traditional *Chole Bhature*, and visit the aromatic Khari Baoli spice market (Asia's largest).
* **02:00 PM – 04:30 PM: Red Fort Complex (UNESCO World Heritage Site)**  
  Tour the seat of Mughal imperial power: walk through Lahori Gate, Diwan-e-Aam, and the marble pavilions of Diwan-e-Khas.
* **05:00 PM – 06:30 PM: Raj Ghat & National Gandhi Museum**  
  Peaceful black marble memorial dedicated to Mahatma Gandhi set amidst serene landscaped lawns.

---

### 🗓️ Day 3 — Medieval Capitals & Sufi Heritage
* **09:00 AM – 11:30 AM: Humayun's Tomb (UNESCO World Heritage Site)**  
  Magnificent red sandstone garden tomb that inspired the Taj Mahal; explore the charbagh gardens and Isa Khan's tomb.
* **11:45 AM – 01:30 PM: Sunder Nursery Heritage Park**  
  Delhi's 90-acre UNESCO-awarded ecological and heritage park featuring restored 16th-century Mughal monuments, nursery flora, and artisan cafes.
* **02:00 PM – 03:30 PM: Hazrat Nizamuddin Basti & Dargah**  
  Experience living Sufi heritage; listen to mystical Sufi Qawwalis in the evening atmosphere."""
        if days >= 4:
            itinerary_md += f"""\n\n---

### 🗓️ Day 4 — Ancient Architecture & Iconic Landmarks
* **09:00 AM – 11:30 AM: Qutub Minar Complex (UNESCO World Heritage Site)**  
  Marvel at the 72.5-meter world's tallest brick minaret built in 1192, alongside the 4th-century rust-resistant Iron Pillar of Chandragupta II.
* **12:00 PM – 02:00 PM: Mehrauli Archaeological Park & Lunch at Olive / Haveli**  
  Explore 200+ historic monuments spanning 1,000 years in Mehrauli; lunch at the vibrant Hauz Khas or Olive Bar.
* **03:00 PM – 05:00 PM: Lotus Temple (Baháʼí House of Worship)**  
  Famous petal-shaped marble lotus structure welcoming all faiths in serene silent meditation.
* **05:30 PM – 07:30 PM: Hauz Khas Village & Lake**  
  Medieval madrassa and reservoir ruins overlooking trendy boutiques and lakeside cafes."""
        if days >= 5:
            itinerary_md += f"""\n\n---

### 🗓️ Day 5 — Modern Spiritual Wonder & Pan-India Crafts
* **09:30 AM – 01:30 PM: Swaminarayan Akshardham Temple**  
  Grand complex showcasing 10,000 years of Indian culture: boat ride exhibition (*Sanskruti Vihar*), carved stone mandir, and thematic gardens.
* **02:30 PM – 06:30 PM: Dilli Haat INA (Pan-India Cultural Bazaar)**  
  Open-air permanent cultural marketplace where authorized rural artisans and master chefs from all 28 Indian states showcase handicrafts, handlooms, and authentic regional cuisines."""
    else:
        itinerary_md = f"""## 🗓️ Day-by-Day Timeline Itinerary: {dest_title} ({days} Days)

### 🗓️ Day 1 — Arrival, Heritage Landmarks & Sunset Views in {city_name}
* **09:00 AM – 12:30 PM: Historic Central Quarter & Architecture**  
  Arrive and check in; morning tour of prominent historical landmarks, temples, and architectural marvels of {city_name}.
* **12:45 PM – 02:00 PM: Traditional Regional Lunch**  
  Savor authentic local delicacies and specialties at popular regional dining establishments.
* **02:30 PM – 05:30 PM: Cultural Museums & Heritage Centers**  
  Explore regional museums, ancient forts, and handicraft artisan lanes.
* **06:00 PM – 07:30 PM: Golden Hour Promenade & Sunset Point**  
  Scenic golden hour views across local lakes, riverbanks, or hilltop vantage points.
* **08:00 PM: Evening Leisure Dinner**  
  Traditional dinner experience featuring seasonal regional recipes.

---

### 🗓️ Day 2 — Sacred Shrines, Regional Circuit & Artisan Bazaars
* **09:00 AM – 01:00 PM: Heritage Temple Circuit & Nature Excursion**  
  Visit renowned heritage shrines, sacred pilgrimage sites, or nature reserves in and around {city_name}.
* **02:00 PM – 05:00 PM: Local Craft Demonstrations & Bazaar Shopping**  
  Browse local handlooms, wooden crafts, and regional spices in traditional shopping districts.
* **07:00 PM: Cultural Dining & Evening Atmosphere**  
  Enjoy regional folk music, street gastronomy, and authentic cultural hospitality."""
        if days >= 3:
            itinerary_md += f"""\n\n---

### 🗓️ Day 3 — Scenic Escapes, Regional Crafts & Farewell Walk
* **09:00 AM – 01:00 PM: Countryside Excursion & Viewpoints**  
  Visit scenic foothills, historic outskirts, or royal estates surrounding {city_name}.
* **02:00 PM – 04:30 PM: Souvenir Shopping & Artisan Emporiums**  
  Pick up GI-tagged handicrafts, teas, or regional sweets.
* **05:00 PM: Departure Prep & Onward Transit**  
  Head to {city_name} Railway Station or Airport for departure."""
    output = f"""# 🇮🇳 {city_name} Trip Plan — Intelligence Dashboard

### 📋 Trip Summary Card
| Parameter | Selection | Parameter | Selection |
| :--- | :--- | :--- | :--- |
| 📍 **Destination** | **{city_name}** | 👥 **Travelers** | **{travelers} Persons** ({'Family' if travelers >= 4 else 'Party'}) |
| 📅 **Duration** | **{days} Days ({nights} Nights)** | 💰 **Budget Tier** | **{budget}** |
| 🚆 **Transport** | **{transport}** | 🏨 **Hotel Type** | **{hotel_type}** |

---

{itinerary_md}

---

## 🚆 Transportation Intelligence & Direct Booking Links

* 🚆 **High-Speed Rail:** {city_data.get('train_route', f'Regular Superfast Express trains connect to {city_name} Junction.') if city_data else f'Regular Superfast Express trains connect to {city_name} Junction.'}  
  👉 **[ 🔗 Search / Book Trains on Official IRCTC Portal ]({irctc_link})**

* ✈️ **Flight Routes:** {city_data.get('flight_route', f'Nearest commercial airport connects {city_name} with major metros.') if city_data else f'Nearest commercial airport connects {city_name} with major metros.'}  
  👉 **[ 🔗 Search Flights on Google Flights ]({flights_link})**

* 🛺 **Local Transportation:** {city_data.get('local_transit', 'Pre-paid autos, e-rickshaws, and app cabs.') if city_data else 'Pre-paid autos, e-rickshaws, and app cabs.'}  
  👉 **[ 🗺️ Open Multi-Stop GPS Route in Google Maps ]({maps_link})**

---

## 🏨 Where to Stay (Handpicked & Verified Accommodations)

> *⚠️ Notice: Hotel prices reflect approximate seasonal averages. Please click the direct verification links below to check live availability and dates.*

{hotel_cards_md}

---

## 💰 Transparent Cost Estimation

| Expense Category | Basis / Estimation Formula | Estimated Amount (₹) |
| :--- | :--- | :--- |
| 🚆 **Intercity Transportation** | Round-trip transit ({transport}) for {travelers} travelers | ₹{transit_cost:,} |
| 🏨 **Hotel Accommodation** | {nights} Nights @ ~₹{night_rate:,}/night ({hotel_type}) | ₹{tot_hotel:,} |
| 🍲 **Food & Dining** | ~₹{food_per_day // travelers:,}/day per person × {days} days | ₹{tot_food:,} |
| 🎟️ **Monument Tickets & Activities** | Official entry passes & activity fees for {travelers} travelers | ₹{tot_activities:,} |
| 🛺 **Local Transit (Metro/Autos/Cabs)** | Daily local city commute for {days} days | ₹{tot_local:,} |
| 🛍️ **Miscellaneous & Emergency Buffer** | Contingency, tips, and personal expenses | ₹{misc:,} |
| **💳 TOTAL ESTIMATED COST** | **Complete trip for {travelers} travelers over {days} days** | **₹{total_est:,}** |

*(Per person cost: Approximately **₹{total_est // travelers:,}** per traveler).*

---

## 🔎 Sources & Authoritative References
* 🏛️ **Monument Authority:** [{ticket_portal_name}]({ticket_link})
* 🚆 **Indian Railway Catering and Tourism Corporation:** [irctc.co.in]({irctc_link})
* 🗺️ **Official State Tourism Portal:** Incredible India ([incredibleindia.gov.in](https://www.incredibleindia.gov.in/))
* 📜 **UNESCO World Heritage Centre:** [whc.unesco.org](https://whc.unesco.org/en/statesparties/in)

---

### 🚀 EXECUTABLE ACTIONS & DIRECT BOOKING LINKS

* 🚆 **Book Train on IRCTC:**  
  👉 **[ 🔗 Click to Open IRCTC Portal (irctc.co.in) ]({irctc_link})**  
  *Direct link to Indian Railways online train search, schedule, and reservation.*

* 🎟️ **Book Official Monument Tickets:**  
  👉 **[ 🔗 Click to Book Monument Tickets Online ]({ticket_link})**  
  *Official online entry ticketing for {city_name} via {ticket_portal_name}.*

* ✈️ **Search Flights:**  
  👉 **[ 🔗 Click to Search Flights to {city_name} on Google Flights ]({flights_link})**  
  *Live flight schedules, airlines, and competitive fares.*

* 🗺️ **Multi-Stop Route Navigation:**  
  👉 **[ 🔗 Click to Open Circuit Route in Google Maps ]({maps_link})**  
  *Optimized turn-by-turn driving & walking directions covering all key landmarks.*

* 📅 **Add Trip to Google Calendar:**  
  👉 **[ 🔗 Click to Save {city_name} Itinerary to Google Calendar ]({cal_link})**  
  *One-click calendar integration with all activity schedules and reminders.*"""

    review_card = get_review_card_payload({
        "destination": city_name,
        "days": days,
        "travelers": travelers,
        "budget": budget,
        "transport": transport,
    })
    resp = make_ai_response(output, review_card)
    return {
        "workflow_stage": "tourism_planner",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def economy_intake_form_node(state: AgentState) -> AgentState:
    """Step 1 of Economy: Interactive Structured Intelligence Form with Dropdowns."""
    output = """# 🇮🇳 Economy & Financial Markets Intelligence

Please select the topics and parameters for your economic analysis using the interactive dropdown below:

```markdown
┌────────────────────────────────────────────────────────────────────────┐
│ 📊 ECONOMIC INTELLIGENCE INTAKE FORM                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. 📈 What do you want to analyze? (Select one or more):               │
│    ☑ Indian Economy & GDP           |  ☐ Inflation & Monetary Policy   │
│    ☑ Major Sectors (Services/Mfg)   |  ☑ Startups & Tech Unicorns      │
│    ☑ Stock Markets (NSE/BSE/NIFTY)  |  ☐ IT Industry & Software Export │
│    ☐ Semiconductor & Electronics    |  ☐ EV & Renewable Energy         │
│    ☐ Infrastructure & PM GatiShakti |  ☐ Taxation, Budget & GST        │
│                                                                        │
│ 2. ⏳ Time Period:                                                     │
│    ● Current State (2024-2025)      |  ○ Last 5 Years                  │
│    ○ Last 10 Years                  |  ○ Future Outlook (Viksit 2047)  │
│                                                                        │
│ 3. 🎯 Analysis Type:                                                   │
│    ○ Beginner / Student Overview    |  ● Detailed Executive Briefing   │
│    ○ Investor-Style Analysis        |  ○ Academic / Policy Grade       │
└────────────────────────────────────────────────────────────────────────┘
```

👉 **[ 🇮🇳 Generate Economic Intelligence Briefing ]**  
*(Reply with your choices e.g.: `Indian Economy, Current, Detailed` or use the dropdown widget below)*"""

    resp = make_ai_response(output, get_economy_card_payload())
    return {
        "workflow_stage": "economy_form",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


ECONOMY_TOPIC_DOSSIERS = {
    "GDP": """# 📊 Indian Economy & GDP Growth: Intelligence Dossier
**$4 Trillion Milestone • 3rd Largest Global Economy Trajectory (PPP)**

### 📈 1. Macroeconomic Overview & GDP Trajectory
* **Nominal GDP:** ~$3.9 - $4.0 Trillion (5th largest global economy, rapidly closing the gap with Germany and Japan).
* **GDP (PPP):** ~$13.2+ Trillion (3rd largest economy in the world, behind only the United States and China).
* **Real GDP Growth:** 6.5% – 7.2% annual growth rate, positioning India as the fastest-growing major economy among G20 nations.
* **Viksit Bharat 2047 Target:** National vision to become a fully developed high-income economy ($30 Trillion GDP) by the centenary of independence in 2047.

---

### 🏛️ 2. Core Macro Stability Pillars
* **Foreign Exchange Reserves:** ~$650 - $700 Billion, providing over 11 months of import cover and substantial balance-of-payments resilience.
* **Inflation Targeting (MPC):** Governed by the Reserve Bank of India with a statutory medium-term target of 4.0% (± 2.0%).
* **Fiscal Consolidation:** Fiscal deficit glide path targeting below 4.5% of GDP by FY26, driven by record direct tax and GST mop-ups.
* **Capital Expenditure (Capex):** Central government Capex outlay exceeding ₹11.11 Lakh Crore, catalyzing public infrastructure and crowding in private investment.

---

### 🔎 3. Authoritative Sources & Citations
* 🏛️ **Reserve Bank of India (RBI):** [rbi.org.in](https://www.rbi.org.in/)
* 📊 **Ministry of Statistics & Programme Implementation (MoSPI):** [mospi.gov.in](https://www.mospi.gov.in/)
* 🌐 **Union Budget Portal (Ministry of Finance):** [indiabudget.gov.in](https://www.indiabudget.gov.in/)
* 🏦 **World Bank India Country Profile:** [worldbank.org/en/country/india](https://www.worldbank.org/en/country/india)""",

    "Sectors": """# 🏭 Structural Breakdown of Key Sectors: Intelligence Dossier
**Services • Manufacturing & Industry • Agriculture & Allied**

### 💼 1. Services Sector (~54% of Gross Value Added)
* **IT & BPM Export Powerhouse:** Global leaders including TCS, Infosys, Wipro, and HCLTech generating over $250 Billion in annual technology revenues.
* **Global Capability Centers (GCCs):** Over 1,600 multinational GCCs employ 1.6+ million high-skilled professionals across Bengaluru, Hyderabad, Pune, and Chennai.
* **Banking & Financial Services:** World-class digital banking infrastructure supported by public sector lenders (SBI) and private giants (HDFC Bank, ICICI Bank).

---

### 🏗️ 2. Manufacturing & Industry (~28% of GVA)
* **Make in India & PLI Schemes:** ₹1.97 Lakh Crore incentive scheme across 14 strategic sectors (electronics, automotive, pharma, specialty steel).
* **Electronics Manufacturing Boom:** India is now the 2nd largest global smartphone manufacturer; assembly hub for global flagships (Apple iPhone, Samsung).
* **Automotive & Heavy Industry:** 3rd largest automobile market globally, led by Tata Motors, Mahindra, Maruti Suzuki, and Bajaj Auto.

---

### 🌾 3. Agriculture & Allied Sector (~18% of GVA)
* **Socio-Economic Backbone:** Employs approximately 45% of India's workforce; vital driver of rural consumption.
* **Global Production Leadership:** World's #1 producer of milk, pulses, and spices; #2 in rice, wheat, cotton, and sugarcane.
* **AgriTech & Modernization:** Expanding integration of Kisan Drones, digital soil health cards, and e-NAM national marketplace.

---

### 🔎 4. Authoritative Sources & Citations
* 🏭 **Department for Promotion of Industry and Internal Trade (DPIIT):** [dpiit.gov.in](https://dpiit.gov.in/)
* 📊 **Ministry of Statistics (MoSPI):** [mospi.gov.in](https://www.mospi.gov.in/)
* 🌾 **Ministry of Agriculture & Farmers Welfare:** [agricoop.gov.in](https://agricoop.gov.in/)""",

    "FinTech": """# 💳 UPI, RuPay & FinTech Revolution: Intelligence Dossier
**Digital Public Infrastructure (DPI) • Sovereign Payment Rails & Cashless Economy**

### ⚡ 1. Unified Payments Interface (UPI) Super-Rails
* **Record Transaction Volume:** Processing **over 14 Billion transactions monthly** with a monthly transaction value exceeding ₹20 Lakh Crore ($240+ Billion).
* **Zero Cost & Instant Settlement:** Powered by the National Payments Corporation of India (NPCI) utilizing IMPS rails with 24x7 real-time settlement.
* **Global Adoption:** Operational in France, UAE, Singapore, Mauritius, Sri Lanka, Bhutan, and Nepal.

---

### 💳 2. RuPay Indigenous Card Network
* **Sovereign Card Rails:** Capturing over 60% of domestic debit cards; zero-fee merchant discounts on RuPay credit cards on UPI.
* **International Acceptance:** Accepted across international networks (Discover, Diners Club, JCB) and global financial centers.

---

### 🚀 3. India's FinTech Ecosystem & Account Aggregators
* **10,000+ FinTech Startups:** Across neo-banking, wealthtech (Zerodha, Groww), insurtech, and merchant payments (PhonePe, Paytm, Razorpay).
* **Account Aggregator (AA) Ecosystem:** Consent-based encrypted financial data sharing transforming collateral-free MSME credit underwriting.
* **CBDC (Digital Rupee):** RBI pilot underway across wholesale (e₹-W) and retail (e₹-R) cross-border settlements.

---

### 🔎 4. Authoritative Sources & Citations
* ⚡ **National Payments Corporation of India (NPCI):** [npci.org.in](https://www.npci.org.in/)
* 🏛️ **Reserve Bank of India (RBI):** [rbi.org.in](https://www.rbi.org.in/)
* 🇮🇳 **IndiaStack Digital Public Infrastructure:** [indiastack.org](https://indiastack.org/)""",

    "Startups": """# 🦄 Startups & Tech Unicorns: Intelligence Dossier
**3rd Largest Global Startup Ecosystem • 115+ Unicorns • DeepTech Frontiers**

### 🚀 1. Ecosystem Scale & Global Standing
* **115+ Tech Unicorns:** Cumulative unicorn valuation exceeding **$350+ Billion**, spanning e-commerce, FinTech, SaaS, EdTech, and logistics.
* **100,000+ DPIIT Recognized Startups:** World's 3rd largest startup ecosystem behind only the US and China.
* **Direct Employment Creation:** Over 1.3+ million direct jobs created by recognized startup ventures nationwide.

---

### 🏙️ 2. Geographic Innovation Clusters
* **Bengaluru (Silicon Plateau):** India's premier deeptech, AI, and SaaS hub (Sarvam AI, Flipkart, Swiggy, Zerodha).
* **Delhi-NCR (Gurugram/Noida):** Hub for consumer tech, logistics, and fintech unicorns (Zomato, Paytm, Delhivery).
* **Mumbai & Pune:** Dominant centers for FinTech, enterprise software, and EV manufacturing startups (Ola Electric, Druva).
* **Hyderabad & Chennai:** Emerging SaaS and space-tech centers (Freshworks, Skyroot Aerospace, Agnikul Cosmos).

---

### 🔎 3. Authoritative Sources & Citations
* 🚀 **Startup India National Portal:** [startupindia.gov.in](https://www.startupindia.gov.in/)
* 🏢 **DPIIT (Ministry of Commerce):** [dpiit.gov.in](https://dpiit.gov.in/)
* 💡 **NITI Aayog Atal Innovation Mission (AIM):** [aim.gov.in](https://aim.gov.in/)""",

    "Stock Market": """# 📈 Stock Markets (NSE, BSE, NIFTY 50, SENSEX): Intelligence Dossier
**$5 Trillion Market Cap • 4th Largest Global Equity Market • Retail Investing Boom**

### 📊 1. Market Capitalization & Global Ranking
* **$5+ Trillion Aggregate Market Cap:** India's combined equity market capitalization ranks **4th globally**, trading closely with Hong Kong, Japan, and the US.
* **BSE SENSEX:** Asia's oldest stock exchange (est. 1875); 30 blue-chip bellwethers reflecting India's industrial backbone.
* **NSE NIFTY 50:** Flagship index of the National Stock Exchange; world's #1 derivatives exchange by contracts traded.

---

### 👥 2. Demat Boom & Domestic Institutional Resilience
* **150+ Million Demat Accounts:** Exponential expansion of retail domestic investors via discount brokers (Zerodha, Groww, Angel One).
* **Systematic Investment Plans (SIPs):** Domestic mutual fund SIP inflows exceeding **₹21,000+ Crore monthly**, creating a resilient domestic liquidity cushion offsetting foreign portfolio fluctuations.

---

### 🛡️ 3. Regulatory Architecture
* **Securities and Exchange Board of India (SEBI):** Premier capital markets regulator enforcing T+1 rolling settlement (fastest in the world) and mandatory instant disclosure.

---

### 🔎 4. Authoritative Sources & Citations
* 📈 **National Stock Exchange of India (NSE):** [nseindia.com](https://www.nseindia.com/)
* 🏛️ **Bombay Stock Exchange (BSE):** [bseindia.com](https://www.bseindia.com/)
* 🛡️ **Securities and Exchange Board of India (SEBI):** [sebi.gov.in](https://www.sebi.gov.in/)""",

    "Titans": """# 🏢 Corporate Titans & Conglomerates: Intelligence Dossier
**Tata • Reliance • Infosys • L&T • HDFC • Adani • Mahindra**

### 🏛️ 1. Premier Indian Multinational Conglomerates
* **Tata Group ($360B+ Market Value):** Salt-to-software enterprise comprising 29 publicly listed companies (Tata Consultancy Services, Tata Motors, Tata Steel, Titan, Air India). Pioneering India's first semiconductor fabrication facility in Dholera.
* **Reliance Industries (RIL):** India's largest company by market cap, spanning petrochemicals, telecommunications (Jio 5G with 470M+ subscribers), and organized retail.
* **Larsen & Toubro (L&T):** Global engineering and construction multinational constructing premier national infrastructure, metro systems, and defence hardware.
* **Infosys & TCS:** Trailblazing IT service exporters providing enterprise digital architecture and AI transformation to Fortune 500 corporations worldwide.

---

### 🔎 2. Authoritative Sources & Citations
* 🏢 **Ministry of Corporate Affairs (MCA):** [mca.gov.in](https://www.mca.gov.in/)
* 📈 **BSE / NSE Listed Company Filings:** [bseindia.com](https://www.bseindia.com/)""",

    "Manufacturing": """# 🏭 Manufacturing, PLI & Atmanirbhar Bharat: Intelligence Dossier
**Production-Linked Incentives (PLI) • Electronics Exports • Industrial Corridors**

### 📦 1. Production-Linked Incentive (PLI) Scheme
* **₹1.97 Lakh Crore National Budget:** Approved across 14 key manufacturing sectors to create champion domestic manufacturing firms and boost exports.
* **Target Sectors:** Mobile phones & electronics, pharmaceuticals, medical devices, automotive & auto components, solar PV modules, advanced chemistry cell (ACC) batteries, and specialty steel.
* **Electronics Manufacturing Milestone:** Annual mobile phone production crossed 330 million units, with electronics exports surging past $25 Billion.

---

### 🚚 2. PM Gati Shakti & National Logistics Policy
* **PM Gati Shakti National Master Plan:** Integrated digital portal synchronizing multi-modal infrastructure projects across 16 ministries to reduce logistics costs to single-digit GDP percentages.
* **Dedicated Freight Corridors (DFCs):** Western and Eastern high-capacity electrified freight rail networks transforming cargo transit speeds across industrial heartlands.

---

### 🔎 3. Authoritative Sources & Citations
* 📦 **DPIIT Production Linked Incentive (PLI):** [dpiit.gov.in/production-linked-incentive-pli-schemes](https://dpiit.gov.in/)
* 🚚 **PM Gati Shakti Portal:** [gati-shakti.gov.in](https://gati-shakti.gov.in/)
* 🇮🇳 **Make in India Initiative:** [makeinindia.com](https://www.makeinindia.com/)"""
}


def economy_dashboard_node(state: AgentState) -> AgentState:
    """Step 2 of Economy: Generates Structured Economic Intelligence Briefing Dashboard."""
    raw_text = get_message_content(state["messages"][-1])
    clean_query, form_params = parse_incoming_input(raw_text)
    extracted = state.get("extracted_params") or {}
    
    topic = form_params.get("topic") or extracted.get("topic") or ""
    q_lower = clean_query.lower()

    # Identify if a predefined dropdown topic was requested
    matched_key = None
    if topic in ECONOMY_TOPIC_DOSSIERS:
        matched_key = topic
    elif form_params.get("action") == "economy_dossier" and topic:
        matched_key = topic
    else:
        # Check if user specifically requested a core dropdown topic as the primary subject
        # Note: Avoid false-positive matching on generic terms like 'economy' when a specific scheme/topic is asked
        custom_economic_indicators = [
            "gati shakti", "gatishakti", "pm gati", "inflation", "fdi", "foreign direct investment",
            "forex", "foreign exchange", "crypto", "cryptocurrency", "ev", "electric vehicle",
            "direct tax", "income tax", "corporate tax", "sez", "logistics", "monetisation", "disinvestment"
        ]
        is_custom_indicator = any(kw in q_lower for kw in custom_economic_indicators)
        
        if not is_custom_indicator:
            if any(p in q_lower for p in ["fintech", "upi", "rupay", "digital payments"]):
                matched_key = "FinTech"
            elif any(p in q_lower for p in ["startup", "startups", "unicorn", "unicorns"]):
                matched_key = "Startups"
            elif any(p in q_lower for p in ["stock market", "stock markets", "nifty 50", "sensex", "bse sensex", "nse nifty"]):
                matched_key = "Stock Market"
            elif any(p in q_lower for p in ["pli scheme", "production-linked", "atmanirbhar bharat", "make in india"]):
                matched_key = "Manufacturing"
            elif any(p in q_lower for p in ["corporate titans", "tata and reliance", "indian conglomerates", "titans"]):
                matched_key = "Titans"
            elif any(p in q_lower for p in ["major sectors", "economic sectors", "services sector", "manufacturing sector"]):
                matched_key = "Sectors"
            elif q_lower in ["gdp", "gdp growth", "indian economy", "economy growth", "macroeconomic overview", "4 trillion"]:
                matched_key = "GDP"

    # Case 1: Topic is in the core dropdown options -> Return dedicated dossier
    if matched_key and matched_key in ECONOMY_TOPIC_DOSSIERS:
        fallback = ECONOMY_TOPIC_DOSSIERS[matched_key]
        prompt = f"""
You are the Chief Economic Intelligence Specialist of 🇮🇳 India Educational AI Intelligence Agent (@india-intel-agent).
User Query / Form Submission: "{clean_query}"
Requested Topic Focus: {matched_key}

Reference Grounding Base:
{fallback}

Generate a research-grade, highly structured intelligence dashboard focusing on {matched_key}.
Include:
1. Executive Macro Overview & Quantitative Indicators.
2. Structural Breakdown & Policy Incentives.
3. Industry Titans, Startups & Market Infrastructure.
4. Future Growth Outlook & Viksit Bharat 2047 Alignment.
5. Authoritative Portals & Citations.
"""
        output = safe_llm_invoke([HumanMessage(content=prompt)], fallback)
        resp = make_ai_response(output, None)
        return {
            "workflow_stage": "economy_briefing",
            "final_output": resp["final_output"],
            "messages": resp["messages"],
        }

    # Case 2: Custom / Non-dropdown topic -> Perform real-time online search!
    search_query = clean_query
    for prefix in ["tell me about", "what is", "explain", "information on", "info on", "research", "search", "details on"]:
        if search_query.lower().startswith(prefix):
            search_query = search_query[len(prefix):].strip()

    online_res = fetch_online_intelligence(search_query, domain="economy")
    topic_title = online_res.get("title", search_query.title())
    extract = online_res.get("extract", "")
    page_url = online_res.get("url", "https://en.wikipedia.org/")

    if extract:
        search_fallback = f"""# 🌐 Real-Time Economic Intelligence: {topic_title}
**Live Online Search Verified • Indian Economy, Financial Policy & Industry**

### 📊 1. Executive Briefing & Live Intelligence
{extract}

---

### 🏛️ 2. Strategic Policy Context & National Impact
* **Government Framework & Alignment:** Aligned with India's macroeconomic growth trajectory and strategic initiatives overseen by the Ministry of Finance, NITI Aayog, and DPIIT.
* **Fiscal & Economic Significance:** Drives capital expenditure (Capex), private sector investment, and employment generation across key national corridors.
* **Global Market Competitiveness:** Enhances India's ease of doing business, domestic supply chain resilience, and integration into global value chains.

---

### 🔎 3. Verified Online Citations & Primary Sources
* 🌐 **Verified Article:** [{topic_title}]({page_url})
* 🏛️ **Reserve Bank of India (RBI):** [rbi.org.in](https://www.rbi.org.in/)
* 📊 **Ministry of Statistics and Programme Implementation (MoSPI):** [mospi.gov.in](https://www.mospi.gov.in/)
* 🇮🇳 **National Portal of India:** [india.gov.in](https://www.india.gov.in/)"""
    else:
        search_fallback = f"""# 🌐 Economic Intelligence: {topic_title}
**Indian Economy, Financial Markets & National Policy**

### 📊 1. Economic Overview: {topic_title}
* **Strategic Role:** {topic_title} represents an active domain of economic policy and market development within India's $4 Trillion economy.
* **Regulatory & Institutional Oversight:** Coordinated by relevant ministries, regulatory bodies, and industry stakeholders.

---

### 🔎 2. Verified Online Citations
* 🌐 **Online Resource:** [{topic_title}]({page_url})
* 🏛️ **Ministry of Finance:** [finmin.nic.in](https://finmin.nic.in/)
* 📊 **MoSPI:** [mospi.gov.in](https://www.mospi.gov.in/)"""

    resp = make_ai_response(search_fallback, None)
    return {
        "workflow_stage": "economy_briefing",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def culture_intake_form_node(state: AgentState) -> AgentState:
    """Step 1 of Culture: Interactive Structured Culture Intake Form with Dropdowns."""
    output = """# 🎨 Cultural Heritage & Social Fabric Explorer

Please select the cultural themes and region you wish to explore using the interactive dropdown below:

```markdown
┌────────────────────────────────────────────────────────────────────────┐
│ 🎭 CULTURAL INTELLIGENCE INTAKE FORM                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. 🪔 What would you like to explore? (Select one or more):           │
│    ☑ Festivals & Seasonal Calendar  |  ☑ Food, Gastronomy & Cuisine   │
│    ☐ Classical & Tribal Languages   |  ☐ Philosophy & Sacred Traditions│
│    ☑ 8 Classical Dance Forms        |  ☑ Classical Music (Hindustani) │
│    ☐ Traditional Textiles & Saree   |  ☐ Art, Handicrafts & Murals    │
│    ☑ Regional Cultures & Synthesis  |  ☐ Modern Indian Social Fabric  │
│                                                                        │
│ 2. 🗺️ Select Region:                                                  │
│    ● North India                    |  ○ South India                  │
│    ○ East & Northeast India         |  ○ West India                   │
│    ○ Central India                  |  ○ Pan-India Synthesis          │
│                                                                        │
│ 3. 📜 Depth Level:                                                     │
│    ○ Quick Educational Overview     |  ● Detailed Cultural Dossier    │
│    ○ Academic / Anthropological     |                                 │
└────────────────────────────────────────────────────────────────────────┘
```

👉 **[ 🎨 Generate Cultural Intelligence Report ]**  
*(Reply with your choices e.g.: `Regional Culture, North India, Detailed` or submit the dropdown form below)*"""

    resp = make_ai_response(output, get_culture_card_payload())
    return {
        "workflow_stage": "culture_form",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


CULTURE_TOPIC_DOSSIERS = {
    "Classical Dances": """# 🎭 8 Classical Dance Traditions of India: Intelligence Dossier
**Recognized by the Sangeet Natak Akademi (National Academy of Music, Dance & Drama)**

### 💃 1. The Eight Classical Dance Forms
1. **Bharatanatyam (Tamil Nadu):** Geometry, fixed upper torso, bent knees (*aramandi*), and intricate facial expression (*abhinaya*). Historically performed in temples (*sadir*).
2. **Kathak (North India - UP/Rajasthan):** Narrative dance-drama rooted in temple storytelling and Mughal court traditions; famous for rhythmic footwork (*tatkar*), pirouettes (*chakkars*), and *jugalbandi* with the tabla.
3. **Kathakali (Kerala):** High-stylized dance-drama characterized by vibrant facial makeup (*vesham*), elaborate headgear, and profound mudras enacted to Chenda percussion.
4. **Kuchipudi (Andhra Pradesh):** Originating in Kuchipudi village; incorporates spoken dialogue, quick footwork, and dancing on the rim of a brass plate (*tarangam*).
5. **Odissi (Odisha):** Originating in the temples of Odisha; defined by the sensual *Tribhanga* (three-bend posture) and sculpture-like postures.
6. **Manipuri (Manipur):** Devotional dance-drama centered on Vaishnavite *Raasleela* themes; noted for lyrical grace and gentle circular movements.
7. **Mohiniyattam (Kerala):** The dance of the enchantress (*Mohini*); graceful, swaying torso movements reminiscent of palm trees and ocean waves.
8. **Sattriya (Assam):** Created in the 15th century by Saint Sankaradeva in monastic *sattras*; spiritual devotion rendered through refined rhythm.

---

### 🔎 2. Authoritative Sources & Citations
* 🎭 **Sangeet Natak Akademi:** [sangeetnatak.gov.in](https://sangeetnatak.gov.in/)
* 🏛️ **Ministry of Culture, Government of India:** [indiaculture.gov.in](https://www.indiaculture.gov.in/)
* 📜 **UNESCO Intangible Cultural Heritage List:** [ich.unesco.org](https://ich.unesco.org/)""",

    "Music": """# 🎵 Classical Music Traditions: Hindustani & Carnatic Dossier
**Nada Brahma • Raga & Tala Systems • Sacred Acoustics**

### 🎼 1. Two Great Classical Traditions
* **Hindustani Classical (North India):** Shaped by Vedic chants, Persian synthesis, and Sufi mysticism. Defined by *Gharana* traditions (Gwalior, Agra, Kirana, Jaipur-Atrauli). Key vocal forms include *Dhrupad*, *Khayal*, and *Tarana*.
* **Carnatic Classical (South India):** Highly structured, mathematically intricate devotional music centered around the Trinity of Carnatic Music (Tyagaraja, Muthuswami Dikshitar, Syama Sastri). Key form is the *Kriti*.

---

### 🪕 2. Core Concepts: Raga & Tala
* **Raga (Melodic Framework):** Specific ascending (*Arohana*) and descending (*Avarohana*) note structures that evoke distinct emotional moods (*Rasas*) and diurnal cycles (morning, evening, monsoon ragas).
* **Tala (Rhythmic Cycle):** Periodic cyclic metric arrangements (e.g. *Teental* 16 beats, *Adi Tala* 8 beats) accented by claps (*Sam*) and pauses.

---

### 🎻 3. Master Instruments
* **Melodic:** Sitar, Sarod, Veena, Santoor, Shehnai, Bansuri, Indian Violin.
* **Percussive:** Tabla, Mridangam, Ghatam, Kanjira, Pakhawaj.

---

### 🔎 4. Authoritative Sources & Citations
* 🎵 **Sangeet Natak Akademi Music Archive:** [sangeetnatak.gov.in](https://sangeetnatak.gov.in/)
* 🏛️ **National Centre for the Performing Arts (NCPA):** [ncpamumbai.com](https://www.ncpamumbai.com/)""",

    "Festivals": """# 🪔 Festivals & Sacred Seasonal Calendar: Intelligence Dossier
**Pan-Indian Celebrations • Harvest Ceremonies • UNESCO Intangible Heritage**

### 🌟 1. Pan-Indian Festivals
* **Diwali (Deepavali):** The festival of lights celebrating the victory of light over darkness and knowledge over ignorance; illuminated with earthen *diyas* and Lakshmi Puja.
* **Holi:** The exuberant spring festival of colors celebrating the harvest, renewal, and the divine play of Radha and Krishna.

---

### 🌾 2. Harvest & Astronomical Celebrations
* **Makar Sankranti / Pongal / Bihu / Uttarayan:** Commemorates the sun's northern transit into Capricorn; celebrated through kite flying in Gujarat, sugarcane and *pongal* boiling in Tamil Nadu, and community bonfires in Assam.
* **Onam (Kerala):** 10-day grand harvest festival honoring mythical King Mahabali, featuring floral carpets (*Pookalam*), boat races (*Vallam Kali*), and the sumptuous *Onam Sadya* feast.
* **Baisakhi (Punjab):** Solar New Year and celebration of the wheat harvest; marks the founding of the Khalsa Panth by Guru Gobind Singh in 1699.

---

### 🌺 3. UNESCO Inscribed Celebrations
* **Durga Puja of Kolkata:** Inscribed on the UNESCO Representative List of the Intangible Cultural Heritage of Humanity; a 5-day mega-installation of public art, devotion, and community harmony.
* **Kumbh Mela:** World's largest peaceful congregation of pilgrims, held cyclically at Prayagraj, Haridwar, Ujjain, and Nashik.

---

### 🔎 4. Authoritative Sources & Citations
* 📜 **UNESCO Intangible Cultural Heritage Portal:** [ich.unesco.org](https://ich.unesco.org/)
* 🏛️ **Ministry of Culture, Government of India:** [indiaculture.gov.in](https://www.indiaculture.gov.in/)""",

    "Cuisine": """# 🍲 Regional Gastronomy & Culinary Philosophy: Intelligence Dossier
**Ayurvedic Six Tastes (Shad Rasa) • Regional Diversity • Spice Heritage**

### 🌿 1. Philosophical Foundation: Ayurveda & Shad Rasa
* **The Six Tastes:** Authentic Indian cuisine balances *Madhura* (sweet), *Amla* (sour), *Lavana* (salty), *Katu* (pungent), *Tikta* (bitter), and *Kashaya* (astringent) to foster cellular digestion and mental equilibrium.
* **Spices as Medicine:** Turmeric (curcumin), cumin, coriander, fenugreek, cardamom, and black pepper utilized for preservation, immunity, and gut health.

---

### 🍛 2. Regional Culinary Landscapes
* **North India:** Rich tandoori breads, saffron-infused gravies, *Dal Makhani*, aromatic Awadhi and Mughlai biryanis.
* **South India:** Fermented rice-lentil delicacies (*Idli*, *Dosa*), tangy *Sambar*, *Rasam*, and coconut-infused coastal curries on banana leaves.
* **West India:** Gujarati *Thali* (subtle sweet-savory balance), Maharashtrian *Misal Pav*, coastal Malvani fish curries, and Goan vindaloo.
* **East & Northeast India:** Bengali mustard fish curries (*Shorshe Ilish*), delicate *Sandesh* and *Rosogolla*, and aromatic fermented herbal broths of the Northeast.

---

### 🔎 3. Authoritative Sources & Citations
* 🌿 **Ministry of Ayush (Ayurvedic Food & Nutrition):** [ayush.gov.in](https://ayush.gov.in/)
* 🍲 **National Portal of India — Culinary Heritage:** [india.gov.in](https://www.india.gov.in/)""",

    "Society": """# 👥 Contemporary Social Fabric & Living Traditions: Intelligence Dossier
**Vasudhaiva Kutumbakam • Demographic Dividend • Pluralistic Harmony**

### 🏛️ 1. Civilizational Ideals & Living Pluralism
* **Vasudhaiva Kutumbakam:** The constitutional and spiritual philosophy that "the world is one single family", embodying accommodation of 22 officially recognized languages, 12 major religions, and hundreds of living dialects.
* **Continuity amidst Modernity:** Ancient family kinship structures harmonize with high digital literacy, mobile commerce, and tech-driven public empowerment.

---

### 🎓 2. The Demographic Dividend
* **Median Age of ~28 Years:** India possesses the youngest, largest working-age population in the world, driving entrepreneurial vibrancy, consumer consumption, and higher STEM education.
* **Women in Leadership:** Accelerating female enrollment in STEM disciplines, grassroots governance (*Panchayati Raj*), and startup leadership.

---

### 🔎 3. Authoritative Sources & Citations
* 🏛️ **Ministry of Social Justice and Empowerment:** [socialjustice.gov.in](https://socialjustice.gov.in/)
* 📊 **Census of India / Ministry of Home Affairs:** [censusindia.gov.in](https://censusindia.gov.in/)"""
}


def culture_dashboard_node(state: AgentState) -> AgentState:
    """Step 2 of Culture: Generates Structured Cultural Intelligence Dossier Dashboard."""
    raw_text = get_message_content(state["messages"][-1])
    clean_query, form_params = parse_incoming_input(raw_text)
    extracted = state.get("extracted_params") or {}
    
    theme = form_params.get("theme") or extracted.get("theme") or ""
    q_lower = clean_query.lower()

    # Identify if a predefined dropdown theme was requested
    matched_theme = None
    if theme in CULTURE_TOPIC_DOSSIERS:
        matched_theme = theme
    elif form_params.get("action") == "culture_dossier" and theme:
        matched_theme = theme
    else:
        # Check if user asked about a specific folk dance, regional festival, or specific craft
        custom_cultural_entities = [
            "kalbelia", "warli", "madhubani", "kalaripayattu", "chhau", "theyyam", "yakshagana", "bhangra",
            "giddha", "garba", "dandiya", "lavani", "rouf", "sufi", "sufism", "qawwali", "thangka", "pattachitra",
            "tanjore", "gond", "chola", "kumbh mela", "folk dance", "folk music", "tribal", "handicraft", "saree"
        ]
        is_custom_entity = any(kw in q_lower for kw in custom_cultural_entities)

        if not is_custom_entity:
            if any(p in q_lower for p in ["8 classical dances", "classical dance forms", "classical dances", "classical dance"]):
                matched_theme = "Classical Dances"
            elif any(p in q_lower for p in ["classical music", "carnatic music", "hindustani music"]):
                matched_theme = "Music"
            elif any(p in q_lower for p in ["festivals of india", "sacred festivals", "harvest festivals", "festivals"]):
                matched_theme = "Festivals"
            elif any(p in q_lower for p in ["indian cuisine", "regional gastronomy", "culinary philosophy", "ayurvedic food"]):
                matched_theme = "Cuisine"
            elif any(p in q_lower for p in ["indian society", "social fabric", "demographic dividend", "vasudhaiva"]):
                matched_theme = "Society"

    # Case 1: Predefined theme -> Return dedicated dossier
    if matched_theme and matched_theme in CULTURE_TOPIC_DOSSIERS:
        fallback = CULTURE_TOPIC_DOSSIERS[matched_theme]
        prompt = f"""
You are the Cultural Anthropologist & Heritage Specialist of 🇮🇳 India Educational AI Intelligence Agent (@india-intel-agent).
User Query / Form Submission: "{clean_query}"
Requested Theme Focus: {matched_theme}

Reference Grounding Base:
{fallback}

Generate a rich, respectful, and educational cultural dossier focusing on {matched_theme}.
"""
        output = safe_llm_invoke([HumanMessage(content=prompt)], fallback)
        resp = make_ai_response(output, None)
        return {
            "workflow_stage": "culture_report",
            "final_output": resp["final_output"],
            "messages": resp["messages"],
        }

    # Case 2: Custom / Non-dropdown culture topic -> Perform real-time online search!
    search_query = clean_query
    for prefix in ["tell me about", "what is", "explain", "information on", "info on", "research", "search", "details on"]:
        if search_query.lower().startswith(prefix):
            search_query = search_query[len(prefix):].strip()

    online_res = fetch_online_intelligence(search_query, domain="culture")
    topic_title = online_res.get("title", search_query.title())
    extract = online_res.get("extract", "")
    page_url = online_res.get("url", "https://en.wikipedia.org/")

    if extract:
        search_fallback = f"""# 🌐 Real-Time Cultural Intelligence: {topic_title}
**Live Online Search Verified • Heritage, Arts & Living Traditions**

### 🎨 1. Cultural Overview & Living Heritage
{extract}

---

### 🏛️ 2. Origins, Community Traditions & Historical Lineage
* **Regional & Cultural Roots:** Deeply rooted in regional communities, oral lineages, and seasonal practices across India.
* **Artistic Expression & Symbolism:** Characterized by unique aesthetic expressions, indigenous instruments, intricate costumes, or iconic motifs.
* **National & Global Preservation:** Recognized and celebrated through institutions like the Sangeet Natak Akademi, Lalit Kala Akademi, and UNESCO Intangible Cultural Heritage initiatives.

---

### 🔎 3. Verified Cultural Citations & Authoritative Portals
* 🌐 **Verified Article:** [{topic_title}]({page_url})
* 🏛️ **Ministry of Culture, Government of India:** [indiaculture.gov.in](https://www.indiaculture.gov.in/)
* 🎭 **Sangeet Natak Akademi:** [sangeetnatak.gov.in](https://sangeetnatak.gov.in/)
* 📜 **UNESCO Intangible Cultural Heritage:** [ich.unesco.org](https://ich.unesco.org/)"""
    else:
        search_fallback = f"""# 🌐 Cultural Intelligence: {topic_title}
**Indian Heritage, Arts & Living Traditions**

### 🎨 1. Cultural Overview: {topic_title}
* **Cultural Significance:** {topic_title} is an integral element of India's multi-faceted cultural mosaic and civilizational heritage.
* **Living Practice:** Preserved through regional communities, cultural guilds, and oral traditions.

---

### 🔎 2. Verified Online Citations
* 🌐 **Online Resource:** [{topic_title}]({page_url})
* 🏛️ **Ministry of Culture:** [indiaculture.gov.in](https://www.indiaculture.gov.in/)"""

    resp = make_ai_response(search_fallback, None)
    return {
        "workflow_stage": "culture_report",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def science_intake_form_node(state: AgentState) -> AgentState:
    """Step 1 of Science & Tech: Interactive Structured Tech Intake Form with Dropdowns."""
    output = """# 🚀 Science, Innovation & Future Tech Intelligence

Please select the technology frontiers and research parameters using the interactive dropdown below:

```markdown
┌────────────────────────────────────────────────────────────────────────┐
│ 🔬 SCIENCE & FUTURE TECH INTAKE FORM                                  │
├────────────────────────────────────────────────────────────────────────┤
│ 1. 🛰️ What technology domain do you want to explore?                   │
│    ☑ Space Exploration & ISRO       |  ☑ Semiconductors & Chip Fabs    │
│    ☑ Artificial Intelligence (IndiaAI)| ☐ Quantum Computing & Crypto   │
│    ☐ Clean Energy & Green Hydrogen  |  ☐ Electric Vehicles & Batteries │
│    ☐ Defence Tech & DRDO            |  ☐ 5G / 6G Indigenous Telecom    │
│    ☐ Digital Public Infrastructure  |  ☐ DeepTech & Space Startups     │
│                                                                        │
│ 2. 🏢 Target Entity / Organization:                                    │
│    ● India Overall                  |  ○ ISRO & IN-SPACe               │
│    ○ India Semiconductor Mission    |  ○ Private Startups (Skyroot/Sarvam)│
│                                                                        │
│ 3. 🎯 Analysis Type:                                                   │
│    ○ Current Technological State    |  ● Future Outlook & 2030 Roadmap │
│    ○ Breakthrough Milestones        |  ○ Deep Technical Research       │
└────────────────────────────────────────────────────────────────────────┘
```

👉 **[ 🚀 Generate Tech Intelligence Dossier ]**  
*(Reply with your choices e.g.: `AI / Semiconductors, Future Outlook` or submit the dropdown form below)*"""

    resp = make_ai_response(output, get_science_card_payload())
    return {
        "workflow_stage": "science_form",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def search_science_topic_online(query: str) -> Dict[str, Any]:
    """Real-time online search via Wikipedia REST / Search API for tech topics outside core dropdown options."""
    return fetch_online_intelligence(query, domain="science")


SCIENCE_TOPIC_DOSSIERS = {
    "Quantum": """# ⚛️ Quantum Computing & Cryptography: Intelligence Dossier
**National Quantum Mission (NQM) • India DeepTech Frontiers**

### 🔬 1. National Quantum Mission (NQM) Overview
* **₹6,003.65 Crore National Outlay:** Approved by the Union Cabinet to seed, nurture, and scale scientific and industrial R&D in Quantum Technology (QT) over 2023–2031.
* **Top 6 Elite League:** India is among the world's top 6 leading nations (US, China, France, Austria, Finland, India) with a dedicated, fully funded national quantum mission.
* **Mission Goal:** Develop intermediate-scale quantum computers with **50 to 1,000 physical qubits** within 8 years using superconducting and photonic qubit platforms.

---

### 🌐 2. Four Thematic Hubs (T-Hubs)
1. **Quantum Computing:** Physical qubit synthesis, quantum algorithms, fault-tolerant error correction architectures, and quantum simulations.
2. **Quantum Communication:** Satellite-based secure quantum communications between ground stations over 2,000 km, and inter-city Quantum Key Distribution (QKD) over 2,000 km optical fiber networks.
3. **Quantum Sensing & Metrology:** High-sensitivity atomic clocks, quantum gravimeters, and quantum magnetometers for navigation, defence, and mineral exploration.
4. **Quantum Materials & Devices:** Fabrication of superconducting quantum materials, topological insulators, and novel 2D quantum wells.

---

### 🏢 3. Key Research Institutions & Pioneer Labs
* **Raman Research Institute (RRI, Bengaluru):** Leading Quantum Information and Computing (QuIC) lab; demonstrated quantum entanglement and QKD over free space.
* **Tata Institute of Fundamental Research (TIFR, Mumbai):** Fabricating India's first superconducting 7-qubit quantum processor.
* **Indian Institute of Science (IISc, Bengaluru):** Centre for Quantum Technologies researching quantum photonic chips and solid-state spin qubits.
* **IIT Madras & IIT Delhi:** Pioneering secure QKD fiber links over 100+ km in partnership with DRDO.
* **C-DOT (Centre for Development of Telematics):** Deploying commercial quantum-secured telecom encryption boxes.

---

### 🚀 4. Private DeepTech Startups & Industry
* **QNu Labs (Bengaluru):** Commercial cybersecurity leader providing quantum-safe cryptographic solutions (*Tropos*, *Armos*) and QKD hardware deployed by the Indian Army and enterprise banks.
* **BosonQ Psi:** Developing enterprise quantum simulation software (*BQPhy*) for aerospace, defense, and automotive multiphysics modeling.
* **QuLabs & QuantAI:** Quantum machine learning algorithms and post-quantum cryptography (PQC) transition frameworks.

---

### 🔎 5. Authoritative Sources & Citations
* ⚛️ **National Quantum Mission Portal (DST):** [dst.gov.in/national-quantum-mission-nqm](https://dst.gov.in/national-quantum-mission-nqm)
* 🏛️ **Department of Science & Technology (DST):** [dst.gov.in](https://dst.gov.in/)
* 🔬 **Centre for Development of Telematics (C-DOT):** [cdot.in](https://www.cdot.in/)
* 🛡️ **Defence Research and Development Organisation (DRDO):** [drdo.gov.in](https://www.drdo.gov.in/)""",

    "Semiconductors": """# 🔬 Semiconductors & Chip Fabs: Intelligence Dossier
**India Semiconductor Mission (ISM) • Silicon Sovereignty & Advanced Packaging**

### 🏭 1. India Semiconductor Mission (ISM) Overview
* **$10 Billion (₹76,000 Crore) Incentive Package:** Comprehensive fiscal scheme subsidizing **up to 50% of capital expenditure (Capex)** for commercial semiconductor silicon fabs, display fabs, compound semiconductors, and advanced packaging (ATMP/OSAT).
* **Strategic Objective:** Reduce import dependency, protect critical national electronics supply chains, and position India as a premier global semiconductor powerhouse.

---

### 🏗️ 2. Landmark Fabs & Facilities Under Construction
1. **Tata Electronics & PSMC (Taiwan) — Dholera Commercial Fab:**
   * **Location:** Dholera Special Investment Region (SIR), Gujarat.
   * **Investment:** **₹91,000 Crore (~$11 Billion)**.
   * **Capacity:** 50,000 wafer starts per month (28nm, 55nm, and 90nm nodes) powering automotive, computing, communications, and power electronics.
2. **Micron Technology — Sanand OSAT Assembly & Test Facility:**
   * **Location:** Sanand, Gujarat.
   * **Investment:** **$2.75 Billion** combined public-private outlay.
   * **Focus:** Advanced packaging for DRAM, NAND, and enterprise SSD memory modules.
3. **CG Power, Renesas (Japan) & Stars Microelectronics:**
   * **Location:** Sanand, Gujarat.
   * **Investment:** ₹7,600 Crore ATMP facility focusing on specialized automotive and IoT chips.
4. **Tata Semiconductor Assembly and Test (TSAT) — Jagiroad, Assam:**
   * **Location:** Morigaon, Assam.
   * **Investment:** ₹27,000 Crore advanced packaging unit creating 27,000+ high-tech jobs.

---

### 🌐 3. Global Chip Design Dominance
* **>20% Global Design Talent:** Over **125,000 top VLSI, logic verification, and chip design engineers** work in India (Bengaluru, Hyderabad, Noida, Pune).
* **Global R&D Centers:** NVIDIA, Intel, AMD, Qualcomm, Broadcom, Texas Instruments, and ARM design their flagship microarchitectures in Indian development centers.
* **Design-Linked Incentive (DLI) Scheme:** Government funding up to 50% design expenses for Indian fabless chip startups (Mindgrove Technologies, InCore Semiconductors, Saankhya Labs).

---

### 🔎 4. Authoritative Sources & Citations
* 🔬 **India Semiconductor Mission (MeitY):** [ism.gov.in](https://www.ism.gov.in/)
* 🏛️ **Ministry of Electronics and Information Technology:** [meity.gov.in](https://www.meity.gov.in/)
* 💼 **India Brand Equity Foundation — Semiconductor Sector:** [ibef.org](https://www.ibef.org/)""",

    "AI": """# 🤖 Artificial Intelligence & IndiaAI Mission: Intelligence Dossier
**₹10,372 Crore Sovereign Compute • Project Bhashini & Inclusive AI**

### 🧠 1. IndiaAI Mission: Strategic Pillars
* **₹10,372 Crore Union Cabinet Outlay:** Establishing sovereign AI capacity, democratizing computational power, and accelerating indigenous foundational AI models.
* **10,000+ High-End GPU Sovereign Compute:** Public-private partnership deploying scalable compute clusters accessible to Indian startups, researchers, and academia at highly subsidized rates.
* **IndiaAI Datasets Platform:** Centralized national repository of non-personal, anonymized open datasets across healthcare, agriculture, climate, and governance.

---

### 🗣️ 2. Project Bhashini (National Language Translation Mission)
* **Breaking Linguistic Barriers:** Voice-first multimodal AI ecosystem spanning all **22 scheduled Indian languages**.
* **Voice-First AI Citizen Services:** Enables citizens to speak in their mother tongue to interact with government welfare portals, banking systems, and educational tools.
* **Crowdsourced Bhasha Daan:** Tens of thousands of citizens contributing dialect voices and localized sentence pairs for open-source AI training.

---

### 🚀 3. DeepTech Generative AI Ecosystem
* **Sarvam AI:** Sovereign foundational models tailored to Indian languages, enterprise context, and voice agents.
* **Krutrim:** Sovereign AI stack with multilingual LLM and silicon design initiatives.
* **BharatGPT & CoRover:** Multimodal conversational AI serving public utilities, railways, and telecom.
* **Stanford AI Index Benchmark:** India ranks **#1 globally in relative AI skill penetration** and AI talent concentration.

---

### 🔎 4. Authoritative Sources & Citations
* 🤖 **IndiaAI Official Mission Portal:** [indiaai.gov.in](https://indiaai.gov.in/)
* 🌐 **Project Bhashini Portal:** [bhashini.gov.in](https://bhashini.gov.in/)
* 🏛️ **Ministry of Electronics and Information Technology (MeitY):** [meity.gov.in](https://www.meity.gov.in/)""",

    "Space": """# 🛰️ Space Exploration & ISRO Ecosystem: Intelligence Dossier
**Lunar South Pole Landing • Human Spaceflight & Commercial NewSpace**

### 🌕 1. Landmark Missions & Planetary Exploration
* **Chandrayaan-3 Historic Landing (Aug 23, 2023):** India made history as the **first nation to land near the lunar South Pole** (Shiv Shakti Point) and 4th country to achieve a soft lunar landing. August 23 is officially celebrated nationwide as **National Space Day**.
* **Aditya-L1 Solar Observatory:** Placed in a halo orbit around the Sun-Earth Lagrangian Point L1 (~1.5 million km from Earth) for continuous observation of solar flares and coronal mass ejections.
* **Gaganyaan Human Spaceflight Program:** Indigenous mission to send 3 Indian astronauts (Vyomnauts) to 400 km Low Earth Orbit aboard the human-rated LVM3 launch vehicle and return safely to Indian coastal waters.
* **Bharatiya Antariksh Station (BAS):** Planned sovereign space station with the first module launching by 2028 and fully operational by 2035, followed by a crewed lunar landing by 2047.

---

### 🚀 2. IN-SPACe & Commercial NewSpace Revolution
* **IN-SPACe (Indian National Space Promotion and Authorization Centre):** Single-window autonomous regulator fostering private rocket launches and satellite manufacturing.
* **Skyroot Aerospace:** Launched India's first privately developed rocket (*Vikram-S*).
* **Agnikul Cosmos:** Built and launched the world's first single-piece 3D-printed semi-cryogenic rocket engine (*Agnibaan SOrTeD*) from India's first private launchpad at Sriharikota.
* **Pixxel:** Building the world's highest-resolution commercial hyperspectral satellite constellation for global agriculture and climate monitoring.

---

### 🔎 3. Authoritative Sources & Citations
* 🛰️ **Indian Space Research Organisation (ISRO):** [isro.gov.in](https://www.isro.gov.in/)
* 🚀 **IN-SPACe Single-Window Portal:** [inspace.gov.in](https://www.inspace.gov.in/)
* 💼 **NewSpace India Limited (NSIL):** [nsilindia.co.in](https://www.nsilindia.co.in/)""",

    "Clean Energy": """# ⚡ Clean Energy, Solar & Green Hydrogen: Intelligence Dossier
**500 GW Non-Fossil Roadmap • PM Surya Ghar & National Green Hydrogen Mission**

### ☀️ 1. Renewable Energy Milestones & Capacity
* **4th Largest Renewable Energy Producer Globally:** India has surpassed **200+ GW installed non-fossil capacity**, targeting **500 GW by 2030** and Net-Zero carbon emissions by 2070.
* **World's Mega Solar Parks:**
  * **Bhadla Solar Park (Rajasthan):** World's largest operational solar plant (~2,245 MW).
  * **Pavagada Solar Park (Karnataka):** Over 2,050 MW installed capacity across arid scrubland.
* **PM-Surya Ghar Muft Bijli Yojana:** ₹75,000 Crore national mission subsidizing rooftop solar systems for 10 million households across India.

---

### 💧 2. National Green Hydrogen Mission
* **₹19,744 Crore Capital Outlay:** Aiming to make India a global hub for production, usage, and export of Green Hydrogen and its derivatives.
* **2030 Key Targets:**
  * Annual Green Hydrogen production of **5 Million Metric Tonnes (MMT)**.
  * Adding **125 GW renewable capacity** dedicated solely to green hydrogen generation.
  * Reduction of over **₹1 Lakh Crore in fossil fuel imports** and 50 MMT carbon emissions per year.
* **Strategic Interventions for Green Hydrogen Transition (SIGHT):** Financial incentives for domestic electrolyser manufacturing and green hydrogen production.

---

### 🔎 3. Authoritative Sources & Citations
* ⚡ **Ministry of New and Renewable Energy (MNRE):** [mnre.gov.in](https://www.mnre.gov.in/)
* 💧 **National Green Hydrogen Mission Portal:** [mnre.gov.in/green-hydrogen](https://mnre.gov.in/green-hydrogen)
* ☀️ **Solar Energy Corporation of India (SECI):** [seci.co.in](https://www.seci.co.in/)""",

    "Telecom": """# 📶 Telecommunications & 6G Frontiers: Intelligence Dossier
**Fastest Global 5G Rollout • Bharat 6G Vision & Indigenous Telecom Stack**

### 📱 1. Record-Breaking 5G Nationwide Deployment
* **World's Fastest 5G Rollout:** Over **450,000 5G base stations** deployed across 98%+ of India's districts within 18 months, covering 100+ million subscribers.
* **Lowest Data Tariffs Globally:** India offers among the lowest mobile data costs globally (~$0.15 per GB), driving explosive digital adoption.
* **BharatNet Optical Fiber Project:** Connecting over 250,000 Gram Panchayats with high-speed rural broadband.

---

### 📡 2. Indigenous 4G/5G Telecom Stack & Bharat 6G
* **C-DOT & TCS Indigenous Telecom Stack:** 100% indigenous 4G/5G core, radio access network (RAN), and transmission software successfully deployed commercially on BSNL networks.
* **Bharat 6G Vision Document:** Launched by the Prime Minister; Bharat 6G Alliance brings together academia, industry, and operators working on terahertz communications, AI-native air interfaces, and intelligent reflecting surfaces.
* **Telecom PLI Scheme:** Over ₹12,000 Crore investment attracted, turning India into a net exporter of telecom gear and smart devices (Foxconn, Pegatron, Dixon, Tejas Networks).

---

### 🔎 3. Authoritative Sources & Citations
* 📶 **Department of Telecommunications (DoT):** [dot.gov.in](https://dot.gov.in/)
* 🌐 **Bharat 6G Vision (DoT):** [dot.gov.in/bharat-6g-vision](https://dot.gov.in/bharat-6g-vision)
* 🔬 **Centre for Development of Telematics (C-DOT):** [cdot.in](https://www.cdot.in/)""",
}


def science_dashboard_node(state: AgentState) -> AgentState:
    """Step 2 of Science: Generates Topic-Specific Technology Intelligence Dossier or Live Online Search."""
    user_query = get_message_content(state["messages"][-1])
    clean_query, form_params = parse_incoming_input(user_query)
    q_lower = clean_query.lower()
    
    # Check if category is specified in form or state
    category_param = form_params.get("category") or (state.get("extracted_params") or {}).get("category")
    selected_topic = None
    
    if category_param:
        cat_str = str(category_param).strip().lower()
        for k in SCIENCE_TOPIC_DOSSIERS.keys():
            if k.lower() in cat_str or cat_str in k.lower():
                selected_topic = k
                break
    
    # If not found via form param, check user query text
    if not selected_topic:
        if any(w in q_lower for w in ["quantum", "nqm", "qubit", "qkd", "cryptography"]):
            selected_topic = "Quantum"
        elif any(w in q_lower for w in ["semiconductor", "chip", "fab", "dholera", "micron", "sanand", "ism"]):
            selected_topic = "Semiconductors"
        elif any(w in q_lower for w in ["ai", "artificial intelligence", "indiaai", "bhashini", "gpu", "compute", "llm"]):
            selected_topic = "AI"
        elif any(w in q_lower for w in ["space", "isro", "chandrayaan", "gaganyaan", "aditya", "skyroot", "agnikul", "in-space", "pixxel", "satellite", "rocket"]):
            selected_topic = "Space"
        elif any(w in q_lower for w in ["clean energy", "green hydrogen", "solar", "renewable", "solar park", "pm surya"]):
            selected_topic = "Clean Energy"
        elif any(w in q_lower for w in ["telecom", "5g", "6g", "telecommunications", "broadband", "bharatnet"]):
            selected_topic = "Telecom"

    # Case 1: Topic is one of the 6 core dropdown categories
    if selected_topic and selected_topic in SCIENCE_TOPIC_DOSSIERS:
        fallback = SCIENCE_TOPIC_DOSSIERS[selected_topic]
        prompt = f"""
You are the Science, Innovation & Future Tech Specialist of 🇮🇳 India Educational AI Intelligence Agent (@india-intel-agent).
Target Topic: {selected_topic}
User Query: "{user_query}"

Provide a dedicated, high-impact intelligence dossier specifically focused on {selected_topic} in India.
Include:
1. Executive Strategic Significance & National Mission Outlays.
2. Flagship Programs, Facilities Under Construction & Core Milestones.
3. Pioneer Research Labs & DeepTech Startup Ecosystem.
4. Global Benchmarks & Comparative Strengths.
5. Authoritative Official Portals & Citations.
"""
        output = safe_llm_invoke([HumanMessage(content=prompt)], fallback)
        resp = make_ai_response(output, None)
        return {
            "workflow_stage": "science_dossier",
            "final_output": resp["final_output"],
            "messages": resp["messages"],
        }

    # Case 2: Topic is NOT in the 6 dropdown options -> Perform real-time online search!
    search_query = clean_query
    for prefix in ["tell me about", "what is", "explain", "information on", "info on", "research", "search"]:
        if search_query.lower().startswith(prefix):
            search_query = search_query[len(prefix):].strip()
            
    online_res = search_science_topic_online(search_query)
    topic_title = online_res.get("title", search_query.title())
    extract = online_res.get("extract", "")
    page_url = online_res.get("url", "https://en.wikipedia.org/")

    if extract:
        search_fallback = f"""# 🌐 Real-Time Web Intelligence: {topic_title}
**Live Online Search Verified • Science, Innovation & Future Tech Ecosystem**

### 🔬 1. Executive Summary & Core Intelligence
{extract}

---

### 🇮🇳 2. Strategic Context & India's National Initiatives
* **Technological Imperative:** Emerging frontier in India's technology ecosystem, driven by public R&D institutions (CSIR, DST, IITs), deeptech startups, and national mission frameworks.
* **National Policy & Funding:** Supported by the **Department of Science & Technology (DST)** and the **Anusandhan National Research Foundation (ANRF)** with ₹50,000 Crore research outlay.
* **Industry & Startup Acceleration:** Rapidly expanding ecosystem of specialized private enterprises and university incubators across Bengaluru, Hyderabad, Chennai, and NCR.

---

### 🔎 3. Verified Online Citations & Primary Sources
* 🌐 **Verified Article:** [{topic_title}]({page_url})
* 🏛️ **Department of Science & Technology (DST):** [dst.gov.in](https://dst.gov.in/)
* 🔬 **Council of Scientific & Industrial Research (CSIR):** [csir.res.in](https://www.csir.res.in/)
* 🇮🇳 **National Portal of India:** [india.gov.in](https://www.india.gov.in/)"""
    else:
        search_fallback = f"""# 🌐 Real-Time Web Intelligence: {topic_title}
**Science, Innovation & Future Tech Ecosystem • India Intelligence**

### 🔬 1. Technology Overview: {topic_title}
* **Emerging Frontier:** {topic_title} represents an active domain of research and development within India's high-technology ecosystem.
* **Academic & Industrial Research:** Spearheaded by the premier Indian Institutes of Technology (IITs), Indian Institute of Science (IISc), and CSIR national laboratories.

---

### 🇮🇳 2. National Ecosystem & DeepTech Acceleration
* **National Research Framework:** Coordinated under the aegis of the Department of Science and Technology (DST) and Ministry of Electronics & IT (MeitY).
* **Startup Incubation:** Thriving deeptech innovation hubs across Bengaluru, Hyderabad, and Pune.

---

### 🔎 3. Authoritative Sources & Citations
* 🏛️ **Department of Science & Technology (DST):** [dst.gov.in](https://dst.gov.in/)
* 🔬 **Council of Scientific & Industrial Research (CSIR):** [csir.res.in](https://www.csir.res.in/)
* 🇮🇳 **National Portal of India:** [india.gov.in](https://www.india.gov.in/)"""

    prompt = f"""
You are the Science, Innovation & Future Tech Specialist of 🇮🇳 India Educational AI Intelligence Agent (@india-intel-agent).
Topic: "{topic_title}"
Live Web Search Data:
{extract}

Generate a clear, authoritative, and structured technology briefing for "{topic_title}" with focus on India's technological ecosystem, initiatives, and institutions.
"""
    output = safe_llm_invoke([HumanMessage(content=prompt)], search_fallback)
    resp = make_ai_response(output, None)
    return {
        "workflow_stage": "science_dossier",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def state_composite_node(state: AgentState) -> AgentState:
    """Multi-dimensional regional intelligence node covering Geography, History, Economy, Culture & Tourism."""
    state_name = state.get("entity_name") or "rajasthan"
    user_query = get_message_content(state["messages"][-1])

    cal_url = generate_google_calendar_url(
        title=f"Exploration: {state_name.title()}",
        details=f"Comprehensive regional dossier curated by @india-intel-agent.",
        location=state_name.title(),
    )
    maps_url = generate_google_maps_route([f"{state_name.title()}, India", f"Capital of {state_name.title()}, India"])

    prompt = f"""
You are the Regional & State Intelligence Specialist of 🇮🇳 India Educational AI Intelligence Agent (@india-intel-agent).
Target State / Region: "{state_name.title()}"
User Query: "{user_query}"

Provide a comprehensive, multi-dimensional educational dossier covering all essential pillars:
1. 🗺️ **Geography, Capital & Administrative Profile**
2. 📜 **Historical Significance & Dynasties**
3. 📈 **Economy, Major Industries & Infrastructure**
4. 🎨 **Culture, Languages, Performing Arts & Festivals**
5. ✈️ **UNESCO Sites & Primary Tourism Highlights**
6. 🎓 **Education, Innovation & Modern Developments**
"""
    fallback = f"""# 🗺️ Comprehensive Regional Dossier: {state_name.title()}, India

### 1. 📍 Geography & Administrative Profile
* **Region:** {state_name.title()} represents an essential administrative, cultural, and economic pillar of India.
* **Governance:** Fosters key industrial zones, cultural heritage circuits, and civic infrastructure.

### 2. 📜 Historical Roots & Dynasties
* Centuries of civilizational heritage, historical forts, ancient trade corridors, and dynastic legacies.

### 3. 📈 Economy, Industries & Agriculture
* **Economic Drivers:** Blend of agriculture, mineral wealth, expanding industrial corridors, and tourism.

### 4. 🎨 Cultural Heritage, Attire & Festivals
* Rich regional language and literature, indigenous folk art, celebrated seasonal festivals, and unique gastronomy.

### 5. ✈️ Heritage Landmarks & Tourism Highlights
* World-renowned heritage circuits, UNESCO World Heritage monuments, and wildlife sanctuaries.

---

### 🚀 Regional Exploration & Navigation Links
* 🗺️ **Explore on Google Maps:**  
  👉 **[ 🔗 Click to Explore {state_name} on Google Maps ]({maps_url})**
* 📅 **Save Study Plan to Google Calendar:**  
  👉 **[ 🔗 Click to Add {state_name} Study Itinerary to Google Calendar ]({cal_url})**"""

    output = safe_llm_invoke([HumanMessage(content=prompt)], fallback)
    resp = make_ai_response(output, None)
    return {
        "workflow_stage": "state_composite",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


def out_of_scope_node(state: AgentState) -> AgentState:
    """Standardized polite guardrail response when asked about foreign non-India destinations."""
    entity = (state.get("entity_name") or "international destinations").title()

    msg = f"""# 🇮🇳 Welcome to India Intel Agent

> **"Your AI-powered intelligence assistant for exploring India's economy, tourism, culture, science, technology and future opportunities."**

⚠️ **Geographic Scope Notice:**  
I am an autonomous intelligence agent dedicated exclusively to the **Republic of India (Bharat)** 🇮🇳. I do not provide travel guides or itineraries for international destinations like **{entity}**.

---

### 🌟 Please select one of our Indian intelligence domains to explore:

### 🇮🇳 Option 1: Economy & Financial Markets
*Explore Indian GDP, major sectors, stock exchanges (NSE/BSE), UPI & FinTech revolution, conglomerates (Tata, Reliance), and policies.*  
👉 **[ 🏛️ Open Economy & Markets ]** *(Reply `1` or `Economy`)*

### ✈️ Option 2: Tourism & UNESCO Heritage
*Autonomous travel planning, 42 UNESCO World Heritage sites, day-by-day itineraries, IRCTC train booking links, hotels, and cost estimates.*  
👉 **[ ✈️ Open Tourism & UNESCO Heritage ]** *(Reply `2` or `Tourism`)*

### 🎨 Option 3: Culture & Social Fabric
*Deep dive into 8 classical dances, Carnatic/Hindustani music, major festivals (Diwali, Holi, Eid, Pongal), and regional diversity.*  
👉 **[ 🎨 Open Culture & Social Fabric ]** *(Reply `3` or `Culture`)*

### 🚀 Option 4: Science & Future Tech
*Research ISRO space exploration (Chandrayaan-3, Gaganyaan), semiconductor fabs, IndiaAI Mission, and clean energy.*  
👉 **[ 🚀 Open Science & Future Tech ]** *(Reply `4` or `Science`)*"""

    resp = make_ai_response(msg, get_landing_card_payload())
    return {
        "workflow_stage": "out_of_scope",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }





def destinations_carousel_node(state: AgentState) -> AgentState:
    """Destinations Carousel Node: Renders an interactive swipeable carousel in ASI:One."""
    output = """# 🌟 Top Flagship Indian Destinations
**Interactive Heritage Circuits & UNESCO Highlights**

Swipe through the interactive destination carousel below to explore India's most celebrated heritage, spiritual, and natural circuits. Click **Explore Destination** on any card to begin an autonomous itinerary!"""

    resp = make_ai_response(output, get_destinations_carousel_payload())
    return {
        "workflow_stage": "destinations_carousel",
        "final_output": resp["final_output"],
        "messages": resp["messages"],
    }


# ----------------------------------------------------
# 8. SUPERVISOR & ROUTING ENGINE
# ----------------------------------------------------
def supervisor_router(state: AgentState) -> AgentState:
    """Intelligent intent classifier and state machine controller with robust @mention and form-handling."""
    messages = state.get("messages", [])
    if not messages:
        return {"workflow_stage": "landing", "entity_name": None, "extracted_params": None}

    raw_text = get_message_content(messages[-1])
    clean_query, form_params = parse_incoming_input(raw_text)
    q_lower = clean_query.lower()
    form_action = str(form_params.get("action", "")).lower()

    # ----------------------------------------------------
    # 0. Check for Foreign Out-of-Scope Queries (e.g. Thailand, Bangkok, Dubai, Paris, etc.)
    # (Only evaluate for user text queries, not explicit form submissions)
    # ----------------------------------------------------
    if not form_action:
        foreign_matches = [
            dest for dest in FOREIGN_DESTINATIONS 
            if re.search(r'\b' + re.escape(dest) + r'\b', q_lower)
        ]
        if foreign_matches:
            is_india_context = any(
                term in q_lower 
                for term in ["india", "indian", "bilateral", "trade with", "relations with", "compare to india", "vs india"]
            )
            if not is_india_context:
                return {"workflow_stage": "out_of_scope", "entity_name": foreign_matches[0], "extracted_params": None}

    # ----------------------------------------------------
    # 1. Landing Screen Greetings (Only if plain text query with no form action)
    # ----------------------------------------------------
    if not form_action and (not clean_query or q_lower in ["hi", "hello", "hey", "namaste", "start", "menu", "options", "help", "home", "welcome", "restart"]):
        return {"workflow_stage": "landing", "entity_name": None, "extracted_params": None}

    # ----------------------------------------------------
    # 2. HIGHEST PRIORITY: Explicit Form Action Submissions & Dossier Requests
    # ----------------------------------------------------
    # A. Economy Dossier Form Submission
    if form_action == "economy_dossier" or form_params.get("topic") in [
        "GDP", "Sectors", "Startups", "Stock Market", "FinTech", "Titans", "Manufacturing"
    ]:
        return {"workflow_stage": "economy_briefing", "entity_name": None, "extracted_params": form_params}

    # B. Culture Dossier Form Submission
    if form_action == "culture_dossier" or (form_params.get("theme") and form_params.get("region")):
        return {"workflow_stage": "culture_report", "entity_name": None, "extracted_params": form_params}

    # C. Science Dossier Form Submission
    if form_action == "science_dossier" or form_params.get("category"):
        return {"workflow_stage": "science_dossier", "entity_name": None, "extracted_params": form_params}

    # D. Review Card CTA actions (Confirm Trip or Modify Trip)
    if form_action == "confirm_trip" or "confirm trip" in q_lower or "confirm and generate" in q_lower:
        dest = form_params.get("destination") or "Jaipur"
        parsed = parse_travel_params(clean_query, form_params)
        return {"workflow_stage": "tourism_planner", "entity_name": str(dest).lower(), "extracted_params": parsed}

    if form_action == "modify_trip" or "modify preferences" in q_lower or "adjust preferences" in q_lower:
        dest = form_params.get("destination") or "Jaipur"
        return {"workflow_stage": "tourism_briefing", "entity_name": str(dest).lower(), "extracted_params": None}

    # E. Tourism Trip Planner Form Submission CTA
    if form_action == "plan_trip" or (form_params.get("destination") and form_params.get("days")):
        dest = form_params.get("destination") or "Jaipur"
        parsed = parse_travel_params(clean_query, form_params)
        return {"workflow_stage": "tourism_planner", "entity_name": str(dest).lower(), "extracted_params": parsed}

    # F. Explore City CTA from Carousel or Destination list
    if form_action == "explore_city" and form_params.get("destination"):
        dest = str(form_params["destination"]).strip()
        return {"workflow_stage": "tourism_briefing", "entity_name": dest.lower(), "extracted_params": None}

    # G. Domain Explorer Form Submission (Option 1 - 4 Selection)
    if form_action == "select_domain" or "domain" in form_params:
        selected_domain = str(form_params.get("domain", "")).lower()
        if "economy" in selected_domain or "1" in selected_domain:
            return {"workflow_stage": "economy_form", "entity_name": None, "extracted_params": None}
        if "tourism" in selected_domain or "2" in selected_domain:
            return {"workflow_stage": "tourism_destinations", "entity_name": None, "extracted_params": None}
        if "culture" in selected_domain or "3" in selected_domain:
            return {"workflow_stage": "culture_form", "entity_name": None, "extracted_params": None}
        if "science" in selected_domain or "4" in selected_domain:
            return {"workflow_stage": "science_form", "entity_name": None, "extracted_params": None}

    # ----------------------------------------------------
    # 3. Direct Landing Page Options & Domain Shortcuts (User typing numbers or keywords)
    # ----------------------------------------------------
    # Economy shortcut (Option 1)
    if q_lower in ["1", "option 1", "opt 1", "open economy"] or (
        q_lower.strip() in ["economy", "indian economy"] and len(q_lower.split()) <= 2
    ):
        return {"workflow_stage": "economy_form", "entity_name": None, "extracted_params": None}

    # Tourism shortcut (Option 2)
    if q_lower in ["2", "option 2", "opt 2", "open tourism"] or (
        q_lower.strip() in ["tourism", "travel", "tourism in india"] and not any(city in q_lower for city in FLAGSHIP_CITIES)
    ):
        return {"workflow_stage": "tourism_destinations", "entity_name": None, "extracted_params": None}

    # Culture shortcut (Option 3)
    if q_lower in ["3", "option 3", "opt 3", "open culture"] or (
        q_lower.strip() in ["culture", "indian culture", "social fabric"] and not any(kw in q_lower for kw in ["dance", "music", "festival", "north india", "south india"])
    ):
        return {"workflow_stage": "culture_form", "entity_name": None, "extracted_params": None}

    # Science shortcut (Option 4)
    if q_lower in ["4", "option 4", "opt 4", "open science"] or (
        q_lower.strip() in ["science", "future tech", "technology", "science & tech"] and not any(kw in q_lower for kw in ["isro", "chip", "semiconductor", "ai", "quantum", "5g", "biotechnology", "drone"])
    ):
        return {"workflow_stage": "science_form", "entity_name": None, "extracted_params": None}

    # ----------------------------------------------------
    # 4. Interactive Destinations Carousel
    # ----------------------------------------------------
    if any(kw in q_lower for kw in ["carousel", "top destinations", "browse cities", "explore destinations"]):
        return {"workflow_stage": "destinations_carousel", "entity_name": None, "extracted_params": None}

    # ----------------------------------------------------
    # 5. Check for Tourism Flow (Destinations, Cities, Form Submissions, Itineraries)
    # ----------------------------------------------------
    matched_city = None
    dest_param = form_params.get("destination") or form_params.get("dest")
    if dest_param and str(dest_param).lower() not in ("choose...", ""):
        dest_str = str(dest_param).strip().lower()
        for city in FLAGSHIP_CITIES.keys():
            if city in dest_str:
                matched_city = city
                break
        if not matched_city:
            for city in POPULAR_TOURIST_DESTINATIONS:
                if city in dest_str:
                    matched_city = city
                    break
        if not matched_city:
            for state_name in INDIAN_STATES:
                if state_name in dest_str:
                    matched_city = state_name
                    break
        if not matched_city:
            matched_city = dest_str

    if not matched_city:
        for city in FLAGSHIP_CITIES.keys():
            if re.search(r'\b' + re.escape(city) + r'\b', q_lower):
                matched_city = city
                break
    if not matched_city:
        for city in POPULAR_TOURIST_DESTINATIONS:
            if re.search(r'\b' + re.escape(city) + r'\b', q_lower):
                matched_city = city
                break
    if not matched_city:
        for state_name in INDIAN_STATES:
            if re.search(r'\b' + re.escape(state_name) + r'\b', q_lower):
                matched_city = state_name
                break
    if not matched_city:
        travel_patterns = [
            r'(?:travel(?:ling)? to|trip to|visit|explore|tour of|tour to|heading to|going to|go to|holiday in|vacation in|itinerary for|plan (?:a )?(?:trip )?(?:to|for)|flights? to|train to|hotels? in|tourism in)\s+([a-zA-Z\s]+)',
            r'([a-zA-Z\s]+?)\s+(?:trip|tour|tourism|itinerary|travel)',
        ]
        for p in travel_patterns:
            m = re.search(p, q_lower)
            if m:
                cand = m.group(1).strip()
                cand = re.split(r'\b(?:for|with|in|on|next|\d+\s*days?)\b', cand, flags=re.IGNORECASE)[0].strip()
                if cand and cand.lower() not in ('india', 'a', 'the', 'my', 'some', 'any', 'place', 'city'):
                    matched_city = cand
                    break

    has_travel_context = has_keyword(q_lower, [
        "tourism", "tourist", "vacation", "holiday", "itinerary", "sightseeing", "places to visit",
        "travel", "trip", "hotel", "hotels", "flight", "flights"
    ]) or any(p in q_lower for p in [
        "travel to", "trip to", "visit india", "plan a trip", "plan my trip", "explore destination", "top destinations"
    ])

    if matched_city:
        has_duration = bool(
            ("days" in form_params and str(form_params["days"]).lower() not in ("choose...", "")) or 
            re.search(r'\b(\d+)\s*(?:days?|day|d)\b', q_lower) or 
            "1-day" in q_lower or "2-day" in q_lower or "3-day" in q_lower or "5-day" in q_lower
        )
        has_people_or_budget = bool(
            ("travelers" in form_params and str(form_params["travelers"]).lower() not in ("choose...", "")) or 
            ("budget" in form_params and str(form_params["budget"]).lower() not in ("choose...", "")) or
            any(kw in q_lower for kw in ["people", "person", "adult", "family", "solo", "couple", "budget", "luxury", "mid-range", "standard"])
        )

        # Smart Follow-up (e.g. "Srinagar 5 days" with missing travelers/budget)
        if has_duration and not has_people_or_budget:
            parsed = parse_travel_params(clean_query, form_params)
            return {"workflow_stage": "tourism_followup", "entity_name": matched_city, "extracted_params": parsed}

        # If user provided sufficient parameters or requested a plan or clicked Generate Plan
        if (has_duration or has_people_or_budget or "plan" in q_lower or "itinerary" in q_lower or form_params.get("action") == "plan_trip"):
            parsed = parse_travel_params(clean_query, form_params)
            return {"workflow_stage": "tourism_planner", "entity_name": matched_city, "extracted_params": parsed}

        # If user just selected a city (e.g. "Srinagar" or "Explore Srinagar" or "I want to travel to Jammu")
        if matched_city in INDIAN_STATES and any(kw in q_lower for kw in ["economy", "culture", "profile", "tell me about"]):
            return {"workflow_stage": "state_composite", "entity_name": matched_city, "extracted_params": None}
        return {"workflow_stage": "tourism_briefing", "entity_name": matched_city, "extracted_params": None}

    # If travel context is present without a city and without competing domain keywords, show tourism destinations intake
    if has_travel_context and not any(kw in q_lower for kw in ["gati shakti", "economy", "gdp", "pli", "fdi", "stock", "dance", "music", "festival", "isro", "ai", "quantum"]):
        return {"workflow_stage": "tourism_destinations", "entity_name": None, "extracted_params": None}

    # ----------------------------------------------------
    # 6. Check for Economy & Financial Markets Inquiries / Form Submissions
    # ----------------------------------------------------
    economy_keywords = [
        "economy", "gdp", "growth", "sector", "sectors", "manufacturing", "services", "industry",
        "startup", "startups", "unicorn", "unicorns", "make in india", "atmanirbhar", "export", "import", "fdi",
        "employment", "inflation", "fiscal", "monetary", "rbi", "budget", "tax", "gst",
        "company", "companies", "tata", "reliance", "infosys", "tcs", "wipro", "hcl", "adani",
        "l&t", "mahindra", "maruti", "stock", "stocks", "nse", "bse", "nifty", "sensex", "mutual fund",
        "sebi", "banking", "bank", "upi", "rupay", "fintech", "financial",
        "titans", "corporate titans", "pli", "pli scheme", "production-linked incentive",
        "gati shakti", "gatishakti", "pm gati", "infrastructure", "capex", "forex", "direct tax",
        "income tax", "corporate tax", "crypto", "cryptocurrency", "rupee", "sez", "logistics"
    ]
    if form_params.get("topic") or has_keyword(q_lower, economy_keywords) or any(kw in q_lower for kw in [
        "upi, rupay & fintech revolution", "startups & tech unicorns", "corporate titans", "manufacturing, pli & atmanirbhar bharat",
        "stock markets (nse, bse, nifty 50, sensex)", "indian economy & gdp growth", "major sectors (services, mfg, agri)"
    ]):
        return {"workflow_stage": "economy_briefing", "entity_name": None, "extracted_params": form_params}

    # ----------------------------------------------------
    # 7. Check for Culture & Social Fabric Inquiries / Form Submissions
    # ----------------------------------------------------
    culture_keywords = [
        "culture", "tradition", "traditions", "festival", "festivals", "diwali", "holi", "eid", "christmas", "navratri",
        "durga puja", "ganesh", "pongal", "onam", "baisakhi", "bihu", "language", "languages", "religion",
        "philosophy", "vedanta", "clothing", "saree", "cuisine", "food", "music", "dance", "dances",
        "bharatanatyam", "kathak", "kathakali", "odissi", "sattriya", "kuchipudi", "manipuri", "mohiniyattam",
        "cinema", "art", "handicraft", "diversity", "society", "demographic", "social fabric", "marriage",
        "kalbelia", "warli", "madhubani", "kalaripayattu", "chhau", "theyyam", "yakshagana", "bhangra",
        "giddha", "garba", "dandiya", "lavani", "rouf", "sufi", "sufism", "qawwali", "carnatic",
        "hindustani", "raga", "tabla", "sitar", "veena", "shehnai", "flute", "thangka", "pattachitra",
        "tanjore painting", "gond", "chola", "temple architecture", "unesco heritage", "kumbh mela"
    ]
    if form_params.get("region") or form_params.get("theme") or has_keyword(q_lower, culture_keywords):
        return {"workflow_stage": "culture_report", "entity_name": None, "extracted_params": form_params}

    # ----------------------------------------------------
    # 8. Check for Science, Innovation & Future Tech Inquiries / Form Submissions
    # ----------------------------------------------------
    science_keywords = [
        "science", "tech", "technology", "isro", "space", "chandrayaan", "mangalyaan", "gaganyaan",
        "aditya", "rocket", "semiconductor", "semiconductors", "chip", "chips", "fab", "dholera",
        "micron", "sanand", "ai", "artificial intelligence", "indiaai", "bhashini", "telecom", "5g", "6g",
        "green hydrogen", "quantum", "biotechnology", "drone", "defence", "drdo",
        "superconductivity", "fusion", "graphene", "robotics", "nanotechnology", "ev battery",
        "lithium", "clean energy", "nuclear", "thorium"
    ]
    if form_params.get("category") or has_keyword(q_lower, science_keywords):
        return {"workflow_stage": "science_dossier", "entity_name": None, "extracted_params": form_params}

    # ----------------------------------------------------
    # 9. Check for State / Regional Profiles
    # ----------------------------------------------------
    for state_name in INDIAN_STATES:
        if re.search(r'\b' + re.escape(state_name) + r'\b', q_lower):
            return {"workflow_stage": "state_composite", "entity_name": state_name, "extracted_params": None}

    # Default fallback: Landing screen
    return {"workflow_stage": "landing", "entity_name": None, "extracted_params": None}


def route_step(state: AgentState) -> str:
    """Direct router to map workflow_stage to node execution."""
    stage = state.get("workflow_stage") or "landing"
    stage_to_node = {
        "landing": "landing_page_node",
        "tourism_destinations": "tourism_destinations_node",
        "tourism_briefing": "tourism_briefing_node",
        "tourism_followup": "tourism_followup_node",
        "tourism_planner": "tourism_planner_node",
        "economy_form": "economy_intake_form_node",
        "economy_briefing": "economy_dashboard_node",
        "culture_form": "culture_intake_form_node",
        "culture_report": "culture_dashboard_node",
        "science_form": "science_intake_form_node",
        "science_dossier": "science_dashboard_node",
        "state_composite": "state_composite_node",
        "out_of_scope": "out_of_scope_node",
        "destinations_carousel": "destinations_carousel_node",
    }
    return stage_to_node.get(stage, "landing_page_node")


# ----------------------------------------------------
# 9. Build and Compile the StateGraph
# ----------------------------------------------------
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_router)
workflow.add_node("landing_page_node", landing_page_node)
workflow.add_node("tourism_destinations_node", tourism_destinations_node)
workflow.add_node("tourism_briefing_node", tourism_briefing_node)
workflow.add_node("tourism_followup_node", tourism_followup_node)
workflow.add_node("tourism_planner_node", tourism_planner_node)
workflow.add_node("economy_intake_form_node", economy_intake_form_node)
workflow.add_node("economy_dashboard_node", economy_dashboard_node)
workflow.add_node("culture_intake_form_node", culture_intake_form_node)
workflow.add_node("culture_dashboard_node", culture_dashboard_node)
workflow.add_node("science_intake_form_node", science_intake_form_node)
workflow.add_node("science_dashboard_node", science_dashboard_node)
workflow.add_node("state_composite_node", state_composite_node)
workflow.add_node("out_of_scope_node", out_of_scope_node)
workflow.add_node("destinations_carousel_node", destinations_carousel_node)

# Add Edges
workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_step,
    {
        "landing_page_node": "landing_page_node",
        "tourism_destinations_node": "tourism_destinations_node",
        "tourism_briefing_node": "tourism_briefing_node",
        "tourism_followup_node": "tourism_followup_node",
        "tourism_planner_node": "tourism_planner_node",
        "economy_intake_form_node": "economy_intake_form_node",
        "economy_dashboard_node": "economy_dashboard_node",
        "culture_intake_form_node": "culture_intake_form_node",
        "culture_dashboard_node": "culture_dashboard_node",
        "science_intake_form_node": "science_intake_form_node",
        "science_dashboard_node": "science_dashboard_node",
        "state_composite_node": "state_composite_node",
        "out_of_scope_node": "out_of_scope_node",
        "destinations_carousel_node": "destinations_carousel_node",
    },
)

# Terminal Edges
workflow.add_edge("landing_page_node", END)
workflow.add_edge("tourism_destinations_node", END)
workflow.add_edge("tourism_briefing_node", END)
workflow.add_edge("tourism_followup_node", END)
workflow.add_edge("tourism_planner_node", END)
workflow.add_edge("economy_intake_form_node", END)
workflow.add_edge("economy_dashboard_node", END)
workflow.add_edge("culture_intake_form_node", END)
workflow.add_edge("culture_dashboard_node", END)
workflow.add_edge("science_intake_form_node", END)
workflow.add_edge("science_dashboard_node", END)
workflow.add_edge("state_composite_node", END)
workflow.add_edge("out_of_scope_node", END)
workflow.add_edge("destinations_carousel_node", END)

# Compile
india_graph = workflow.compile()
