"""
LangGraph Multi-Agent Workflow powered by langchain_google_genai (Gemini)
"""

import os
from typing import Annotated, Dict, List, Literal, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ----------------------------------------------------
# 1. Initialize Google Gemini LLM
# ----------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)

# ----------------------------------------------------
# 2. State Definition
# ----------------------------------------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]
    city_name: str | None
    route_type: Literal["flagship", "dynamic_city", "economy", "general"] | None
    city_data: Dict | None
    final_output: str | None

# ----------------------------------------------------
# 3. Verified Knowledge Base for Flagship Cities
# ----------------------------------------------------
FLAGSHIP_CITIES = {
    "agra": {
        "title": "Agra — City of the Taj",
        "heritage": ["Taj Mahal (UNESCO)", "Agra Fort (UNESCO)", "Fatehpur Sikri (UNESCO)"],
        "ticket_info": "Taj Mahal: ₹50 (Indians), ₹1,100 (Foreign tourists). Agra Fort: ₹50 / ₹650.",
        "best_hours": "Sunrise (6:00 AM – 8:30 AM) for Taj Mahal. NOTE: Closed on Fridays!",
        "culinary": "Petha (sweets at Panchhi Petha), Bedmi Puri breakfast, and Mughlai dishes.",
        "transit": "Gatimaan Express (Fastest: 1 hr 40 min from Hazrat Nizamuddin, Delhi).",
    },
    "jaipur": {
        "title": "Jaipur — The Pink City of Rajasthan",
        "heritage": ["Amer Fort", "Hawa Mahal", "Jantar Mantar (UNESCO)", "City Palace", "Nahargarh Fort"],
        "ticket_info": "Composite 2-day ticket (5 monuments): ₹100 (Indians), ₹500 (Foreign tourists).",
        "best_hours": "Early morning (8:00 AM) for Amer Fort ascent; sunset at Nahargarh Fort.",
        "culinary": "Dal Baati Churma, Pyaaz Kachori (Rawat Mishtan Bhandar), Ghevar.",
        "transit": "Vande Bharat Express from Delhi (4 hours) or NH48 Expressway.",
    },
    "delhi": {
        "title": "New Delhi — The Sovereign Capital",
        "heritage": ["Qutub Minar (UNESCO)", "Humayun's Tomb (UNESCO)", "Red Fort (UNESCO)", "India Gate"],
        "ticket_info": "Most ASI monuments: ₹35-₹50 (Indians), ₹550-₹600 (Foreign tourists).",
        "best_hours": "Mornings for Old Delhi heritage walks; evenings at India Gate & Kartavya Path.",
        "culinary": "Chandni Chowk street food (Paranthe Wali Gali), Chole Bhature, Butter Chicken.",
        "transit": "Delhi Metro (Yellow & Airport Express lines) connects all heritage hubs.",
    },
    "mumbai": {
        "title": "Mumbai (Bombay) — City of Dreams & Commerce",
        "heritage": ["Gateway of India", "Elephanta Caves (UNESCO)", "Victorian Gothic & Art Deco Ensemble (UNESCO)"],
        "ticket_info": "Elephanta Ferry: ₹260 return. Caves entry: ₹40 (Indians), ₹600 (Foreigners).",
        "best_hours": "Sunset at Marine Drive (Queen's Necklace) and Colaba heritage walk.",
        "culinary": "Vada Pav, Pav Bhaji, Bombay Duck, Coastal Malvani seafood.",
        "transit": "Western Railway Local, Mumbai Metro Line 3, and Coastal Road.",
    },
}

# ----------------------------------------------------
# 4. LangGraph Nodes
# ----------------------------------------------------

def supervisor_router(state: AgentState) -> AgentState:
    """Classifies user intent and routes to Flagship, Dynamic, Economy, or General."""
    user_query = state["messages"][-1].content.lower()

    # Check for 4 flagship cities
    for city in FLAGSHIP_CITIES.keys():
        if city in user_query or (city == "delhi" and "new delhi" in user_query) or (city == "mumbai" and "bombay" in user_query):
            return {"city_name": city, "route_type": "flagship"}

    # Check for economy / tech
    if any(k in user_query for k in ["economy", "upi", "tech", "gdp", "isro", "startup", "finance"]):
        return {"city_name": None, "route_type": "economy"}

    # Check if user is asking about ANY other city / location
    prompt = f"""Analyze this query: "{user_query}"
Does it ask about travel, tourism, or exploring a specific Indian city, state, or location (like Varanasi, Goa, Bengaluru, Kerala, Amritsar, Kolkata, etc.)?
Reply strictly with the city/location name, or "NONE" if not a city/travel request."""

    result = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()

    if result != "none" and len(result) < 30:
        return {"city_name": result, "route_type": "dynamic_city"}

    return {"city_name": None, "route_type": "general"}


def flagship_city_node(state: AgentState) -> AgentState:
    """Loads verified, curated data for one of the flagship 4 cities."""
    city = state.get("city_name")
    data = FLAGSHIP_CITIES.get(city, {})
    return {"city_data": data}


def dynamic_city_node(state: AgentState) -> AgentState:
    """Uses Google Gemini to research and construct an action itinerary for ANY other city."""
    city = state.get("city_name")
    user_query = state["messages"][-1].content

    prompt = f"""
You are the India Travel & Heritage AI Agent.
The user is asking about the Indian destination: "{city}".
User Query: "{user_query}"

Generate a structured, actionable travel intelligence briefing:
1. 🏛️ **Top Must-Visit Landmarks & Heritage Sites**
2. 🎟️ **Estimated Ticket Costs & Timings (Indians vs Foreigners)**
3. 💡 **Crucial Travel Pro-Tips (Closing days, dress codes, best visiting hours)**
4. 🍲 **Must-Try Local Culinary Specialties**
5. 🗺️ **Recommended 1-Day Hour-by-Hour Itinerary**

Format with clean bullet points and clear emojis. Be factual, concise, and helpful.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_output": response.content}


def economy_node(state: AgentState) -> AgentState:
    """Briefing on India's modern economy and Digital Public Infrastructure."""
    summary = (
        "📈 **India Economy & Digital Public Infrastructure (DPI)**:\n\n"
        "• **UPI (Unified Payments Interface):** Processes 14+ billion real-time transactions every month.\n"
        "• **Tech & Startup Powerhouse:** 3rd largest startup ecosystem globally with 115+ unicorns.\n"
        "• **Space Exploration:** ISRO's ultra-frugal lunar landing (Chandrayaan-3) & solar observatory (Aditya-L1).\n"
        "• **Manufacturing Hub:** Expanding semiconductor fabrication units and mobile electronics manufacturing."
    )
    return {"final_output": summary}


def general_node(state: AgentState) -> AgentState:
    """General orientation for the agent."""
    summary = (
        "🇮🇳 **Namaste! Welcome to India Intelligence & Travel Concierge (@india)**\n\n"
        "I am an autonomous agent powered by Google Gemini & the ASI Alliance.\n\n"
        "**What I can do for you:**\n"
        "• 🕌 **Flagship City Guides:** Detailed itineraries, ticket fees, and Friday closure checks for **Agra**, **Jaipur**, **New Delhi**, and **Mumbai**.\n"
        "• 🌏 **Any City in India:** Ask for travel plans, routes, and food trails for **any** destination (e.g., *Varanasi, Goa, Amritsar, Bengaluru*).\n"
        "• 📈 **Economic & Tech Briefing:** Insights into India's DPI (UPI), space tech, and startup clusters.\n\n"
        "*Try asking: 'I want a 1-day itinerary for Agra with train options from Delhi under ₹2000.'*"
    )
    return {"final_output": summary}


def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesizes verified flagship city data with Gemini into an actionable itinerary."""
    data = state.get("city_data")
    user_query = state["messages"][-1].content

    prompt = f"""
You are the AI Ambassador for India (@india) on ASI:One.
User Query: "{user_query}"

Synthesize an engaging, actionable itinerary response using this verified data:
- Destination: {data.get('title')}
- Heritage Sites: {', '.join(data.get('heritage', []))}
- Entry Fees: {data.get('ticket_info')}
- Best Visiting Windows: {data.get('best_hours')}
- Food Specialties: {data.get('culinary')}
- Transit Options: {data.get('transit')}

Include:
1. 🏛️ Overview & Verified Entry Fees
2. ⏱️ 1-Day Structured Hour-by-Hour Action Plan
3. ⚠️ Critical Warnings (e.g., Friday closures or crowd advice)
4. 🍲 Local Food Recommendations

Keep it crisp, formatted with clear markdown and emojis.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_output": response.content}


# ----------------------------------------------------
# 5. Routing Logic
# ----------------------------------------------------
def route_step(state: AgentState) -> Literal["flagship_city_node", "dynamic_city_node", "economy_node", "general_node"]:
    route = state.get("route_type")
    if route == "flagship":
        return "flagship_city_node"
    elif route == "dynamic_city":
        return "dynamic_city_node"
    elif route == "economy":
        return "economy_node"
    return "general_node"


# ----------------------------------------------------
# 6. Build the StateGraph
# ----------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_router)
workflow.add_node("flagship_city_node", flagship_city_node)
workflow.add_node("dynamic_city_node", dynamic_city_node)
workflow.add_node("economy_node", economy_node)
workflow.add_node("general_node", general_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_step,
    {
        "flagship_city_node": "flagship_city_node",
        "dynamic_city_node": "dynamic_city_node",
        "economy_node": "economy_node",
        "general_node": "general_node",
    },
)

workflow.add_edge("flagship_city_node", "synthesizer")
workflow.add_edge("synthesizer", END)
workflow.add_edge("dynamic_city_node", END)
workflow.add_edge("economy_node", END)
workflow.add_edge("general_node", END)

# Compile
india_graph = workflow.compile()