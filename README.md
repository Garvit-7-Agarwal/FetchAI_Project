![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

# 🇮🇳 India Intel Agent (@india-intel-agent)
### Autonomous Multi-Domain Educational & Research Intelligence Platform

> **"Your AI-powered intelligence assistant for exploring India's economy, tourism, culture, science, technology and future opportunities."**

Powered by **LangGraph**, **Google Gemini 2.5 Flash**, and the **Fetch.ai Agentverse / ASI Alliance** ecosystem.

---

### ⚠️ OFFICIAL DISCLAIMER
*This autonomous agent is an independent educational, training, and informational intelligence agent operating on Agentverse and powered by the ASI Alliance Network. This agent is **NOT** an official government entity, embassy, or state representative of the Government of India. Its mission is strictly to provide structured domain knowledge, public insights, educational training, research assistance, and actionable cultural exploration.*

---

## 🌟 The Core Experience: Intelligent Agent vs. Chatbot

Unlike conventional conversational chatbots that produce long scrolling walls of generic text, **India Intel Agent** operates as a true **interactive intelligence application**:

1. **Initial Welcome Landing Screen:**
   Prominently opens with the official welcome banner and **4 Large Interactive Domain Cards/Buttons**:
   * 🇮🇳 **Option 1: Economy & Financial Markets**
   * ✈️ **Option 2: Tourism & UNESCO Heritage**
   * 🎨 **Option 3: Culture & Social Fabric**
   * 🚀 **Option 4: Science & Future Tech**

2. **Form & Intake-Driven Interaction:**
   Users navigate through structured menus, searchable destination selectors, intake cards, duration options, and travel parameter grids.

3. **Autonomous Planning & Research Engine:**
   When an intake form is submitted, the AI backend acts as an autonomous planning agent:
   * Coordinates day-by-day itineraries tailored to pacing, travel times, and monument opening hours.
   * Generates transportation routes with direct, real booking links (IRCTC, Google Flights, Google Maps).
   * Generates curated accommodation cards with ⭐ ratings, estimated price tiers, location, and verified lookup links.
   * Calculates a transparent, itemized cost estimation breakdown.
   * Provides authoritative official sources (ASI, ISRO, RBI, Ministry of Culture, etc.).

4. **Smart Follow-Up:**
   If a user submits an incomplete request (e.g. `Jaipur 5 days` without specifying budget or travelers), the agent intelligently pauses and renders a targeted follow-up card rather than making blind assumptions.

5. **Strict Geographic Guardrails:**
   If asked about foreign destinations (e.g. Thailand, Dubai, Paris), the agent politely enforces its geographic scope and presents the 4 core Indian intelligence domains.

---

## 🏛️ Four Core Knowledge Domains

```
                         🇮🇳 India Intel Agent
                                  │
    ┌──────────────────┬──────────┴───────────┬─────────────────┐
    ▼                  ▼                      ▼                 ▼
Option 1:          Option 2:              Option 3:         Option 4:
🇮🇳 Economy &       ✈️ Tourism &           🎨 Culture &      🚀 Science &
   Financial Mkts     UNESCO Heritage        Social Fabric     Future Tech
```

### 1. 🇮🇳 Economy & Financial Markets
* **Macroeconomics:** 5th largest nominal GDP (~$3.9–$4.0T), 3rd by PPP (~$13T+), real GDP growth (6.5%–7.2%), and structural breakdown across Services (~54%), Manufacturing (~28%), and Agriculture (~18%).
* **Industrial Policies:** Make in India, Production-Linked Incentive (PLI in 14 strategic sectors), PM Gati Shakti national logistics masterplan.
* **Corporate Titans:** In-depth profiles of Reliance Industries, Tata Group, Infosys, TCS, Larsen & Toubro, HDFC Bank, Adani, and Mahindra.
* **Capital Markets & Rails:** BSE (S&P BSE SENSEX), NSE (NIFTY 50), SEBI, RBI monetary policy (inflation targeting 4% ± 2%), and UPI processing >14 billion monthly transactions.

### 2. ✈️ Tourism, Landmarks & UNESCO Heritage
* **Destination Intelligence Grid:** Covers flagship hubs (Jaipur, Agra, New Delhi, Mumbai, Varanasi, Udaipur, Goa, Kochi, Amritsar, Srinagar, Rishikesh, Darjeeling) plus all 28 Indian states.
* **Intake Form:** Captures duration, number of travelers, budget category (Low Budget, Budget, Mid-range, Luxury), transportation mode, hotel preference, and travel interests.
* **Actionable Execution Tools:**
  * 📅 **1-Click Google Calendar Sync:** Saves day-by-day schedules into calendar.
  * 🗺️ **Multi-Stop GPS Routes:** Direct multi-stop navigation in Google Maps.
  * 🎟️ **Government Ticket Portals:** Official Archaeological Survey of India (ASI) & state monuments booking gateways.
  * 🚆 **Rail & Air Booking Links:** Direct IRCTC train reservation and Google Flights search.
  * 🏨 **Hotel Recommendation Cards:** Rated accommodations with direct lookup links and transparent pricing disclosures.

### 3. 🎨 Cultural Heritage & Social Fabric
* **8 Classical Dance Forms:** Bharatanatyam, Kathak, Kathakali, Kuchipudi, Odissi, Manipuri, Mohiniyattam, Sattriya (Sangeet Natak Akademi).
* **Music Traditions:** Comparative Hindustani and Carnatic classical traditions.
* **Festive Calendar:** Pan-Indian (Diwali, Holi) and regional harvest celebrations (Pongal, Onam, Baisakhi, Bihu, Chhath Puja).
* **Social Fabric:** Demographic dividend (median age ~28), traditional family systems harmonizing with modern digital literacy, and women-led entrepreneurship.

### 4. 🚀 Science, Innovation & Future Technology
* **ISRO Space Exploration:** Chandrayaan-3 south pole landing (National Space Day on Aug 23), Gaganyaan crewed human spaceflight, Aditya-L1 solar observatory, and private space startups (Skyroot, Agnikul, Pixxel).
* **India Semiconductor Mission (ISM):** $10B incentive framework, Tata-PSMC commercial fab in Dholera, Micron OSAT in Sanand, and India's >20% global chip design engineering workforce.
* **IndiaAI Mission:** ₹10,300+ crore outlay approving 10,000+ GPU sovereign compute infrastructure and Project Bhashini multilingual translation.
* **Frontier Tech:** National Green Hydrogen Mission, Bharat 6G Vision, and the National Quantum Mission.

---

## 🧪 Verification & Automated Test Suites

The project includes automated verification suites for both traditional queries and interactive workflows:

```bash
# 1. Activate virtual environment
source myenv/bin/activate

# 2. Run the Interactive Workflow Test Suite (10 Comprehensive Scenarios)
python3 test_interactive_workflows.py

# 3. Run the Core Domain Query Test Suite
python3 test_graph.py
```

### Verified Test Scenarios:
1. `hi` → Welcome landing screen with 4 interactive category cards
2. `Tourism` → Destination grid and searchable city picker
3. `Explore Jaipur` → Jaipur Destination Intelligence Card + Trip Planning Form
4. `Jaipur 5 days` → Smart Follow-Up prompting for missing travelers and budget
5. `one day trip at thailand` → Scope guardrail with polite redirection to India domains
6. `Tourism → Jaipur → 3 days → 2 people → Budget` → Full 3-day travel dashboard
7. `Tourism → Delhi → 5 days → Family` → 5-day Delhi family travel dashboard
8. `Culture → Regional Culture → North India` → Cultural intelligence dossier
9. `Economy → Indian Economy → Current` → Macroeconomic dashboard
10. `Science → AI / Semiconductors → Future Outlook` → Technology dossier

---

## 🚀 Running the Live Agent

```bash
# Start the Fetch.ai Agentverse LangGraph adapter with Mailbox relay
python3 agent.py
```

You can also interact with `@india-intel-agent` live on **ASI:One** ([asi1.ai](https://asi1.ai/)).
