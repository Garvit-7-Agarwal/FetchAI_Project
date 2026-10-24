![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

# 🇮🇳 India Intel Agent (@india-intel-agent)
### Autonomous Multi-Domain Educational & Research Intelligence Platform
**Registered on Agentverse • Discoverable & Usable on ASI:One (asi1.ai) • Built with LangGraph & Fetch.ai uAgents**

> **"Your AI-powered intelligence assistant for exploring India's economy, tourism, culture, science, technology and future opportunities."**

Powered by **LangGraph**, **Google Gemini 2.5 Flash**, **Fetch.ai uAgents Protocol**, and the **ASI Alliance** ecosystem.

---

### ⚠️ OFFICIAL DISCLAIMER
*This autonomous agent is an independent educational, training, and informational intelligence agent operating on Agentverse and powered by the ASI Alliance Network. This agent is **NOT** an official government entity, embassy, or state representative of the Government of India. Its mission is strictly to provide structured domain knowledge, public insights, educational training, research assistance, and actionable cultural exploration.*

---

## 🌟 What We Built: Intelligent Agent vs. Chatbot

Unlike conventional chatbots that produce long scrolling walls of generic text, **India Intel Agent** operates as an **interactive, multi-agent intelligence platform**:

1. **Native ASI:One Interactive Cards:**
   Renders rich, guided visual widgets directly inside the ASI:One chat window without requiring an external custom frontend:
   * **Form Cards (`FormCardPayload`):** Domain Explorer, Trip Planning Intake, Currency Converter, and Payment Checkout.
   * **Review Cards (`ReviewCardPayload`):** Interactive confirmation screens with key summary rows and 1-click **Approve** (`🚀 Confirm`) or **Reject** (`🎛️ Adjust`) CTAs.
   * **Carousel Cards (`CarouselCardPayload`):** Swipeable cards showcasing flagship heritage circuits with direct action buttons.

2. **Effective Multi-Agent Collaboration:**
   A dual-agent collective featuring the master **India Intel Supervisor Agent** (`agent.py`) and an autonomous companion **Transit & Mobility Specialist Subagent** (`transit_agent.py`) communicating via the Fetch.ai Agent Chat Protocol (ACP).

3. **Fetch.ai Payment Protocol & Monetization Model:**
   Tokenized micropayments (in **ASI** or **FET** tokens) to unlock premium services:
   * **Free Tier:** Public educational dossiers, 3-day itineraries, cultural overviews, and macro indicators.
   * **Tier 1 (1.0 ASI):** **Verified VIP All-India Expedited Travel Pass** with digital security seal, 24/7 IRCTC Tatkal assistance guidelines, Tourist Police 1363 hotline, and offline printable summary.
   * **Tier 2 (2.0 ASI):** **Institutional Investor Economic Dossier** featuring 5-year econometric forecast models (FY2026-FY2030) and sectoral FDI equity caps.

4. **Creative Real-Time Data & External Services:**
   * **Live Forex Engine:** Real-time exchange rates (USD, EUR, GBP, AED, JPY, CAD, AUD to INR) via Open ER API with local purchasing power equivalents.
   * **Live Meteorological Weather:** Real-time temperature (°C), conditions, and environmental travel advisories via Open-Meteo for 20+ Indian destinations.
   * **Dynamic Deep-Links:** Real-time IRCTC station-to-station train booking links, Google Flights, Google Maps multi-point routes, and Google Calendar 1-click sync.

5. **Strong Reliability & Error Recovery:**
   * Dual-engine resilience: when Gemini quotas are available, generates dynamic AI synthesis; under quota exhaustion (HTTP 429) or network hiccups, immediately serves high-fidelity deterministic dossiers.
   * Single-card wire gateway in `agent.py` ensuring zero duplicate cards in the ASI:One UI.

---

## 🏗️ Multi-Agent Architecture & Protocol Flow

```
                               ┌────────────────────────────────┐
                               │       ASI:One User Chat        │
                               │  (Interactive Cards & Messages)│
                               └───────────────┬────────────────┘
                                               │ Inbound ACP Envelope
                                               ▼
                               ┌────────────────────────────────┐
                               │   India Intel Supervisor Agent │
                               │  (@india-intel-agent / agent.py│
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               │                               │                               │
               ▼                               ▼                               ▼
 ┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
 │ Companion Transit Subagent│   │ Real-Time Services Engine │   │ Fetch.ai Payment Protocol │
 │     (transit_agent.py)    │   │       (services.py)       │   │       (payment.py)        │
 ├───────────────────────────┤   ├───────────────────────────┤   ├───────────────────────────┤
 │ • Vande Bharat Schedules  │   │ • Live Forex (USD/EUR/INR)│   │ • Treasury LocalWallet    │
 │ • Rail Distance & Fares   │   │ • Live Weather & Advisory │   │ • VIP Travel Pass Seal    │
 │ • IRCTC Deep-Link Payload │   │ • Dynamic Google Maps/Cal │   │ • Institutional Research  │
 └───────────────────────────┘   └───────────────────────────┘   └───────────────────────────┘
```

---

## 🏛️ Core Intelligence Domains

```
                         🇮🇳 India Intel Agent
                                  │
     ┌──────────────────┬─────────┴─────────┬─────────────────┐
     ▼                  ▼                   ▼                 ▼
  Option 1:          Option 2:           Option 3:         Option 4:
  Economy &          Tourism &           Culture &         Science &
Financial Mkts     UNESCO Heritage     Social Fabric      Future Tech
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
  * 🌤️ **Live Weather Advisory:** Temperature, atmospheric humidity, and seasonal packing advisories.
  * 💱 **Foreign Currency Breakdown:** Real-time budget conversion in USD, EUR, GBP, and AED.
  * 🚆 **Transit Specialist Intelligence:** Vande Bharat train numbers, schedules, and direct IRCTC booking links.
  * 📅 **1-Click Google Calendar Sync:** Saves day-by-day schedules into calendar.
  * 🗺️ **Multi-Stop GPS Routes:** Direct multi-stop navigation in Google Maps.
  * 🎟️ **Government Ticket Portals:** Official Archaeological Survey of India (ASI) & state monuments booking gateways.

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

## 💳 Fetch.ai Payment Protocol & Monetization

The agent includes a native on-chain monetization model built on the Fetch.ai Network:
* **Treasury Wallet Address:** `fetch1kxzxsfw5ykc26l8528d4nvxxq74cvwvvqjgknr` (derived via Cosmpy `LocalWallet`)
* **Supported Rails:** Fetch.ai Mainnet (`ASI`), Fetch.ai Dorado Testnet (`FET`), and Instant Demo Escrow.

```
╔═════════════════════════════════════════════════════════════════════════════════╗
║                      🇮🇳 ALL-INDIA VIP EXPEDITED TRAVEL PASS                     ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║ PASS ID     : IND-VIP-482019       ISSUE DATE  : 24 October 2026, 11:16 IST     ║
║ DESTINATION : Jaipur               DURATION    : 3 Days (2 Travelers)           ║
║ BUDGET TIER : Budget               TREASURY    : fetch1kxzxsfw5yk...            ║
║ PAYMENT     : 1.0 ASI (VERIFIED)   TX HASH     : 0x8a9bf32e...4819              ║
║ STATUS      : ✅ ACTIVE & VERIFIED DIGITAL SEAL: 8A9BF32E4819C204A1F6...        ║
╚═════════════════════════════════════════════════════════════════════════════════╝
```

---

## 🧪 Automated Verification & Test Suites

The project includes automated verification test suites covering all core workflows and hackathon bonus criteria:

```bash
# 1. Activate virtual environment
source myenv/bin/activate

# 2. Run the Expanded 15-Scenario Interactive Workflow Suite (Core + All 5 Bonus Criteria)
python3 test_interactive_workflows.py

# 3. Run the Domain Knowledge Verification Suite
python3 test_graph.py
```

### Verified Test Scenarios (15/15 Passed):
1. `hi` → Welcome landing screen with 6 interactive domain options
2. `Tourism` → Destination grid and searchable city picker
3. `Explore Jaipur` → Jaipur Destination Intelligence Card + Trip Planning Form
4. `Jaipur 5 days` → Smart Follow-Up prompting for missing travelers and budget
5. `one day trip at thailand` → Scope guardrail with polite redirection to India domains
6. `Tourism → Jaipur → 3 days → 2 people → Budget` → Full 3-day travel dashboard with Live Weather + Forex + Review Card
7. `Tourism → Delhi → 5 days → Family` → 5-day Delhi family travel dashboard
8. `Culture → Regional Culture → North India` → Cultural intelligence dossier
9. `Economy → Indian Economy → Current` → Macroeconomic dashboard
10. `Science → AI / Semiconductors → Future Outlook` → Technology dossier
11. **BONUS 1 (Multi-Agent Transit):** `train to jaipur` → Routes to companion transit agent returning Vande Bharat schedule and booking link
12. **BONUS 2 (Real-Time Forex):** `convert 100 USD to INR` → Live exchange rate ($1 = ~₹95.04) and local purchasing power benchmarks
13. **BONUS 3 (Payment Protocol):** `pay 1 ASI for VIP Pass` → Issues cryptographically signed VIP pass with emergency helplines
14. **BONUS 4 (Destinations Carousel):** `show top destinations carousel` → Renders interactive swipeable carousel in ASI:One
15. **BONUS 5 (Review Card CTA):** `{"action": "confirm_trip"}` → Processes native ReviewCard approval action

---

## 🚀 Running the Live Agent

```bash
# 1. Start the companion Transit Specialist uAgent (Optional standalone process)
python3 transit_agent.py

# 2. Start the primary Fetch.ai Agentverse LangGraph adapter with Mailbox relay
python3 agent.py
```

You can interact with `@india-intel-agent` live on **ASI:One** ([asi1.ai](https://asi1.ai/)).


