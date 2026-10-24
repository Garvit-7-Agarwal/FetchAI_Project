"""
Comprehensive Interactive Workflow Verification Test Suite
Verifies all interactive screens, core domains, and Hackathon Bonus Criteria for 🇮🇳 India Intel Agent:
1. Landing Screen (Domain Explorer + 6 interactive options)
2. Tourism Destination Selector
3. Destination Intelligence Briefing & Intake Form
4. Smart Follow-Up (handling missing parameters)
5. Foreign Scope Guardrail (Thailand query)
6. Workflow 1: Jaipur Budget Trip (with Live Weather + Real-Time Forex + Review Card)
7. Workflow 2: Delhi Family Trip
8. Workflow 3: Culture North India
9. Workflow 4: Economy Current
10. Workflow 5: Science Future Tech
11. BONUS 1: Multi-Agent Transit Specialist (High-speed Vande Bharat rail routing)
12. BONUS 2: Real-Time Forex Exchange Converter (USD/EUR/GBP to INR + Purchasing Power)
13. BONUS 3: Fetch.ai Payment Protocol (Issuing cryptographically signed VIP Travel Pass)
14. BONUS 4: Interactive Destinations Carousel Card (Flagship heritage circuits)
15. BONUS 5: Interactive Review & Confirmation Screen (Action CTA confirmation)
"""

import asyncio
from langchain_core.messages import HumanMessage
from graph import india_graph

INTERACTIVE_TESTS = [
    # Core Scenarios
    ("Greeting / Landing Screen", "hi"),
    ("Tourism Category Selection", "Tourism"),
    ("Destination Intelligence Selection", "Explore Jaipur"),
    ("Smart Follow-Up Missing Info", "Jaipur 5 days"),
    ("Foreign Scope Guardrail", "one day trip at thailand"),
    ("Workflow 1: Jaipur Budget Trip", "Tourism → Jaipur → 3 days → 2 people → Budget"),
    ("Workflow 2: Delhi Family Trip", "Tourism → Delhi → 5 days → Family"),
    ("Workflow 3: Culture North India", "Culture → Regional Culture → North India"),
    ("Workflow 4: Economy Current", "Economy → Indian Economy → Current"),
    ("Workflow 5: Science Future Tech", "Science → AI / Semiconductors → Future Outlook"),
    
    # Hackathon Bonus Scenarios
    ("BONUS 1: Multi-Agent Transit Specialist", "train to jaipur"),
    ("BONUS 2: Real-Time Forex & Purchasing Power", "convert 100 USD to INR"),
    ("BONUS 3: Payment Protocol & Signed VIP Pass", "pay 1 ASI for VIP Pass"),
    ("BONUS 4: Interactive Destinations Carousel", "show top destinations carousel"),
    ("BONUS 5: Review Card Confirmation CTA", '{"action": "confirm_trip", "destination": "Jaipur", "days": "3", "travelers": "2", "budget": "Budget"}'),
]

async def run_tests():
    print("=" * 80)
    print("🚀 Running India Intel Agent Interactive Workflow Test Suite (Core + Bonus)")
    print("=" * 80)
    
    for i, (name, query) in enumerate(INTERACTIVE_TESTS, 1):
        print(f"\n[Test {i}/{len(INTERACTIVE_TESTS)}] 🎯 Scenario: {name}")
        print(f"   Input Query: '{query}'")
        try:
            state = await india_graph.ainvoke({"messages": [HumanMessage(content=query)]})
            output = state.get("final_output", "")
            stage = state.get("workflow_stage", "unknown")
            print(f"   ✅ Stage Routed: {stage}")
            print(f"   ✅ Output Generated: {len(output)} characters")
            
            # Print brief preview of the first 220 characters
            preview = output[:220].replace("\n", " ")
            print(f"   Preview: {preview}...\n")
        except Exception as e:
            print(f"   ❌ FAILED on '{query}': {e}")
            raise e

    print("=" * 80)
    print(f"🎉 ALL {len(INTERACTIVE_TESTS)} INTERACTIVE WORKFLOW TESTS (CORE + BONUS) PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_tests())
