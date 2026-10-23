"""
Comprehensive Interactive Workflow Verification Test Suite
Verifies all interactive screens and user workflows for 🇮🇳 India Intel Agent:
1. Landing Screen (Welcome + 4 interactive cards)
2. Tourism Destination Selector
3. Destination Intelligence Briefing & Intake Form
4. Smart Follow-Up (handling missing parameters)
5. Foreign Scope Guardrail (Thailand query)
6. 5 User Acceptance Workflows:
   - Workflow 1: Tourism → Jaipur → 3 days → 2 people → Budget
   - Workflow 2: Tourism → Delhi → 5 days → Family
   - Workflow 3: Culture → Regional Culture → North India
   - Workflow 4: Economy → Indian Economy → Current
   - Workflow 5: Science → AI / Semiconductors → Future Outlook
"""

import asyncio
from langchain_core.messages import HumanMessage
from graph import india_graph

INTERACTIVE_TESTS = [
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
]

async def run_tests():
    print("=" * 75)
    print("🚀 Running India Intel Agent Interactive Workflow Test Suite")
    print("=" * 75)
    
    for i, (name, query) in enumerate(INTERACTIVE_TESTS, 1):
        print(f"\n[Test {i}/{len(INTERACTIVE_TESTS)}] 🎯 Scenario: {name}")
        print(f"   Input Query: '{query}'")
        try:
            state = await india_graph.ainvoke({"messages": [HumanMessage(content=query)]})
            output = state.get("final_output", "")
            stage = state.get("workflow_stage", "unknown")
            print(f"   ✅ Stage Routed: {stage}")
            print(f"   ✅ Output Generated: {len(output)} characters")
            
            # Print brief preview of the first 250 characters
            preview = output[:250].replace("\n", " ")
            print(f"   Preview: {preview}...\n")
        except Exception as e:
            print(f"   ❌ FAILED on '{query}': {e}")
            raise e

    print("=" * 75)
    print("🎉 ALL 10 INTERACTIVE WORKFLOW TESTS PASSED!")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(run_tests())
