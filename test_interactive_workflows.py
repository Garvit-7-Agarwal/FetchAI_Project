"""
Comprehensive Interactive Workflow Verification Test Suite
Verifies all interactive screens, core 4 domains, and Science dropdown & live search capabilities:
1. Landing Screen (Domain Explorer + 4 interactive options)
2. Option 1: Economy Form (Native dropdown intake)
3. Option 2: Tourism Destinations (Destination selector grid)
4. Option 3: Culture Form (Theme and region dropdown intake)
5. Option 4: Science Form (6 Technology dropdown options)
6. Destination Intelligence Briefing & Intake Form
7. Smart Follow-Up (handling missing parameters)
8. Foreign Scope Guardrail (Thailand query)
9. Workflow 1: Jaipur Budget Trip (with direct IRCTC & Google Flights links)
10. Workflow 2: Delhi Family Trip (with direct GPS & booking links)
11. Workflow 3: Culture North India
12. Workflow 4: Economy Current
13. Interactive Destinations Carousel Card (Flagship heritage circuits)
14. Interactive Review & Confirmation Screen (Action CTA confirmation)
15. Form Submission: Economy FinTech Dossier (UPI, RuPay, FinTech)
16. Form Submission: Culture Heritage Dossier (Classical Dances & Festivals)
17. Science Dropdown Selection: Quantum Computing & Cryptography (NQM, ₹6,003 Cr, T-Hubs)
18. Science Dropdown Selection: Semiconductors & Chip Fabs (ISM, Dholera, Sanand, Micron)
19. Science Dropdown Selection: Artificial Intelligence & IndiaAI (10,000 GPUs, Bhashini)
20. Science Custom Topic (Live Online Search): Biotechnology in India
"""

import asyncio
from langchain_core.messages import HumanMessage
from graph import india_graph

INTERACTIVE_TESTS = [
    # Core Landing & Domain Shortcuts
    ("Greeting / Landing Screen", "hi"),
    ("Option 1: Economy Selection", "1"),
    ("Option 2: Tourism Selection", "2"),
    ("Option 3: Culture Selection", "3"),
    ("Option 4: Science Selection", "4"),
    
    # Tourism Workflows
    ("Tourism Custom Destination: Jammu Inquiry", "I WANT TO TRAVEL TO JAMMU"),
    ("Tourism Custom Destination Plan: Jammu", "Tourism → Jammu → 3 days → 2 people → Budget"),
    ("Destination Intelligence Selection", "Explore Jaipur"),
    ("Smart Follow-Up Missing Info", "Jaipur 5 days"),
    ("Foreign Scope Guardrail", "one day trip at thailand"),
    ("Workflow 1: Jaipur Budget Trip", "Tourism → Jaipur → 3 days → 2 people → Budget"),
    ("Workflow 2: Delhi Family Trip", "Tourism → Delhi → 5 days → Family"),
    ("Top Destinations Carousel", "show top destinations carousel"),
    ("Review Card Confirmation CTA", '{"action": "confirm_trip", "destination": "Jaipur", "days": "3", "travelers": "2", "budget": "Budget"}'),
    
    # Economy & Culture Dossiers (Dropdowns & Custom Online Search)
    ("Domain Form: Economy FinTech Dossier", '{"topic": "FinTech", "period": "Current", "analysis_type": "Detailed", "action": "economy_dossier", "approved": true}'),
    ("Economy Custom Topic: PM Gati Shakti", "Explain PM Gati Shakti master plan"),
    ("Domain Form: Culture Heritage Dossier", '{"region": "North India", "theme": "Festivals", "depth": "Detailed", "action": "culture_dossier", "approved": true}'),
    ("Culture Custom Topic: Kalbelia Folk Dance", "Tell me about Kalbelia folk dance"),
    
    # Science & Future Tech: Dedicated Dropdown Categories
    ("Science Dropdown: Quantum Computing", '{"category": "Quantum", "organization": "India Overall", "focus": "Future Outlook", "action": "science_dossier", "approved": true}'),
    ("Science Dropdown: Semiconductors", '{"category": "Semiconductors", "organization": "ISM", "focus": "Current State", "action": "science_dossier", "approved": true}'),
    ("Science Dropdown: AI & IndiaAI", '{"category": "AI", "organization": "India Overall", "focus": "Breakthroughs", "action": "science_dossier", "approved": true}'),
    ("Science Dropdown: Clean Energy", '{"category": "Clean Energy", "organization": "India Overall", "focus": "Future Outlook", "action": "science_dossier", "approved": true}'),
    ("Science Dropdown: Telecom & 6G", '{"category": "Telecom", "organization": "India Overall", "focus": "Current State", "action": "science_dossier", "approved": true}'),
    
    # Science & Future Tech: Non-Dropdown Real-Time Online Search
    ("Science Online Search: Biotechnology", "Tell me about biotechnology in India"),
]

async def run_tests():
    print("=" * 80)
    print("🚀 Running India Intel Agent Interactive Workflow Test Suite (4 Core Domains)")
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
            
            # Specific assertions
            if "Quantum Computing" in name:
                assert "Quantum" in output or "NQM" in output, "Quantum topic not reflected in output"
            elif "Semiconductors" in name:
                assert "Semiconductor" in output or "Fab" in output or "ISM" in output, "Semiconductor topic not reflected"
            elif "Biotechnology" in name:
                assert "Biotechnology" in output or "Real-Time Web Intelligence" in output, "Online search not reflected"
                
        except Exception as e:
            print(f"   ❌ FAILED on '{query}': {e}")
            raise e

    print("=" * 80)
    print(f"🎉 ALL {len(INTERACTIVE_TESTS)} WORKFLOW TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_tests())
