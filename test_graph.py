"""
Comprehensive Verification Test Suite for 🇮🇳 India Educational AI Intelligence Agent
Tests all 4 major domains + composite state profiles + tourism actions.
"""

import asyncio
from langchain_core.messages import HumanMessage
from graph import india_graph

TEST_QUERIES = [
    # Domain 1: Economy & Financial Markets
    "What are the major sectors of India's economy?",
    "What is UPI and why is it important?",
    
    # Domain 2: Science, Innovation & Future Tech
    "Explain India's semiconductor mission.",
    "Tell me about ISRO and India's major space missions.",
    "Explain India's AI ecosystem.",
    "Explain India's digital public infrastructure.",
    
    # Domain 3: Cultural Heritage & Social Fabric
    "Explain India's cultural diversity.",
    "What are India's major festivals?",
    
    # Domain 4: Tourism, Landmarks & UNESCO Heritage
    "What are the most important UNESCO World Heritage Sites in India?",
    "Plan a 1-day trip to the Taj Mahal in Agra on Saturday with train options and booking buttons.",
    
    # Domain 5: Composite State Dossier
    "Tell me about Rajasthan's economy, culture and tourism.",
]

async def run_domain_tests():
    print("=" * 70)
    print("🚀 Starting India Educational AI Intelligence Agent Verification Tests")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test {i}/{len(TEST_QUERIES)}] ❓ Query: '{query}'")
        try:
            state_result = await india_graph.ainvoke({"messages": [HumanMessage(content=query)]})
            output = state_result.get("final_output", "")
            
            # Print brief sample of the output to verify quality
            preview = output[:300].replace("\n", " ")
            print(f"✅ Success! Generated {len(output)} chars.")
            print(f"   Preview: {preview}...\n")
        except Exception as e:
            print(f"❌ Error on query '{query}': {e}")
            raise e

    print("=" * 70)
    print("🎉 ALL DOMAIN TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

async def run_single_test(idx: int):
    if not (0 <= idx < len(TEST_QUERIES)):
        print(f"❌ Invalid index {idx + 1}. Must be between 1 and {len(TEST_QUERIES)}.")
        return
    query = TEST_QUERIES[idx]
    print(f"\n[Test {idx + 1}/{len(TEST_QUERIES)}] ❓ Query: '{query}'")
    try:
        state_result = await india_graph.ainvoke({"messages": [HumanMessage(content=query)]})
        output = state_result.get("final_output", "")
        print(f"\n✅ Output:\n{output}\n")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        asyncio.run(run_single_test(int(sys.argv[1]) - 1))
    else:
        asyncio.run(run_domain_tests())
