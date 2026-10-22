import asyncio
from langchain_core.messages import HumanMessage
from graph import india_graph

async def run_tests():
    test_queries = [
        "I want to visit the Taj Mahal in Agra on Friday. What are the rules and ticket prices?",
        "Plan a 1-day trip to Jaipur covering Amer Fort and local food.",
        "Plan a quick 1-day cultural and ghats itinerary for Varanasi.",
    ]

    for query in test_queries:
        print("\n" + "="*60)
        print(f"👉 Query: {query}")
        print("="*60)
        
        state_result = await india_graph.ainvoke({"messages": [HumanMessage(content=query)]})
        output = state_result.get("final_output", "No response generated.")
        
        print(output)
        print("\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
