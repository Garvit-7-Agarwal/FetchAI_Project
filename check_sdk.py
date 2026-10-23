import inspect
import agentverse_sdk
from agentverse_sdk.langgraph import init as agentverse_init

print("agentverse_sdk location:", agentverse_sdk.__file__)
print("agentverse_init signature:", inspect.signature(agentverse_init))
print("\n--- SOURCE CODE OF agentverse_init ---")
try:
    print(inspect.getsource(agentverse_init))
except Exception as e:
    print("Could not get source:", e)
