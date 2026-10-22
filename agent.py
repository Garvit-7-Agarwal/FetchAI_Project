import os
import asyncio
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from langchain_core.messages import HumanMessage
from graph import india_graph

load_dotenv()

# --- Asyncio event loop fix for Python 3.12 / 3.14 ---
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ----------------------------------------------------

MAILBOX_KEY = os.getenv("AGENTVERSE_MAILBOX_KEY")
SEED = os.getenv("AGENT_SEED", "india_gemini_agent_secret_seed_phrase_2026")

is_valid_mailbox = bool(MAILBOX_KEY and "your_agentverse_mailbox" not in MAILBOX_KEY)

agent = Agent(
    name="india_intelligence_agent",
    seed=SEED,
    mailbox=f"{MAILBOX_KEY}@https://agentverse.ai" if is_valid_mailbox else None,
    loop=loop,
)

# Setup standard Agent Chat Protocol (ACP)
chat_proto = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )

# Message Handler for ASI:One
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    # Acknowledge receipt immediately
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.utcnow(), acknowledged_msg_id=msg.msg_id),
    )

    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started with {sender}")
            welcome_text = (
                "🇮🇳 **Namaste! Welcome to India Intelligence & Travel Concierge (@india)**\n\n"
                "Powered by Google Gemini & LangGraph on the ASI Alliance network.\n\n"
                "Ask me to plan a heritage trip to **Agra**, **Jaipur**, **Delhi**, **Mumbai**, "
                "or **any city across India**!"
            )
            await ctx.send(sender, create_text_chat(welcome_text))

        elif isinstance(item, TextContent):
            user_prompt = item.text
            ctx.logger.info(f"Received query: '{user_prompt}' from {sender}")

            try:
                # Run the LangGraph StateGraph asynchronously
                inputs = {"messages": [HumanMessage(content=user_prompt)]}
                state_result = await india_graph.ainvoke(inputs)

                reply_text = state_result.get(
                    "final_output", "I was unable to process the itinerary. Please try again."
                )
            except Exception as e:
                ctx.logger.error(f"Error during LangGraph execution: {e}")
                reply_text = f"An error occurred while compiling your itinerary: {str(e)}"

            await ctx.send(sender, create_text_chat(reply_text))

        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended with {sender}")

# Include protocol & publish manifest to Agentverse
agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    print(f"\n========================================================")
    print(f"🚀 India Intelligence Agent is running!")
    print(f"📍 Agent Address: {agent.address}")
    print(f"========================================================\n")
    agent.run()