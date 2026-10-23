import json
import os
from dotenv import load_dotenv
from uagents_core.contrib.protocols.chat import ChatMessage, MetadataContent
from uagents_core.contrib.protocols.chat.cards import extract_card_response
import agentverse_sdk.langgraph.content as av_content
from agentverse_sdk.langgraph import init as agentverse_init
from graph import india_graph

load_dotenv()

# 1. Ensure MetadataContent (Interactive Cards for ASI:One) is enabled for outbound responses
_orig_convert = av_content._convert_content_block
async def _patched_convert(block, upload=None):
    if block.get("type") == "metadata":
        return MetadataContent(type="metadata", metadata=block.get("metadata", {}))
    return await _orig_convert(block, upload)
av_content._convert_content_block = _patched_convert

# 2. Ensure inbound form submissions from ASI:One interactive cards are forwarded to LangGraph
_orig_chat_message_text = ChatMessage.text
def _safe_chat_message_text(self) -> str:
    # 1. Check for interactive card submission in MetadataContent FIRST!
    for block in self.content:
        if isinstance(block, MetadataContent):
            resp = extract_card_response(block)
            if resp:
                if resp.selection:
                    return json.dumps(resp.selection)
                if resp.text:
                    return resp.text
            meta = getattr(block, "metadata", {}) or {}
            if "selection" in meta:
                sel = meta["selection"]
                return sel if isinstance(sel, str) else json.dumps(sel)
            if "card_payload" in meta and "selection" in str(meta.get("card_payload", "")):
                try:
                    payload = json.loads(meta["card_payload"])
                    if "selection" in payload:
                        return json.dumps(payload["selection"])
                except Exception:
                    pass

    # 2. If no card submission found, return raw text content
    return _orig_chat_message_text(self).strip()
ChatMessage.text = _safe_chat_message_text

AGENT_URI = os.getenv(
    "AGENT_URI",
    "https://india-intel-agent:533ce0d7182046a6bdd46a621da5bdfa@agentverse.ai/India%20Intel%20Agent",
)

# 1. Initialize Agentverse LangGraph adapter with Mailbox relay
agentverse_init(AGENT_URI, mailbox=True)

# 2. Expose the compiled LangGraph workflow for Agentverse
graph = india_graph

if __name__ == "__main__":
    import sys
    from agentverse_sdk.langgraph import run as agentverse_run

    # Default to 'dev' command if no args passed
    if len(sys.argv) == 1:
        sys.argv.append("dev")

    agentverse_run()