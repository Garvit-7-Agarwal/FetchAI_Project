import json
import os
from typing import Any, List
from dotenv import load_dotenv
from uagents_core.contrib.protocols.chat import AgentContent, ChatMessage, MetadataContent, TextContent
from uagents_core.contrib.protocols.chat.cards import extract_card_response
import agentverse_sdk.langgraph.content as av_content
import agentverse_sdk.langgraph._app as lg_app
from agentverse_sdk.langgraph import init as agentverse_init
from graph import india_graph

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Ensure MetadataContent (Interactive Cards for ASI:One) is properly mapped
# ---------------------------------------------------------------------------
_orig_convert = av_content._convert_content_block
async def _patched_convert(block, upload=None):
    if block.get("type") == "metadata":
        return MetadataContent(type="metadata", metadata=block.get("metadata", {}))
    return await _orig_convert(block, upload)
av_content._convert_content_block = _patched_convert


# ---------------------------------------------------------------------------
# 2. Strict Single-Card Enforcement & Robust Turn Slicing (Prevents Duplicate Forms)
# ---------------------------------------------------------------------------
def _is_human_message(msg: Any) -> bool:
    """Detects human/user message regardless of whether it is a dict or LangChain message object."""
    if isinstance(msg, dict):
        mtype = str(msg.get("type", "")).lower()
        role = str(msg.get("role", "")).lower()
        return mtype in ("human", "user") or role in ("human", "user")
    if hasattr(msg, "type") and getattr(msg, "type") in ("human", "user"):
        return True
    cls = msg.__class__.__name__.lower()
    return "human" in cls or "user" in cls


def clean_single_turn_content(content: List[AgentContent]) -> List[AgentContent]:
    """
    Ensures at most ONE interactive form card (MetadataContent) is delivered per response.
    If historical cards from earlier turns leak through thread history, only the active
    card and its accompanying message from the current turn are preserved.
    """
    if not content:
        return content

    def is_card(b: Any) -> bool:
        if isinstance(b, MetadataContent):
            return True
        if isinstance(b, dict) and b.get("type") == "metadata":
            return True
        if getattr(b, "type", None) == "metadata":
            return True
        return False

    last_card_idx = -1
    for i, block in enumerate(content):
        if is_card(block):
            last_card_idx = i

    if last_card_idx < 0:
        return content

    prev_card_idx = -1
    for i in range(last_card_idx - 1, -1, -1):
        if is_card(content[i]):
            prev_card_idx = i
            break

    turn_start_idx = prev_card_idx + 1 if prev_card_idx >= 0 else 0
    turn_content = content[turn_start_idx:]

    result = []
    card_found = False
    for block in reversed(turn_content):
        if is_card(block):
            if not card_found:
                result.append(block)
                card_found = True
        else:
            result.append(block)
    result.reverse()
    return result


def _robust_turn_messages_since_last_human(messages: list[Any]) -> list[Any]:
    """Slices thread messages so only messages from the latest human turn are processed."""
    last_human_idx = -1
    for i, msg in enumerate(messages):
        if _is_human_message(msg):
            last_human_idx = i
    if last_human_idx < 0:
        # Fallback: return only the last message rather than leaking the entire history
        return messages[-1:] if messages else []
    return messages[last_human_idx + 1 :]

av_content._turn_messages_since_last_human = _robust_turn_messages_since_last_human

# Patch extract_ai_content in both content module and app module
_orig_extract_ai_content = av_content.extract_ai_content
async def _patched_extract_ai_content(data: dict[str, Any], upload=None) -> list[AgentContent]:
    messages = data.get("messages")
    if isinstance(messages, list) and messages:
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if _is_human_message(msg):
                last_human_idx = i
        if last_human_idx >= 0:
            data = dict(data)
            data["messages"] = messages[last_human_idx + 1 :]
        else:
            data = dict(data)
            data["messages"] = messages[-1:]

    content = await _orig_extract_ai_content(data, upload)
    return clean_single_turn_content(content)

av_content.extract_ai_content = _patched_extract_ai_content
lg_app.extract_ai_content = _patched_extract_ai_content

# Patch application _send_reply as the final gateway before envelope dispatch
_orig_send_reply = lg_app.AgentverseLangGraphApplication._send_reply
async def _safe_send_reply(self, env, content, sender_identity):
    clean_content = clean_single_turn_content(content)
    return await _orig_send_reply(self, env, clean_content, sender_identity)
lg_app.AgentverseLangGraphApplication._send_reply = _safe_send_reply


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