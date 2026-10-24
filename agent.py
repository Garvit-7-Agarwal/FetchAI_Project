import json
import os
import socket
import sys
import time
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
# 0. Singleton Process Guard: Prevents Multiple agent.py Daemons from Running
# ---------------------------------------------------------------------------
_SINGLETON_SOCKET = None

def acquire_singleton_lock(port: int = 49155) -> None:
    """Ensures only ONE instance of agent.py runs at any time, preventing duplicate mailbox handlers."""
    global _SINGLETON_SOCKET
    _SINGLETON_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _SINGLETON_SOCKET.bind(("127.0.0.1", port))
    except OSError:
        print(f"⚠️ [SINGLETON GUARD] An agent.py daemon is already running on port {port}. Exiting to prevent duplicate responses.")
        sys.exit(0)


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
    STRICT SINGLE-CARD ENFORCEMENT:
    Ensures that at most ONE interactive form card (MetadataContent) is present in ANY reply.
    If multiple cards exist, keeps only the latest one and discards all duplicates.
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

    last_card = None
    non_cards = []

    for block in content:
        if is_card(block):
            last_card = block  # Always keep ONLY the latest card
        else:
            non_cards.append(block)

    result = non_cards
    if last_card is not None:
        result.append(last_card)

    return result


def _robust_turn_messages_since_last_human(messages: list[Any]) -> list[Any]:
    """Slices thread messages so only messages from the latest human turn are processed."""
    last_human_idx = -1
    for i, msg in enumerate(messages):
        if _is_human_message(msg):
            last_human_idx = i
    if last_human_idx < 0:
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


# ---------------------------------------------------------------------------
# 3. Message Deduplication Gateways (Envelope & Outgoing Reply Caches)
# ---------------------------------------------------------------------------
_SEEN_MSG_IDS = {}
_RECENT_REPLIES = {}

_orig_process_and_reply = lg_app.AgentverseLangGraphApplication._process_and_reply
async def _safe_process_and_reply(self, env, msg):
    msg_id_str = str(msg.msg_id)
    now = time.time()

    # Prune old IDs older than 120 seconds
    expired = [k for k, t in _SEEN_MSG_IDS.items() if now - t > 120]
    for k in expired:
        _SEEN_MSG_IDS.pop(k, None)

    if msg_id_str in _SEEN_MSG_IDS:
        lg_app.logger.warning(f"Duplicate incoming message {msg_id_str} ignored to prevent duplicate card responses.")
        return

    _SEEN_MSG_IDS[msg_id_str] = now
    return await _orig_process_and_reply(self, env, msg)

lg_app.AgentverseLangGraphApplication._process_and_reply = _safe_process_and_reply


_orig_send_reply = lg_app.AgentverseLangGraphApplication._send_reply
async def _safe_send_reply(self, env, content, sender_identity):
    clean_content = clean_single_turn_content(content)
    session_key = str(env.session or env.sender)
    now = time.time()

    # Suppress duplicate identical card within a 2.0-second window
    card_meta = None
    for b in clean_content:
        if isinstance(b, MetadataContent):
            card_meta = str(getattr(b, "metadata", ""))
            break

    if card_meta:
        last_meta, last_time = _RECENT_REPLIES.get(session_key, (None, 0))
        if last_meta == card_meta and (now - last_time < 2.0):
            lg_app.logger.warning("Duplicate card reply suppressed within 2.0s window.")
            clean_content = [b for b in clean_content if not isinstance(b, MetadataContent)]
            if not clean_content:
                return

        _RECENT_REPLIES[session_key] = (card_meta, now)

    return await _orig_send_reply(self, env, clean_content, sender_identity)

lg_app.AgentverseLangGraphApplication._send_reply = _safe_send_reply


# ---------------------------------------------------------------------------
# 4. Form Submission Forwarding
# ---------------------------------------------------------------------------
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
    acquire_singleton_lock()
    from agentverse_sdk.langgraph import run as agentverse_run

    # Default to 'dev' command if no args passed
    if len(sys.argv) == 1:
        sys.argv.append("dev")

    agentverse_run()