# core/brain.py
from typing import Any, Dict, List
import random
import os
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from core.prompts import PROMPTS

# -------------------------
# ✅ System identity
# -------------------------
SYSTEM_PROMPT = """You are Nova Human.
You are calm, direct, non-motivational.
You do not repeat yourself.
You ask one focused question at a time.
You help convert vague thoughts into concrete next steps.
"""

# -------------------------
# ✅ Provider config
# -------------------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()

# -------------------------
# ✅ Gemini (lazy init, env-config)
# -------------------------
import google.generativeai as genai
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest").strip()



_gemini_ready = False
_gemini_model = None


def _ensure_gemini():
    global _gemini_ready, _gemini_model
    if _gemini_ready:
        return

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    genai.configure(api_key=key)
    _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    _gemini_ready = True


def _gemini_generate(text: str) -> str:
    _ensure_gemini()

    # free-tier friendliness
    time.sleep(0.35)

    try:
        resp = _gemini_model.generate_content(
            text,
            generation_config={"temperature": 0.4, "max_output_tokens": 600},
        )
        return (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        print("🔥 GEMINI ERROR (attempt 1):", e)
        time.sleep(0.6)
        resp = _gemini_model.generate_content(
            text,
            generation_config={"temperature": 0.4, "max_output_tokens": 600},
        )
        return (getattr(resp, "text", "") or "").strip()


# -------------------------
# ✅ OpenAI (lazy init, env-config)
# -------------------------
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

_openai_client = None


def _ensure_openai():
    global _openai_client
    if _openai_client is not None:
        return
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    _openai_client = OpenAI(api_key=key)


def _openai_generate(text: str) -> str:
    _ensure_openai()
    # Responses API
    resp = _openai_client.responses.create(
        model=OPENAI_MODEL,
        input=text,
    )
    # SDK exposes output_text for convenience in Responses API
    out = getattr(resp, "output_text", "") or ""
    return out.strip()


# -------------------------
# ✅ LLM Call Router (OpenAI primary, Gemini fallback)
# -------------------------
def call_llm(prompt: str) -> str:
    # primary = env selection
    primary = MODEL_PROVIDER

    # fallback strategy (keep both)
    fallback = "gemini" if primary == "openai" else "openai"

    def _call(provider: str) -> str:
        if provider == "openai":
            return _openai_generate(prompt)
        if provider == "gemini":
            return _gemini_generate(prompt)
        raise RuntimeError(f"Unknown provider: {provider}")

    try:
        return _call(primary)
    except Exception as e:
        print(f"⚠️ {primary.upper()} failed, falling back to {fallback.upper()}:", e)
        try:
            return _call(fallback)
        except Exception as e2:
            print("🔥 Both providers failed:", e2)
            return "I paused for a moment. Ask again."


# -------------------------
# ✅ Prompt formatting helpers
# -------------------------
def format_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Gemini generate_content() here is single-string, so we convert messages[] into a prompt.
    Keeps your context builder exactly as-is.
    """
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}:\n{content}")
    parts.append("ASSISTANT:\n")  # strong anchor
    return "\n\n".join(parts)


def _last_assistant_text(recent_messages):
    """Get the most recent assistant message to avoid repetition."""
    for m in reversed(recent_messages or []):
        if m.get("role") == "assistant":
            return (m.get("content") or "").strip()
    return ""


def pick_prompt(state: str, recent_messages) -> str:
    """Pick a prompt from the state pool, avoiding repeating the last assistant text."""
    pool = PROMPTS.get(state) or PROMPTS.get("no_checkin", [])
    last = _last_assistant_text(recent_messages)

    choices = [p for p in pool if p.strip() and p.strip() != last]
    if not choices:
        choices = pool
    return random.choice(choices).strip() if choices else ""


def detect_state(payload: dict) -> str:
    """Minimal, safe heuristics to determine user state."""
    summary = (payload.get("summary") or "").strip()
    direction = payload.get("direction")
    today_action = payload.get("todayAction") or payload.get("today_action")

    if direction:
        return "direction_locked"
    if today_action:
        return "progressing"
    if summary:
        return "stuck_or_idle"
    return "no_checkin"


# -------------------------
# ✅ Main entry
# -------------------------
def mentor_reply(payload: Dict[str, Any]) -> str:
    user_message_raw: str = (payload.get("message") or "").strip()
    user_message_lower: str = user_message_raw.lower().strip()
    task: str = (payload.get("task") or "").strip()

    if not user_message_raw and task != "summarize":
        return "Say one sentence. What do you need help with?"

    # Greeting guard: state-aware greeting responses
    greetings = {"hi", "hello", "hey", "hi nova", "hello nova", "hey nova"}
    if user_message_lower in greetings:
        state = detect_state(payload)
        prompt = pick_prompt(state, payload.get("recent_messages") or [])
        if prompt:
            return prompt

        summary = (payload.get("summary") or "").strip()
        if summary:
            return "I'm here. Last time, you were focused on something important. What do you want to continue?"
        return "I'm here. What's the one thing you want help with right now?"

       # -------------------------
    # ✅ Identity / About guard (NO LLM CALL)
    # -------------------------
    about_triggers = {
        "tell me about yourself",
        "who are you",
        "what are you",
        "what is nova",
        "who is nova",
        "who built you",
        "who created you",
        "who made you",
        "who is your founder",
        "who is your creator",
    }

    # normalize input (strip punctuation + extra spaces)
    normalized = user_message_lower.strip().rstrip("?.!")

    if normalized in about_triggers:
        return (
            "I’m Nova Human. I help people slow down their thinking, "
            "clarify what matters, and take one grounded step at a time. "
            "I don’t motivate or overwhelm. I focus on execution.\n\n"
            "I was created by Suman Singh Dhami."
        )


    # Continue with normal message handling
    summary: str = (payload.get("summary") or "").strip()
    recent: List[Dict[str, str]] = payload.get("recent_messages", []) or []

    # Build messages according to required structure
    messages: List[Dict[str, str]] = []

    # 1️⃣ System identity (always include)
    SYSTEM_INSTRUCTION = """
You are Nova Human — an AI Life Architect focused on clarity, discipline, and follow-through.

Tone:
- Calm, direct, disciplined.
- Slightly philosophical, but never poetic or motivational.
- Speak like a serious mentor, not a chatbot.

Core Goal:
Help the user achieve Mental Clarity and consistent execution through:
1) Direction (30-day focus)
2) Daily action (check-in)
3) Supporting habits (automaticity/effort)

Non-negotiable Rules:
1) Be concise: 2–6 sentences by default. If complexity requires more, use bullet points.
2) Ask at most ONE question per reply (unless the user explicitly asks for multiple things).
3) Do not repeat the same opening prompt in the same session. Rotate prompts.
4) Use memory: If Direction exists, anchor advice to it. If Habits/effort exist, reference them.
5) If the user greets or is vague, do not do generic small talk. Transition into Direction/Habits/Today.
6) If the user is stuck, propose ONE named Archetype and one next step.

Archetypes (use only when appropriate):
- MVP Ship: reduce scope to one shippable step today.
- Discipline Reset: remove distractions; choose one non-negotiable action.
- Clarity Audit: identify the real problem, then pick the next step.
- Momentum Lock: define a 15-minute start and protect it.

Behavior by state:
- If Direction is locked: connect response to Direction + today step.
- If no check-in today: ask for one action the user will complete today.
- If habits exist: ask effort (1–5) only when relevant; do not gamify.
- If user is idle/vague: ask one precise question that forces clarity.

Output style:
- Prefer plain language.
- No emojis.
- No long disclaimers.
""".strip()

    # If you want the model to be aware of long-term memory, prepend summary text
    if summary:
        SYSTEM_INSTRUCTION = f"{SYSTEM_INSTRUCTION}\n\nLong-term memory summary:\n{summary}".strip()

    messages.append({"role": "system", "content": SYSTEM_INSTRUCTION})

    # 3️⃣ Short-term memory (recent chat)
    for m in recent:
        if task == "summarize":
            content_lower = (m.get("content") or "").lower()
            if "summary" in content_lower or "long-term memory" in content_lower:
                continue
        messages.append(
            {
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
            }
        )

    # 4️⃣ Current user input
    messages.append({"role": "user", "content": user_message_raw})

    # Summarize task stays local (KEEP AS-IS)
    if task == "summarize":
        texts = [m.get("content", "") for m in recent]
        filtered = [
            t
            for t in texts
            if "summary" not in (t or "").lower()
            and "long-term memory" not in (t or "").lower()
        ]
        joined = "\n".join(filtered).strip()
        if not joined:
            return ""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", joined)
        if len(sentences) >= 2:
            return " ".join(sentences[:2]).strip()
        return joined[:1000].strip()

    # ✅ Model call (OpenAI/Gemini via router)
    final_prompt = format_messages_to_prompt(messages)
    reply = call_llm(final_prompt)
    return reply
