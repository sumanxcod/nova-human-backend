# main.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, date
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ✅ OpenAI
from openai import OpenAI

# ✅ (Optional) Gemini fallback
import google.generativeai as genai


# Load .env for local dev (Render uses dashboard env vars)
load_dotenv()

# -------------------------
# App
# -------------------------
ENV = os.getenv("ENV", "development").strip().lower()
app = FastAPI(title="Nova Human Backend")

# -------------------------
# CORS
# -------------------------
# IMPORTANT:
# Do NOT add any custom middleware that returns early for OPTIONS.
# CORSMiddleware must handle OPTIONS so it can attach CORS headers.
cors_raw = (os.getenv("CORS_ORIGINS") or "").strip()
if cors_raw:
    allow_origins = [o.strip().rstrip("/") for o in cors_raw.split(",") if o.strip()]
else:
    allow_origins = [
        "https://nova-human-frontend-4.onrender.com",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,  # demo-safe
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Gemini (optional configure once)
# -------------------------
gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
if gemini_key:
    genai.configure(api_key=gemini_key)

# -------------------------
# OpenAI lazy client
# -------------------------
_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip())
    return _openai_client


def openai_answer(text: str) -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return "I’m temporarily unavailable. Please try again in a moment."

    # Pick a sane default; you can override via Render env var OPENAI_MODEL
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    client = _get_openai_client()

    resp = client.responses.create(
        model=model,
        instructions=(
            "You are Nova Human. Calm, direct, helpful. "
            "Answer the user normally (not coaching) unless they ask for coaching."
        ),
        input=text,
    )
    out = (getattr(resp, "output_text", "") or "").strip()
    return out or "I’m here. Tell me what you want to talk about in one sentence."


def simple_answer(text: str) -> str:
    """
    General Q&A (non-coaching) answer.
    Default provider: OpenAI
    Optional fallback: Gemini
    """
    provider = (os.getenv("MODEL_PROVIDER") or "openai").strip().lower()

    # ✅ default: openai
    if provider == "openai":
        try:
            return openai_answer(text)
        except Exception as e:
            print("🔥 openai simple_answer failed:", e)
            return "I’m here. Tell me what you want to talk about in one sentence."

    # optional: gemini fallback
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not gemini_key:
        return "I’m temporarily unavailable. Please try again in a moment."

    model_name = (os.getenv("GEMINI_MODEL") or "models/gemini-flash-lite-latest").strip()
    m = genai.GenerativeModel(model_name)

    prompt = (
        "You are Nova Human. Calm, direct, helpful.\n"
        "Answer the user directly.\n\n"
        f"User: {text}"
    )

    last_err = None
    for delay in (0, 1.0, 2.0):
        try:
            if delay:
                time.sleep(delay)
            r = m.generate_content(prompt)
            out = (getattr(r, "text", "") or "").strip()
            if out:
                return out
            last_err = RuntimeError("Empty response from model")
        except Exception as e:
            last_err = e

    print("🔥 gemini simple_answer failed:", last_err)
    return "I’m here. Tell me what you want to talk about in one sentence."


# -------------------------
# DB + init
# -------------------------
from core.db import connect
from core.models import init_db

# AI brain + context
from core.brain import mentor_reply
from core.context import build_chat_context

# Chat storage (SQLite)
from core.chat_store_sqlite import (
    add_message,
    get_messages,
    list_sessions,
    clear_session,
    delete_session,
    touch_session,
    save_summary,
    maybe_set_title_from_text,
)

# -------------------------
# Helpers
# -------------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat()

def today_iso() -> str:
    return date.today().isoformat()

def compute_end_date(start_day: str, duration_days: int) -> str:
    d0 = date.fromisoformat(start_day)
    return (d0 + timedelta(days=max(1, int(duration_days)) - 1)).isoformat()

def normalize_sid(raw: Optional[str]) -> str:
    s = (raw or "").strip()
    return s if s else "default"

# -------------------------
# Demo-friendly intent routing
# -------------------------
def is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return t in {"hi", "hello", "hey", "hi nova", "hello nova", "hey nova"}

def is_about(text: str) -> bool:
    t = text.strip().lower()
    return t in {
        "who are you",
        "what are you",
        "tell me about yourself",
        "can you tell me about you",
        "what is nova",
        "who is nova",
    }

def looks_like_coaching(text: str) -> bool:
    t = text.lower()
    coaching_words = [
        "stuck", "discipline", "focus", "direction", "habit", "habits",
        "procrast", "goal", "goals", "plan", "planning", "motivation",
        "routine", "consistency", "addiction", "dopamine", "lazy", "burnout"
    ]
    return any(w in t for w in coaching_words)

# -------------------------
# SQLite tuning (stability)
# -------------------------
def _sqlite_pragmas():
    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA busy_timeout=5000;")
        conn.commit()
        conn.close()
    except Exception:
        pass

def migrate_db():
    """
    Adds missing columns safely (idempotent).
    """
    conn = connect()
    cur = conn.cursor()

    try:
        cur.execute("PRAGMA table_info(direction)")
        cols = {row["name"] for row in cur.fetchall()}
    except Exception:
        cols = set()

    def add_col(sql: str):
        try:
            cur.execute(sql)
        except Exception:
            pass

    if "emotion_30" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN emotion_30 TEXT NOT NULL DEFAULT ''")
    if "consequence" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN consequence TEXT NOT NULL DEFAULT ''")

    if "duration_days" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN duration_days INTEGER NOT NULL DEFAULT 30")
    if "status" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN status TEXT NOT NULL DEFAULT 'draft'")
    if "created_at" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN created_at TEXT")
    if "calibration_ends_at" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN calibration_ends_at TEXT")
    if "locked_at" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN locked_at TEXT")
    if "start_date" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN start_date TEXT")
    if "end_date" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN end_date TEXT")
    if "metric_name" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN metric_name TEXT NOT NULL DEFAULT 'Days completed'")
    if "metric_target" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN metric_target INTEGER NOT NULL DEFAULT 30")
    if "metric_progress" not in cols:
        add_col("ALTER TABLE direction ADD COLUMN metric_progress INTEGER NOT NULL DEFAULT 0")

    conn.commit()
    conn.close()

def ensure_direction_row_exists():
    """
    Ensures direction row id=1 exists.
    """
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT id FROM direction WHERE id = 1")
    row = cur.fetchone()
    if not row:
        created_at = now_iso()
        metric_name = "Days completed"
        cur.execute(
            """
            INSERT INTO direction (
              id, title, emotion_30, consequence, duration_days, status, created_at,
              metric_name, metric_target, metric_progress
            ) VALUES (1,'','','',30,'draft',?,?,30,0)
            """,
            (created_at, metric_name),
        )
        conn.commit()

    conn.close()

def get_today_step():
    conn = connect()
    cur = conn.cursor()
    day = today_iso()
    cur.execute("SELECT day,text,estimate_min,done FROM direction_steps WHERE day = ?", (day,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "date": r["day"],
        "text": r["text"],
        "estimate_min": int(r["estimate_min"] or 25),
        "done": bool(r["done"]),
    }

def get_direction_with_step():
    ensure_direction_row_exists()
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM direction WHERE id = 1")
    r = cur.fetchone()
    conn.close()

    ts = get_today_step()

    return {
        "title": r["title"],
        "emotion_30": r["emotion_30"],
        "consequence": r["consequence"],
        "duration_days": int(r["duration_days"] or 30),

        "status": r["status"] or "draft",
        "created_at": r["created_at"],
        "calibration_ends_at": r["calibration_ends_at"],
        "locked_at": r["locked_at"],

        "start_date": r["start_date"],
        "end_date": r["end_date"],

        "metric_name": r["metric_name"] or "Days completed",
        "metric_target": int(r["metric_target"] or 30),
        "metric_progress": int(r["metric_progress"] or 0),

        "today_step": ts,
    }

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
def _startup():
    _sqlite_pragmas()
    init_db()
    migrate_db()
    ensure_direction_row_exists()

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/health/full")
def full_health():
    try:
        conn = connect()
        conn.execute("SELECT 1")
        conn.close()
        return {"ok": True, "db": "ok"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -------------------------
# Models (chat)
# -------------------------
class ChatSend(BaseModel):
    sid: Optional[str] = None
    message: Optional[str] = None
    system: Optional[str] = None
    context: Optional[dict] = None

class ChatClear(BaseModel):
    sid: Optional[str] = None

class ChatDelete(BaseModel):
    sid: Optional[str] = None

# -------------------------
# Sessions list endpoints
# -------------------------
@app.get("/memory/chats")
def get_chats():
    items = list_sessions()
    return {"items": items}

@app.get("/memory/sessions")
def get_sessions():
    items = list_sessions()
    return {"sessions": items, "items": items}

# -------------------------
# READ: must NOT create/touch
# -------------------------
@app.get("/memory/chat")
def read_chat(sid: str = Query(default="default")):
    sid = normalize_sid(sid)
    msgs = get_messages(sid)
    return {"messages": msgs}

# -------------------------
# WRITE: touch/create session ONLY here
# -------------------------
@app.post("/memory/chat")
def send_chat(payload: ChatSend):
    sid = normalize_sid(payload.sid)
    text = (payload.message or "").strip()

    if not text:
        return {"ok": False, "error": "message required"}

    # Always record user message
    touch_session(sid)
    add_message(sid, "user", text)

    # Title best-effort
    try:
        maybe_set_title_from_text(sid, text)
    except Exception:
        pass

    # Demo-friendly short-circuits
    if is_greeting(text):
        assistant = "Hey — I’m here. What do you want to work on?"
        add_message(sid, "assistant", assistant)
        touch_session(sid)
        return {"ok": True, "sid": sid, "assistant_message": assistant, "messages": get_messages(sid)}

    if is_about(text):
        assistant = "I’m Nova Human — calm, direct, and practical. What do you want to work on right now?"
        add_message(sid, "assistant", assistant)
        touch_session(sid)
        return {"ok": True, "sid": sid, "assistant_message": assistant, "messages": get_messages(sid)}

    # Build context (safe)
    try:
        ctx = build_chat_context(sid)
    except Exception:
        ctx = {}

    # Route: coaching vs general
    try:
        if looks_like_coaching(text):
            assistant = mentor_reply(
                {
                    "message": text,
                    "sid": sid,
                    "summary": ctx.get("summary", ""),
                    "recent_messages": ctx.get("recent_messages", []),

                    "direction": ctx.get("direction"),
                    "todayAction": ctx.get("todayAction") or ctx.get("today_action"),
                    "tone": ctx.get("tone"),

                    "system": payload.system,
                    "context": payload.context,
                }
            )
        else:
            assistant = simple_answer(text)

    except Exception as e:
        print("🔥 mentor/general ERROR:", e)
        # Absolute fallback: try general again
        try:
            assistant = simple_answer(text)
        except Exception as e2:
            print("🔥 simple_answer ERROR:", e2)
            assistant = "I hit a temporary error. Please try again."

    if not isinstance(assistant, str) or not assistant.strip():
        assistant = "I’m here. Tell me what you want to do next."

    add_message(sid, "assistant", assistant)
    touch_session(sid)

    # Optional summarization (demo-safe)
    try:
        msgs_for_summary = get_messages(sid, limit=12)
        if len(msgs_for_summary) >= 6:
            summary = mentor_reply({"task": "summarize", "messages": msgs_for_summary})
            if isinstance(summary, str) and summary.strip():
                save_summary(sid, summary)
    except Exception:
        pass

    msgs = get_messages(sid)
    return {
        "ok": True,
        "sid": sid,

        "assistant_message": assistant,
        "assistant_text": assistant,
        "message": assistant,
        "content": assistant,
        "assistant": {"role": "assistant", "content": assistant},

        "messages": msgs,
    }

@app.post("/memory/chat/clear")
def clear_chat_route(payload: ChatClear):
    sid = normalize_sid(payload.sid)
    clear_session(sid)
    return {"ok": True, "sid": sid, "messages": []}

@app.post("/memory/chat/delete")
def delete_chat_route(payload: ChatDelete):
    sid = normalize_sid(payload.sid)
    if sid == "default":
        return {"ok": False, "error": "cannot delete default"}
    delete_session(sid)
    return {"ok": True, "sid": sid}

# -------------------------
# Direction routes (SQLite)
# -------------------------
class DirectionDraftPayload(BaseModel):
    title: Optional[str] = ""
    emotion_30: Optional[str] = ""
    consequence: Optional[str] = ""
    duration_days: Optional[int] = 30
    metric_name: Optional[str] = "Days completed"
    metric_target: Optional[int] = 30

@app.get("/memory/direction")
def get_direction():
    return {"direction": get_direction_with_step()}

@app.post("/memory/direction/draft")
def draft_direction(payload: DirectionDraftPayload):
    ensure_direction_row_exists()
    conn = connect()
    cur = conn.cursor()

    dur = int(payload.duration_days or 30)
    metric_target = int(payload.metric_target or dur)

    cur.execute(
        """
        UPDATE direction SET
          title = ?,
          emotion_30 = ?,
          consequence = ?,
          duration_days = ?,
          status = 'draft',
          metric_name = ?,
          metric_target = ?
        WHERE id = 1
        """,
        (
            payload.title or "",
            payload.emotion_30 or "",
            payload.consequence or "",
            dur,
            payload.metric_name or "Days completed",
            metric_target,
        ),
    )

    conn.commit()
    conn.close()
    return {"direction": get_direction_with_step()}

@app.post("/memory/direction/lock")
def lock_direction(payload: DirectionDraftPayload):
    ensure_direction_row_exists()
    conn = connect()
    cur = conn.cursor()

    dur = int(payload.duration_days or 30)
    metric_target = int(payload.metric_target or dur)
    ends = (datetime.utcnow() + timedelta(hours=24)).isoformat()

    cur.execute(
        """
        UPDATE direction SET
          title = ?,
          emotion_30 = ?,
          consequence = ?,
          duration_days = ?,
          status = 'calibration',
          calibration_ends_at = ?,
          metric_name = ?,
          metric_target = ?
        WHERE id = 1
        """,
        (
            payload.title or "",
            payload.emotion_30 or "",
            payload.consequence or "",
            dur,
            ends,
            payload.metric_name or "Days completed",
            metric_target,
        ),
    )
    conn.commit()
    conn.close()
    return {"direction": get_direction_with_step()}

@app.post("/memory/direction/finalize")
def finalize_direction():
    ensure_direction_row_exists()
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT duration_days, start_date FROM direction WHERE id = 1")
    r = cur.fetchone()
    dur = int((r["duration_days"] or 30))
    start = (r["start_date"] or today_iso())
    end = compute_end_date(start, dur)

    cur.execute(
        """
        UPDATE direction SET
          status = 'locked',
          locked_at = ?,
          start_date = ?,
          end_date = ?,
          calibration_ends_at = NULL
        WHERE id = 1
        """,
        (now_iso(), start, end),
    )

    conn.commit()
    conn.close()
    return {"direction": get_direction_with_step()}

class ProgressAddPayload(BaseModel):
    delta: int = 1

@app.post("/memory/direction/progress/add")
def add_progress(payload: ProgressAddPayload):
    ensure_direction_row_exists()
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT metric_progress FROM direction WHERE id = 1")
    r = cur.fetchone()
    cur_progress = int(r["metric_progress"] or 0)
    new_progress = max(0, cur_progress + int(payload.delta or 0))

    cur.execute("UPDATE direction SET metric_progress = ? WHERE id = 1", (new_progress,))
    conn.commit()
    conn.close()
    return {"metric_progress": new_progress}

class TodayStepPayload(BaseModel):
    text: str
    estimate_min: int = 25

@app.post("/memory/direction/today_step")
def set_today_step(payload: TodayStepPayload):
    day = today_iso()
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT done FROM direction_steps WHERE day = ?", (day,))
    r = cur.fetchone()
    done_val = int(r["done"]) if r else 0

    cur.execute(
        """
        INSERT INTO direction_steps(day,text,estimate_min,done,updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(day) DO UPDATE SET
          text=excluded.text,
          estimate_min=excluded.estimate_min,
          done=?,
          updated_at=excluded.updated_at
        """,
        (day, payload.text.strip(), int(payload.estimate_min or 25), done_val, now_iso(), done_val),
    )

    conn.commit()
    conn.close()
    return {"today_step": get_today_step()}

@app.post("/memory/direction/today_step/done")
def done_today_step():
    day = today_iso()
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT day FROM direction_steps WHERE day = ?", (day,))
    exists = cur.fetchone()
    if not exists:
        cur.execute(
            """
            INSERT INTO direction_steps(day,text,estimate_min,done,updated_at)
            VALUES(?,?,?,?,?)
            """,
            (day, "", 25, 1, now_iso()),
        )
    else:
        cur.execute(
            """
            UPDATE direction_steps
            SET done = 1, updated_at = ?
            WHERE day = ?
            """,
            (now_iso(), day),
        )

    conn.commit()
    conn.close()
    return {"today_step": get_today_step()}

# -------------------------
# Habits routes (SQLite)
# -------------------------
@app.get("/memory/habits")
def get_habits():
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT id,name FROM habits ORDER BY id ASC")
    habits = cur.fetchall()

    out = []
    for h in habits:
        cur.execute("SELECT day,done FROM habit_days WHERE habit_id = ?", (h["id"],))
        rows = cur.fetchall()
        days = {r["day"]: int(r["done"]) for r in rows}

        effort = {}
        try:
            cur.execute("SELECT day, effort FROM habit_effort WHERE habit_id = ?", (h["id"],))
            eff_rows = cur.fetchall()
            effort = {r["day"]: int(r["effort"]) for r in eff_rows}
        except Exception:
            pass

        out.append({"id": str(h["id"]), "name": h["name"], "days": days, "effort": effort})

    conn.close()
    return {"habits": out}

class HabitTogglePayload(BaseModel):
    habit_id: str
    habit_name: str
    value: int  # 0|1

@app.post("/memory/habits/toggle")
def toggle_habit(payload: HabitTogglePayload):
    day = today_iso()
    conn = connect()
    cur = conn.cursor()

    hid = None
    if payload.habit_id and payload.habit_id.isdigit():
        hid = int(payload.habit_id)
        cur.execute("SELECT id FROM habits WHERE id = ?", (hid,))
        if not cur.fetchone():
            hid = None

    if hid is None:
        name = (payload.habit_name or "").strip()
        if not name:
            conn.close()
            return {"ok": False, "error": "habit_name required"}
        cur.execute("INSERT OR IGNORE INTO habits(name, created_at) VALUES(?,?)", (name, now_iso()))
        cur.execute("SELECT id FROM habits WHERE name = ?", (name,))
        hid = int(cur.fetchone()["id"])

    cur.execute(
        """
        INSERT INTO habit_days(habit_id, day, done, updated_at)
        VALUES(?,?,?,?)
        ON CONFLICT(habit_id, day) DO UPDATE SET
          done=excluded.done,
          updated_at=excluded.updated_at
        """,
        (hid, day, 1 if int(payload.value) == 1 else 0, now_iso()),
    )

    conn.commit()
    conn.close()
    return {"ok": True}

class HabitEffortPayload(BaseModel):
    habit_id: str
    day: str
    effort: int  # 1..5

@app.post("/memory/habits/effort")
def set_habit_effort(payload: HabitEffortPayload):
    if not payload.habit_id or not payload.habit_id.isdigit():
        return {"ok": False, "error": "habit_id required"}

    hid = int(payload.habit_id)
    day = (payload.day or today_iso()).strip()
    e = int(payload.effort or 0)
    if e < 1 or e > 5:
        return {"ok": False, "error": "effort must be 1..5"}

    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO habit_effort(habit_id, day, effort, updated_at)
        VALUES(?,?,?,?)
        ON CONFLICT(habit_id, day) DO UPDATE SET
          effort=excluded.effort,
          updated_at=excluded.updated_at
        """,
        (hid, day, e, now_iso()),
    )
    conn.commit()
    conn.close()
    return {"ok": True}

class HabitCreatePayload(BaseModel):
    cue: str
    action: str
    note: str = ""

@app.post("/memory/habits/create")
def create_habit(payload: HabitCreatePayload):
    cue = (payload.cue or "").strip()
    action = (payload.action or "").strip()
    if not cue or not action:
        return {"ok": False, "error": "cue and action required"}

    conn = connect()
    cur = conn.cursor()

    name = f"After {cue}, I will {action}"
    cur.execute("INSERT INTO habits(name, created_at) VALUES(?,?)", (name, now_iso()))
    hid = cur.lastrowid

    conn.commit()
    conn.close()
    return {"ok": True, "id": str(hid), "name": name}

class HabitSuggestPayload(BaseModel):
    sid: str = "default"

@app.post("/memory/habits/suggest_from_direction")
def suggest_habit_from_direction(payload: HabitSuggestPayload):
    sid = normalize_sid(payload.sid)

    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT title, status FROM direction WHERE id = 1")
    r = cur.fetchone()
    conn.close()

    direction_title = (r["title"] if r else "") or ""
    direction_status = (r["status"] if r else "") or ""

    if not direction_title.strip():
        return {"ok": False, "error": "No direction set yet."}

    prompt = f"""
You are Nova Human. Create ONE supporting habit for this Direction.

Direction: {direction_title}
Status: {direction_status}

Rules:
- Return JSON ONLY with keys: cue, action
- cue must start with "After ..."
- action must start with a verb and be small (5–20 minutes)

Example:
{{"cue":"After I finish breakfast","action":"study focused for 15 minutes"}}
"""

    try:
        out = mentor_reply({"task": "habit_suggest", "message": prompt, "sid": sid})
    except Exception:
        out = ""

    if not isinstance(out, str) or "cue" not in out or "action" not in out:
        return {"ok": True, "cue": "After I sit down at my desk", "action": "work on my direction for 15 minutes"}

    import json
    try:
        data = json.loads(out)
        cue = (data.get("cue") or "").strip()
        action = (data.get("action") or "").strip()
        if cue and action:
            return {"ok": True, "cue": cue, "action": action}
    except Exception:
        pass

    return {"ok": True, "cue": "After I sit down at my desk", "action": "work on my direction for 15 minutes"}

# -------------------------
# Check-in routes (SQLite)
# -------------------------
@app.get("/memory/checkin/today")
def get_today_checkin():
    day = today_iso()
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT day,moved_forward,today_action,note FROM checkins WHERE day = ?", (day,))
    r = cur.fetchone()
    conn.close()

    checkin = None
    if r:
        checkin = {
            "date": r["day"],
            "moved_forward": bool(r["moved_forward"]),
            "today_action": r["today_action"] or "",
            "note": r["note"] or "",
        }

    return {"date": day, "checkin": checkin, "escalation_level": 0, "tone": "neutral"}

class TodayCheckinPayload(BaseModel):
    moved_forward: int  # 0|1
    today_action: str = ""
    note: str = ""

@app.post("/memory/checkin/today")
def set_today_checkin(payload: TodayCheckinPayload):
    day = today_iso()
    moved = 1 if int(payload.moved_forward or 0) == 1 else 0
    action = (payload.today_action or "").strip()
    note = (payload.note or "").strip()

    conn = connect()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO checkins(day, moved_forward, today_action, note, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(day) DO UPDATE SET
          moved_forward=excluded.moved_forward,
          today_action=excluded.today_action,
          note=excluded.note,
          updated_at=excluded.updated_at
        """,
        (day, moved, action, note, now_iso()),
    )

    conn.commit()
    conn.close()

    return {
        "ok": True,
        "date": day,
        "checkin": {
            "date": day,
            "moved_forward": bool(moved),
            "today_action": action,
            "note": note,
        },
    }

# -------------------------
# Compatibility: Day history (Sidebar expects these)
# -------------------------
class HistoryDeletePayload(BaseModel):
    day: str

@app.get("/memory/history")
def list_day_history():
    return []

@app.get("/memory/history/{day}")
def read_day_history(day: str):
    return {"messages": []}

@app.post("/memory/history/delete")
def delete_day_history(payload: HistoryDeletePayload):
    return {"ok": True}

# -------------------------
# Misc / debug
# -------------------------
@app.get("/favicon.ico")
def favicon():
    return {}

@app.get("/")
def root():
    return {"ok": True, "status": "Nova Human backend is running"}

@app.get("/debug/version")
def debug_version():
    return {
        "service": "backend-1",
        "cors_origins": allow_origins,
        "ts": datetime.utcnow().isoformat(),
    }
