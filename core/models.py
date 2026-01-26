# core/models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from fastapi import Query
from core.db import connect


class ChatSend(BaseModel):
    message: str
    sid: str = Field(default="default")
    system: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatClear(BaseModel):
    sid: str

class ChatDelete(BaseModel):
    sid: str
from typing import List, Dict, Any, Optional, Literal

class HabitsPayload(BaseModel):
    habits: List[Dict[str, Any]] = []

DirectionStatus = Literal["draft", "calibration", "locked"]

class TodayStep(BaseModel):
    text: str
    estimate_min: int = 25
    done: bool = False
    date: str  # YYYY-MM-DD

class Direction(BaseModel):
    title: str
    emotion_30: str = ""
    consequence: str = ""
    duration_days: int = 30

    status: DirectionStatus = "draft"
    created_at: Optional[str] = None
    calibration_ends_at: Optional[str] = None
    locked_at: Optional[str] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    metric_name: str = "Progress"
    metric_target: int = 30
    metric_progress: int = 0

    today_step: Optional[TodayStep] = None


def init_db():
    conn = connect()
    cur = conn.cursor()

    # ---- chat ----
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        sid TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sid TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT,
        FOREIGN KEY(sid) REFERENCES chat_sessions(sid) ON DELETE CASCADE
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_summaries (
    sid TEXT PRIMARY KEY,
    summary TEXT,
    updated_at TEXT
)
""")


    # ---- direction ----
    cur.execute("""
    CREATE TABLE IF NOT EXISTS direction (
        id INTEGER PRIMARY KEY,
        title TEXT,
        emotion_30 TEXT,
        consequence TEXT,
        duration_days INTEGER,
        status TEXT,
        created_at TEXT,
        calibration_ends_at TEXT,
        locked_at TEXT,
        start_date TEXT,
        end_date TEXT,
        metric_name TEXT,
        metric_target INTEGER,
        metric_progress INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS direction_steps (
        day TEXT PRIMARY KEY,
        text TEXT,
        estimate_min INTEGER,
        done INTEGER,
        updated_at TEXT
    )
    """)

    # ---- habits ----
    cur.execute("""
    CREATE TABLE IF NOT EXISTS habits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS habit_days (
        habit_id INTEGER,
        day TEXT,
        done INTEGER,
        updated_at TEXT,
        PRIMARY KEY (habit_id, day),
        FOREIGN KEY(habit_id) REFERENCES habits(id) ON DELETE CASCADE
    )
    """)
    cur.execute("""
CREATE TABLE IF NOT EXISTS habit_effort (
  habit_id INTEGER NOT NULL,
  day TEXT NOT NULL,
  effort INTEGER NOT NULL,
  updated_at TEXT,
  PRIMARY KEY (habit_id, day),
  FOREIGN KEY(habit_id) REFERENCES habits(id) ON DELETE CASCADE
)
""")

    # ---- checkins ----
    cur.execute("""
    CREATE TABLE IF NOT EXISTS checkins (
        day TEXT PRIMARY KEY,
        moved_forward INTEGER,
        today_action TEXT,
        note TEXT,
        updated_at TEXT
    )
    """)

    conn.commit()
    conn.close()

