# core/chat_store_sqlite.py
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.db import connect

def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

def ensure_session(sid: str, title: str = "New chat"):
    sid = (sid or "default").strip() or "default"
    with connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO chat_sessions (sid, title, created_at, updated_at)
            VALUES (?, ?, datetime('now'), datetime('now'))
            """,
            (sid, title),
        )
        conn.commit()

def touch_session(sid: str, title_if_empty: Optional[str] = None):
    sid = (sid or "default").strip() or "default"
    ensure_session(sid)
    with connect() as conn:
        conn.execute("UPDATE chat_sessions SET updated_at = datetime('now') WHERE sid = ?", (sid,))
        if title_if_empty:
            conn.execute(
                """
                UPDATE chat_sessions
                SET title = CASE
                    WHEN title IS NULL OR trim(title) = '' OR title = 'New chat' THEN ?
                    ELSE title
                END
                WHERE sid = ?
                """,
                (title_if_empty.strip()[:28], sid),
            )
        conn.commit()

def add_message(sid: str, role: str, content: str):
    sid = (sid or "default").strip() or "default"
    if role not in ("user", "assistant"):
        raise ValueError("role must be 'user' or 'assistant'")
    content = (content or "").strip()
    if not content:
        return

    ensure_session(sid)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (sid, role, content, created_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (sid, role, content),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = datetime('now') WHERE sid = ?", (sid,))
        conn.commit()

def get_messages(sid: str, limit: int = 2000) -> List[Dict[str, Any]]:
    sid = (sid or "default").strip() or "default"
    ensure_session(sid)
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE sid = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (sid, limit),
        ).fetchall()

    return [{"role": r["role"], "content": r["content"], "ts": r["created_at"]} for r in rows]

def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    with connect() as conn:
        sessions = conn.execute(
            """
            SELECT sid, title, created_at, updated_at
            FROM chat_sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for s in sessions:
            last = conn.execute(
                """
                SELECT content
                FROM chat_messages
                WHERE sid = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (s["sid"],),
            ).fetchone()

            count = conn.execute(
                "SELECT COUNT(*) as c FROM chat_messages WHERE sid = ?",
                (s["sid"],),
            ).fetchone()["c"]

            out.append(
                {
                    "id": s["sid"],
                    "sid": s["sid"],
                    "title": s["title"] or "New chat",
                    "last": (last["content"][:80] if last else ""),
                    "updated_at": s["updated_at"],
                    "count": int(count),
                }
            )

    return out

def clear_session(sid: str):
    sid = (sid or "default").strip() or "default"
    ensure_session(sid)
    with connect() as conn:
        conn.execute("DELETE FROM chat_messages WHERE sid = ?", (sid,))
        conn.execute("UPDATE chat_sessions SET updated_at = datetime('now') WHERE sid = ?", (sid,))
        conn.commit()

def delete_session(sid: str):
    sid = (sid or "").strip()
    if not sid:
        return
    with connect() as conn:
        conn.execute("DELETE FROM chat_messages WHERE sid = ?", (sid,))
        conn.execute("DELETE FROM chat_sessions WHERE sid = ?", (sid,))
        conn.commit()
def maybe_set_title_from_text(sid: str, text: str):
    title = (text or "").strip()
    if not title:
        return
    title = title[:28]
    touch_session(sid, title_if_empty=title)

def get_summary(sid: str) -> str:
    with connect() as conn:
        r = conn.execute(
            "SELECT summary FROM chat_summaries WHERE sid = ?",
            (sid,),
        ).fetchone()
    return r["summary"] if r and r["summary"] else ""

def save_summary(sid: str, summary: str):
    if not summary.strip():
        return
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_summaries(sid, summary, updated_at)
            VALUES(?,?,datetime('now'))
            ON CONFLICT(sid) DO UPDATE SET
              summary=excluded.summary,
              updated_at=excluded.updated_at
            """,
            (sid, summary.strip()),
        )
        conn.commit()

def clear_summary(sid: str):
    with connect() as conn:
        conn.execute("DELETE FROM chat_summaries WHERE sid = ?", (sid,))
        conn.commit()
