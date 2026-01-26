# core/context.py
from typing import Any, Dict
from datetime import date

from core.db import connect
from core.chat_store_sqlite import get_summary, get_messages


def today_key() -> str:
    return date.today().isoformat()

def build_chat_context(sid: str = "default") -> Dict[str, Any]:
    sid = (sid or "default").strip() or "default"
    today = today_key()
    summary = get_summary(sid)
    recent = get_messages(sid, limit=12)

    return {
    "summary": summary,
    "recent_messages": recent,
}


    # ---- chat (SQLite) ----
    chat = get_messages(sid)
    recent_user_messages = [m["content"] for m in chat if m.get("role") == "user"][-5:]

    # ---- direction (SQLite) ----
    conn = connect()
    cur = conn.cursor()


    cur.execute("SELECT * FROM direction WHERE id = 1")
    d = cur.fetchone()

    direction = {}
    if d:
        direction = {
            "title": d["title"] or "",
            "emotion_30": d["emotion_30"] or "",
            "consequence": d["consequence"] or "",
            "duration_days": int(d["duration_days"] or 30),
            "status": d["status"] or "draft",
            "created_at": d["created_at"],
            "calibration_ends_at": d["calibration_ends_at"],
            "locked_at": d["locked_at"],
            "start_date": d["start_date"],
            "end_date": d["end_date"],
            "metric_name": d["metric_name"] or "Days completed",
            "metric_target": int(d["metric_target"] or 30),
            "metric_progress": int(d["metric_progress"] or 0),
        }

    # ---- today step (SQLite) ----
    cur.execute(
        "SELECT day,text,estimate_min,done FROM direction_steps WHERE day = ?",
        (today,),
    )
    s = cur.fetchone()
    today_step = None
    if s:
        today_step = {
            "date": s["day"],
            "text": s["text"] or "",
            "estimate_min": int(s["estimate_min"] or 25),
            "done": bool(s["done"]),
        }

    # ---- habits summary (SQLite) ----
    cur.execute("SELECT id FROM habits")
    habit_rows = cur.fetchall()
    total_habits = len(habit_rows)
    habits_done = 0
    for hr in habit_rows:
        cur.execute(
            "SELECT done FROM habit_days WHERE habit_id = ? AND day = ?",
            (hr["id"], today),
        )
        r = cur.fetchone()
        if r and int(r["done"] or 0) == 1:
            habits_done += 1

    # ---- checkin today (SQLite) ----
    cur.execute(
        "SELECT moved_forward,today_action,note FROM checkins WHERE day = ?",
        (today,),
    )
    c = cur.fetchone()
    today_checkin = None
    if c:
        today_checkin = {
            "date": today,
            "moved_forward": bool(c["moved_forward"]),
            "today_action": c["today_action"] or "",
            "note": c["note"] or "",
        }
    conn.close()
    chat = get_messages(sid, limit=30)
    return {
        "sid": sid,
        "today": today,
        "chat": chat,
        "recent_user_messages": recent_user_messages,

        "direction": direction,
        "today_step": today_step,

        "habits_done": habits_done,
        "habits_total": total_habits,

        "today_checkin": today_checkin,
    }