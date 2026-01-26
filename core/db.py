import sqlite3
from pathlib import Path
import os
import sqlite3
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "nova.db"

def connect():
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=10,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn



# absolute path: /<project-root>/data/nova.db
ROOT = Path(__file__).resolve().parent.parent   # core/.. = project root
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "nova.db"

def connect():
    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=10,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")

    return conn
