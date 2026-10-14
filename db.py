"""Tiny database layer: Postgres when DATABASE_URL is set (persistent), otherwise a local SQLite file.
Free web hosts wipe local files on restart, so SQLite there means accounts are lost."""
import os
import sqlite3
from contextlib import contextmanager

from .config import DATABASE_URL, DATA_DIR

PG = DATABASE_URL.startswith("postgresql://")
WARNING = None
if PG:
    try:
        import psycopg2
        INTEGRITY_ERRORS = (psycopg2.IntegrityError,)
    except ImportError:  # requirements not installed: fall back instead of crashing
        PG = False
        WARNING = "psycopg2 is not installed, so DATABASE_URL is ignored and accounts are NOT saved permanently."
if not PG:
    INTEGRITY_ERRORS = (sqlite3.IntegrityError,)
    if WARNING is None and os.getenv("RENDER"):
        WARNING = ("No DATABASE_URL set: accounts are stored in a temporary file and will be lost "
                   "whenever the server restarts.")

SQLITE_PATH = DATA_DIR / "users.db"


@contextmanager
def connect():
    if PG:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    else:
        conn = sqlite3.connect(SQLITE_PATH, timeout=10)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run(conn, sql, params=()):
    """Execute with '?' placeholders on both backends."""
    cur = conn.cursor()
    cur.execute(sql.replace("?", "%s") if PG else sql, params)
    return cur


def init():
    pk = "SERIAL PRIMARY KEY" if PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with connect() as c:
        run(c, f"CREATE TABLE IF NOT EXISTS users (id {pk}, email TEXT UNIQUE NOT NULL, "
               "pw_hash TEXT NOT NULL, created_at BIGINT NOT NULL)")
        run(c, "CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, "
               "user_id INTEGER NOT NULL, expires_at BIGINT NOT NULL)")
        run(c, "CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id)")
        run(c, f"CREATE TABLE IF NOT EXISTS slips (id {pk}, user_id INTEGER NOT NULL, fixture_id BIGINT NOT NULL, "
               "match_date TEXT NOT NULL, league TEXT, home TEXT, away TEXT, kickoff TEXT, market TEXT NOT NULL, "
               "bet_key TEXT NOT NULL, label TEXT, probability DOUBLE PRECISION, created_at BIGINT NOT NULL, "
               "status TEXT NOT NULL DEFAULT 'pending', hg INTEGER, ag INTEGER)")
        run(c, "CREATE INDEX IF NOT EXISTS slips_user ON slips(user_id)")
        run(c, "CREATE TABLE IF NOT EXISTS pred_log (fixture_id BIGINT PRIMARY KEY, match_date TEXT NOT NULL, "
               "league TEXT, home TEXT, away TEXT, kickoff TEXT, pick TEXT, pick_prob DOUBLE PRECISION, "
               "p_home DOUBLE PRECISION, p_draw DOUBLE PRECISION, p_away DOUBLE PRECISION, "
               "surest INTEGER NOT NULL DEFAULT 0, s_market TEXT, s_key TEXT, s_label TEXT, s_prob DOUBLE PRECISION, "
               "logged_at BIGINT NOT NULL, hg INTEGER, ag INTEGER, settled INTEGER NOT NULL DEFAULT 0)")
        run(c, "CREATE INDEX IF NOT EXISTS pred_log_date ON pred_log(match_date)")


def kind():
    return "postgres" if PG else "sqlite"
