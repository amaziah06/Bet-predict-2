"""Accounts: email + password. Passwords are salted PBKDF2 hashes (never stored in plain text);
logins are random session tokens kept in an HttpOnly cookie, stored hashed in the database."""
import hashlib
import hmac
import os
import re
import secrets
import time

from . import db
from .config import SESSION_DAYS

ITER = 200_000
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s.]{2,}$")


class EmailExists(Exception):
    pass


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email)) and len(email) <= 254


def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ITER)
    return f"pbkdf2${ITER}${salt.hex()}${h.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, iters, salt, h = stored.split("$")
        calc = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(calc.hex(), h)
    except Exception:
        return False


_DUMMY = hash_password("dummy-password-for-timing")


def create_user(email: str, password: str) -> int:
    pw_hash = hash_password(password)
    try:
        with db.connect() as c:
            if db.run(c, "SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                raise EmailExists()
            if db.PG:
                cur = db.run(c, "INSERT INTO users (email, pw_hash, created_at) VALUES (?, ?, ?) RETURNING id",
                             (email, pw_hash, int(time.time())))
                return cur.fetchone()[0]
            cur = db.run(c, "INSERT INTO users (email, pw_hash, created_at) VALUES (?, ?, ?)",
                         (email, pw_hash, int(time.time())))
            return cur.lastrowid
    except db.INTEGRITY_ERRORS:
        raise EmailExists()


def authenticate(email: str, password: str):
    with db.connect() as c:
        row = db.run(c, "SELECT id, email, pw_hash FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        verify_password(password, _DUMMY)  # keep timing similar whether or not the email exists
        return None
    if not verify_password(password, row[2]):
        return None
    return {"id": row[0], "email": row[1]}


def _th(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with db.connect() as c:
        db.run(c, "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
               (_th(token), user_id, now + SESSION_DAYS * 86400))
    return token


def user_for_token(token):
    if not token:
        return None
    with db.connect() as c:
        row = db.run(c, "SELECT u.id, u.email, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id "
                        "WHERE s.token_hash = ?", (_th(token),)).fetchone()
    if not row or row[2] < time.time():
        return None
    return {"id": row[0], "email": row[1]}


def delete_session(token):
    if token:
        with db.connect() as c:
            db.run(c, "DELETE FROM sessions WHERE token_hash = ?", (_th(token),))


def purge_expired():
    with db.connect() as c:
        db.run(c, "DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),))
