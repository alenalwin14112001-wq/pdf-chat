"""
auth_db.py — PostgreSQL user management for report_gen
Handles: table creation, register, login, session tokens
"""
import os
import hashlib
import secrets
import psycopg2
import bcrypt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB", "reportgen_db"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Create users and sessions tables if they don't exist."""
    sql = """
    CREATE TABLE IF NOT EXISTS users (
        id          SERIAL PRIMARY KEY,
        full_name   TEXT NOT NULL,
        email       TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at  TIMESTAMP DEFAULT NOW(),
        is_active   BOOLEAN DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS sessions (
        id          SERIAL PRIMARY KEY,
        user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
        token       TEXT UNIQUE NOT NULL,
        expires_at  TIMESTAMP NOT NULL,
        created_at  TIMESTAMP DEFAULT NOW()
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


# ── Registration ────────────────────────────────────────────────────────────

def register_user(full_name: str, email: str, password: str) -> dict:
    """
    Register a new user.
    Returns {"success": True, "user_id": int} or {"success": False, "error": str}
    """
    if len(password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters."}

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (full_name, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
                    (full_name.strip(), email.strip().lower(), password_hash),
                )
                user_id = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "user_id": user_id}
    except psycopg2.errors.UniqueViolation:
        return {"success": False, "error": "An account with this email already exists."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Login ────────────────────────────────────────────────────────────────────

def login_user(email: str, password: str) -> dict:
    """
    Verify credentials and create a session token.
    Returns {"success": True, "token": str, "user": dict} or {"success": False, "error": str}
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, full_name, email, password_hash, is_active FROM users WHERE email = %s",
                    (email.strip().lower(),),
                )
                row = cur.fetchone()

        if not row:
            return {"success": False, "error": "Invalid email or password."}

        user_id, full_name, user_email, password_hash, is_active = row

        if not is_active:
            return {"success": False, "error": "This account has been deactivated."}

        if not bcrypt.checkpw(password.encode(), password_hash.encode()):
            return {"success": False, "error": "Invalid email or password."}

        # Create session token (expires in 24 hours)
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sessions (user_id, token, expires_at) VALUES (%s, %s, %s)",
                    (user_id, token, expires_at),
                )
            conn.commit()

        return {
            "success": True,
            "token": token,
            "user": {"id": user_id, "full_name": full_name, "email": user_email},
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Session validation ───────────────────────────────────────────────────────

def validate_session(token: str) -> dict | None:
    """
    Return user dict if token is valid and not expired, else None.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT u.id, u.full_name, u.email
                    FROM sessions s
                    JOIN users u ON s.user_id = u.id
                    WHERE s.token = %s AND s.expires_at > NOW() AND u.is_active = TRUE
                    """,
                    (token,),
                )
                row = cur.fetchone()
        if row:
            return {"id": row[0], "full_name": row[1], "email": row[2]}
        return None
    except Exception:
        return None


def logout_user(token: str):
    """Delete session token from DB."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE token = %s", (token,))
            conn.commit()
    except Exception:
        pass