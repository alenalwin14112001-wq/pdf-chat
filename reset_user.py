"""
reset_user.py — Password reset flow for report_gen
Handles: reset token generation, email dispatch, token validation, password update.

Flow:
  1. User enters email → generate_reset_token(email)
  2. Email sent with ?token=<token> link
  3. User clicks link → validate_reset_token(token)
  4. User enters new password → reset_password(token, new_password)

Uses:
  - PostgreSQL (shared DB config from auth_db.py)
  - smtplib for sending email (configure SMTP in .env)
  - bcrypt for re-hashing the new password
"""

import os
import secrets
import smtplib
import bcrypt
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB", "reportgen_db"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", 587))
SMTP_USER     = os.getenv("SMTP_USER", "")           # your Gmail / SMTP address
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")       # app password
APP_BASE_URL  = os.getenv("APP_BASE_URL", "http://localhost:8501")
RESET_TTL_MINUTES = int(os.getenv("RESET_TTL_MINUTES", 30))


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# ── Table init ────────────────────────────────────────────────────────────────

def init_reset_table():
    """
    Creates the password_reset_tokens table if it doesn't exist.
    Call once at startup (add to auth_db.init_db if preferred).
    """
    sql = """
    CREATE TABLE IF NOT EXISTS password_reset_tokens (
        id          SERIAL PRIMARY KEY,
        user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
        token       TEXT UNIQUE NOT NULL,
        expires_at  TIMESTAMP NOT NULL,
        used        BOOLEAN DEFAULT FALSE,
        created_at  TIMESTAMP DEFAULT NOW()
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


# ── Step 1: Generate reset token ──────────────────────────────────────────────

def generate_reset_token(email: str) -> dict:
    """
    Look up the user by email, create a time-limited reset token,
    and send a reset email.

    Returns {"success": True} always — never reveal whether email exists
    (prevents user enumeration attacks).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, full_name FROM users WHERE email = %s AND is_active = TRUE",
                    (email.strip().lower(),),
                )
                row = cur.fetchone()

        if not row:
            # Return success anyway — do not reveal missing accounts
            return {"success": True}

        user_id, full_name = row
        token = secrets.token_urlsafe(48)
        expires_at = datetime.now() + timedelta(minutes=RESET_TTL_MINUTES)

        # Invalidate any existing unused tokens for this user
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE password_reset_tokens SET used = TRUE WHERE user_id = %s AND used = FALSE",
                    (user_id,),
                )
                cur.execute(
                    "INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (%s, %s, %s)",
                    (user_id, token, expires_at),
                )
            conn.commit()

        _send_reset_email(email.strip().lower(), full_name, token)
        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Step 2: Validate token ────────────────────────────────────────────────────

def validate_reset_token(token: str) -> dict:
    """
    Check that the token exists, has not been used, and has not expired.

    Returns {"valid": True, "user_id": int, "email": str} or {"valid": False, "error": str}
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT rt.user_id, u.email
                    FROM password_reset_tokens rt
                    JOIN users u ON rt.user_id = u.id
                    WHERE rt.token = %s
                      AND rt.used = FALSE
                      AND rt.expires_at > NOW()
                    """,
                    (token,),
                )
                row = cur.fetchone()

        if not row:
            return {"valid": False, "error": "This reset link is invalid or has expired."}

        return {"valid": True, "user_id": row[0], "email": row[1]}

    except Exception as e:
        return {"valid": False, "error": str(e)}


# ── Step 3: Reset password ────────────────────────────────────────────────────

def reset_password(token: str, new_password: str) -> dict:
    """
    Validate the token, hash the new password, update the user record,
    and mark the token as used.

    Returns {"success": True} or {"success": False, "error": str}
    """
    if len(new_password) < 8:
        return {"success": False, "error": "Password must be at least 8 characters."}

    validation = validate_reset_token(token)
    if not validation["valid"]:
        return {"success": False, "error": validation["error"]}

    user_id = validation["user_id"]
    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Update password
                cur.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (new_hash, user_id),
                )
                # Mark token as used
                cur.execute(
                    "UPDATE password_reset_tokens SET used = TRUE WHERE token = %s",
                    (token,),
                )
                # Invalidate all active sessions for this user (security best practice)
                cur.execute(
                    "DELETE FROM sessions WHERE user_id = %s",
                    (user_id,),
                )
            conn.commit()

        return {"success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Email sender ──────────────────────────────────────────────────────────────

def _send_reset_email(to_email: str, full_name: str, token: str):
    """
    Send a password reset email via SMTP.
    Configure SMTP_HOST / SMTP_USER / SMTP_PASSWORD in .env
    """
    reset_url = f"{APP_BASE_URL}?reset_token={token}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Reset your ReportGen password"
    msg["From"]    = f"ReportGen <{SMTP_USER}>"
    msg["To"]      = to_email

    plain = f"""Hi {full_name},

We received a request to reset your ReportGen password.

Click the link below to choose a new password (valid for {RESET_TTL_MINUTES} minutes):

{reset_url}

If you didn't request this, you can safely ignore this email.

— The ReportGen Team
"""

    html = f"""
<html><body style="font-family:'DM Sans',sans-serif;background:#0d1117;color:#c8d4de;padding:40px 20px;">
  <div style="max-width:460px;margin:0 auto;background:#13191f;border:0.5px solid #2a3140;border-radius:16px;padding:2rem;">
    <h2 style="font-family:'DM Serif Display',serif;color:#e8edf2;margin:0 0 8px">Reset your password</h2>
    <p style="color:#5a6a7a;font-size:14px;margin:0 0 24px">Hi {full_name}, click the button below to set a new password. This link expires in <strong style="color:#8a9bac">{RESET_TTL_MINUTES} minutes</strong>.</p>
    <a href="{reset_url}"
       style="display:inline-block;background:#3ecf8e;color:#0d1117;text-decoration:none;
              font-weight:500;font-size:14px;padding:10px 24px;border-radius:8px;">
      Reset password →
    </a>
    <p style="color:#3a4a58;font-size:12px;margin:24px 0 0">If you didn't request this, ignore this email. Your password won't change.</p>
  </div>
</body></html>
"""

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, to_email, msg.as_string())


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def show_forgot_password_ui():
    """
    Renders the 'Forgot password?' flow inside Streamlit.
    Handles both:
      - Requesting a reset link (step 1)
      - Entering a new password after clicking the email link (step 3)

    Call this from login_page.py or wherever you want the reset UI.
    """
    import streamlit as st

    # Init reset table on first call
    init_reset_table()

    # ── Check if a reset token is in the URL query params ────────────────────
    params = st.query_params
    token = params.get("reset_token", None)

    if token:
        # ── Step 3 UI: enter new password ────────────────────────────────────
        validation = validate_reset_token(token)

        if not validation["valid"]:
            st.error(validation["error"])
            if st.button("Request a new reset link"):
                st.query_params.clear()
                st.rerun()
            return

        st.subheader("Choose a new password")
        st.caption(f"Resetting password for **{validation['email']}**")

        with st.form("reset_password_form"):
            new_pw  = st.text_input("New password", type="password", placeholder="Min. 8 characters")
            conf_pw = st.text_input("Confirm new password", type="password", placeholder="Re-enter password")
            submitted = st.form_submit_button("Update password →")

            if submitted:
                if not new_pw or not conf_pw:
                    st.error("Please fill in both fields.")
                elif new_pw != conf_pw:
                    st.error("Passwords do not match.")
                else:
                    result = reset_password(token, new_pw)
                    if result["success"]:
                        st.success("Password updated! You can now sign in.")
                        st.query_params.clear()
                    else:
                        st.error(result["error"])

    else:
        # ── Step 1 UI: request reset email ───────────────────────────────────
        st.subheader("Reset your password")
        st.caption("Enter your email and we'll send you a reset link.")

        with st.form("forgot_password_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            submitted = st.form_submit_button("Send reset link →")

            if submitted:
                if not email:
                    st.error("Please enter your email address.")
                else:
                    with st.spinner("Sending..."):
                        generate_reset_token(email)
                    # Always show success — don't reveal if email exists
                    st.success("If that email is registered, a reset link has been sent.")