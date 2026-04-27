"""
login_page.py — Streamlit login & registration UI for report_gen
Call show_login_page() from app.py before rendering the main app.
"""
import streamlit as st
from auth_db import init_db, login_user, register_user, validate_session, logout_user

# ── Init DB on first run ─────────────────────────────────────────────────────
init_db()


def show_login_page():
    """
    Renders login/register UI.
    Returns True if the user is authenticated, False otherwise.
    Also sets st.session_state.user and st.session_state.auth_token.
    """
    # ── Check existing session ───────────────────────────────────────────────
    if "auth_token" in st.session_state and st.session_state.auth_token:
        user = validate_session(st.session_state.auth_token)
        if user:
            st.session_state.user = user
            return True
        else:
            # Token expired — clear state
            st.session_state.auth_token = None
            st.session_state.user = None

    # ── Page styling ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: #0d1117;
    }

    .auth-card {
        max-width: 420px;
        margin: 3rem auto 0;
        background: #13191f;
        border: 0.5px solid #2a3140;
        border-radius: 16px;
        padding: 2.5rem 2rem;
    }

    .auth-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 26px;
        color: #e8edf2;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .auth-logo span { color: #3ecf8e; }

    .auth-sub {
        font-size: 13px;
        color: #5a6a7a;
        margin-bottom: 1.8rem;
    }

    .stTextInput > label {
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: #8a9bac !important;
    }
    .stTextInput > div > div > input {
        background: #0d1117 !important;
        border: 0.5px solid #2a3140 !important;
        border-radius: 8px !important;
        color: #c8d4de !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3ecf8e !important;
        box-shadow: 0 0 0 1px #3ecf8e22 !important;
    }

    .stButton > button {
        width: 100%;
        background: #3ecf8e !important;
        color: #0d1117 !important;
        font-weight: 500 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover { background: #4ddfa0 !important; }

    .pg-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #0d1117;
        border: 0.5px solid #1e2a35;
        border-radius: 8px;
        padding: 9px 12px;
        margin-top: 1rem;
        font-size: 12px;
        color: #5a6a7a;
    }
    .pg-dot {
        width: 7px; height: 7px;
        background: #3ecf8e;
        border-radius: 50%;
        flex-shrink: 0;
    }

    div[data-testid="stHorizontalBlock"] { gap: 0; }

    footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── Centered card ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-logo">Report<span>Gen</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">AI-powered report generation</div>', unsafe_allow_html=True)

        # Tab switcher (Login / Register)
        tab_login, tab_register = st.tabs(["Sign in", "Register"])

        # ── LOGIN TAB ─────────────────────────────────────────────────────────
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email", placeholder="you@example.com", key="login_email")
                password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
                submitted = st.form_submit_button("Sign in →")

                if submitted:
                    if not email or not password:
                        st.error("Please fill in all fields.")
                    else:
                        with st.spinner("Signing in..."):
                            result = login_user(email, password)
                        if result["success"]:
                            st.session_state.auth_token = result["token"]
                            st.session_state.user = result["user"]
                            st.success(f"Welcome back, {result['user']['full_name']}!")
                            st.rerun()
                        else:
                            st.error(result["error"])

        # ── REGISTER TAB ──────────────────────────────────────────────────────
        with tab_register:
            with st.form("register_form", clear_on_submit=False):
                full_name = st.text_input("Full name", placeholder="Alena Smith", key="reg_name")
                reg_email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                reg_password = st.text_input("Password", type="password", placeholder="Min. 8 characters", key="reg_password")
                reg_confirm = st.text_input("Confirm password", type="password", placeholder="Re-enter password", key="reg_confirm")
                submitted = st.form_submit_button("Create account →")

                if submitted:
                    if not all([full_name, reg_email, reg_password, reg_confirm]):
                        st.error("Please fill in all fields.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        with st.spinner("Creating account..."):
                            result = register_user(full_name, reg_email, reg_password)
                        if result["success"]:
                            st.success("Account created! Please sign in.")
                        else:
                            st.error(result["error"])

        # ── Security badge ────────────────────────────────────────────────────
        st.markdown("""
        <div class="pg-badge">
            <div class="pg-dot"></div>
            <span>Passwords hashed with <strong>bcrypt</strong> · Sessions stored in <strong>PostgreSQL</strong></span>
        </div>
        """, unsafe_allow_html=True)

    return False  # Not authenticated yet


def show_logout_button():
    """Call this in your sidebar when user is logged in."""
    user = st.session_state.get("user", {})
    st.sidebar.markdown(f"**{user.get('full_name', 'User')}**")
    st.sidebar.caption(user.get("email", ""))
    st.sidebar.divider()
    if st.sidebar.button("Sign out"):
        logout_user(st.session_state.get("auth_token", ""))
        st.session_state.auth_token = None
        st.session_state.user = None
        st.rerun()