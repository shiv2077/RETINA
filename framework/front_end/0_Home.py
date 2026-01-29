# app.py (Home)
# ==============================================================
import streamlit as st
from client import APIClient, AuthExpired  

st.set_page_config(page_title="Home", layout="wide")

API_BASE = st.secrets.get("api_base", "http://127.0.0.1:8000")

# Supported theme keys
PRIMARY = st.get_option("theme.primaryColor")
PRIBG   = st.get_option("theme.backgroundColor")
SECBG   = st.get_option("theme.secondaryBackgroundColor")
TEXT    = st.get_option("theme.textColor")

# Extra (custom) colors from secrets; safe defaults
extras   = st.secrets.get("theme_extras", {})
SECTEXT  = extras.get("secondaryTextColor", "#475569")
SUCCESS  = extras.get("successColor",        "#10B981")
WARNING  = extras.get("warningColor",        "#F59E0B")
ERROR    = extras.get("errorColor",          "#DC2626")
PRILIGHT = extras.get("primaryLight",        "#60A5FA")
PRIDARK  = extras.get("primaryDark",         "#1E40AF")

# Small helpers for visual blocks (inline styles respect theme colors)
def card(title: str, body: str = ""):
    st.markdown(
        f"""
        <div style="
            background:{SECBG};
            color:{TEXT};
            padding:18px 18px 12px;
            border-radius:14px;
            border:1px solid rgba(0,0,0,.06);
        ">
            <div style="font-size:0.95rem; opacity:.8">{title}</div>
            <div style="font-size:1.8rem; font-weight:600; margin-top:6px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def note(message: str, muted: bool = True):
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:18px;
            border-radius:12px;
            background:{PRIBG};
            color:{SECTEXT if muted else TEXT};
            opacity:{0.9 if muted else 1};
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )


if "client" not in st.session_state:
    st.session_state.client = APIClient(API_BASE)
if "anomaly_amount" not in st.session_state:
    st.session_state.anomaly_amount = 0
if "prev_amount" not in st.session_state:
    st.session_state.prev_amount = 0

st.title("MV4QC Project Website")
st.caption(f"API base: {API_BASE}")

with st.sidebar:
    st.subheader("Authenticate Yourself")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    colL, colR = st.columns(2)
    if colL.button("Login"):
        try:
            data = st.session_state.client.login(username, password)
            st.success("Logged in.")
            st.caption(f"Token: {str(st.session_state.client.token)[:12]}...")
            st.toast("Authenticated", icon="✅")
        except Exception as e:
            st.error(f"Login failed: {e}")
    with st.expander("Register new account"):
        r_user = st.text_input("New username", key="r_user")
        r_pass = st.text_input("New password", type="password", key="r_pass")
        if st.button("Register"):
            try:
                resp = st.session_state.client.register(r_user, r_pass or None)
                st.success("Registered. Now login.")
                st.json(resp)
            except Exception as e:
                st.error(f"Register failed: {e}")

st.write("")  # small spacing
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    card("Deployed Models", "3")
with c2:
    st.markdown(
        f"""
        <div style="
            background:{SECBG};
            color:{TEXT};
            padding:18px 18px 12px;
            border-radius:14px;
            border:1px solid rgba(0,0,0,.06);
        ">
            <div style="opacity:.8; font-size:.95rem;">About the TETRA Project</div>
            <div style="margin-top:6px;">
                <a href="https://mv4qc.be/" target="_blank" style="text-decoration:none; color:{PRIDARK}; font-size:1.8rem; font-weight:600;">
                    mv4qc.be ↗
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    card("Detected Anomalies", str(st.session_state.anomaly_amount))

st.divider()
note("Use the pages on the left to navigate.")
