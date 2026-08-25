from __future__ import annotations

import streamlit as st

from api_client import API_URL, ApiClient, ApiError, init_auth, logout

st.set_page_config(page_title="Profile & Settings", page_icon="⚙️", layout="wide")

client = init_auth()
if not client or not st.session_state.get("token"):
    st.warning("Please log in from the home page first.")
    st.stop()

st.title("⚙️ Profile & Settings")

st.subheader("Profile")
try:
    user = client.me()
    st.write(f"**Email:** {user['email']}")
    st.write(f"**Name:** {user.get('full_name') or '—'}")
    st.write(f"**User ID:** `{user['id']}`")
except ApiError as exc:
    st.error(f"Could not load profile: {exc.detail}")

st.divider()
st.subheader("Connection")
st.write(f"**Backend API:** `{API_URL}`")
st.caption("Change this by setting the KNOWLEDGE_ASSISTANT_API_URL environment variable before starting Streamlit.")

st.divider()
st.subheader("Session")
if st.button("Log out", type="primary"):
    logout()
    st.switch_page("app.py")
