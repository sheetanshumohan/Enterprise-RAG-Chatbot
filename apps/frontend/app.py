"""
AI Knowledge Assistant - Streamlit frontend entrypoint.

This file only handles auth (login/register) and session bootstrap.
Once logged in, the real UI lives in pages/ (Streamlit's multipage
convention: files in pages/ show up in the sidebar automatically).
"""
from __future__ import annotations

import streamlit as st

from api_client import ApiClient, ApiError, init_auth, logout

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠", layout="wide")


def _init_session_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("active_collection_id", None)
    st.session_state.setdefault("active_session_id", None)


_init_session_state()
st.session_state._active_page_tag = "home"
client = init_auth()


def _login_register_view() -> None:
    st.title("🧠 AI Knowledge Assistant")
    st.caption("Your personal, private RAG-powered knowledge base.")

    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            try:
                token = ApiClient().login(email, password)
                st.session_state.token = token
                st.session_state.user = ApiClient(token).me()
                if hasattr(st, "query_params"):
                    st.query_params["auth"] = token
                st.rerun()
            except ApiError as exc:
                st.error(f"Login failed: {exc.detail}")
            except Exception as exc:  # connection errors, etc.
                st.error(f"Could not reach the backend API: {exc}")

    with tab_register:
        with st.form("register_form"):
            full_name = st.text_input("Full name (optional)", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password (min 8 characters)", type="password", key="reg_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            try:
                token = ApiClient().register(email, password, full_name)
                st.session_state.token = token
                st.session_state.user = ApiClient(token).me()
                if hasattr(st, "query_params"):
                    st.query_params["auth"] = token
                st.success("Account created!")
                st.rerun()
            except ApiError as exc:
                st.error(f"Registration failed: {exc.detail}")
            except Exception as exc:
                st.error(f"Could not reach the backend API: {exc}")


def _dashboard_view() -> None:
    user = st.session_state.user
    st.title("🧠 AI Knowledge Assistant")
    st.write(f"Welcome back, **{user.get('full_name') or user['email']}**.")

    col1, col2, col3 = st.columns(3)
    active_client = ApiClient(st.session_state.token)
    try:
        collections = active_client.list_collections()
        documents = active_client.list_documents()
        sessions = active_client.list_sessions()
    except ApiError as exc:
        st.error(f"Could not load dashboard data: {exc.detail}")
        return

    col1.metric("Collections", len(collections))
    col2.metric("Documents", len(documents))
    col3.metric("Conversations", len(sessions))

    st.divider()
    st.subheader("Get started")
    st.markdown(
        "- **Collections** — organize your documents into folders like *Work*, *Research*, *Personal*\n"
        "- **Upload Documents** — add PDF, DOCX, Markdown, or TXT files\n"
        "- **AI Chat** — ask questions grounded in your own documents, with citations\n"
        "- **My Documents** — view, tag, and manage what you've uploaded\n\n"
        "Use the sidebar to navigate."
    )

    if st.button("Log out"):
        logout()
        st.rerun()


if st.session_state.token and st.session_state.user:
    _dashboard_view()
else:
    _login_register_view()
