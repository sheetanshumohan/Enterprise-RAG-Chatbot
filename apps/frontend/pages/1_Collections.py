from __future__ import annotations

import streamlit as st

from api_client import ApiClient, ApiError, init_auth

st.set_page_config(page_title="Collections", page_icon="📁", layout="wide")

client = init_auth()
st.session_state._active_page_tag = "collections"
if not client or not st.session_state.get("token"):
    st.warning("Please log in from the home page first.")
    st.stop()

st.title("📁 Collections")
st.caption("Organize your documents. The AI Chat only searches the collection you select (or all of them).")

with st.form("new_collection"):
    c1, c2 = st.columns([2, 3])
    name = c1.text_input("Collection name", placeholder="e.g. Research Papers")
    description = c2.text_input("Description (optional)", placeholder="e.g. ML papers I'm reading")
    submitted = st.form_submit_button("Create collection")
if submitted:
    if not name.strip():
        st.error("Collection name is required.")
    else:
        try:
            client.create_collection(name.strip(), description.strip())
            st.success(f"Created collection '{name}'")
            st.rerun()
        except ApiError as exc:
            st.error(f"Could not create collection: {exc.detail}")

st.divider()

try:
    collections = client.list_collections()
except ApiError as exc:
    st.error(f"Could not load collections: {exc.detail}")
    st.stop()

if not collections:
    st.info("No collections yet. Create one above, e.g. **Work**, **Research Papers**, **Personal**.")
else:
    for collection in collections:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 5, 1])
            c1.markdown(f"**{collection['name']}**")
            c2.caption(collection["description"] or "_no description_")
            if c3.button("Delete", key=f"del_{collection['id']}"):
                try:
                    with st.spinner("Deleting collection..."):
                        client.delete_collection(collection["id"])
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Could not delete: {exc.detail}")
                except Exception as exc:
                    st.error(f"Could not delete collection: {exc}")
