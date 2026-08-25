from __future__ import annotations

import streamlit as st

from api_client import ApiClient, ApiError, format_asia_time, init_auth

st.set_page_config(page_title="My Documents", page_icon="📄", layout="wide")

client = init_auth()
st.session_state._active_page_tag = "documents"
if not client or not st.session_state.get("token"):
    st.warning("Please log in from the home page first.")
    st.stop()

st.title("📄 My Documents")

try:
    collections = client.list_collections()
except ApiError as exc:
    st.error(f"Could not load collections: {exc.detail}")
    st.stop()

collection_names = {"All collections": None, **{c["name"]: c["id"] for c in collections}}
selected_name = st.selectbox("Filter by collection", list(collection_names.keys()))
collection_id = collection_names[selected_name]

try:
    documents = client.list_documents(collection_id)
except ApiError as exc:
    st.error(f"Could not load documents: {exc.detail}")
    st.stop()

if not documents:
    st.info("No documents yet. Go to **Upload Documents** to add some.")
    st.stop()

status_icons = {"pending": "⏳", "processing": "⚙️", "indexed": "✅", "failed": "❌"}

for document in documents:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
        c1.markdown(f"**{document['filename']}**")
        uploaded_time = format_asia_time(document.get("created_at"))
        c1.caption(f"Type: {document['doc_type'].upper()} · Version {document['version']} · {uploaded_time}")
        c2.write(f"{status_icons.get(document['status'], '')} {document['status'].title()}")
        chunk_count = document.get("metadata", {}).get("chunk_count")
        c3.caption(f"{chunk_count} chunks" if chunk_count else "")
        if c4.button("🗑️", key=f"del_{document['id']}", help="Delete document"):
            try:
                with st.spinner("Deleting document..."):
                    client.delete_document(document["id"])
                st.rerun()
            except ApiError as exc:
                st.error(f"Could not delete: {exc.detail}")
            except Exception as exc:
                st.error(f"Could not delete document: {exc}")

        if document["tags"]:
            c1.caption("Tags: " + ", ".join(document["tags"]))
