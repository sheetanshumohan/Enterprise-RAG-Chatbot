from __future__ import annotations

import streamlit as st

from api_client import ApiClient, ApiError, init_auth

st.set_page_config(page_title="Upload Documents & Research Papers", page_icon="⬆️", layout="wide")

client = init_auth()
st.session_state._active_page_tag = "upload"
if not client or not st.session_state.get("token"):
    st.warning("Please log in from the home page first.")
    st.stop()

st.title("⬆️ Ingest Documents & Research Papers")
st.caption("Upload multiple research papers (PDF, DOCX, Markdown, TXT) or import directly from arXiv. Documents are automatically parsed, chunked, embedded, and indexed for hybrid search.")

try:
    collections = client.list_collections()
except ApiError as exc:
    st.error(f"Could not load collections: {exc.detail}")
    st.stop()

if not collections:
    st.warning("Create a collection first (see the **Collections** page in the sidebar) before uploading documents.")
    st.stop()

# Collection selector
collection_names = {c["name"]: c["id"] for c in collections}
col_select, col_info = st.columns([2, 3])
with col_select:
    selected_name = st.selectbox("🎯 Target Collection", list(collection_names.keys()))
    collection_id = collection_names[selected_name]

with col_info:
    desc = next((c.get("description") for c in collections if c["id"] == collection_id), "")
    st.info(f"Targeting: **{selected_name}**" + (f" — _{desc}_" if desc else ""))

tab_files, tab_arxiv = st.tabs(["📚 Upload Files (PDF / Docs)", "🚀 Import from arXiv by ID / URL"])

with tab_files:
    st.subheader("Batch Upload Research Papers & Files")
    
    col_upload, col_meta = st.columns([3, 2])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Choose research papers or files to index (supports multi-select)",
            type=["pdf", "docx", "md", "markdown", "txt"],
            accept_multiple_files=True,
            help="Select one or multiple research papers / documentation files to index.",
        )
    
    with col_meta:
        paper_tags = st.text_input(
            "🏷️ Research Tags (comma-separated)",
            value="research-paper, arxiv",
            help="E.g. LLM, RAG, survey, transformer, cvpr-2025",
        )
        ingest_mode = st.radio(
            "⚡ Ingestion Mode",
            ["Direct Instant Indexing", "Background Worker Queue (Celery)"],
            help="Instant Indexing processes immediately. Background Queue is ideal for massive 100+ page papers.",
        )

    if uploaded_files:
        total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
        st.caption(f"📁 Selected **{len(uploaded_files)}** file(s) — Total size: **{total_size_mb:.2f} MB**")

        if st.button(f"🚀 Upload & Index {len(uploaded_files)} Paper(s)", type="primary"):
            progress_bar = st.progress(0.0, text="Starting batch ingestion...")
            results = []

            for i, f in enumerate(uploaded_files):
                pct = i / len(uploaded_files)
                progress_bar.progress(pct, text=f"Processing paper ({i+1}/{len(uploaded_files)}): {f.name}...")
                file_bytes = f.getvalue()

                try:
                    if ingest_mode == "Background Worker Queue (Celery)":
                        res = client.upload_document_async(collection_id, f.name, file_bytes)
                        results.append({"name": f.name, "status": "⏳ Queued in Worker", "detail": f"Task ID: {res.get('task_id', '')[:8]}...", "chunks": "-"})
                    else:
                        doc = client.upload_document(collection_id, f.name, file_bytes, tags=paper_tags)
                        chunk_count = doc.get("metadata", {}).get("chunk_count", 0)
                        results.append({"name": f.name, "status": "✅ Indexed", "detail": f"Version {doc.get('version', 1)}", "chunks": chunk_count})
                except ApiError as exc:
                    if exc.status_code == 409:
                        results.append({"name": f.name, "status": "⚠️ Duplicate", "detail": "Already indexed in collection", "chunks": "-"})
                    else:
                        results.append({"name": f.name, "status": "❌ Failed", "detail": exc.detail, "chunks": "-"})
                except Exception as exc:
                    results.append({"name": f.name, "status": "❌ Error", "detail": str(exc), "chunks": "-"})

            progress_bar.progress(1.0, text="Batch processing complete!")
            st.success(f"Processed {len(uploaded_files)} paper(s) successfully.")

            # Summary Display Table
            st.write("### 📊 Ingestion Summary")
            for res in results:
                status = res["status"]
                with st.container(border=True):
                    c1, c2, c3 = st.columns([4, 2, 2])
                    c1.markdown(f"**{res['name']}**")
                    c1.caption(res["detail"])
                    c2.write(f"**Status:** {status}")
                    c3.write(f"**Chunks:** {res['chunks']}")

            st.write("")
            if st.button("💬 Start/Continue AI Chat with these papers", key="goto_chat_files", type="primary"):
                st.session_state.active_collection_id = collection_id
                st.switch_page("pages/4_AI_Chat.py")

with tab_arxiv:
    st.subheader("Direct arXiv Research Paper Ingestion")
    st.write("Enter arXiv paper IDs or URLs (one per line or comma-separated). The system will fetch paper abstracts & metadata, download the PDF, chunk it, and index it into your selected collection.")

    arxiv_input = st.text_area(
        "arXiv IDs or URLs",
        placeholder="e.g.\n1706.03762\nhttps://arxiv.org/abs/2312.10997\n2005.11401",
        height=120,
    )
    arxiv_tags_input = st.text_input(
        "🏷️ Tags for arXiv Papers",
        value="arxiv, research-paper, ai-survey",
    )

    if st.button("📥 Fetch & Index arXiv Paper(s)", type="primary"):
        import re

        # Extract all arXiv IDs from input
        raw_ids = re.findall(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arxiv_input)
        if not raw_ids:
            st.error("Please enter at least one valid arXiv ID or URL (e.g. `1706.03762` or `https://arxiv.org/abs/2312.10997`).")
        else:
            tags = [t.strip() for t in arxiv_tags_input.split(",") if t.strip()]
            progress = st.progress(0.0, text="Fetching arXiv papers...")
            arxiv_results = []

            for i, aid in enumerate(raw_ids):
                progress.progress(i / len(raw_ids), text=f"Downloading and indexing arXiv:{aid} ({i+1}/{len(raw_ids)})...")
                try:
                    doc = client.import_arxiv(collection_id, aid, tags=tags)
                    chunks = doc.get("metadata", {}).get("chunk_count", "?")
                    arxiv_results.append({"id": aid, "title": doc.get("filename", aid), "status": "✅ Indexed", "chunks": chunks})
                except ApiError as exc:
                    if exc.status_code == 409:
                        arxiv_results.append({"id": aid, "title": aid, "status": "⚠️ Duplicate", "chunks": "Already indexed"})
                    else:
                        arxiv_results.append({"id": aid, "title": aid, "status": "❌ Failed", "chunks": exc.detail})
                except Exception as exc:
                    arxiv_results.append({"id": aid, "title": aid, "status": "❌ Error", "chunks": str(exc)})

            progress.progress(1.0, text="arXiv ingestion completed!")
            st.success(f"Processed {len(raw_ids)} arXiv paper(s).")

            st.write("### 📜 arXiv Ingestion Results")
            for item in arxiv_results:
                with st.container(border=True):
                    c1, c2 = st.columns([5, 2])
                    c1.markdown(f"**{item['title']}**")
                    c1.caption(f"arXiv ID: {item['id']}")
                    c2.write(f"{item['status']} ({item['chunks']} chunks)")

            st.write("")
            if st.button("💬 Start/Continue AI Chat with these arXiv papers", key="goto_chat_arxiv", type="primary"):
                st.session_state.active_collection_id = collection_id
                st.switch_page("pages/4_AI_Chat.py")
