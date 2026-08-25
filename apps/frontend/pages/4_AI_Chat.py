from __future__ import annotations

import streamlit as st

from api_client import ApiClient, ApiError, format_asia_time, init_auth

st.set_page_config(page_title="AI Research Chat", page_icon="💬", layout="wide")

client = init_auth()
if not client or not st.session_state.get("token"):
    st.warning("Please log in from the home page first.")
    st.stop()

# Track page tag and per-session paper selections
st.session_state._active_page_tag = "ai_chat"
st.session_state.setdefault("conversation_papers", {})

# Fetch all user sessions
try:
    sessions = client.list_sessions()
except ApiError as exc:
    st.error(f"Could not load conversations: {exc.detail}")
    sessions = []

# Ensure active session exists and is valid
session_ids = [s["id"] for s in sessions]
if not st.session_state.get("active_session_id") or st.session_state.active_session_id not in session_ids:
    if sessions:
        st.session_state.active_session_id = sessions[0]["id"]
    else:
        try:
            new_sess = client.create_session(st.session_state.get("active_collection_id"))
            st.session_state.active_session_id = new_sess["id"]
            sessions = client.list_sessions()
        except Exception:
            pass

session_id = st.session_state.get("active_session_id")
active_session_obj = next((s for s in sessions if s["id"] == session_id), None) if session_id else None

# Sync collection on session change
if active_session_obj:
    sess_coll_id = active_session_obj.get("collection_id")
    if st.session_state.get("_last_synced_session_id") != session_id:
        st.session_state._last_synced_session_id = session_id
        if sess_coll_id is not None:
            st.session_state.active_collection_id = sess_coll_id

# Load history early for active session to extract memory & referenced research papers
history: list[dict] = []
referenced_doc_ids: set[str] = set()
referenced_doc_names: dict[str, str] = {}

if session_id:
    try:
        history = client.get_history(session_id)
        for msg in history:
            for cite in msg.get("citations", []):
                d_id = cite.get("document_id")
                d_name = cite.get("document_filename")
                if d_id:
                    referenced_doc_ids.add(d_id)
                    if d_name:
                        referenced_doc_names[d_id] = d_name
    except ApiError:
        history = []

# --- Sidebar: Collection Scope + Automatic Paper Linkage + Conversations ---
with st.sidebar:
    st.subheader("🎯 Retrieval Scope")
    try:
        collections = client.list_collections()
    except ApiError as exc:
        st.error(f"Could not load collections: {exc.detail}")
        collections = []

    scope_names = {"Search all collections": None, **{c["name"]: c["id"] for c in collections}}
    scope_keys = list(scope_names.keys())

    # Find the index corresponding to the active collection
    cur_coll = st.session_state.get("active_collection_id")
    default_idx = 0
    for idx, (name, cid) in enumerate(scope_names.items()):
        if cid == cur_coll:
            default_idx = idx
            break

    scope_choice = st.selectbox(
        "📁 Collection",
        scope_keys,
        index=default_idx,
        key=f"scope_sel_{session_id}",
    )
    st.session_state.active_collection_id = scope_names[scope_choice]

    # Fetch available documents in this collection/all
    try:
        available_docs = client.list_documents(st.session_state.active_collection_id)
    except ApiError:
        available_docs = []

    selected_doc_ids: list[str] | None = None
    if available_docs:
        doc_options = {f"📄 {d['filename']}": d["id"] for d in available_docs}
        
        # Restore papers for this session: from saved session state OR referenced citations
        saved_papers = st.session_state.conversation_papers.get(session_id)
        if saved_papers is not None:
            default_papers = [lbl for lbl in saved_papers if lbl in doc_options]
        elif referenced_doc_ids:
            default_papers = [lbl for lbl, did in doc_options.items() if did in referenced_doc_ids]
            st.session_state.conversation_papers[session_id] = default_papers
        else:
            default_papers = []

        selected_paper_labels = st.multiselect(
            "📚 Active Research Papers",
            options=list(doc_options.keys()),
            default=default_papers,
            key=f"papers_ms_{session_id}",
            placeholder="All papers indexed in scope (or pick specific)",
            help="Select specific research papers to focus the AI retrieval and memory.",
        )
        # Update session memory of selected papers
        if session_id:
            st.session_state.conversation_papers[session_id] = selected_paper_labels

        if selected_paper_labels:
            selected_doc_ids = [doc_options[lbl] for lbl in selected_paper_labels if lbl in doc_options]
            st.caption(f"🔬 Scoped to **{len(selected_doc_ids)}** specific paper(s).")
        else:
            st.caption(f"🌐 Searching across all **{len(available_docs)}** paper(s) in collection.")
    else:
        st.caption("_No indexed documents in this scope._")

    st.divider()
    st.subheader("💬 Conversations")

    if st.button("➕ New conversation", use_container_width=True, type="secondary"):
        try:
            new_session = client.create_session(st.session_state.active_collection_id)
            st.session_state.active_session_id = new_session["id"]
            st.session_state.conversation_papers[new_session["id"]] = []
            st.rerun()
        except Exception as exc:
            st.error(f"Could not create conversation: {exc}")

    for s in sessions:
        s_id = s["id"]
        title = s.get("title") or "New conversation"
        is_active = (s_id == session_id)

        sc1, sc2 = st.columns([5, 1])
        with sc1:
            btn_label = f"👉 {title}" if is_active else f"💬 {title}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(btn_label, key=f"sess_btn_{s_id}", use_container_width=True, type=btn_type):
                st.session_state.active_session_id = s_id
                st.rerun()
        with sc2:
            if st.button("🗑️", key=f"sess_del_{s_id}", help="Delete this conversation"):
                try:
                    client.delete_session(s_id)
                    st.session_state.conversation_papers.pop(s_id, None)
                    if st.session_state.get("active_session_id") == s_id:
                        st.session_state.active_session_id = None
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Could not delete: {exc.detail}")

# --- Main Conversation View ---
if not session_id:
    st.info("Select or create a conversation to start chatting.")
    st.stop()

active_title = active_session_obj.get("title") if active_session_obj else "New conversation"

# Header with Title and Rename option
h_col1, h_col2 = st.columns([4, 1])
with h_col1:
    st.title(f"💬 {active_title}")
    if active_session_obj and active_session_obj.get("created_at"):
        st.caption(f"🕒 Conversation started: **{format_asia_time(active_session_obj['created_at'])}**")
with h_col2:
    with st.popover("✏️ Rename"):
        rename_input = st.text_input("Topic Name", value=active_title, key=f"rename_val_{session_id}")
        if st.button("Save Name", key=f"save_rename_{session_id}", use_container_width=True):
            if rename_input.strip():
                try:
                    client.update_session_title(session_id, rename_input.strip())
                    st.success("Renamed!")
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Error: {exc.detail}")

# --- Conversation Memory & Research Papers Status Card ---
with st.container(border=True):
    mem_c1, mem_c2, mem_c3 = st.columns([3, 4, 2])
    with mem_c1:
        st.markdown(f"🧠 **Conversation Memory:** `{len(history)}` messages preserved")
        st.caption("Context carries over seamlessly across questions.")
    with mem_c2:
        if referenced_doc_names:
            paper_list_str = ", ".join(f"`{name}`" for name in list(referenced_doc_names.values())[:3])
            more_count = len(referenced_doc_names) - 3
            if more_count > 0:
                paper_list_str += f" (+{more_count} more)"
            st.markdown(f"📚 **Referenced Papers:** {paper_list_str}")
        elif selected_doc_ids:
            st.markdown(f"📚 **Scoped Papers:** `{len(selected_doc_ids)}` papers active in retrieval")
        else:
            st.markdown("📚 **Collection Papers:** All indexed collection documents active")
        st.caption("Uploaded papers remain indexed permanently in the database.")
    with mem_c3:
        if st.button("⬆️ Ingest More Papers", key="btn_quick_upload", use_container_width=True):
            st.switch_page("pages/2_Upload_Documents.py")


def render_message(message: dict) -> None:
    role = message["role"]
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(message["content"])
        if role == "assistant":
            meta_cols = st.columns([1, 1, 2, 2])
            if message.get("confidence") is not None:
                meta_cols[0].caption(f"🎯 Confidence: **{message['confidence']:.0%}**")
            if message.get("citations"):
                meta_cols[1].caption(f"📎 **{len(message['citations'])}** source(s)")
            if message.get("created_at"):
                meta_cols[2].caption(f"🕒 {format_asia_time(message['created_at'])}")

            if message.get("reasoning_summary"):
                with st.expander("🧭 Comprehensive Reasoning & Verification Protocol", expanded=False):
                    st.markdown(message["reasoning_summary"])

            if message.get("citations"):
                with st.expander("📎 Verified Citations & Evidence Chunks", expanded=False):
                    for i, c in enumerate(message["citations"], start=1):
                        score_pct = int(c["score"] * 100)
                        st.markdown(f"**[{i}] 📄 {c['document_filename']}** · `Relevance: {score_pct}%`")
                        st.info(c["snippet"])

            if message.get("suggested_followups"):
                st.write("**Suggested Research Follow-ups:**")
                cols = st.columns(len(message["suggested_followups"]))
                for i, followup in enumerate(message["suggested_followups"]):
                    if cols[i].button(followup, key=f"followup_{message['id']}_{i}"):
                        st.session_state["_pending_question"] = followup
                        st.rerun()


for message in history:
    render_message(message)

typed_question = st.chat_input("Ask a question, continue discussion, or compare research papers...")
pending = st.session_state.pop("_pending_question", None)
question = pending or typed_question

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        thought_container = st.container()
        thought_placeholder = thought_container.empty()
        answer_placeholder = st.empty()
        answer_text = ""
        final_payload = None
        thought_steps: list[dict] = []

        def render_live_thoughts(steps: list[dict], in_progress: bool = True):
            with thought_placeholder.container():
                with st.expander("🧠 Agent Live Thought Process & Retrieval Path", expanded=in_progress):
                    for step in steps:
                        st.markdown(f"**{step.get('title', 'Step')}**")
                        if step.get("detail"):
                            st.caption(step["detail"])
                        if step.get("reasoning"):
                            st.write(f"> _{step['reasoning']}_")
                    if in_progress:
                        st.spinner("Agent analyzing and synthesizing response...")

        try:
            for event in client.ask_stream(
                session_id,
                st.session_state.active_collection_id,
                question,
                document_ids=selected_doc_ids,
            ):
                if event["type"] == "thought":
                    thought_steps.append(event["data"])
                    render_live_thoughts(thought_steps, in_progress=True)
                elif event["type"] == "status":
                    thought_steps.append({"title": f"🔎 {event['data']}"})
                    render_live_thoughts(thought_steps, in_progress=True)
                elif event["type"] == "token":
                    answer_text += event["data"]
                    answer_placeholder.markdown(answer_text + "▌")
                elif event["type"] == "final":
                    final_payload = event["data"]

            # Collapse thought container upon completion
            if thought_steps:
                render_live_thoughts(thought_steps, in_progress=False)

            answer_placeholder.markdown(answer_text)
            if final_payload is None:
                st.warning(
                    "The response ended unexpectedly before finishing (connection interrupted). "
                    "The partial answer above may be incomplete -- try asking again."
                )
        except ApiError as exc:
            thought_placeholder.empty()
            st.error(f"Something went wrong: {exc.detail}")
            final_payload = None

        if final_payload:
            meta_cols = st.columns([1, 1, 4])
            meta_cols[0].caption(f"🎯 Confidence: **{final_payload['confidence']:.0%}**")
            meta_cols[1].caption(f"📎 **{len(final_payload['citations'])}** source(s)")

            if final_payload.get("reasoning_summary"):
                with st.expander("🧭 Comprehensive Reasoning & Verification Protocol", expanded=False):
                    st.markdown(final_payload["reasoning_summary"])

            if final_payload.get("citations"):
                with st.expander("📎 Verified Citations & Evidence Chunks", expanded=False):
                    for i, c in enumerate(final_payload["citations"], start=1):
                        score_pct = int(c["score"] * 100)
                        st.markdown(f"**[{i}] 📄 {c['document_filename']}** · `Relevance: {score_pct}%`")
                        st.info(c["snippet"])

            if final_payload.get("suggested_followups"):
                st.write("**Suggested Research Follow-ups:**")
                cols = st.columns(len(final_payload["suggested_followups"]))
                for i, followup in enumerate(final_payload["suggested_followups"]):
                    if cols[i].button(followup, key=f"live_followup_{i}"):
                        st.session_state["_pending_question"] = followup
                        st.rerun()

    # Rerun once after receiving response so dynamic title shows up in sidebar immediately
    if active_title in ("New conversation", "Untitled", "", None):
        st.rerun()

MERMAID_SCRIPT = """
<div id="chat-bottom-anchor"></div>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({
    startOnLoad: false,
    theme: 'neutral',
    themeVariables: {
        fontSize: '14px',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        primaryColor: '#e0e7ff',
        primaryTextColor: '#1e293b',
        primaryBorderColor: '#6366f1',
        lineColor: '#64748b'
    },
    securityLevel: 'loose'
});

async function processMermaidBlocks() {
    const targetDoc = window.parent ? window.parent.document : document;
    if (!targetDoc) return;
    
    const codeBlocks = targetDoc.querySelectorAll('pre code');
    for (let i = 0; i < codeBlocks.length; i++) {
        const codeEl = codeBlocks[i];
        if (codeEl.getAttribute('data-mermaid-rendered') === 'true') continue;
        
        const text = codeEl.textContent.trim();
        const isMermaid = codeEl.classList.contains('language-mermaid') || 
                          text.startsWith('graph ') || 
                          text.startsWith('flowchart ') || 
                          text.startsWith('sequenceDiagram') || 
                          text.startsWith('classDiagram') || 
                          text.startsWith('stateDiagram') || 
                          text.startsWith('erDiagram');
                          
        if (!isMermaid) continue;
        
        const preEl = codeEl.closest('pre');
        if (!preEl) continue;
        
        codeEl.setAttribute('data-mermaid-rendered', 'true');
        const container = targetDoc.createElement('div');
        container.className = 'mermaid-rendered-container';
        container.style.cssText = 'background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin: 14px 0; overflow-x: auto; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);';
        
        try {
            const id = 'mermaid-diag-' + Math.random().toString(36).substring(2, 9);
            const { svg } = await mermaid.render(id, text);
            container.innerHTML = svg;
            preEl.parentNode.insertBefore(container, preEl);
            preEl.style.display = 'none';
        } catch (e) {
            codeEl.removeAttribute('data-mermaid-rendered');
            container.remove();
        }
    }

    const anchor = targetDoc.getElementById("chat-bottom-anchor");
    if (anchor && window._shouldScrollToBottom) {
        anchor.scrollIntoView({behavior: "smooth", block: "end"});
        window._shouldScrollToBottom = false;
    }
}

setInterval(processMermaidBlocks, 700);
</script>
"""

if hasattr(st, "html"):
    st.html(MERMAID_SCRIPT)
else:
    st.markdown(MERMAID_SCRIPT, unsafe_allow_html=True)

