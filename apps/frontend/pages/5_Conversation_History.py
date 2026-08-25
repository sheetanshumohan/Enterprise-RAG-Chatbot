from __future__ import annotations

import streamlit as st

from api_client import ApiClient, ApiError, format_asia_time, init_auth

st.set_page_config(page_title="Conversation History", page_icon="🕘", layout="wide")

client = init_auth()
st.session_state._active_page_tag = "history"
st.session_state.setdefault("conversation_papers", {})

if not client or not st.session_state.get("token"):
    st.warning("Please log in from the home page first.")
    st.stop()

st.title("🕘 Conversation History")
st.caption("Review, continue, or manage your past research conversations. All messages, citations, and research papers are preserved.")

try:
    sessions = client.list_sessions()
except ApiError as exc:
    st.error(f"Could not load conversations: {exc.detail}")
    st.stop()

if not sessions:
    st.info("No conversations yet. Start one from the **💬 AI Chat** page.")
    st.stop()


for s in sessions:
    s_id = s["id"]
    title = s.get("title") or "New conversation"
    time_label = format_asia_time(s.get("created_at", ""))

    c_card, c_del = st.columns([11, 1])
    with c_del:
        if st.button("🗑️", key=f"hist_del_{s_id}", help="Delete this entire conversation"):
            try:
                client.delete_session(s_id)
                st.session_state.conversation_papers.pop(s_id, None)
                if st.session_state.get("active_session_id") == s_id:
                    st.session_state.active_session_id = None
                st.success("Conversation deleted.")
                st.rerun()
            except ApiError as exc:
                st.error(f"Could not delete: {exc.detail}")

    with c_card:
        try:
            messages = client.get_history(s_id)
        except ApiError as exc:
            st.error(f"Could not load messages: {exc.detail}")
            continue

        # Extract all referenced research papers in this conversation
        referenced_papers = []
        seen_docs = set()
        for m in messages:
            for c in m.get("citations", []):
                doc_name = c.get("document_filename")
                if doc_name and doc_name not in seen_docs:
                    seen_docs.add(doc_name)
                    referenced_papers.append(doc_name)

        paper_preview = f" · 📚 {len(referenced_papers)} paper(s) referenced" if referenced_papers else ""
        with st.expander(f"💬 **{title}**  ·  _{time_label}_{paper_preview}", expanded=False):
            if referenced_papers:
                st.markdown("**📚 Research Papers in Memory:** " + ", ".join(f"`{p}`" for p in referenced_papers))

            if not messages:
                st.caption("_Empty conversation (no messages)._")
            else:
                for m in messages:
                    with st.chat_message("user" if m["role"] == "user" else "assistant"):
                        st.markdown(m["content"])
                        if m.get("citations"):
                            st.caption(f"📎 **Sources:** {', '.join(c['document_filename'] for c in m['citations'])}")

            st.write("")
            act_col1, act_col2 = st.columns([3, 1])
            with act_col1:
                if st.button("▶️ Continue this conversation in AI Chat", key=f"continue_{s_id}", type="primary"):
                    st.session_state.active_session_id = s_id
                    st.session_state.active_collection_id = s.get("collection_id")
                    if referenced_papers:
                        st.session_state.conversation_papers[s_id] = [f"📄 {p}" for p in referenced_papers]
                    st.switch_page("pages/4_AI_Chat.py")
            with act_col2:
                if st.button("🗑️ Delete conversation", key=f"exp_del_{s_id}"):
                    try:
                        client.delete_session(s_id)
                        st.session_state.conversation_papers.pop(s_id, None)
                        if st.session_state.get("active_session_id") == s_id:
                            st.session_state.active_session_id = None
                        st.success("Conversation deleted.")
                        st.rerun()
                    except ApiError as exc:
                        st.error(f"Could not delete: {exc.detail}")

MERMAID_SCRIPT = """
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
}

setInterval(processMermaidBlocks, 800);
</script>
"""

if hasattr(st, "html"):
    st.html(MERMAID_SCRIPT)
else:
    st.markdown(MERMAID_SCRIPT, unsafe_allow_html=True)

