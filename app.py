import os
import time
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingest_jobs" not in st.session_state:
    st.session_state.ingest_jobs = {}   # {job_id: {filename, status, message}}

# ---------------------------------------------------------------------------
# Sidebar — Document Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.caption("Upload documents to expand the assistant's knowledge.")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "doc", "txt", "xlsx", "xls", "pptx", "ppt", "png", "jpg", "jpeg", "md", "csv"],
        key="doc_uploader",
        help="Upload company documents, policies, technical guides, spreadsheets, or images.",
    )

    upload_btn = st.button("⬆️ Upload & Ingest", use_container_width=True, type="primary")

    # -- Handle upload --
    if upload_btn and uploaded_file:
        with st.spinner(f"Uploading {uploaded_file.name}…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")},
                    timeout=30,
                )
                resp.raise_for_status()
                job = resp.json()
                job_id = job["job_id"]
                st.session_state.ingest_jobs[job_id] = {
                    "filename": uploaded_file.name,
                    "status": "queued",
                    "message": job.get("detail", ""),
                }
                st.success(f"✅ '{uploaded_file.name}' uploaded! Ingesting in background…")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot reach the API server. Is `uvicorn main:app --reload --port 8001` running?")
            except Exception as e:
                st.error(f"Upload failed: {e}")
    elif upload_btn and not uploaded_file:
        st.warning("Please select a document first.")

    # -- Ingestion job status panel --
    if st.session_state.ingest_jobs:
        st.divider()
        st.subheader("📋 Ingestion Jobs")

        refresh = st.button("🔄 Refresh Status", use_container_width=True)

        for job_id, info in list(st.session_state.ingest_jobs.items()):
            if refresh:
                try:
                    r = requests.get(f"{API_BASE}/ingest/status/{job_id}", timeout=5)
                    if r.ok:
                        updated = r.json()
                        info["status"] = updated.get("status", info["status"])
                        info["message"] = updated.get("message", "")
                        chunks = updated.get("chunks_added", 0)
                        if chunks:
                            info["message"] += f" ({chunks} chunks)"
                except Exception:
                    pass

            status = info["status"]
            icon = {"queued": "⏳", "processing": "⚙️", "done": "✅", "failed": "❌"}.get(status, "❓")
            color = {"done": "green", "failed": "red", "processing": "orange"}.get(status, "grey")

            with st.expander(f"{icon} {info['filename']}", expanded=(status != "done")):
                st.markdown(f"**Status:** :{color}[{status.upper()}]")
                if info.get("message"):
                    st.caption(info["message"])

    # -----------------------------------------------------------------------
    # Ingested Documents Section (Persisted permanently from DB)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📑 Ingested Documents")
    
    def _fetch_stored_docs():
        try:
            r = requests.get(f"{API_BASE}/documents", timeout=5)
            if r.ok:
                return r.json().get("documents", [])
        except Exception:
            pass
        return []

    stored_docs = _fetch_stored_docs()

    if stored_docs:
        with st.expander(f"📚 Stored Files ({len(stored_docs)})", expanded=True):
            for d in stored_docs:
                st.markdown(f"📄 **{d['filename']}**")
                st.caption(f"🧩 Chunks: `{d['chunks']}`")
                st.markdown("---")
    else:
        st.info("No documents currently stored in the knowledge base.")

    # -----------------------------------------------------------------------
    # Remove Document Section (with confirmation popup)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🗑️ Remove Document")
    if stored_docs:
        doc_names = [d["filename"] for d in stored_docs]
        doc_to_delete = st.selectbox(
            "Select document to remove",
            options=doc_names,
            key="delete_doc_selector",
            help="Choose a file to permanently delete from ChromaDB."
        )

        # Confirmation popup via popover (or fallback for older Streamlit versions)
        if hasattr(st, "popover"):
            with st.popover("🗑️ Delete Selected Document", use_container_width=True):
                st.warning(f"⚠️ Are you sure you want to delete **{doc_to_delete}**?")
                st.caption("All associated vector chunks will be deleted, and the search retriever will be reloaded.")
                if st.button("Yes, Delete Permanently", type="primary", use_container_width=True):
                    with st.spinner(f"Deleting {doc_to_delete}…"):
                        try:
                            del_resp = requests.delete(f"{API_BASE}/documents/{doc_to_delete}", timeout=10)
                            if del_resp.ok:
                                st.success(f"'{doc_to_delete}' deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Failed to delete: {del_resp.text}")
                        except Exception as e:
                            st.error(f"Error during deletion: {e}")
        else:
            if st.button("🗑️ Delete Selected Document", use_container_width=True):
                st.session_state["pending_delete"] = doc_to_delete

            if st.session_state.get("pending_delete") == doc_to_delete:
                st.warning(f"⚠️ Confirm permanent deletion of **{doc_to_delete}**?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Confirm", type="primary", use_container_width=True):
                        with st.spinner(f"Deleting {doc_to_delete}…"):
                            try:
                                del_resp = requests.delete(f"{API_BASE}/documents/{doc_to_delete}", timeout=10)
                                if del_resp.ok:
                                    st.session_state["pending_delete"] = None
                                    st.success(f"'{doc_to_delete}' deleted!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Failed: {del_resp.text}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                with c2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state["pending_delete"] = None
                        st.rerun()
    else:
        st.caption("No documents available to remove.")

    st.divider()
    st.caption("Ask questions about any uploaded document in the chat →")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("📚 Intelligent Document Assistant")
st.markdown("Ask questions, analyze research papers, and search through your uploaded documents.")

# Re-draw history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"🔍 Sources  |  Confidence: {message['confidence']:.0%}"):
                for idx, source in enumerate(message["sources"]):
                    st.write(f"{idx + 1}. **{source['document']}** — Page {source['page']}")

# Chat input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    st.chat_message("user").markdown(prompt)
    payload = {"question": prompt, "history": st.session_state.messages}
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
                response.raise_for_status()

                data     = response.json()
                answer   = data.get("answer", "Something went wrong. Please try again.")
                sources  = data.get("sources", [])
                confidence = data.get("confidence", 0.0)

                st.markdown(answer)

                if sources:
                    with st.expander(f"🔍 Sources  |  Confidence: {confidence:.0%}"):
                        for idx, source in enumerate(sources):
                            st.write(f"{idx + 1}. **{source['document']}** — Page {source['page']}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })

            except requests.exceptions.ConnectionError:
                st.error("⚠️ API Connection Error: Make sure `uvicorn main:app --reload` is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"⚠️ API Error ({e.response.status_code}): {e.response.text}")
            except Exception as e:
                st.error(f"⚠️ Unexpected error: {e}")