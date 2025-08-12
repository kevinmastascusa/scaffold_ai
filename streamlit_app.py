"""
Streamlit UI for Scaffold AI

This app provides an interface similar to the Enhanced Flask UI with:
- Syllabus PDF upload and analysis
- Conversational chat with session-based memory
- Cited answers and sources display
- Clear memory and clear conversation actions
- Simple feedback capture
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st

# Ensure project root is on path for module imports
PROJECT_ROOT = Path(__file__).parent.absolute()
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))


@st.cache_resource(show_spinner=True)
def get_query_system():
    """Lazily import and initialize the improved enhanced query system.

    Returns the module and a callable for querying.
    """
    from scaffold_core.vector.enhanced_query_improved import (
        improved_enhanced_query_system,
        query_enhanced_improved,
    )

    if not improved_enhanced_query_system.initialized:
        improved_enhanced_query_system.initialize()

    return improved_enhanced_query_system, query_enhanced_improved


def ensure_session_id() -> str:
    """Ensure a stable session id for chat memory across interactions."""
    if "session_id" not in st.session_state or not st.session_state.session_id:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def ensure_directories() -> Dict[str, Path]:
    """Ensure required directories exist and return their paths."""
    uploads = PROJECT_ROOT / "uploads"
    conversations = PROJECT_ROOT / "conversations"
    feedback = PROJECT_ROOT / "ui_feedback"
    uploads.mkdir(exist_ok=True)
    conversations.mkdir(exist_ok=True)
    feedback.mkdir(exist_ok=True)
    return {
        "uploads": uploads,
        "conversations": conversations,
        "feedback": feedback,
    }


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


def _conversation_file(session_id: str) -> Path:
    dirs = ensure_directories()
    return dirs["conversations"] / f"{session_id}.json"


def get_conversation_history(session_id: str) -> List[Dict]:
    file_path = _conversation_file(session_id)
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_conversation_history(session_id: str, conversation: List[Dict]) -> None:
    file_path = _conversation_file(session_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            conversation,
            f,
            indent=2,
            default=str,
        )


def append_message(session_id: str, message: Dict) -> List[Dict]:
    conversation = get_conversation_history(session_id)
    conversation.append(message)
    save_conversation_history(session_id, conversation)
    return conversation


def format_sources(sources: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for s in sources or []:
        src = s.get("source", {})
        file_name = src.get("name", src.get("title", "Unknown"))
        preview = (s.get("text_preview", "") or "").strip()
        rows.append(
            {
                "Score": round(float(s.get("score", 0.0)), 4),
                "File": file_name,
                "Year": src.get("year", ""),
                "DOI": src.get("doi", ""),
                "Preview": preview,
            }
        )
    return rows


def render_chat_history(session_id: str) -> None:
    conversation = get_conversation_history(session_id)
    for msg in conversation:
        msg_type = msg.get("type")
        if msg_type == "user":
            with st.chat_message("user"):
                st.markdown(msg.get("content", ""))
        elif msg_type == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg.get("content", ""))
                sources = msg.get("sources", [])
                if sources:
                    st.caption("Sources")
                    rows = format_sources(sources)
                    st.dataframe(
                        rows,
                        use_container_width=True,
                        hide_index=True,
                    )
        elif msg_type == "syllabus_context":
            with st.chat_message("assistant"):
                st.info("Syllabus context added for this session.")


def handle_upload(session_id: str) -> None:
    st.subheader("Upload Syllabus PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF", type=["pdf"], key="syllabus_upload"
    )
    if uploaded_file is None:
        return

    if not allowed_file(uploaded_file.name or ""):
        st.warning("Only PDF files are allowed.")
        return

    dirs = ensure_directories()
    filename = uploaded_file.name or "syllabus.pdf"
    safe_name = filename.replace("/", "_").replace("\\", "_")
    out_path = dirs["uploads"] / f"{session_id}_{safe_name}"

    file_bytes = uploaded_file.read()
    with open(out_path, "wb") as f:
        f.write(file_bytes)

    try:
        from scaffold_core.pdf_processor import process_syllabus_upload
        result = process_syllabus_upload(str(out_path), session_id)

        if result.get("processing_status") == "success":
            analysis = result.get("analysis", {})
            info = analysis.get("course_info", {})
            course_title = info.get("title", "Unknown Course")
            course_code = info.get("code", "N/A")
            topics = ", ".join(analysis.get("topics", [])[:5])
            summary = (result.get("text_content", "") or "")[:500]
            syllabus_context = (
                "UPLOADED SYLLABUS CONTEXT:\n"
                f"Course: {course_title}\n"
                f"Course Code: {course_code}\n"
                f"Topics: {topics}\n"
                f"Content Summary: {summary}..."
            )
            append_message(
                session_id,
                {
                    "id": str(uuid.uuid4()),
                    "type": "syllabus_context",
                    "content": syllabus_context,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "filename": filename,
                },
            )
            st.success("Syllabus uploaded and analyzed successfully.")
            suggestions = result.get("sustainability_suggestions")
            if suggestions:
                with st.expander("Suggestions"):
                    st.json(suggestions)
        else:
            st.warning("Syllabus uploaded but analysis failed.")
            err = result.get("error_message", "Unknown error")
            st.caption(f"Error: {err}")
    except Exception:
        st.error("Upload processing failed. See logs for details.")


def render_feedback(session_id: str) -> None:
    st.subheader("Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        rating = st.slider(
            "How helpful was the answer?",
            min_value=1,
            max_value=5,
            value=4,
        )
        comments = st.text_area("Comments (optional)")
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            ensure_directories()
            feedback_file = (
                PROJECT_ROOT
                / "ui_feedback"
                / (
                    f"feedback_"
                    f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            )
            feedback = {
                "timestamp": datetime.datetime.now().isoformat(),
                "rating": rating,
                "comments": comments,
                "session_id": session_id,
            }
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(feedback, f, indent=2)
            st.success("Feedback saved. Thank you!")


def main():
    st.set_page_config(
        page_title="Scaffold AI - Enhanced (Streamlit)",
        page_icon="📚",
        layout="wide",
    )
    st.title("Scaffold AI (Streamlit)")
    st.caption("Enhanced UI with upload, chat, and cited answers")

    session_id = ensure_session_id()
    ensure_directories()

    with st.sidebar:
        st.header("Session")
        new_id = st.text_input("Session ID", value=session_id)
        if new_id != session_id:
            st.session_state.session_id = new_id or session_id
            session_id = st.session_state.session_id

        st.divider()
        if st.button("Clear conversation"):
            file_path = _conversation_file(session_id)
            if file_path.exists():
                file_path.unlink()
            st.success("Conversation cleared")

        if st.button("Clear memory"):
            try:
                system, _ = get_query_system()
                system.clear_memory(session_id)
                st.success("Memory cleared for this session")
            except Exception as e:
                st.warning(f"Unable to clear memory: {e}")

        st.divider()
        handle_upload(session_id)

    with st.spinner("Loading models and index..."):
        system, query_fn = get_query_system()

    st.divider()
    render_chat_history(session_id)

    prompt = st.chat_input("Type your message")
    if prompt and prompt.strip():
        append_message(
            session_id,
            {
                "id": str(uuid.uuid4()),
                "type": "user",
                "content": prompt.strip(),
                "timestamp": datetime.datetime.now().isoformat(),
            },
        )

        with st.chat_message("assistant"):
            with st.spinner(
                "Retrieving, reranking, and generating answer..."
            ):
                result: Dict = query_fn(prompt.strip(), session_id=session_id)

            assistant_message = {
                "id": str(uuid.uuid4()),
                "type": "assistant",
                "content": result.get("response", "(no response)"),
                "sources": result.get("sources", [])[:5],
                "timestamp": datetime.datetime.now().isoformat(),
            }
            append_message(session_id, assistant_message)

            st.markdown(assistant_message["content"])
            sources = assistant_message.get("sources", [])
            if sources:
                st.caption("Sources")
                rows = format_sources(sources)
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                )

    st.divider()
    render_feedback(session_id)


if __name__ == "__main__":
    main()
