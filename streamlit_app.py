"""
Streamlit UI for Scaffold AI

This app provides a simple interface to query the vector index and generate
answers with sources using the existing improved enhanced query system.
"""

import os
import uuid
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
    # Deferred import so Streamlit reloads are fast
    from scaffold_core.vector.enhanced_query_improved import (
        improved_enhanced_query_system,
        query_enhanced_improved,
    )

    # Initialize once per process
    if not improved_enhanced_query_system.initialized:
        improved_enhanced_query_system.initialize()

    return improved_enhanced_query_system, query_enhanced_improved


def ensure_session_id() -> str:
    """Ensure a stable session id for chat memory across interactions."""
    if "session_id" not in st.session_state or not st.session_state.session_id:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    return st.session_state.session_id


def format_sources(sources: List[Dict]) -> List[Dict]:
    """Format sources list for display in a table."""
    rows: List[Dict] = []
    for s in sources or []:
        src = s.get("source", {})
        rows.append(
            {
                "Score": round(float(s.get("score", 0.0)), 4),
                "File": src.get("name", "Unknown"),
                "Page": src.get("page", ""),
                "Path": src.get("path", ""),
                "Preview": (s.get("text_preview", "") or "").strip(),
            }
        )
    return rows


def main():
    st.set_page_config(
        page_title="Scaffold AI - Streamlit",
        page_icon="📚",
        layout="wide",
    )

    st.title("Scaffold AI (Streamlit)")
    st.caption("Semantic search over your research corpus with cited answers")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        session_id = ensure_session_id()
        session_id = st.text_input("Session ID", value=session_id)
        st.session_state.session_id = session_id

        st.divider()
        st.markdown("Model selection is managed via `SC_LLM_KEY` env var.")
        st.markdown("Using FAISS index and metadata from `vector_outputs/`.")

        if st.button("Clear conversation memory"):
            try:
                from scaffold_core.vector.enhanced_query_improved import (
                    improved_enhanced_query_system,
                )
                if improved_enhanced_query_system.initialized:
                    improved_enhanced_query_system.clear_memory(session_id)
                st.success("Cleared memory for this session.")
            except Exception as e:
                st.warning(f"Unable to clear memory: {e}")

    # Load system once
    with st.spinner("Loading models and index..."):
        system, query_fn = get_query_system()

    # Display small status
    col1, col2, col3 = st.columns(3)
    with col1:
        vectors = getattr(getattr(system, "faiss_index", None), "ntotal", 0)
        st.metric("Vectors", f"{vectors:,}")
    with col2:
        meta_count = len(getattr(system, "metadata", []) or [])
        st.metric("Metadata entries", f"{meta_count:,}")
    with col3:
        st.metric("Device", "CPU")

    st.divider()

    # Query input
    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Ask a question",
            placeholder=(
                "e.g., Strategies to integrate sustainability into first-year "
                "engineering curricula?"
            ),
            height=120,
        )
        submitted = st.form_submit_button(
            "Search and Answer", use_container_width=True
        )

    if submitted and question.strip():
        with st.spinner("Retrieving, reranking, and generating answer..."):
            result: Dict = query_fn(question.strip(), session_id=session_id)

        # Answer
        st.subheader("Answer")
        st.markdown(result.get("response", "(no response)"))

        # Sources
        sources = result.get("sources", [])
        st.subheader(f"Sources ({len(sources)})")
        rows = format_sources(sources)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No sources found.")

        # Raw details expander
        with st.expander("Details (JSON)"):
            st.json(result)


if __name__ == "__main__":
    main()
