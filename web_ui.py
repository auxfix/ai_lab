import os
import sys

import streamlit as st

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import CodeRAGOrchestrator

# Page config
st.set_page_config(page_title="Code RAG Assistant", page_icon="🤖", layout="wide")

# Initialize session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    repo_path = st.text_input("Repository Path", value=".")
    reindex = st.checkbox("Force Reindex", value=False)

    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Setting up Code RAG..."):
            orchestrator = CodeRAGOrchestrator(repo_path)
            if orchestrator.setup(reindex=reindex):
                st.session_state.orchestrator = orchestrator
                st.session_state.initialized = True
                st.success("✅ System ready!")
            else:
                st.error("❌ Failed to initialize")

    st.divider()

    if st.session_state.initialized:
        st.success("System is ready!")

        # Stats
        stats = st.session_state.orchestrator.vectorizer.get_stats()
        st.metric("Chunks in DB", stats["total_chunks"])

        # Settings
        n_chunks = st.slider("Chunks to retrieve", 1, 10, 5)
        show_sources = st.checkbox("Show sources", value=True)

# Main content
st.title("🤖 Code RAG Assistant")
st.markdown("Ask questions about your codebase!")

if not st.session_state.initialized:
    st.info("👈 Initialize the system from the sidebar first.")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your codebase..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.orchestrator.query_one(prompt)

            if result:
                # Display answer
                st.markdown(result["answer"])

                # Display sources if enabled
                if show_sources and result["sources"]:
                    with st.expander(f"📚 Sources ({len(result['sources'])})"):
                        for i, source in enumerate(result["sources"], 1):
                            st.markdown(f"**{i}. {source['file']}**")
                            st.caption(f"Match: {source['similarity']:.2f}")
                            st.code(source["preview"], language="text")

                # Add to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )
            else:
                st.error("Failed to get response")
