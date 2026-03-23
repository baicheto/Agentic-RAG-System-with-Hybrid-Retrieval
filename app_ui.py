import streamlit as st
import os
from dotenv import load_dotenv

# Import all existing functions from agentic_rag.py
from agentic_rag import (
    setup_vector_db,
    build_local_context_summary,
    setup_web_scraping_crew,
    process_query_agentic
)

load_dotenv()

st.title("Agentic RAG System")
st.write("Ask anything about your document, the system will search locally and on the web.")

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Query Input
query = st.text_input("Ask a question:", placeholder="e.g. What is Keyword Extraction?")

# Run Button
if st.button("Search") and uploaded_file and query:

    # Save the uploaded PDF temporarily
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Setting up vector database..."):
        vector_db = setup_vector_db(pdf_path)
        local_context_summary = build_local_context_summary(vector_db)
        web_crew = setup_web_scraping_crew()

    with st.spinner("Processing your query..."):
        result = process_query_agentic(query, vector_db, local_context_summary, web_crew)

    # Results
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Pipeline Details")
    col1, col2, col3 = st.columns(3)
    col1.metric("Route", result["route"])
    col2.metric("Sources Used", ", ".join(result["actual_sources_used"]))
    col3.metric("Rewrites", result["rewrites"])

    if result["grounding_warning"]:
        st.warning("Answer could not be fully verified against retrieved context.")
    else:
        st.success("Answer is grounded and verified.")

    # Clean temp file
    os.remove(pdf_path)