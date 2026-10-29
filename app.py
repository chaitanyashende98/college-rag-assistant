import streamlit as st
from pathlib import Path
from document_loader import extract_pdf_pages
from chunker import chunk_pages
from vector_store import get_collection, index_chunks, search_chunks, clear_collection
from rag_service import answer_question
from config import DATA_DIR, CHROMA_DIR, CHAT_MODEL, EMBEDDING_MODEL

st.set_page_config(page_title="College Document Q&A Assistant", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main {max-width: 1150px; margin: auto;}
.hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#10243a,#244d68); color: white;}
.hero h1 {margin-bottom: .25rem;}
.small-muted {color: #64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🎓 College Document Q&A Assistant</h1><p>Ask questions about your college PDFs. Answers are grounded in the documents you provide.</p></div>', unsafe_allow_html=True)
st.caption(f"Local RAG • Ollama model: {CHAT_MODEL} • Embeddings: {EMBEDDING_MODEL}")
DATA_DIR.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.header("⚙️ Setup")
    st.write("1. Start Ollama.")
    st.code("ollama pull llama3.2:3b\nollama pull nomic-embed-text", language="bash")
    st.write("2. Upload PDFs and build the searchable index.")
    st.divider()
    st.subheader("Index controls")
    if st.button("Clear indexed documents", use_container_width=True):
        clear_collection()
        st.success("The local index was cleared.")
    st.caption("Your documents and vector index stay on this computer.")

tab_ask, tab_docs, tab_about = st.tabs(["💬 Ask a question", "📚 Add documents", "ℹ️ How it works"])

with tab_docs:
    st.subheader("Add college documents")
    st.write("Upload searchable PDF files such as academic regulations, examination guidelines, or a student handbook.")
    uploaded = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process and index PDFs", type="primary", disabled=not uploaded):
        all_chunks = []
        progress = st.progress(0)
        status = st.empty()
        for i, file in enumerate(uploaded):
            try:
                pages = extract_pdf_pages(file.getvalue(), file.name)
                chunks = chunk_pages(pages)
                all_chunks.extend(chunks)
                status.write(f"Read **{file.name}**: {len(pages)} pages, {len(chunks)} chunks.")
            except Exception as exc:
                st.error(f"Could not process {file.name}: {exc}")
            progress.progress((i + 1) / len(uploaded))
        if all_chunks:
            try:
                with st.spinner("Creating embeddings and saving the local index…"):
                    count = index_chunks(all_chunks)
                st.success(f"Indexed {count} text chunks. You can now ask questions.")
            except Exception as exc:
                st.error(f"Indexing failed. Check that Ollama and the embedding model are available. Details: {exc}")
        else:
            st.warning("No readable text was found. Scanned/image-only PDFs may need OCR before use.")

with tab_ask:
    st.subheader("Ask your college documents")
    question = st.text_input("Your question", placeholder="e.g., What attendance is required for semester examinations?")
    top_k = st.slider("Number of document sections to retrieve", min_value=1, max_value=8, value=4)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Searching documents and generating an answer…"):
                result = answer_question(question.strip(), top_k=top_k)
            st.markdown("### Answer")
            st.write(result["answer"])
            if result["sources"]:
                st.markdown("### Sources")
                for idx, source in enumerate(result["sources"], start=1):
                    with st.expander(f'{idx}. {source["source"]} — page {source["page"]}'):
                        st.write(source["text"])
                        st.caption(f'Retrieval distance: {source["distance"]:.3f}')
            else:
                st.info("No sufficiently relevant source sections were found.")
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")
            st.info("Confirm Ollama is running and both models have been downloaded.")
    st.caption("Answers may be incomplete if the supplied PDFs are incomplete or contain scanned pages.")

with tab_about:
    st.subheader("Retrieval-Augmented Generation (RAG)")
    st.markdown("""
    1. **PDF extraction:** read text and page numbers from each PDF.
    2. **Chunking:** split text into overlapping sections.
    3. **Embeddings:** convert each section and question into semantic vectors using Ollama.
    4. **Vector search:** ChromaDB retrieves relevant sections.
    5. **Answer generation:** a local LLM answers from retrieved context only.
    6. **Citations and safe refusal:** sources are shown; if evidence is insufficient, the assistant says it was not found.

    This application does not connect to Jira. Jira is used separately to manage project tasks and sprints, as described in the project document.
    """)
    st.markdown("**Local storage:** `data/` contains uploaded PDFs if you choose to save them; `chroma_db/` contains the local vector index.")
