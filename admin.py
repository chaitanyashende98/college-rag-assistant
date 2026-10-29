
import hmac
import os

import streamlit as st

from chunker import chunk_pages
from document_loader import extract_pdf_pages
from vector_store import index_chunks


st.set_page_config(
    page_title="College RAG Admin",
    page_icon="🎓",
)

st.title("College RAG Assistant — Admin")
st.write("Upload college PDFs to the cloud document index.")

admin_password = os.getenv("ADMIN_PASSWORD")

if not admin_password:
    st.error("Admin access is not configured. Contact the administrator.")
    st.stop()

password = st.text_input("Admin password", type="password")

if not hmac.compare_digest(password, admin_password):
    st.info("Enter the admin password to continue.")
    st.stop()

st.success("Authenticated")

uploaded_files = st.file_uploader(
    "Choose PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process and index PDFs", disabled=not uploaded_files):
    all_chunks = []

    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):
        try:
            pages = extract_pdf_pages(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )
            chunks = chunk_pages(pages)
            all_chunks.extend(chunks)

            status.write(
                f"{uploaded_file.name}: "
                f"{len(pages)} pages, {len(chunks)} chunks"
            )
        except Exception as exc:
            st.error(f"Could not process {uploaded_file.name}: {exc}")

        progress.progress((index + 1) / len(uploaded_files))

    if all_chunks:
        try:
            with st.spinner("Creating embeddings and indexing in Qdrant Cloud..."):
                count = index_chunks(all_chunks)

            st.success(f"Successfully indexed {count} chunks.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
    else:
        st.warning("No readable PDF text was found.")
