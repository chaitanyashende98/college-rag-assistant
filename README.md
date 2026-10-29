# College Document Q&A Assistant (Local RAG)

A final-year mini project that answers natural-language questions from college PDF documents using Retrieval-Augmented Generation (RAG). It is designed to run locally without OpenAI, Gemini, Claude, paid APIs, or hosted databases.

## Technology stack

- Python
- Streamlit user interface
- PyPDF for PDF text extraction
- Ollama with `llama3.2:3b` for local answer generation
- Ollama `nomic-embed-text` for embeddings
- ChromaDB for persistent local vector storage
- pytest for basic unit tests

## Features

- Upload one or more PDFs through the UI
- Extract text page-by-page and preserve filenames/page numbers
- Split text into overlapping chunks
- Generate local embeddings and store them in ChromaDB
- Retrieve semantically related sections for a question
- Generate answers from retrieved context only
- Display source filename, page, retrieved excerpt, and retrieval distance
- Return `Information not found in the available documents.` when no sufficiently relevant context is retrieved
- Basic automated tests for chunking and refusal behaviour

## Requirements

Install Python 3.10 or later and [Ollama](https://ollama.com/). The first model downloads require an internet connection; inference and document retrieval run locally afterward. Model performance and speed depend on your computer's memory and CPU/GPU.

## Run on Windows

Open Command Prompt or PowerShell in this project folder.

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
ollama serve
```

If Ollama is already running as a background service, you do not need to run `ollama serve` again. Open a second terminal in the project folder, activate the environment, and start Streamlit:

```powershell
venv\Scripts\activate
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser, upload PDF files in **Add documents**, and click **Process and index PDFs**. Then go to **Ask a question**.

## Run on macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
streamlit run app.py
```

Install and start Ollama for your operating system before asking questions.

## Project structure

```text
college-rag-assistant/
├── app.py
├── config.py
├── document_loader.py
├── chunker.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── llm_service.py
├── rag_service.py
├── requirements.txt
├── README.md
├── data/
├── chroma_db/       # generated locally; ignored by Git
└── tests/
    ├── test_chunker.py
    └── test_rag_service.py
```

## Evaluation plan

The accompanying project brief recommends at least 50 questions in three groups:
1. Direct questions
2. Paraphrased questions
3. Unanswerable questions

Record the expected document, retrieved document, and whether retrieval was correct. Calculate:

`Retrieval Accuracy = Correct Retrievals / Total Questions × 100`

For unanswerable questions, calculate:

`Safe Refusal Rate = Correct Refusals / Unanswerable Questions × 100`

Have human evaluators score answer correctness, relevance, groundedness, clarity, and completeness from 1–5. The app's distance cutoff is a practical starting heuristic, not a calibrated confidence guarantee; tune it using your own test set. Scanned/image-only PDFs are not OCR-processed and may need text extraction/OCR before uploading.

## Jira project management

Use Jira separately (the application does not need to connect to Jira), as specified in the brief. Suggested epics:
- Requirements and Dataset
- Local LLM Setup
- RAG Pipeline
- User Interface
- Testing and Evaluation
- Documentation and Demonstration

Suggested workflow: To Do → In Progress → Code Review → Testing → Done. Track implementation, bugs, evaluation, and report/demo work as Jira issues.

## Privacy and limitations

- The vector database is stored in `chroma_db/` on the local machine.
- Do not upload confidential student records or sensitive personal data.
- PDF extraction works only when the PDF contains extractable text; scanned PDFs need OCR.
- The LLM can still make mistakes. Verify important college rules against the cited original document.
- The initial Ollama model downloads require internet access; the app itself is designed for local use.

## Tests

```powershell
pytest -q
```
