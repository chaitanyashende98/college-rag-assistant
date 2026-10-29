from io import BytesIO
from pypdf import PdfReader

def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract readable text page-by-page, preserving source filename and page number."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"source": filename, "page": page_number, "text": text})
    return pages
