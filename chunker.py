from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split each page into overlapping chunks while preserving source and page metadata."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks = []
    for page in pages:
        text = page["text"].strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "source": page["source"],
                    "page": int(page["page"]),
                })
            if end >= len(text):
                break
            start = end - overlap
    return chunks
