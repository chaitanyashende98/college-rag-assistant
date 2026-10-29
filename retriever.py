from vector_store import search_chunks

def retrieve(question: str, top_k: int = 4) -> list[dict]:
    return search_chunks(question, top_k=top_k)
