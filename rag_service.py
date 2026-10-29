
from retriever import retrieve

REFUSAL = "Information not found in the available documents."


def get_relevant_context(
    question: str,
    top_k: int = 4,
) -> dict:
    """Retrieve relevant college-document passages for an MCP client."""
    question = question.strip()

    if not question:
        return {
            "answer": REFUSAL,
            "sources": [],
        }

    sources = retrieve(question, top_k=top_k)

    # Cosine distance is a heuristic, not a calibrated confidence score.
    relevant = [
        source
        for source in sources
        if source["distance"] <= 0.55
    ]

    if not relevant:
        return {
            "answer": REFUSAL,
            "sources": [],
        }

    return {
        "answer": None,
        "sources": relevant,
    }


def answer_question(question: str, top_k: int = 4) -> dict:
    """Compatibility wrapper; Claude will generate the final answer."""
    return get_relevant_context(question, top_k=top_k)
