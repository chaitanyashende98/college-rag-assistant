
import os
import cohere

EMBEDDING_MODEL = "embed-english-v3.0"

_client = None


def _get_client():
    global _client

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY environment variable is not set."
        )

    if _client is None:
        _client = cohere.ClientV2(api_key=api_key)

    return _client


def embed_texts(
    texts: list[str],
    input_type: str = "search_document",
) -> list[list[float]]:
    """Create embeddings using Cohere's hosted embedding API."""
    if not texts:
        return []

    response = _get_client().embed(
        model=EMBEDDING_MODEL,
        texts=texts,
        input_type=input_type,
        embedding_types=["float"],
    )

    return response.embeddings.float_
