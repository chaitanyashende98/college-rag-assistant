
import hashlib
import uuid

from qdrant_client import QdrantClient, models

from config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY
from embeddings import embed_texts

_client = None


def _get_client():
    global _client

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError(
            "Set QDRANT_URL and QDRANT_API_KEY environment variables."
        )

    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )

    return _client


def _ensure_collection(vector_size: int):
    client = _get_client()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def get_collection():
    """Return the Qdrant client for compatibility with existing imports."""
    return _get_client()


def index_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    client = _get_client()
    total = 0
    batch_size = 32

    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        texts = [item["text"] for item in batch]

        vectors = embed_texts(
            texts,
            input_type="search_document",
        )

        if len(vectors) != len(batch):
            raise RuntimeError(
                "Cohere returned an unexpected number of embeddings."
            )

        _ensure_collection(len(vectors[0]))
        points = []

        for item, text, vector in zip(batch, texts, vectors):
            raw = (
                f'{item["source"]}|{item["page"]}|{text}'
            )
            digest = hashlib.sha256(
                raw.encode("utf-8")
            ).hexdigest()

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, digest)
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": text,
                        "source": item["source"],
                        "page": int(item["page"]),
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        total += len(batch)

    return total


def search_chunks(
    question: str,
    top_k: int = 4,
) -> list[dict]:
    client = _get_client()

    if not client.collection_exists(COLLECTION_NAME):
        return []

    query_vectors = embed_texts(
        [question],
        input_type="search_query",
    )

    if not query_vectors:
        return []

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vectors[0],
        limit=top_k,
        with_payload=True,
    ).points

    found = []

    for point in results:
        payload = point.payload or {}

        found.append({
            "text": payload.get("text", ""),
            "source": payload.get("source", "Unknown source"),
            "page": payload.get("page", "?"),
            "distance": 1.0 - float(point.score),
        })

    return found


def clear_collection():
    """Delete the indexed collection if it exists."""
    client = _get_client()

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
