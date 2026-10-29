
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "college_documents"

# Cohere hosted embeddings
EMBEDDING_MODEL = "embed-english-v3.0"

# Text chunking
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
