import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = "llama-3.3-70b-versatile"  # Free & powerful!

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Indexing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5

    # Hybrid retrieval weights
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    # Paths
    PDF_DIR: str = "data/pdfs"
    INDEX_DIR: str = "indexes"
    FAISS_INDEX_PATH: str = "indexes/faiss_index"
    BM25_INDEX_PATH:  str = "indexes/bm25_index.pkl"
    CHUNKS_PATH: str = "indexes/chunks.pkl"
    CHUNK_PAGES_PATH: str = "indexes/chunk_pages.pkl"

config = Config()