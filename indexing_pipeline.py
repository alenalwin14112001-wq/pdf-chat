import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import config

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def load_all_pdfs(pdf_dir: str) -> list[str]:
    texts = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDFs found in {pdf_dir}")
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        if text.strip():
            texts.append(text)
        else:
            print(f"  Warning: No text extracted from {pdf_file.name}")
    return texts


def chunk_texts(texts: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Encoding {len(texts)} chunks locally...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return np.array(embeddings, dtype="float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def run_indexing():
    os.makedirs(config.INDEX_DIR, exist_ok=True)

    print("\n Step 1 — Loading PDFs...")
    texts = load_all_pdfs(config.PDF_DIR)
    print(f"   {len(texts)} PDF(s) loaded")

    print("\n Step 2 — Chunking text...")
    chunks = chunk_texts(texts)
    print(f"   {len(chunks)} chunks created")

    print("\n Step 3 — Embedding chunks (may take a minute)...")
    embeddings = embed_texts(chunks)
    print(f"   Embeddings shape: {embeddings.shape}")

    print("\n Step 4 — Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)
    faiss.write_index(faiss_index, config.FAISS_INDEX_PATH)
    print(f"   Saved to {config.FAISS_INDEX_PATH}")

    print("\n Step 5 — Building BM25 index...")
    bm25_index = build_bm25_index(chunks)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_index, f)
    with open(config.CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"   Saved to {config.BM25_INDEX_PATH}")

    print(f"\n Indexing complete — {len(chunks)} chunks indexed and ready!")


if __name__ == "__main__":
    run_indexing()