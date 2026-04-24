import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from config import config

# Load indexes once at startup
_model = SentenceTransformer("all-MiniLM-L6-v2")
_faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)
with open(config.CHUNKS_PATH, "rb") as f:
    _chunks = pickle.load(f)
with open(config.BM25_INDEX_PATH, "rb") as f:
    _bm25 = pickle.load(f)


def retrieve_chunks(query: str, k: int = None) -> list[str]:
    k = k or config.TOP_K

    # 1. FAISS vector search
    query_vec = np.array([_model.encode(query)], dtype="float32")
    distances, indices = _faiss_index.search(query_vec, k)
    faiss_hits = list(indices[0])

    # 2. BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_hits = np.argsort(bm25_scores)[::-1][:k].tolist()

    # 3. Hybrid fusion — combine scores
    scores = {}
    for rank, idx in enumerate(faiss_hits):
        scores[idx] = scores.get(idx, 0) + config.VECTOR_WEIGHT * (1 / (rank + 1))
    for rank, idx in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0) + config.BM25_WEIGHT * (1 / (rank + 1))

    top_indices = sorted(scores, key=scores.get, reverse=True)[:k]
    return [_chunks[i] for i in top_indices]


def answer_query(query: str) -> str:
    chunks = retrieve_chunks(query)
    context = "\n\n---\n\n".join(chunks)

    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = client.models.generate_content(
         model=config.LLM_MODEL,
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    print("PDF Query System ready. Type 'quit' to exit.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue
        print("\nSearching...\n")
        answer = answer_query(query)
        print(f"Answer: {answer}\n")
        print("-" * 60 + "\n")