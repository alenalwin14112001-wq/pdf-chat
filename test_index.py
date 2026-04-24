import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import config

# Load everything
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(config.FAISS_INDEX_PATH)
with open(config.CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"Chunks loaded: {len(chunks)}")
print(f"FAISS vectors: {index.ntotal}")

# Test a query
query = "what is this document about?"
query_vec = np.array([model.encode(query)], dtype="float32")
distances, indices = index.search(query_vec, k=3)

print(f"\nTop 3 results for: '{query}'\n")
for rank, idx in enumerate(indices[0]):
    print(f"[{rank+1}] {chunks[idx][:300]}")
    print()