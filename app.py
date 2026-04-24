import streamlit as st
import pickle
import faiss
import numpy as np
import os
import google.generativeai as genai
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from config import config

st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄", 
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f1117;
    border-right: 1px solid #2a2a3d;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* Upload button */
[data-testid="stFileUploader"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #a0a0b0 !important;
}
[data-testid="stFileUploader"] {
    background: #1a1a2e;
    border: 1px dashed #3a3a5c;
    border-radius: 10px;
    padding: 10px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #1a3a5c;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 1px solid #2a2a3d !important;
    background: #1a1a2e !important;
    color: #e0e0e0 !important;
    padding: 12px 18px !important;
    font-size: 14px !important;
}

/* Expander (sources) */
[data-testid="stExpander"] {
    border: 1px solid #2a2a3d !important;
    border-radius: 10px !important;
    background: #1a1a2e !important;
}

/* Success/info boxes */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #4a90d9 !important;
}

/* Buttons */
[data-testid="stButton"] button {
    border-radius: 8px !important;
    border: 1px solid #2a2a3d !important;
    background: #1a1a2e !important;
    color: #a0a0b0 !important;
    font-size: 13px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    border-color: #4a90d9 !important;
    color: #4a90d9 !important;
}

/* Page tag badge */
.page-tag {
    display: inline-block;
    background: #1a3a5c;
    color: #4a90d9;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 4px;
}

/* Chunk header */
.chunk-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}

/* Source chunk card */
.source-card {
    background: #12121e;
    border: 1px solid #2a2a3d;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}

/* PDF badge */
.pdf-badge {
    background: #1a3a5c;
    border: 1px solid #2a4a6c;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12px;
    color: #4a90d9;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 4px;
}

/* Topbar */
.topbar {
    background: #12121e;
    border-bottom: 1px solid #2a2a3d;
    padding: 12px 0;
    margin-bottom: 16px;
}
.topbar h2 {
    font-size: 22px;
    font-weight: 600;
    color: #ffffff;
    margin: 0;
}
.topbar p {
    font-size: 13px;
    color: #a0a0b0;
    margin: 2px 0 0;
}

/* Chunks indexed badge */
.chunks-badge {
    background: #0d2d1a;
    border: 1px solid #1a5c2a;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12px;
    color: #4aaa6a;
    text-align: center;
    margin-top: 4px;
}

/* Light mode overrides */
@media (prefers-color-scheme: light) {
    [data-testid="stSidebar"] {
        background-color: #f5f5f7;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * { color: #1a1a2e !important; }
    [data-testid="stFileUploader"] { background: #ffffff; border-color: #c0c0d0; }
    [data-testid="stChatInput"] textarea { background: #ffffff !important; border-color: #c0c0d0 !important; color: #1a1a2e !important; }
    [data-testid="stExpander"] { background: #ffffff !important; border-color: #e0e0e0 !important; }
    [data-testid="stButton"] button { background: #ffffff !important; border-color: #c0c0d0 !important; color: #555 !important; }
    .source-card { background: #f9f9fb; border-color: #e0e0e0; }
    .pdf-badge { background: #e8f0fb; border-color: #b0c8f0; color: #185fa5; }
    .chunks-badge { background: #eafaf1; border-color: #a0d8b0; color: #1a7a3a; }
    .page-tag { background: #e8f0fb; color: #185fa5; }
    .topbar { background: #ffffff; border-color: #e0e0e0; }
    .topbar h2 { color: #1a1a2e; }
    .topbar p { color: #666; }
}
</style>
""", unsafe_allow_html=True)


# ── Load embedding model ───────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Index PDF ──────────────────────────────────────────────
def index_pdf(pdf_path: str):
    import PyPDF2
    pages_text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                pages_text.append((i + 1, extracted))

    if not pages_text:
        return False, "No text could be extracted from this PDF."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = []
    chunk_pages = []

    for page_num, text in pages_text:
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            chunk_pages.append(page_num)

    model = load_model()
    embeddings = np.array(model.encode(chunks, show_progress_bar=False), dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.BM25_INDEX_PATH, "wb") as f: pickle.dump(bm25, f)
    with open(config.CHUNKS_PATH, "wb") as f: pickle.dump(chunks, f)
    with open(config.CHUNK_PAGES_PATH, "wb") as f: pickle.dump(chunk_pages, f)

    return True, len(chunks)


# ── Load indexes ───────────────────────────────────────────
def load_indexes():
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CHUNKS_PATH, "rb") as f: chunks = pickle.load(f)
    with open(config.BM25_INDEX_PATH, "rb") as f: bm25 = pickle.load(f)
    with open(config.CHUNK_PAGES_PATH, "rb") as f: chunk_pages = pickle.load(f)
    return index, chunks, bm25, chunk_pages


# ── Retrieve chunks ────────────────────────────────────────
def retrieve_chunks(query, model, index, chunks, bm25, chunk_pages):
    k = config.TOP_K
    query_vec = np.array([model.encode(query)], dtype="float32")
    _, indices = index.search(query_vec, k)
    faiss_hits = list(indices[0])
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_hits = np.argsort(bm25_scores)[::-1][:k].tolist()
    scores = {}
    for rank, idx in enumerate(faiss_hits):
        scores[idx] = scores.get(idx, 0) + config.VECTOR_WEIGHT * (1 / (rank + 1))
    for rank, idx in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0) + config.BM25_WEIGHT * (1 / (rank + 1))
    top_indices = sorted(scores, key=scores.get, reverse=True)[:k]
    return [(chunks[i], chunk_pages[i]) for i in top_indices]


# ── Answer query ───────────────────────────────────────────
def answer_query(query, model, index, chunks, bm25, chunk_pages, chat_history):
    retrieved = retrieve_chunks(query, model, index, chunks, bm25, chunk_pages)
    retrieved_chunks = [r[0] for r in retrieved]
    retrieved_pages = [r[1] for r in retrieved]
    context = "\n\n---\n\n".join(retrieved_chunks)

    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful professional assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Previous conversation:
{history_text}
Question: {query}

Answer:"""

    try:
        genai.configure(api_key=config.GOOGLE_API_KEY)
        gemini = genai.GenerativeModel(config.LLM_MODEL)
        response = gemini.generate_content(prompt)
        return response.text, retrieved_chunks, retrieved_pages
    except Exception as e:
        error_msg = str(e)
        if "ResourceExhausted" in error_msg or "429" in error_msg:
            return "⚠️ API quota exceeded. Please try again after a few minutes or contact the app owner.", retrieved_chunks, retrieved_pages
        elif "APIConnectionError" in error_msg:
            return "⚠️ Connection error. Please check your internet connection and try again.", retrieved_chunks, retrieved_pages
        else:
            return f"⚠️ An error occurred: {error_msg[:200]}", retrieved_chunks, retrieved_pages
# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Chat")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="visible")

    if uploaded_file:
        pdf_path = f"data/pdfs/{uploaded_file.name}"
        os.makedirs("data/pdfs", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            success, result = index_pdf(pdf_path)
        if success:
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chunk_count = result
            st.session_state.messages = []
        else:
            st.error(result)

    if "pdf_name" in st.session_state:
        st.markdown(f"""<div class="pdf-badge">📄 {st.session_state.pdf_name}</div>""", unsafe_allow_html=True)

    if "chunk_count" in st.session_state:
        st.markdown(f"""<div class="chunks-badge">✓ {st.session_state.chunk_count:,} chunks indexed</div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ── Main area ──────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <h2>Chat with your PDF</h2>
  <p>Upload a document and ask anything about it</p>
</div>
""", unsafe_allow_html=True)

indexes_exist = (
    Path(config.FAISS_INDEX_PATH).exists() and
    Path(config.CHUNKS_PATH).exists() and
    Path(config.BM25_INDEX_PATH).exists() and
    Path(config.CHUNK_PAGES_PATH).exists()
)

if not indexes_exist:
    st.info("📂 Upload a PDF in the sidebar to get started.")
    st.stop()

model = load_model()
index, chunks, bm25, chunk_pages = load_indexes()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, sources, pages = answer_query(
                query, model, index, chunks, bm25, chunk_pages,
                st.session_state.messages
            )
        st.markdown(answer)

        unique_pages = sorted(set(pages))
        page_tags = " ".join([f'<span class="page-tag">pg {p}</span>' for p in unique_pages])
        st.markdown(f"**Sources:** {page_tags}", unsafe_allow_html=True)

        with st.expander(f"📚 {len(sources)} source chunks used"):
            for i, (chunk, page) in enumerate(zip(sources, pages)):
                st.markdown(f"""
<div class="source-card">
  <div class="chunk-header">
    <strong>Chunk {i+1}</strong>
    <span class="page-tag">📄 Page {page}</span>
  </div>
  <small style="color:#a0a0b0;">{chunk[:400] + "..." if len(chunk) > 400 else chunk}</small>
</div>
""", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })