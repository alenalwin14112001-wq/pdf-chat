# PDF Chat 📄

An AI-powered PDF chat application built with Streamlit and Groq.

## Features
- Upload any PDF and ask questions about it
- Hybrid search using FAISS + BM25
- Page number references for every answer
- Multi-turn chat history
- Professional dark/light UI

## Tech Stack
- Streamlit
- Groq (LLaMA 3.3 70B)
- FAISS + BM25 hybrid retrieval
- SentenceTransformers
- LangChain

## How to Run Locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GROQ_API_KEY` to a `.env` file
4. Run: `streamlit run app.py`