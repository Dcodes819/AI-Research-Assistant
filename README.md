AI Research Assistant:
A Retrieval Augmented Generation (RAG) application that answers questions from uploaded PDF documents using semantic search and Gemini.

Features
- Upload PDFs
- Extract and chunk text
- Generate embeddings using Sentence Transformers
- Retrieve relevant context using cosine similarity
- Generate grounded answers using Gemini

Tech Stack
- FastAPI
- SentenceTransformers
- Gemini API
- NumPy
- Python

Run locally
```bash
git clone <repo>
cd ai-research-assistant

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python -m uvicorn app.main:app --reload
