from fastapi import FastAPI, UploadFile, File
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunk_service import chunk_text
from app.services.embedding_service import (
    generate_embeddings,
    generate_query_embedding,
    find_most_similar
)
from app.services.llm_service import generate_answer
import os

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Temporary in-memory storage
stored_chunks = []
stored_embeddings = []


@app.get("/")
def root():
    return {"message": "Lock-in started 🚀"}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global stored_chunks, stored_embeddings

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded PDF
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract text
    extracted_text = extract_text_from_pdf(file_path)

    # Chunk text
    chunks = chunk_text(extracted_text)

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Store for retrieval
    stored_chunks = chunks
    stored_embeddings = embeddings

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "embedding_dimension": len(embeddings[0]),
        "preview_chunk": chunks[0][:300]
    }


@app.get("/ask/")
async def ask_question(question: str):
    global stored_chunks, stored_embeddings

    if not stored_chunks:
        return {
            "error": "No documents uploaded yet."
        }

    query_embedding = generate_query_embedding(question)

    results = find_most_similar(
        query_embedding,
        stored_embeddings,
        stored_chunks
    )

    contexts = [result["content"] for result in results]

    answer = generate_answer(
        question,
        contexts
    )

    return {
        "question": question,
        "answer": answer,
        "sources": results
    }