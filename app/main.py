from fastapi import FastAPI, UploadFile, File
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunk_service import chunk_text
from app.services.embedding_service import generate_embeddings
import os

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Lock-in started 🚀"}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    extracted_text = extract_text_from_pdf(file_path)

    chunks = chunk_text(extracted_text)

    embeddings = generate_embeddings(chunks)

    return {
        "filename": file.filename,
        "chunks_created": len(chunks),
        "embedding_dimension": len(embeddings[0]),
        "preview_chunk": chunks[0][:300]
       }