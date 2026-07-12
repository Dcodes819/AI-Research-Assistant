from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    return model.encode(chunks)


def generate_query_embedding(query):
    return model.encode(query)


def find_most_similar(query_embedding, chunk_embeddings, chunks, top_k=3):
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) *
        np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "similarity_score": float(similarities[idx]),
            "content": chunks[idx]
        })

    return results