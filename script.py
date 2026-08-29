from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = [...]
vectors = model.encode(chunks)

def retrieve(query, k=3):
    q = model.encode([query])[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top_k = np.argsort(scores)[-k:][::-1]
    return [chunks[i] for i in top_k]