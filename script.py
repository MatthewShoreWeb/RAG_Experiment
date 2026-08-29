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

# 1. Load raw text
with open("knowledge.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Chunk text by paragraph
chunks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

# Now pass chunks into your existing script
vectors = model.encode(chunks)