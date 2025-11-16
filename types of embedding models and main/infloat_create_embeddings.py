from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import numpy as np

# ---------------------------
# Folders
# ---------------------------
DOCS_FOLDER = os.path.join("backend", "data", "docs")
MODEL_FOLDER = os.path.join("models")
os.makedirs(MODEL_FOLDER, exist_ok=True)


# ---------------------------
# Load documents
# ---------------------------
docs = []
for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".txt"):
        path = os.path.join(DOCS_FOLDER, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                docs.append(text)

if not docs:
    raise ValueError(f"No text files found in {DOCS_FOLDER}")

print(f"[INFO] Loaded {len(docs)} documents.")

# ---------------------------
# Create embeddings
# ---------------------------
embedding_model = SentenceTransformer('intfloat/e5-base-v2')  # better semantic quality
embeddings = embedding_model.encode(docs, convert_to_numpy=True).astype('float32')

# Normalize embeddings for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print("[INFO] Created and normalized embeddings.")

# ---------------------------
# Create FAISS index (cosine similarity)
# ---------------------------
index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product ≈ cosine similarity
index.add(embeddings)
print(f"[INFO] FAISS index created with {index.ntotal} vectors.")

# ---------------------------
# Save index and metadata
# ---------------------------
faiss.write_index(index, os.path.join(MODEL_FOLDER, "index.faiss"))
with open(os.path.join(MODEL_FOLDER, "metadata.pkl"), "wb") as f:
    pickle.dump(docs, f)

print(f"[INFO] Saved FAISS index and metadata to '{MODEL_FOLDER}'.")
