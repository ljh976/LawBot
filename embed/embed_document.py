import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# File paths
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
input_file = data_dir / "texas_family_code.jsonl"
index_file = data_dir / "index.faiss"
meta_file = data_dir / "doc_meta.json"

# Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_TOKEN_SIZE = 1000  # Max tokens per chunk (approximate)

# Rough token estimator (1 word ≈ 1 token)
def split_text(text, max_tokens=CHUNK_TOKEN_SIZE):
    words = text.split()
    chunks = []
    current = []
    count = 0

    for word in words:
        count += 1
        current.append(word)
        if count >= max_tokens:
            chunks.append(" ".join(current))
            current = []
            count = 0

    if current:
        chunks.append(" ".join(current))
    return chunks

# Get OpenAI embedding for a given text
def get_embedding(text, model=EMBEDDING_MODEL):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)

# Initialize storage
embeddings = []
metadata = []

# Read and process documents
with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Embedding documents"):
        doc = json.loads(line)
        text_chunks = split_text(doc["text"])
        for i, chunk in enumerate(text_chunks):
            embedding = get_embedding(chunk)
            embeddings.append(embedding)
            metadata.append({
                "id": f"{doc['id']}_{i}",
                "title": doc.get("title", "Untitled"),
                "text": chunk
            })

# Build FAISS index
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))
faiss.write_index(index, str(index_file))

# Save metadata
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("✅ Embedding complete. Saved FAISS index and metadata.")