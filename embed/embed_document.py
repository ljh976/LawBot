import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths (absolute)
base_dir = Path(__file__).resolve().parent.parent  # LawBot/
data_dir = base_dir / "data"
input_file = data_dir / "texas_family_code.jsonl"
index_file = data_dir / "index.faiss"
meta_file = data_dir / "doc_meta.json"

# Model config
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text, model=EMBEDDING_MODEL):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)

# Load JSONL
documents = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        documents.append(json.loads(line))

# Compute embeddings
embeddings = []
for doc in tqdm(documents, desc="Embedding documents"):
    embeddings.append(get_embedding(doc["text"]))

# Create FAISS index
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# Save index and metadata
faiss.write_index(index, str(index_file))
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2)

print("✅ Embedding complete. Saved index and metadata.")
