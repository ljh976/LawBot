import os
import json
import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Init OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
index_path = data_dir / "index.faiss"
meta_path = data_dir / "doc_meta.json"

# Load FAISS index and metadata
index = faiss.read_index(str(index_path))
with open(meta_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Helper
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)

def search_documents(query, k=3):
    query_vec = get_embedding(query).reshape(1, -1)
    _, indices = index.search(query_vec, k)
    return [documents[i] for i in indices[0]]

def build_prompt(docs, question):
    context = "\n\n".join([f"[{d['id']}] {d['title']}\n{d['text']}" for d in docs])
    return f"""You are a legal assistant. Answer based only on the documents below.

Documents:
{context}

Question:
{question}

Only answer if the documents contain the answer.
"""

def get_answer(query):
    docs = search_documents(query)
    prompt = build_prompt(docs, query)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip(), docs

# UI
st.set_page_config(page_title="Texas LawBot", layout="wide")
st.title("📘 Texas LawBot")
st.caption("Ask a question based on Texas Family Law")

query = st.text_area("Your legal question", height=120)

if st.button("Submit") and query.strip():
    with st.spinner("Thinking..."):
        answer, sources = get_answer(query)
        st.markdown("### 💬 Answer")
        st.write(answer)

        with st.expander("📚 Source Documents"):
            for doc in sources:
                st.markdown(f"**{doc['id']} — {doc['title']}**")
                st.write(doc['text'])
