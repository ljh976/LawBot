import os
import json
import requests
import streamlit as st
import firebase_admin
from openai import OpenAI
from dotenv import load_dotenv
from firebase_admin import auth, credentials, initialize_app
from pathlib import Path
import faiss
import numpy as np

# Firebase Admin Init
cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Load env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Get API Keys
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
FIREBASE_API_KEY = st.secrets.get("FIREBASE_API_KEY", os.getenv("FIREBASE_API_KEY"))
client = OpenAI(api_key=OPENAI_API_KEY)

# Firebase REST Signup/Login
def signup(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    res = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True})
    if res.status_code == 200:
        return res.json()
    else:
        raise Exception(res.json()["error"]["message"])

def login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    res = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True})
    if res.status_code == 200:
        return res.json()
    else:
        raise Exception(res.json()["error"]["message"])

# FAISS + RAG
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
index_path = data_dir / "index.faiss"
meta_path = data_dir / "doc_meta.json"

index = faiss.read_index(str(index_path))
with open(meta_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

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

# Streamlit UI
st.set_page_config(page_title="Texas LawBot", layout="wide")
st.title("📘 Texas LawBot")

# Auth UI
if "user" not in st.session_state:
    mode = st.sidebar.radio("Choose", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button(mode):
        try:
            if mode == "Sign Up":
                user = signup(email, password)
            else:
                user = login(email, password)
            st.session_state["user"] = user
            st.rerun()
        except Exception as e:
            st.sidebar.error(str(e))
    st.stop()

# Logged in
st.sidebar.success(f"Logged in as: {st.session_state['user']['email']}")
if st.sidebar.button("Log out"):
    del st.session_state["user"]
    st.rerun()

# App UI
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
