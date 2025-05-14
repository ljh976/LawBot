import os
import json
import faiss
import numpy as np
import streamlit as st
import pyrebase
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load .env (local)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Load secrets (for deployment)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

FIREBASE_CONFIG = {
    "apiKey": st.secrets.get("FIREBASE_API_KEY", ""),
    "authDomain": st.secrets.get("FIREBASE_AUTH_DOMAIN", ""),
    "projectId": st.secrets.get("FIREBASE_PROJECT_ID", ""),
    "storageBucket": st.secrets.get("FIREBASE_STORAGE_BUCKET", ""),
    "messagingSenderId": st.secrets.get("FIREBASE_MESSAGING_SENDER_ID", ""),
    "appId": st.secrets.get("FIREBASE_APP_ID", ""),
    "databaseURL": ""
}

# Firebase init
firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
auth = firebase.auth()

# OpenAI init
client = OpenAI(api_key=OPENAI_API_KEY)

# Load FAISS index and metadata
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
index_path = data_dir / "index.faiss"
meta_path = data_dir / "doc_meta.json"

index = faiss.read_index(str(index_path))
with open(meta_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Embedding/search
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

# Sidebar auth
st.sidebar.title("Account")
if "user" not in st.session_state:
    auth_mode = st.sidebar.radio("Mode", ["Login", "Sign Up"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if auth_mode == "Login":
        if st.sidebar.button("Log In"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state["user"] = user
                st.rerun()
            except Exception as e:
                st.sidebar.error("Login failed")
                st.sidebar.code(str(e))
    else:
        if st.sidebar.button("Create Account"):
            try:
                auth.create_user_with_email_and_password(email, password)
                st.sidebar.success("Account created. Please log in.")
            except Exception as e:
                st.sidebar.error("Sign Up failed")
                st.sidebar.code(str(e))
    st.stop()
else:
    st.sidebar.success(f"Logged in as: {st.session_state['user']['email']}")
    if st.sidebar.button("Log out"):
        del st.session_state["user"]
        st.rerun()

# Main app
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
