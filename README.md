# 📘 LawBot

A RAG-based legal assistant for Texas Family Code built with OpenAI, FAISS, and Streamlit.

> 🔒 This project is proprietary. Not licensed for reuse.

---

## 🧠 What It Does

This application allows users to ask questions related to Texas family law and receive accurate, document-grounded answers using:

- **Retrieval-Augmented Generation (RAG)** structure
- **FAISS** for vector-based document similarity search
- **OpenAI GPT** for natural language generation
- **Streamlit** for a user-friendly web interface

---

## 🏗️ Tech Stack

- **Python 3.10+**
- [OpenAI](https://platform.openai.com/docs/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [dotenv](https://pypi.org/project/python-dotenv/)

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI API key
echo OPENAI_API_KEY=sk-... > .env

# 3. Generate document embeddings
python embed_all.py

# 4. Launch the web UI
python run_ui.py
