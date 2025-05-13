import subprocess
from pathlib import Path

# Base path
base = Path(__file__).resolve().parent

# Step 1: Convert txt → jsonl
print("📄 Converting TXT to JSONL...")
convert_script = base / "data" / "convert_code_to_json.py"
subprocess.run(["python", str(convert_script)], check=True)

# Step 2: Embed JSONL into FAISS
print("📦 Embedding documents into FAISS index...")
embed_script = base / "embed" / "embed_document.py"
subprocess.run(["python", str(embed_script)], check=True)

print("✅ All done.")
