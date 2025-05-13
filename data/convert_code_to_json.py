import json
import re
from pathlib import Path

# Use absolute base directory (LawBot/data)
base_dir = Path(__file__).resolve().parent
input_path = base_dir / "texas_family_code.txt"
output_path = base_dir / "texas_family_code.jsonl"

with open(input_path, "r", encoding="utf-8") as infile:
    raw_text = infile.read()

pattern = r"Sec\. (\d+\.\d+)\.\s+(.*?)\.\s+(.*?)(?=\n\nSec\.|\Z)"
matches = re.findall(pattern, raw_text, re.DOTALL)

with open(output_path, "w", encoding="utf-8") as outfile:
    for sec_id, title, body in matches:
        item = {
            "id": f"Sec. {sec_id}",
            "title": title.strip(),
            "text": body.strip()
        }
        outfile.write(json.dumps(item) + "\n")
