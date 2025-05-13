import subprocess
from pathlib import Path

# Resolve path to app/streamlit_app.py
base_dir = Path(__file__).resolve().parent
app_path = base_dir / "app" / "streamlit_app.py"

# Run Streamlit app in headless mode
subprocess.run([
    "python", "-m", "streamlit", "run", str(app_path),
    "--server.headless", "true"
], check=True)
