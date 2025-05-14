import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__) / ".env"
load_dotenv(dotenv_path=env_path)

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

app_path = Path(__file__).resolve().parent / "app" / "lawbot_app.py"

subprocess.run([
    "python", "-m", "streamlit", "run", str(app_path),
    "--server.headless", "true"
], check=True, env=env)
