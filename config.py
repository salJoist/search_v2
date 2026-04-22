
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
ES_HOSTNAME = os.getenv("ES_HOSTNAME", "http://localhost")
ES_PORT = os.getenv("ES_PORT", "9200")
ES_USERNAME = os.getenv("ES_USERNAME") or os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD") or os.getenv("ES_PASS")
