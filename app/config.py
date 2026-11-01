"""Central config. Sab kuch environment variables (.env) se aata hai - secrets code me nahi."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


PROMPTS_DIR = ROOT / "prompts"
DB_PATH = _path("DB_PATH", "data/jobs.db")
CHROMA_DIR = _path("CHROMA_DIR", "data/chroma")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

EXTRACT_PROMPT_VERSION = os.getenv("EXTRACT_PROMPT_VERSION", "extract_v2")
SQL_PROMPT_VERSION = os.getenv("SQL_PROMPT_VERSION", "text2sql_v1")
QUERY_PARSE_PROMPT_VERSION = "query_parse_v1"
ANSWER_PROMPT_VERSION = "answer_v1"

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")  # chroma | memory
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "st")  # st | hash
EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.88"))

MAX_EXTRACT_ATTEMPTS = 3
MAX_SQL_RETRIES = 3

# USD per 1M tokens (input, output). List prices - apne provider ke pricing page se verify karo.
# Free tier me actual cost 0 hoti hai; ye numbers "agar paid hota to" estimate ke liye hain.
PRICING = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
}
