# config.py
import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# --- File Paths ---
PAPERS_DIR = "data/papers"
CHROMA_DIR = "data/chroma_db"

# --- Embedding Model ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cpu"    # Switch to "cpu" if CUDA unavailable

# --- LLM via Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.1-8b-instant"

# --- Chunking ---
CHUNK_SIZE    = 512    # max characters per chunk
CHUNK_OVERLAP = 64     # characters shared between adjacent chunks

# --- Retrieval ---
TOP_K = 5              # number of chunks to retrieve per query