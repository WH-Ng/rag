# src/config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "lectures")
IMAGE_STORE = os.path.join(PROJECT_ROOT, "data", "processed", "lecture_images")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "results")
DB_DIR = os.getenv("DB_DIR", os.path.join(PROJECT_ROOT, "storage", "chroma_db"))

COLLECTION_NAME = "lectures"
EMBED_MODEL = "bge-m3"
VISION_MODEL = "qwen2.5vl:7b"
LLM_MODEL = "qwen2.5:7b"
TOP_K = 4