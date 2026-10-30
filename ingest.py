"""Run once (or to re-index): python ingest.py [path/to/file.pdf]"""
import sys, shutil, os
from app.rag import build_index, CHROMA_DIR, PDF_PATH
from app.graph import CACHE_FILE

if os.path.isdir(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
if os.path.exists(CACHE_FILE):
    os.remove(CACHE_FILE)  # cached answers belong to the old index
build_index(sys.argv[1] if len(sys.argv) > 1 else PDF_PATH)
print("Index built.")