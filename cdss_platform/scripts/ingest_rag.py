import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.rag.rag_engine import rag_engine

print("Starting ChromaDB ingestion...")
guidelines_dir = Path("data/guidelines")
count = rag_engine.ingest_guidelines_directory(guidelines_dir)
print(f"Done. Total chunks ingested: {count}")
print(f"Collection count: {rag_engine.collection_count()}")
