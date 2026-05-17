"""
CDSS Platform – Server Entry Point
Run with: python run.py
Or:       uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import sys
from pathlib import Path

# Ensure we're in the project root
os.chdir(Path(__file__).parent)

# Load .env
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from app.core.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    print("=" * 60)
    print(f"  {settings.app_name}")
    print(f"  v{settings.app_version} | {settings.app_env.upper()}")
    print(f"  http://{settings.api_host}:{settings.api_port}")
    print(f"  Docs: http://localhost:{settings.api_port}/docs")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info",
        access_log=True,
    )
