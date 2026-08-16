"""
Phase 5: FastAPI Backend.

Provides a /chat REST endpoint for the MedManglish-RAG pipeline.
Handles CORS, request validation, structured responses, and health checks.

Usage (on Ubuntu):
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import get_config, setup_logging

logger = setup_logging("api")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request schema for the /chat endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="User medical query (Manglish/English/Marathi).")
    language_mode: str = Field("manglish", description="Output language: 'manglish', 'english', 'marathi'.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")


class CitationItem(BaseModel):
    """A single citation reference."""
    chunk_id: int
    source: str
    document_id: str
    text_snippet: str
    umls_cuis: List[str] = []


class ChatResponse(BaseModel):
    """Response schema for the /chat endpoint."""
    answer: str
    thought_process: str = ""
    risk_level: str = "LOW"
    disclaimer: str = ""
    citations: List[CitationItem] = []
    language_mode: str = "manglish"
    query_info: Dict[str, Any] = {}
    timings: Dict[str, float] = {}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedManglish-RAG API",
    description=(
        "Script-Aware Retrieval-Augmented Generation for "
        "Marathi-English Code-Mixed Medical QA"
    ),
    version="2.0.0",
)

# CORS
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialize pipeline (avoid loading on import)
_pipeline = None


def _get_pipeline():
    """Lazy-load the RAG pipeline on first request.

    Returns:
        Initialized RAGPipeline instance.
    """
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline
        logger.info("Initializing RAG pipeline (first request)...")
        _pipeline = RAGPipeline()
        logger.info("RAG pipeline ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse with system status.
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        components={
            "api": "running",
            "pipeline": "initialized" if _pipeline else "not_loaded",
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a medical query through the RAG pipeline.

    Args:
        request: ChatRequest with query and language mode.

    Returns:
        ChatResponse with answer, citations, and metadata.

    Raises:
        HTTPException: On pipeline errors.
    """
    logger.info(
        "Chat request: query='%s', lang='%s'",
        request.query[:80], request.language_mode,
    )

    try:
        pipeline = _get_pipeline()
        result = pipeline.run(request.query, request.language_mode)

        # Build citations
        citations = []
        for chunk in result.citations:
            citations.append(CitationItem(
                chunk_id=chunk.get("chunk_id", 0),
                source=chunk.get("source", "unknown"),
                document_id=chunk.get("document_id", ""),
                text_snippet=chunk.get("text", "")[:200],
                umls_cuis=chunk.get("umls_cuis", []),
            ))

        return ChatResponse(
            answer=result.answer,
            thought_process=result.thought_process,
            risk_level=result.risk_level,
            disclaimer=result.disclaimer,
            citations=citations,
            language_mode=result.language_mode,
            query_info=result.query_info,
            timings=result.timings,
        )

    except Exception as exc:
        logger.error("Chat endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get query statistics from SQLite logs.

    Returns:
        Dict with aggregate statistics.
    """
    import sqlite3

    config = get_config()
    db_path = _PROJECT_ROOT / config["paths"]["sqlite_db"]

    if not db_path.exists():
        return {"total_queries": 0, "message": "No queries logged yet."}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    stats = {}
    cursor.execute("SELECT COUNT(*) FROM query_logs")
    stats["total_queries"] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(total_ms) FROM query_logs")
    avg_ms = cursor.fetchone()[0]
    stats["avg_latency_ms"] = round(avg_ms, 2) if avg_ms else 0

    cursor.execute(
        "SELECT risk_level, COUNT(*) FROM query_logs GROUP BY risk_level"
    )
    stats["risk_distribution"] = dict(cursor.fetchall())

    cursor.execute(
        "SELECT language_toggle, COUNT(*) FROM query_logs GROUP BY language_toggle"
    )
    stats["language_distribution"] = dict(cursor.fetchall())

    conn.close()
    return stats


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True,
    )
