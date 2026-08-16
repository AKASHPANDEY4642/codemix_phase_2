"""
Phase 4: Hybrid Retrieval & HyDE Pipeline.

Retrieves and fuses context using all representations from the UnifiedQuery:
    1. HyDE (Hypothetical Document Embeddings) via external API
    2. BM25 sparse search (transliterated Marathi + medical terms)
    3. FAISS dense search (BAAI/bge-m3 local, API-offloaded secondary)
    4. Reciprocal Rank Fusion (RRF) to merge rankings
    5. Cross-encoder reranking (BAAI/bge-reranker-v2-m3)

VRAM management: models are loaded/unloaded sequentially.

Usage (on Ubuntu):
    from src.retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.retrieve("mala stomach pain hotoy")
"""

import gc
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src import get_config, get_project_root, setup_logging
from src.query_processor import QueryProcessor, UnifiedQuery

logger = setup_logging("retriever")


# ---------------------------------------------------------------------------
# VRAM Management Helpers
# ---------------------------------------------------------------------------

def _unload_model(model: Any, label: str = "model") -> None:
    """Safely unload a PyTorch model from GPU/RAM.

    Args:
        model: The model object to unload.
        label: Human-readable label for logging.
    """
    import torch

    try:
        if hasattr(model, "cpu"):
            model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded %s. VRAM freed.", label)
    except Exception as exc:
        logger.warning("Error unloading %s: %s", label, exc)


# ---------------------------------------------------------------------------
# HyDE Generator
# ---------------------------------------------------------------------------

class HyDEGenerator:
    """Generates a hypothetical document using a lightweight LLM API.

    The HyDE document is a synthetic pure-English medical answer that
    is used as the query for dense search — matching the semantic space
    of the knowledge base more closely than the raw Manglish query.
    """

    def __init__(self) -> None:
        """Initialize HyDE with Groq API configuration."""
        config = get_config()
        self._model = config["retrieval"]["hyde_model"]
        self._max_tokens = config["retrieval"]["hyde_max_tokens"]
        self._temperature = config["retrieval"]["hyde_temperature"]
        logger.info("HyDEGenerator initialized (model=%s).", self._model)

    def generate(self, query: str) -> str:
        """Generate a hypothetical medical document for the query.

        Args:
            query: Pure English or code-mixed query text.

        Returns:
            A synthetic medical answer in pure English.
        """
        try:
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                logger.warning("GROQ_API_KEY not set. Returning query as HyDE doc.")
                return query

            client = Groq(api_key=api_key)
            prompt = (
                "You are a medical expert. Provide a brief, factual, "
                "evidence-based medical answer to the following query. "
                "Use only standard medical English. Be concise (2-3 sentences).\n\n"
                f"Query: {query}\n\nAnswer:"
            )

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a medical knowledge assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            hyde_doc = response.choices[0].message.content.strip()
            logger.debug("HyDE document generated (%d chars).", len(hyde_doc))
            return hyde_doc

        except Exception as exc:
            logger.warning("HyDE generation failed: %s. Using query as fallback.", exc)
            return query


# ---------------------------------------------------------------------------
# Sparse Search (BM25)
# ---------------------------------------------------------------------------

class BM25Searcher:
    """BM25 sparse searcher over the knowledge base chunks.

    Uses the pre-built BM25 index (rank_bm25.BM25Okapi).
    """

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        """Load the BM25 index and metadata.

        Args:
            index_dir: Directory containing BM25 index files.
        """
        if index_dir is None:
            index_dir = get_project_root() / "data" / "index"

        bm25_path = index_dir / "bm25_index.pkl"
        metadata_path = index_dir / "kb_metadata.json"

        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")

        with open(bm25_path, "rb") as fh:
            data = pickle.load(fh)
            self._bm25 = data["bm25"]

        with open(metadata_path, "r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

        logger.info("BM25 searcher loaded (%d documents).", len(self._metadata))

    def search(self, query: str, top_k: int = 30) -> List[Tuple[int, float]]:
        """Search using BM25 scoring.

        Args:
            query: Search query (tokenized by whitespace).
            top_k: Number of top results to return.

        Returns:
            List of (chunk_index, bm25_score) tuples, sorted by score descending.
        """
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ---------------------------------------------------------------------------
# Dense Search (FAISS)
# ---------------------------------------------------------------------------

class DenseSearcher:
    """FAISS dense vector searcher using a local embedding model.

    Uses BAAI/bge-m3 as the local embedding model.
    VRAM-aware: model is loaded and unloaded explicitly.
    """

    def __init__(self, index_dir: Optional[Path] = None) -> None:
        """Load the FAISS index (vectors only, model loaded on-demand).

        Args:
            index_dir: Directory containing FAISS index files.
        """
        import faiss

        if index_dir is None:
            index_dir = get_project_root() / "data" / "index"

        faiss_path = index_dir / "dense_index.faiss"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

        self._index = faiss.read_index(str(faiss_path))
        self._index_dir = index_dir
        logger.info("FAISS index loaded (%d vectors).", self._index.ntotal)

    def search(self, query_text: str, top_k: int = 30) -> List[Tuple[int, float]]:
        """Search using dense embeddings.

        Loads the embedding model, encodes the query, searches, then unloads.

        Args:
            query_text: Query text to embed and search.
            top_k: Number of top results to return.

        Returns:
            List of (chunk_index, distance) tuples, sorted by similarity.
        """
        import torch
        from sentence_transformers import SentenceTransformer

        config = get_config()
        model_name = config["retrieval"]["local_embedding_model"]

        logger.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name, trust_remote_code=True)

        if torch.cuda.is_available():
            model = model.to("cuda")

        query_vector = model.encode(
            [query_text], convert_to_numpy=True, show_progress_bar=False
        )

        # Unload immediately
        _unload_model(model, "embedding model")

        distances, indices = self._index.search(query_vector, top_k)
        results = [
            (int(indices[0][i]), float(distances[0][i]))
            for i in range(len(indices[0]))
            if indices[0][i] != -1
        ]
        return results

    def search_with_api(
        self, query_text: str, top_k: int = 30
    ) -> List[Tuple[int, float]]:
        """Search using API-offloaded embeddings (e.g., OpenAI).

        Args:
            query_text: Query text to embed via API.
            top_k: Number of top results.

        Returns:
            List of (chunk_index, distance) tuples.
        """
        config = get_config()
        provider = config["retrieval"]["api_embedding_provider"]

        if provider == "openai":
            return self._search_via_openai(query_text, top_k)
        else:
            logger.warning("API embedding provider '%s' not supported. Falling back to local.", provider)
            return self.search(query_text, top_k)

    def _search_via_openai(self, query_text: str, top_k: int) -> List[Tuple[int, float]]:
        """Embed query via OpenAI API and search FAISS index.

        Args:
            query_text: Query to embed.
            top_k: Number of results.

        Returns:
            List of (chunk_index, distance) tuples.
        """
        try:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set. Falling back to local.")
                return self.search(query_text, top_k)

            client = OpenAI(api_key=api_key)
            config = get_config()
            model = config["retrieval"]["api_embedding_model"]

            response = client.embeddings.create(input=[query_text], model=model)
            query_vector = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

            # Note: dimension mismatch is possible if API model != local index model
            if query_vector.shape[1] != self._index.d:
                logger.warning(
                    "Dimension mismatch: API=%d, index=%d. Falling back to local.",
                    query_vector.shape[1], self._index.d,
                )
                return self.search(query_text, top_k)

            distances, indices = self._index.search(query_vector, top_k)
            return [
                (int(indices[0][i]), float(distances[0][i]))
                for i in range(len(indices[0]))
                if indices[0][i] != -1
            ]

        except Exception as exc:
            logger.warning("OpenAI embedding failed: %s. Falling back to local.", exc)
            return self.search(query_text, top_k)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *result_lists: List[Tuple[int, float]],
    k: int = 60,
    top_k: int = 30,
) -> List[Tuple[int, float]]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        result_lists: Variable number of (chunk_index, score) lists.
        k: RRF constant (default 60).
        top_k: Number of top results to return after fusion.

    Returns:
        List of (chunk_index, rrf_score) tuples, sorted by RRF score descending.
    """
    rrf_scores: Dict[int, float] = {}

    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_fused[:top_k]


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Reranks candidate chunks using a cross-encoder model.

    Uses BAAI/bge-reranker-v2-m3. VRAM-aware: loads and unloads explicitly.
    """

    def __init__(self) -> None:
        """Initialize reranker configuration."""
        config = get_config()
        self._model_name = config["retrieval"]["reranker_model"]
        self._top_k = config["retrieval"]["reranker_top_k"]
        logger.info("CrossEncoderReranker configured (model=%s, top_k=%d).",
                     self._model_name, self._top_k)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks against the query.

        Loads the cross-encoder, scores all candidates, returns top-k, then unloads.

        Args:
            query: The query text (use UnifiedQuery representations).
            candidates: List of chunk metadata dicts (must have 'text' key).
            top_k: Number of top candidates to return. Defaults to config value.

        Returns:
            Top-k candidates sorted by reranker score, each with 'rerank_score' added.
        """
        import torch
        from sentence_transformers import CrossEncoder

        if top_k is None:
            top_k = self._top_k

        if not candidates:
            return []

        logger.info("Loading reranker model: %s", self._model_name)
        model = CrossEncoder(self._model_name, trust_remote_code=True)

        if torch.cuda.is_available():
            model.model.to("cuda")

        # Prepare query-document pairs
        pairs = [(query, cand["text"]) for cand in candidates]

        # Score all pairs
        scores = model.predict(pairs, show_progress_bar=False)

        # Unload immediately
        if hasattr(model, "model"):
            _unload_model(model.model, "reranker model")
        del model
        gc.collect()

        # Sort by score
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_results: List[Dict[str, Any]] = []
        for cand, score in scored[:top_k]:
            result = cand.copy()
            result["rerank_score"] = float(score)
            top_results.append(result)

        logger.info(
            "Reranking complete. Top %d of %d candidates returned.",
            len(top_results), len(candidates),
        )
        return top_results


# ---------------------------------------------------------------------------
# Main Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Full hybrid retrieval pipeline with HyDE, BM25, FAISS, RRF, and reranking.

    Manages VRAM by loading and unloading models sequentially.
    """

    def __init__(self) -> None:
        """Initialize all retrieval components."""
        self._query_processor = QueryProcessor()
        self._hyde = HyDEGenerator()
        self._reranker = CrossEncoderReranker()

        # Load metadata for result enrichment
        config = get_config()
        root = get_project_root()
        metadata_path = root / "data" / "index" / "kb_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            logger.info("HybridRetriever initialized (%d chunks in KB).", len(self._metadata))
        else:
            self._metadata = []
            logger.warning("KB metadata not found. Run ingestion + indexing first.")

    def retrieve(
        self,
        raw_query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Execute the full retrieval pipeline.

        Pipeline:
            1. Process query → UnifiedQuery
            2. Generate HyDE document
            3. BM25 sparse search (transliterated + medical terms)
            4. FAISS dense search (HyDE document)
            5. RRF fusion (top 30)
            6. Cross-encoder reranking (top 5)

        Args:
            raw_query: Raw user query (Manglish).
            top_k: Final number of retrieved chunks.

        Returns:
            Dictionary with query_info, hyde_document, retrieved_context,
            rrf_scores, and timing information.
        """
        timings: Dict[str, float] = {}
        config = get_config()

        # Step 1: Process query
        t0 = time.time()
        unified = self._query_processor.process(raw_query)
        timings["query_processing_ms"] = (time.time() - t0) * 1000

        # Step 2: HyDE
        t0 = time.time()
        hyde_doc = self._hyde.generate(unified.semantic_english)
        timings["hyde_generation_ms"] = (time.time() - t0) * 1000

        # Step 3: BM25 Sparse Search
        t0 = time.time()
        try:
            bm25 = BM25Searcher()
            # Build search query from transliterated + medical terms
            medical_terms_str = " ".join(
                t["term"] for t in unified.extracted_medical_terms
            )
            bm25_query = f"{unified.transliterated_marathi} {medical_terms_str}".strip()
            sparse_results = bm25.search(bm25_query, top_k=config["retrieval"]["bm25_top_k"])
        except FileNotFoundError:
            logger.warning("BM25 index not found. Skipping sparse search.")
            sparse_results = []
        timings["bm25_search_ms"] = (time.time() - t0) * 1000

        # Step 4: FAISS Dense Search
        t0 = time.time()
        try:
            dense = DenseSearcher()
            # Use HyDE document for primary dense search
            dense_results = dense.search(hyde_doc, top_k=config["retrieval"]["dense_top_k"])

            # Optional: API-offloaded secondary search with semantic_english
            api_provider = config["retrieval"].get("api_embedding_provider", "none")
            if api_provider != "none":
                try:
                    api_results = dense.search_with_api(
                        unified.semantic_english,
                        top_k=config["retrieval"]["dense_top_k"],
                    )
                    # Merge API results into dense results
                    dense_results = dense_results + api_results
                except Exception as exc:
                    logger.warning("API dense search failed: %s", exc)
        except FileNotFoundError:
            logger.warning("FAISS index not found. Skipping dense search.")
            dense_results = []
        timings["dense_search_ms"] = (time.time() - t0) * 1000

        # Step 5: Reciprocal Rank Fusion
        t0 = time.time()
        fused = reciprocal_rank_fusion(
            sparse_results,
            dense_results,
            k=config["retrieval"]["rrf_k"],
            top_k=config["retrieval"]["rrf_top_k"],
        )
        timings["rrf_fusion_ms"] = (time.time() - t0) * 1000

        # Build candidate chunks with metadata
        candidates: List[Dict[str, Any]] = []
        for chunk_idx, rrf_score in fused:
            if chunk_idx < len(self._metadata):
                chunk = self._metadata[chunk_idx].copy()
                chunk["rrf_score"] = rrf_score
                candidates.append(chunk)

        # Step 6: Cross-Encoder Reranking
        t0 = time.time()
        # Use multiple query representations for reranking
        rerank_query = f"{unified.semantic_english} {unified.transliterated_marathi}"
        top_chunks = self._reranker.rerank(rerank_query, candidates, top_k=top_k)
        timings["reranking_ms"] = (time.time() - t0) * 1000

        total_ms = sum(timings.values())
        timings["total_retrieval_ms"] = total_ms

        logger.info(
            "Retrieval complete: %d chunks in %.0fms (BM25=%.0f, Dense=%.0f, Rerank=%.0f)",
            len(top_chunks), total_ms,
            timings["bm25_search_ms"], timings["dense_search_ms"],
            timings["reranking_ms"],
        )

        return {
            "query_info": {
                "original": unified.original_manglish,
                "transliterated": unified.transliterated_marathi,
                "semantic_english": unified.semantic_english,
                "script_type": unified.script_type,
                "medical_terms": unified.extracted_medical_terms,
            },
            "hyde_document": hyde_doc,
            "retrieved_context": top_chunks,
            "rrf_scores": [(idx, score) for idx, score in fused[:10]],
            "timings": timings,
        }


# ---------------------------------------------------------------------------
# Index Builder (run on Ubuntu)
# ---------------------------------------------------------------------------

def build_indices() -> None:
    """Build BM25 and FAISS indices from the knowledge base.

    This must be run on Ubuntu after ingestion. It:
    1. Loads KB chunks
    2. Builds BM25 index
    3. Loads embedding model, generates vectors, builds FAISS index
    4. Unloads embedding model
    """
    import faiss
    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    root = get_project_root()
    config = get_config()

    # Load knowledge base chunks
    kb_path = root / "data" / "knowledge_base" / "chunks.json"
    with open(kb_path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)

    logger.info("Building indices for %d chunks.", len(chunks))
    index_dir = root / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata mapping
    metadata_path = index_dir / "kb_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    # Build BM25
    logger.info("Building BM25 index...")
    corpus_texts = [c["text"] for c in chunks]
    tokenized = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    bm25_path = index_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as fh:
        pickle.dump({"bm25": bm25}, fh)
    logger.info("BM25 index saved: %s", bm25_path)

    # Build FAISS
    model_name = config["retrieval"]["local_embedding_model"]
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)

    if torch.cuda.is_available():
        model = model.to("cuda")

    logger.info("Generating embeddings in batches...")
    batch_size = 64
    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs)
        if (i // batch_size) % 10 == 0:
            logger.info("  Embedded %d / %d chunks", min(i + batch_size, len(corpus_texts)), len(corpus_texts))

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # Unload embedding model
    _unload_model(model, "embedding model (indexing)")

    # Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine with normalized vectors
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    faiss_path = index_dir / "dense_index.faiss"
    faiss.write_index(faiss_index, str(faiss_path))
    logger.info("FAISS index saved: %s (%d vectors, dim=%d)", faiss_path, faiss_index.ntotal, dimension)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4: Retrieval operations.")
    parser.add_argument("--build-index", action="store_true", help="Build BM25 + FAISS indices.")
    parser.add_argument("--test-query", type=str, help="Test retrieval with a query.")
    args = parser.parse_args()

    if args.build_index:
        build_indices()

    if args.test_query:
        retriever = HybridRetriever()
        results = retriever.retrieve(args.test_query)
        print(f"\nQuery: {args.test_query}")
        print(f"HyDE: {results['hyde_document'][:200]}")
        print(f"\nTop {len(results['retrieved_context'])} chunks:")
        for i, chunk in enumerate(results["retrieved_context"], 1):
            print(f"  [{i}] (score={chunk.get('rerank_score', 0):.4f}) {chunk['text'][:100]}...")
        print(f"\nTimings: {results['timings']}")
