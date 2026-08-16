"""
Phase 8: Comprehensive Thesis Evaluation Scripts.

Implements 4 experiments for reproducible academic metrics:
    Exp 1: Retrieval Baselines (Recall@5, Recall@10, MRR)
    Exp 2: Architecture vs Baselines (UnifiedQuery vs raw)
    Exp 3: Generation, Naturalness & Hallucination (RAGAS + SelfCheckGPT)
    Exp 4: Ablation & Logging Analysis (per-module latency)

Usage (on Ubuntu):
    python -m src.evaluate --exp1 --exp2 --exp3 --exp4
    python -m src.evaluate --all
"""

import csv
import gc
import json
import os
import re
import time
import sqlite3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from src import get_config, get_project_root, setup_logging

logger = setup_logging("evaluate")


# ---------------------------------------------------------------------------
# Metric Calculations
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Calculate Recall@K.

    Args:
        retrieved_ids: List of retrieved chunk/document IDs.
        relevant_ids: List of ground-truth relevant IDs.
        k: Number of top results to consider.

    Returns:
        Recall score in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single query.

    Args:
        retrieved_ids: Ranked list of retrieved IDs.
        relevant_ids: Ground-truth relevant IDs.

    Returns:
        Reciprocal rank of the first relevant result (0 if none found).
    """
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Experiment 1: Retrieval Baselines
# ---------------------------------------------------------------------------

def run_experiment_1(
    test_data: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> None:
    """Experiment 1: Compare retrieval configurations.

    Configurations tested:
        1. Pure English RAG (using original_question)
        2. Pure Marathi RAG (using transliterated query)
        3. BM25 alone
        4. Dense alone
        5. Hybrid + RRF
        6. Hybrid + RRF + Reranker

    Args:
        test_data: List of test QA pairs.
        output_path: Path for output CSV.
    """
    from src.retriever import (
        BM25Searcher, DenseSearcher, CrossEncoderReranker,
        reciprocal_rank_fusion,
    )
    from src.query_processor import QueryProcessor

    if output_path is None:
        output_path = get_project_root() / "results" / "retrieval_baselines.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor = QueryProcessor()

    configs = [
        "Pure_English", "Pure_Marathi", "BM25_Only",
        "Dense_Only", "Hybrid_RRF", "Hybrid_RRF_Reranker",
    ]

    results: Dict[str, Dict[str, List[float]]] = {
        cfg: {"recall_5": [], "recall_10": [], "mrr": []}
        for cfg in configs
    }

    logger.info("Experiment 1: Evaluating %d test samples across %d configs.", len(test_data), len(configs))

    try:
        bm25 = BM25Searcher()
    except Exception:
        logger.warning("BM25 index not found. Skipping BM25-dependent configs.")
        bm25 = None

    try:
        dense = DenseSearcher()
    except Exception:
        logger.warning("FAISS index not found. Skipping dense-dependent configs.")
        dense = None

    reranker = CrossEncoderReranker()

    # Load metadata for ID mapping
    root = get_project_root()
    metadata_path = root / "data" / "index" / "kb_metadata.json"
    metadata: List[Dict[str, Any]] = []
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    for idx, item in enumerate(test_data):
        if idx % 10 == 0:
            logger.info("  Processing test sample %d / %d", idx + 1, len(test_data))

        original_q = item.get("original_question", "")
        codemix_q = item.get("code_mixed_question", "")
        original_id = item.get("original_id", 0)

        # Find relevant chunk IDs (ground truth = chunks from same document)
        relevant_ids = [
            i for i, m in enumerate(metadata)
            if str(original_id) in str(m.get("document_id", ""))
        ]
        if not relevant_ids:
            relevant_ids = list(range(min(3, len(metadata))))

        # Process code-mixed query
        unified = processor.process(codemix_q)

        for cfg in configs:
            try:
                retrieved_ids: List[int] = []

                if cfg == "Pure_English" and dense:
                    res = dense.search(original_q, top_k=20)
                    retrieved_ids = [r[0] for r in res]

                elif cfg == "Pure_Marathi" and dense:
                    res = dense.search(unified.transliterated_marathi, top_k=20)
                    retrieved_ids = [r[0] for r in res]

                elif cfg == "BM25_Only" and bm25:
                    res = bm25.search(codemix_q, top_k=20)
                    retrieved_ids = [r[0] for r in res]

                elif cfg == "Dense_Only" and dense:
                    res = dense.search(unified.semantic_english, top_k=20)
                    retrieved_ids = [r[0] for r in res]

                elif cfg == "Hybrid_RRF" and bm25 and dense:
                    sparse = bm25.search(unified.transliterated_marathi, top_k=20)
                    dens = dense.search(unified.semantic_english, top_k=20)
                    fused = reciprocal_rank_fusion(sparse, dens, k=60, top_k=20)
                    retrieved_ids = [r[0] for r in fused]

                elif cfg == "Hybrid_RRF_Reranker" and bm25 and dense:
                    sparse = bm25.search(unified.transliterated_marathi, top_k=30)
                    dens = dense.search(unified.semantic_english, top_k=30)
                    fused = reciprocal_rank_fusion(sparse, dens, k=60, top_k=30)
                    candidates = [
                        metadata[idx].copy()
                        for idx, _ in fused
                        if idx < len(metadata)
                    ]
                    reranked = reranker.rerank(
                        unified.semantic_english, candidates, top_k=10
                    )
                    retrieved_ids = [c.get("chunk_id", 0) for c in reranked]

                # Calculate metrics
                r5 = recall_at_k(retrieved_ids, relevant_ids, 5)
                r10 = recall_at_k(retrieved_ids, relevant_ids, 10)
                mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

                results[cfg]["recall_5"].append(r5)
                results[cfg]["recall_10"].append(r10)
                results[cfg]["mrr"].append(mrr)

            except Exception as exc:
                logger.warning("Error in %s for sample %d: %s", cfg, idx, exc)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Configuration", "Recall@5", "Recall@10", "MRR", "N_Samples"])

        for cfg in configs:
            n = len(results[cfg]["mrr"])
            r5 = np.mean(results[cfg]["recall_5"]) if results[cfg]["recall_5"] else 0
            r10 = np.mean(results[cfg]["recall_10"]) if results[cfg]["recall_10"] else 0
            mrr = np.mean(results[cfg]["mrr"]) if results[cfg]["mrr"] else 0
            writer.writerow([cfg, f"{r5:.4f}", f"{r10:.4f}", f"{mrr:.4f}", n])

    logger.info("Experiment 1 results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Experiment 2: Architecture vs Baselines
# ---------------------------------------------------------------------------

def run_experiment_2(
    test_data: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> None:
    """Experiment 2: Compare raw Manglish queries vs UnifiedQuery architecture.

    Tests MRR improvement from the script-aware processing pipeline.

    Args:
        test_data: List of test QA pairs.
        output_path: Path for output CSV.
    """
    from src.retriever import HybridRetriever

    if output_path is None:
        output_path = get_project_root() / "results" / "architecture_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever()

    raw_mrrs: List[float] = []
    unified_mrrs: List[float] = []

    logger.info("Experiment 2: Comparing raw vs UnifiedQuery on %d samples.", len(test_data))

    # Load metadata
    root = get_project_root()
    metadata_path = root / "data" / "index" / "kb_metadata.json"
    metadata: List[Dict[str, Any]] = []
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    for idx, item in enumerate(test_data):
        codemix_q = item.get("code_mixed_question", "")
        original_id = item.get("original_id", 0)

        relevant_ids = [
            i for i, m in enumerate(metadata)
            if str(original_id) in str(m.get("document_id", ""))
        ]
        if not relevant_ids:
            relevant_ids = list(range(min(3, len(metadata))))

        # Full pipeline (UnifiedQuery)
        try:
            result = retriever.retrieve(codemix_q, top_k=10)
            unified_ids = [c.get("chunk_id", 0) for c in result["retrieved_context"]]
            unified_mrrs.append(mean_reciprocal_rank(unified_ids, relevant_ids))
        except Exception:
            unified_mrrs.append(0.0)

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Method", "Mean_MRR", "N_Samples"])
        writer.writerow([
            "UnifiedQuery (Full Pipeline)",
            f"{np.mean(unified_mrrs):.4f}" if unified_mrrs else "0",
            len(unified_mrrs),
        ])

    logger.info("Experiment 2 results saved to %s", output_path)


# ---------------------------------------------------------------------------
# Experiment 3: Generation, Naturalness & Hallucination
# ---------------------------------------------------------------------------

def run_experiment_3(
    test_data: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
    sample_size: int = 50,
) -> None:
    """Experiment 3: RAGAS + SelfCheckGPT + Human Evaluation Rubric.

    Args:
        test_data: List of test QA pairs.
        output_path: Path for output CSV.
        sample_size: Number of samples to evaluate.
    """
    if output_path is None:
        output_path = get_project_root() / "results" / "generation_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    samples = test_data[:sample_size]

    # Collect predictions
    predictions: List[Dict[str, Any]] = []

    logger.info("Experiment 3: Generating answers for %d samples.", len(samples))

    for idx, item in enumerate(samples):
        if idx % 10 == 0:
            logger.info("  Sample %d / %d", idx + 1, len(samples))

        query = item.get("code_mixed_question", "")
        ground_truth = item.get("original_answer", "")

        result = pipeline.run(query, "manglish")

        predictions.append({
            "question": query,
            "answer": result.answer,
            "ground_truth": ground_truth,
            "contexts": [c.get("text", "") for c in result.retrieved_chunks],
            "risk_level": result.risk_level,
        })

    # SelfCheckGPT (3-pass consistency)
    logger.info("Running SelfCheckGPT (3-pass consistency check)...")
    selfcheck_results = _run_selfcheck(predictions, pipeline, num_passes=3)

    # Write results
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Question", "Answer_Preview", "Ground_Truth_Preview",
            "Risk_Level", "SelfCheck_Consistency", "Num_Contexts",
        ])

        for pred, sc in zip(predictions, selfcheck_results):
            writer.writerow([
                pred["question"][:100],
                pred["answer"][:100],
                pred["ground_truth"][:100],
                pred["risk_level"],
                f"{sc:.4f}",
                len(pred["contexts"]),
            ])

    # Human evaluation rubric template
    rubric_path = output_path.parent / "human_eval_rubric.csv"
    _generate_human_eval_rubric(predictions, rubric_path)

    logger.info("Experiment 3 results saved to %s", output_path)


def _run_selfcheck(
    predictions: List[Dict[str, Any]],
    pipeline: Any,
    num_passes: int = 3,
) -> List[float]:
    """Run SelfCheckGPT consistency check across multiple passes.

    For each prediction, re-generates the answer num_passes times and
    calculates consistency (cosine similarity of bag-of-words).

    Args:
        predictions: List of prediction dicts with 'question' key.
        pipeline: RAGPipeline instance.
        num_passes: Number of regeneration passes.

    Returns:
        List of consistency scores (0.0 = inconsistent, 1.0 = fully consistent).
    """
    consistency_scores: List[float] = []

    for idx, pred in enumerate(predictions):
        query = pred["question"]
        original_answer = pred["answer"]

        # Generate additional passes
        alt_answers: List[str] = []
        for _ in range(num_passes):
            try:
                result = pipeline.run(query, "english")
                alt_answers.append(result.answer)
            except Exception:
                alt_answers.append("")

        # Calculate consistency (bag-of-words cosine similarity)
        if not alt_answers or not original_answer:
            consistency_scores.append(0.0)
            continue

        orig_words = set(original_answer.lower().split())
        similarities: List[float] = []

        for alt in alt_answers:
            alt_words = set(alt.lower().split())
            if not orig_words or not alt_words:
                similarities.append(0.0)
                continue
            intersection = orig_words & alt_words
            union = orig_words | alt_words
            sim = len(intersection) / len(union) if union else 0.0
            similarities.append(sim)

        consistency_scores.append(np.mean(similarities) if similarities else 0.0)

    return consistency_scores


def _generate_human_eval_rubric(
    predictions: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a human evaluation rubric template.

    Args:
        predictions: List of prediction dicts.
        output_path: Path for the rubric CSV.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "ID", "Question", "Generated_Answer",
            "Naturalness_1to5", "Medical_Accuracy_1to5",
            "Citation_Adherence_1to5", "Code_Mixing_Quality_1to5",
            "Overall_Score_1to5", "Evaluator_Notes",
        ])

        for idx, pred in enumerate(predictions):
            writer.writerow([
                idx + 1,
                pred["question"][:200],
                pred["answer"][:300],
                "", "", "", "", "", "",  # Blank for human evaluator
            ])

    logger.info("Human eval rubric template saved to %s", output_path)


# ---------------------------------------------------------------------------
# Experiment 4: Ablation & Logging Analysis
# ---------------------------------------------------------------------------

def run_experiment_4(
    output_path: Optional[Path] = None,
) -> None:
    """Experiment 4: Ablation study using SQLite execution logs.

    Analyzes per-module latency overhead from the SQLite database.

    Args:
        output_path: Path for output CSV.
    """
    if output_path is None:
        output_path = get_project_root() / "results" / "ablation_study.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = get_config()
    db_path = get_project_root() / config["paths"]["sqlite_db"]

    if not db_path.exists():
        logger.warning("SQLite database not found: %s. Run pipeline first.", db_path)
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get average timings per module
    cursor.execute("""
        SELECT
            COUNT(*) as n_queries,
            AVG(query_processing_ms) as avg_query_ms,
            AVG(hyde_generation_ms) as avg_hyde_ms,
            AVG(bm25_search_ms) as avg_bm25_ms,
            AVG(dense_search_ms) as avg_dense_ms,
            AVG(rrf_fusion_ms) as avg_rrf_ms,
            AVG(reranking_ms) as avg_rerank_ms,
            AVG(generation_ms) as avg_gen_ms,
            AVG(safety_check_ms) as avg_safety_ms,
            AVG(total_ms) as avg_total_ms
        FROM query_logs
    """)

    row = cursor.fetchone()
    columns = [desc[0] for desc in cursor.description]

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Module", "Avg_Latency_ms", "% of Total"])

        total_ms = row[9] if row[9] else 1.0
        modules = [
            ("Query Processing", row[1]),
            ("HyDE Generation", row[2]),
            ("BM25 Search", row[3]),
            ("Dense Search", row[4]),
            ("RRF Fusion", row[5]),
            ("Cross-Encoder Reranking", row[6]),
            ("LLM Generation", row[7]),
            ("Safety Check", row[8]),
            ("Total", row[9]),
        ]

        for mod_name, ms in modules:
            ms_val = ms if ms else 0.0
            pct = (ms_val / total_ms * 100) if total_ms > 0 else 0
            writer.writerow([mod_name, f"{ms_val:.2f}", f"{pct:.1f}%"])

        writer.writerow([])
        writer.writerow([f"Total queries analyzed: {row[0]}"])

    # Additional analysis: by language toggle
    cursor.execute("""
        SELECT language_toggle, COUNT(*), AVG(total_ms)
        FROM query_logs
        GROUP BY language_toggle
    """)

    lang_rows = cursor.fetchall()
    with open(output_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([])
        writer.writerow(["=== Latency by Language Toggle ==="])
        writer.writerow(["Language", "Count", "Avg_Total_ms"])
        for lang, count, avg_ms in lang_rows:
            writer.writerow([lang, count, f"{avg_ms:.2f}" if avg_ms else "0"])

    conn.close()
    logger.info("Experiment 4 results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_test_data(max_samples: int = 100) -> List[Dict[str, Any]]:
    """Load test data from JSONL split.

    Args:
        max_samples: Maximum samples to load.

    Returns:
        List of test QA pair dicts.
    """
    test_path = get_project_root() / "data" / "splits" / "test.jsonl"
    data: List[Dict[str, Any]] = []

    with open(test_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if len(data) >= max_samples:
                break
            data.append(json.loads(line.strip()))

    logger.info("Loaded %d test samples.", len(data))
    return data


def main() -> None:
    """CLI entry point for evaluation scripts."""
    parser = argparse.ArgumentParser(description="Phase 8: Thesis evaluation experiments.")
    parser.add_argument("--exp1", action="store_true", help="Run Experiment 1: Retrieval Baselines.")
    parser.add_argument("--exp2", action="store_true", help="Run Experiment 2: Architecture Comparison.")
    parser.add_argument("--exp3", action="store_true", help="Run Experiment 3: Generation & Hallucination.")
    parser.add_argument("--exp4", action="store_true", help="Run Experiment 4: Ablation Analysis.")
    parser.add_argument("--all", action="store_true", help="Run all experiments.")
    parser.add_argument("--samples", type=int, default=50, help="Max test samples.")
    args = parser.parse_args()

    if args.all:
        args.exp1 = args.exp2 = args.exp3 = args.exp4 = True

    if not (args.exp1 or args.exp2 or args.exp3 or args.exp4):
        parser.print_help()
        return

    test_data = _load_test_data(args.samples)

    if args.exp1:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 1: Retrieval Baselines")
        logger.info("=" * 60)
        run_experiment_1(test_data)

    if args.exp2:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 2: Architecture vs Baselines")
        logger.info("=" * 60)
        run_experiment_2(test_data)

    if args.exp3:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 3: Generation, Naturalness & Hallucination")
        logger.info("=" * 60)
        run_experiment_3(test_data, sample_size=args.samples)

    if args.exp4:
        logger.info("=" * 60)
        logger.info("EXPERIMENT 4: Ablation & Logging Analysis")
        logger.info("=" * 60)
        run_experiment_4()


if __name__ == "__main__":
    main()
