"""
Phase 1: Data Loader Utilities for JSONL Splits.

Provides typed data classes and loader functions for reading, validating,
and iterating over the synthetic code-mixed QA dataset splits.

Usage:
    from src.data_loader import load_split, MedQAPair
    pairs = load_split("train")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from src import get_project_root, setup_logging

logger = setup_logging("data_loader")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MedQAPair:
    """A single code-mixed medical QA pair.

    Attributes:
        original_id: Unique identifier from the source dataset.
        original_question: Pure English question.
        original_answer: Pure English answer.
        code_mixed_question: Code-mixed (Manglish) question.
        code_mixed_answer: Code-mixed (Manglish) answer.
        source: Data source name (e.g., 'PubMedQA').
        cmi_question: Code-Mixing Index of the question.
        cmi_answer: Code-Mixing Index of the answer.
        cmi_avg: Average CMI across question and answer.
        cmi_tier: CMI generation tier ('light', 'medium', 'heavy').
    """
    original_id: int
    original_question: str
    original_answer: str
    code_mixed_question: str
    code_mixed_answer: str
    source: str = "PubMedQA"
    cmi_question: float = 0.0
    cmi_answer: float = 0.0
    cmi_avg: float = 0.0
    cmi_tier: str = "medium"


@dataclass
class ChunkMetadata:
    """Metadata for a single knowledge base chunk.

    Attributes:
        document_id: Unique document identifier.
        source: Data source name.
        title: Document title (truncated to 120 chars).
        chunk_id: Global chunk sequence number.
        text: The chunk text content.
        umls_cuis: List of UMLS Concept Unique Identifiers found in the chunk.
    """
    document_id: str
    source: str
    title: str
    chunk_id: int
    text: str
    umls_cuis: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------

def load_split(
    split_name: str,
    splits_dir: Optional[Path] = None,
) -> List[MedQAPair]:
    """Load a JSONL dataset split into typed MedQAPair objects.

    Args:
        split_name: One of 'train', 'val', 'test'.
        splits_dir: Optional override for the splits directory path.

    Returns:
        List of MedQAPair instances.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If split_name is not one of the valid names.
    """
    valid_splits = {"train", "val", "test"}
    if split_name not in valid_splits:
        raise ValueError(f"Invalid split name '{split_name}'. Must be one of {valid_splits}.")

    if splits_dir is None:
        splits_dir = get_project_root() / "data" / "splits"

    split_path = splits_dir / f"{split_name}.jsonl"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            "Run `python -m src.generate_synthetic_data --split` first."
        )

    pairs: List[MedQAPair] = []

    with open(split_path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                pair = MedQAPair(
                    original_id=entry.get("original_id", 0),
                    original_question=entry.get("original_question", ""),
                    original_answer=entry.get("original_answer", ""),
                    code_mixed_question=entry.get("code_mixed_question", ""),
                    code_mixed_answer=entry.get("code_mixed_answer", ""),
                    source=entry.get("source", "PubMedQA"),
                    cmi_question=entry.get("cmi_question", 0.0),
                    cmi_answer=entry.get("cmi_answer", 0.0),
                    cmi_avg=entry.get("cmi_avg", 0.0),
                    cmi_tier=entry.get("cmi_tier", "medium"),
                )
                pairs.append(pair)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping malformed line %d in %s: %s", line_num, split_path, exc)

    logger.info("Loaded %d pairs from %s split.", len(pairs), split_name)
    return pairs


def iter_split(
    split_name: str,
    splits_dir: Optional[Path] = None,
) -> Iterator[MedQAPair]:
    """Lazily iterate over a JSONL split file (memory-efficient).

    Args:
        split_name: One of 'train', 'val', 'test'.
        splits_dir: Optional override for the splits directory path.

    Yields:
        MedQAPair instances, one at a time.
    """
    if splits_dir is None:
        splits_dir = get_project_root() / "data" / "splits"

    split_path = splits_dir / f"{split_name}.jsonl"

    with open(split_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                yield MedQAPair(
                    original_id=entry.get("original_id", 0),
                    original_question=entry.get("original_question", ""),
                    original_answer=entry.get("original_answer", ""),
                    code_mixed_question=entry.get("code_mixed_question", ""),
                    code_mixed_answer=entry.get("code_mixed_answer", ""),
                    source=entry.get("source", "PubMedQA"),
                    cmi_question=entry.get("cmi_question", 0.0),
                    cmi_answer=entry.get("cmi_answer", 0.0),
                    cmi_avg=entry.get("cmi_avg", 0.0),
                    cmi_tier=entry.get("cmi_tier", "medium"),
                )
            except (json.JSONDecodeError, KeyError):
                continue


def load_knowledge_base(
    kb_path: Optional[Path] = None,
) -> List[ChunkMetadata]:
    """Load the knowledge base chunks from JSON.

    Args:
        kb_path: Optional path to chunks.json. Defaults to data/knowledge_base/chunks.json.

    Returns:
        List of ChunkMetadata instances.
    """
    if kb_path is None:
        kb_path = get_project_root() / "data" / "knowledge_base" / "chunks.json"

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {kb_path}. Run `python -m src.ingest --rebuild` first."
        )

    with open(kb_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    chunks = [
        ChunkMetadata(
            document_id=item["document_id"],
            source=item["source"],
            title=item["title"],
            chunk_id=item["chunk_id"],
            text=item["text"],
            umls_cuis=item.get("umls_cuis", []),
        )
        for item in data
    ]
    logger.info("Loaded %d knowledge base chunks.", len(chunks))
    return chunks


def get_split_stats(split_name: str) -> dict:
    """Get statistics about a dataset split.

    Args:
        split_name: One of 'train', 'val', 'test'.

    Returns:
        Dictionary with count, avg_question_length, avg_answer_length, avg_cmi.
    """
    pairs = load_split(split_name)

    if not pairs:
        return {"count": 0}

    q_lengths = [len(p.code_mixed_question.split()) for p in pairs]
    a_lengths = [len(p.code_mixed_answer.split()) for p in pairs]
    cmis = [p.cmi_avg for p in pairs if p.cmi_avg > 0]

    return {
        "count": len(pairs),
        "avg_question_words": sum(q_lengths) / len(q_lengths),
        "avg_answer_words": sum(a_lengths) / len(a_lengths),
        "avg_cmi": sum(cmis) / len(cmis) if cmis else 0.0,
    }
