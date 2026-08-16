"""
Phase 1: Medical Corpus Ingestion, Semantic Chunking & UMLS Enrichment.

Parses medical datasets (PubMedQA, NHP India, Ayushman Bharat),
applies medical-aware semantic chunking, and enriches every chunk
with UMLS Concept Unique Identifiers (CUIs).

Usage (on Ubuntu):
    python -m src.ingest --rebuild
"""

import gc
import json
import os
import re
import hashlib
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src import get_config, get_project_root, setup_logging

logger = setup_logging("ingest")

# ---------------------------------------------------------------------------
# UMLS CUI Enrichment (Local Dictionary Fallback)
# ---------------------------------------------------------------------------
# A curated subset of UMLS CUIs for common medical concepts.
# On Ubuntu, this can be extended by querying the UMLS REST API.

UMLS_LOCAL_DICT: Dict[str, str] = {
    # Symptoms
    "fever": "C0015967", "headache": "C0018681", "cough": "C0010200",
    "pain": "C0030193", "nausea": "C0027497", "vomiting": "C0042963",
    "diarrhea": "C0011991", "fatigue": "C0015672", "dizziness": "C0012833",
    "shortness of breath": "C0013404", "chest pain": "C0008031",
    "abdominal pain": "C0000737", "back pain": "C0004604",
    "sore throat": "C0242429", "runny nose": "C1260880",
    "body ache": "C0281856", "weakness": "C0004093",
    "swelling": "C0013604", "rash": "C0015230", "itching": "C0033774",
    "bleeding": "C0019080", "breathlessness": "C0013404",
    "weight loss": "C1262477", "weight gain": "C0043094",
    "insomnia": "C0917801", "anxiety": "C0003467",
    "depression": "C0011570", "palpitations": "C0030252",

    # Diseases
    "diabetes": "C0011849", "hypertension": "C0020538",
    "asthma": "C0004096", "tuberculosis": "C0041296",
    "malaria": "C0024530", "dengue": "C0011311",
    "typhoid": "C0041466", "cholera": "C0008354",
    "pneumonia": "C0032285", "bronchitis": "C0006277",
    "arthritis": "C0003864", "cancer": "C0006826",
    "stroke": "C0038454", "heart attack": "C0027051",
    "myocardial infarction": "C0027051",
    "kidney disease": "C0022658", "liver disease": "C0023890",
    "anemia": "C0002871", "thyroid": "C0040136",
    "epilepsy": "C0014544", "migraine": "C0149931",
    "covid": "C5203670", "influenza": "C0021400",
    "hiv": "C0019693", "hepatitis": "C0019158",
    "obesity": "C0028754", "copd": "C0024117",

    # Medications
    "metformin": "C0025598", "paracetamol": "C0000970",
    "ibuprofen": "C0020740", "aspirin": "C0004057",
    "amoxicillin": "C0002645", "azithromycin": "C0052796",
    "insulin": "C0021641", "atorvastatin": "C0286651",
    "omeprazole": "C0028978", "amlodipine": "C0051696",
    "losartan": "C0126174", "ciprofloxacin": "C0008809",
    "prednisone": "C0032952", "dexamethasone": "C0011777",
    "cetirizine": "C0055147", "montelukast": "C0298130",
    "pantoprazole": "C0081876", "ranitidine": "C0034614",

    # Body parts / systems
    "blood pressure": "C0005823", "blood sugar": "C0005802",
    "heart": "C0018787", "lung": "C0024109", "liver": "C0023884",
    "kidney": "C0022646", "brain": "C0006104", "stomach": "C0038351",
    "intestine": "C0021853", "bone": "C0262950", "muscle": "C0026845",
    "skin": "C0037267", "eye": "C0015392", "ear": "C0013443",
    "throat": "C0031354", "nose": "C0028429",

    # Procedures
    "surgery": "C0543467", "biopsy": "C0005558",
    "ct scan": "C0040405", "mri": "C0024485", "x-ray": "C0043299",
    "ultrasound": "C0041618", "ecg": "C0013798",
    "blood test": "C0005841", "urine test": "C0042014",
    "vaccination": "C0042196", "injection": "C0021485",
    "dialysis": "C0011946", "chemotherapy": "C0013216",
    "radiation": "C0034618", "transplant": "C0040732",
}


def extract_umls_cuis(text: str) -> List[str]:
    """Extract UMLS CUIs from text using local dictionary matching.

    Uses longest-match-first strategy for multi-word terms.

    Args:
        text: Input text to scan for medical concepts.

    Returns:
        List of unique UMLS CUI strings found in the text.
    """
    text_lower = text.lower()
    found_cuis: List[str] = []

    # Sort terms by length (longest first) for greedy matching
    sorted_terms = sorted(UMLS_LOCAL_DICT.keys(), key=len, reverse=True)

    for term in sorted_terms:
        # Word boundary match
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, text_lower):
            cui = UMLS_LOCAL_DICT[term]
            if cui not in found_cuis:
                found_cuis.append(cui)

    return found_cuis


# ---------------------------------------------------------------------------
# Medical-Aware Semantic Chunking
# ---------------------------------------------------------------------------

# Regex patterns for medical section boundaries (should NOT be split across chunks)
_MEDICAL_BOUNDARY_PATTERNS = [
    re.compile(r"(?i)\b(symptoms?|signs?)\s*:", re.MULTILINE),
    re.compile(r"(?i)\b(dosage|dose|posology)\s*:", re.MULTILINE),
    re.compile(r"(?i)\b(side\s*effects?|adverse\s*effects?)\s*:", re.MULTILINE),
    re.compile(r"(?i)\b(contra\s*-?\s*indications?)\s*:", re.MULTILINE),
    re.compile(r"(?i)\b(diagnosis|treatment|prognosis)\s*:", re.MULTILINE),
    re.compile(r"(?i)\b(warnings?|precautions?)\s*:", re.MULTILINE),
    re.compile(r"(?i)\b(ingredients?|composition)\s*:", re.MULTILINE),
    re.compile(r"(?i)\b(indications?|uses?)\s*:", re.MULTILINE),
]


def _find_section_boundaries(text: str) -> List[int]:
    """Find character positions of medical section headers.

    Args:
        text: Full document text.

    Returns:
        Sorted list of character offsets where medical sections begin.
    """
    boundaries: List[int] = []
    for pat in _MEDICAL_BOUNDARY_PATTERNS:
        for m in pat.finditer(text):
            boundaries.append(m.start())
    return sorted(set(boundaries))


def _estimate_token_count(text: str) -> int:
    """Estimate token count using whitespace splitting (4 chars ≈ 1 token heuristic).

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    # Use word count as proxy; average English word ≈ 1.3 tokens
    words = text.split()
    return max(1, int(len(words) * 1.3))


def semantic_chunk(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 65,
    min_chunk_size: int = 100,
) -> List[str]:
    """Split text into semantically aware chunks respecting medical boundaries.

    Keeps medical definitions, symptom lists, and dosage instructions together.
    Avoids blind splitting in the middle of medical sections.

    Args:
        text: The full document text.
        chunk_size: Target chunk size in estimated tokens (300-500).
        chunk_overlap: Overlap between chunks in estimated tokens (50-80).
        min_chunk_size: Minimum acceptable chunk size.

    Returns:
        List of chunk text strings.
    """
    if not text or not text.strip():
        return []

    # Split into sentences first
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    # Find medical section boundaries
    section_starts = _find_section_boundaries(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0

    for sent in sentences:
        sent_size = _estimate_token_count(sent)

        # Check if this sentence starts a medical section
        sent_pos = text.find(sent)
        is_section_start = any(
            abs(sent_pos - boundary) < 10 for boundary in section_starts
        ) if sent_pos >= 0 else False

        # If adding this sentence would exceed chunk_size
        if current_size + sent_size > chunk_size and current_chunk:
            # But if this is NOT a section start, and current chunk is small, keep going
            if is_section_start or current_size >= min_chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Overlap: keep last few sentences
                overlap_sents: List[str] = []
                overlap_size = 0
                for prev_sent in reversed(current_chunk):
                    prev_size = _estimate_token_count(prev_sent)
                    if overlap_size + prev_size > chunk_overlap:
                        break
                    overlap_sents.insert(0, prev_sent)
                    overlap_size += prev_size

                current_chunk = overlap_sents
                current_size = overlap_size

        current_chunk.append(sent)
        current_size += sent_size

    # Flush remaining
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if chunks and _estimate_token_count(chunk_text) < min_chunk_size:
            # Merge with last chunk if too small
            chunks[-1] = chunks[-1] + " " + chunk_text
        else:
            chunks.append(chunk_text)

    return chunks


# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------

def _generate_doc_id(source: str, raw_id: Any) -> str:
    """Generate a stable document ID from source and raw identifier.

    Args:
        source: Data source name (e.g., 'PubMedQA').
        raw_id: Original record identifier.

    Returns:
        A deterministic hash-based document ID string.
    """
    raw = f"{source}:{raw_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def ingest_pubmed_qa(raw_path: Path) -> List[Dict[str, Any]]:
    """Ingest PubMedQA raw data into chunked, UMLS-enriched metadata.

    Args:
        raw_path: Path to pubmed_qa_raw.json.

    Returns:
        List of chunk metadata dictionaries conforming to the schema.
    """
    with open(raw_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    logger.info("Ingesting %d PubMedQA records from %s", len(data), raw_path)
    config = get_config()
    chunk_cfg = config["ingestion"]

    all_chunks: List[Dict[str, Any]] = []
    global_chunk_id = 0

    for item in data:
        raw_id = item.get("id", "unknown")
        doc_id = _generate_doc_id("PubMedQA", raw_id)
        title = item.get("question", "")[:120]
        full_text = f"Question: {item['question']}\n\nAnswer: {item['answer']}"

        text_chunks = semantic_chunk(
            full_text,
            chunk_size=chunk_cfg["chunk_size_tokens"],
            chunk_overlap=chunk_cfg["chunk_overlap_tokens"],
            min_chunk_size=chunk_cfg["min_chunk_size_tokens"],
        )

        for chunk_text in text_chunks:
            cuis = extract_umls_cuis(chunk_text)

            chunk_meta: Dict[str, Any] = {
                "document_id": doc_id,
                "source": "PubMedQA",
                "title": title,
                "chunk_id": global_chunk_id,
                "text": chunk_text,
                "umls_cuis": cuis,
            }
            all_chunks.append(chunk_meta)
            global_chunk_id += 1

    logger.info("Generated %d chunks from PubMedQA.", len(all_chunks))
    return all_chunks


def ingest_nhp_india(raw_path: Path) -> List[Dict[str, Any]]:
    """Ingest NHP (National Health Portal) India data.

    Expects a JSON array of {title, content, source} objects.

    Args:
        raw_path: Path to NHP India JSON data.

    Returns:
        List of chunk metadata dictionaries.
    """
    if not raw_path.exists():
        logger.warning("NHP India data not found at %s. Skipping.", raw_path)
        return []

    with open(raw_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    logger.info("Ingesting %d NHP India records.", len(data))
    config = get_config()
    chunk_cfg = config["ingestion"]

    all_chunks: List[Dict[str, Any]] = []
    global_chunk_id = 100000  # Offset to avoid ID collisions

    for idx, item in enumerate(data):
        doc_id = _generate_doc_id("NHP_India", idx)
        title = item.get("title", "")[:120]
        content = item.get("content", "")

        text_chunks = semantic_chunk(
            content,
            chunk_size=chunk_cfg["chunk_size_tokens"],
            chunk_overlap=chunk_cfg["chunk_overlap_tokens"],
            min_chunk_size=chunk_cfg["min_chunk_size_tokens"],
        )

        for chunk_text in text_chunks:
            cuis = extract_umls_cuis(chunk_text)
            chunk_meta: Dict[str, Any] = {
                "document_id": doc_id,
                "source": "NHP_India",
                "title": title,
                "chunk_id": global_chunk_id,
                "text": chunk_text,
                "umls_cuis": cuis,
            }
            all_chunks.append(chunk_meta)
            global_chunk_id += 1

    logger.info("Generated %d chunks from NHP India.", len(all_chunks))
    return all_chunks


def ingest_ayushman_bharat(raw_path: Path) -> List[Dict[str, Any]]:
    """Ingest Ayushman Bharat scheme data.

    Expects a JSON array of {title, content, source} objects.

    Args:
        raw_path: Path to Ayushman Bharat JSON data.

    Returns:
        List of chunk metadata dictionaries.
    """
    if not raw_path.exists():
        logger.warning("Ayushman Bharat data not found at %s. Skipping.", raw_path)
        return []

    with open(raw_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    logger.info("Ingesting %d Ayushman Bharat records.", len(data))
    config = get_config()
    chunk_cfg = config["ingestion"]

    all_chunks: List[Dict[str, Any]] = []
    global_chunk_id = 200000

    for idx, item in enumerate(data):
        doc_id = _generate_doc_id("AyushmanBharat", idx)
        title = item.get("title", "")[:120]
        content = item.get("content", "")

        text_chunks = semantic_chunk(
            content,
            chunk_size=chunk_cfg["chunk_size_tokens"],
            chunk_overlap=chunk_cfg["chunk_overlap_tokens"],
            min_chunk_size=chunk_cfg["min_chunk_size_tokens"],
        )

        for chunk_text in text_chunks:
            cuis = extract_umls_cuis(chunk_text)
            chunk_meta: Dict[str, Any] = {
                "document_id": doc_id,
                "source": "AyushmanBharat",
                "title": title,
                "chunk_id": global_chunk_id,
                "text": chunk_text,
                "umls_cuis": cuis,
            }
            all_chunks.append(chunk_meta)
            global_chunk_id += 1

    logger.info("Generated %d chunks from Ayushman Bharat.", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Full Ingestion Pipeline
# ---------------------------------------------------------------------------

def run_full_ingestion() -> List[Dict[str, Any]]:
    """Run the complete ingestion pipeline across all data sources.

    Returns:
        List of all chunk metadata dictionaries.
    """
    root = get_project_root()

    all_chunks: List[Dict[str, Any]] = []

    # PubMedQA (primary)
    pubmed_path = root / "data" / "raw" / "pubmed_qa_raw.json"
    if pubmed_path.exists():
        all_chunks.extend(ingest_pubmed_qa(pubmed_path))

    # NHP India (supplementary)
    nhp_path = root / "data" / "raw" / "nhp_india.json"
    all_chunks.extend(ingest_nhp_india(nhp_path))

    # Ayushman Bharat (supplementary)
    ayushman_path = root / "data" / "raw" / "ayushman_bharat.json"
    all_chunks.extend(ingest_ayushman_bharat(ayushman_path))

    logger.info("Total chunks from all sources: %d", len(all_chunks))

    # Save knowledge base
    kb_dir = root / "data" / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    kb_path = kb_dir / "chunks.json"

    with open(kb_path, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
    logger.info("Knowledge base saved to %s", kb_path)

    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for corpus ingestion."""
    parser = argparse.ArgumentParser(description="Ingest medical corpora and build knowledge base.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of knowledge base.")
    args = parser.parse_args()

    root = get_project_root()
    kb_path = root / "data" / "knowledge_base" / "chunks.json"

    if kb_path.exists() and not args.rebuild:
        logger.info("Knowledge base already exists at %s. Use --rebuild to force.", kb_path)
        return

    run_full_ingestion()


if __name__ == "__main__":
    main()
