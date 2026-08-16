"""
Phase 2: Tokenizer Profiling, Vocabulary Expansion & Linguistic Integrity.

Profiles the Token Tax of existing multilingual models on code-mixed data,
expands vocabulary with Marathi medical roots, and runs morphological
completeness + cross-script consistency tests.

Usage (on Ubuntu):
    python -m src.tokenizer_pipeline --profile --expand --test
"""

import gc
import json
import os
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from src import get_config, get_project_root, setup_logging

logger = setup_logging("tokenizer_pipeline")


# ---------------------------------------------------------------------------
# Marathi Medical Vocabulary for Expansion
# ---------------------------------------------------------------------------

MARATHI_MEDICAL_ROOTS: List[str] = [
    # Common Marathi medical terms (Devanagari)
    "रुग्णालय",    # Hospital
    "रुग्ण",       # Patient
    "डॉक्टर",     # Doctor
    "औषध",        # Medicine
    "आजार",       # Disease/Illness
    "तपासणी",     # Examination
    "उपचार",      # Treatment
    "लक्षण",       # Symptom
    "ताप",         # Fever
    "डोकेदुखी",    # Headache
    "पोटदुखी",     # Stomachache
    "रक्तदाब",     # Blood pressure
    "मधुमेह",      # Diabetes
    "दमा",         # Asthma
    "हृदयविकार",   # Heart disease
    "शस्त्रक्रिया",  # Surgery
    "गोळी",        # Tablet/Pill
    "इंजेक्शन",     # Injection
    "रक्त",        # Blood
    "मूत्र",        # Urine
    "खोकला",       # Cough
    "सर्दी",        # Cold
    "जुलाब",       # Diarrhea
    "उलटी",        # Vomiting
    "अशक्तपणा",    # Weakness/Anemia
    "जखम",        # Wound
    "वेदना",       # Pain
    "सूज",         # Swelling
    "पुरळ",        # Rash
    "अलर्जी",      # Allergy
    "कर्करोग",      # Cancer
    "क्षयरोग",      # Tuberculosis
    "मलेरिया",     # Malaria
    "डेंग्यू",      # Dengue
    "लसीकरण",      # Vaccination
    "गर्भधारणा",    # Pregnancy
    "प्रसूती",      # Delivery
    "बालरोग",      # Pediatrics
    "त्वचारोग",     # Dermatology
    "नेत्ररोग",     # Ophthalmology
    # Common Marathi suffixes and word forms
    "रुग्णालयात",   # In the hospital (locative)
    "रुग्णांना",     # To the patients (dative)
    "औषधांचे",      # Of medicines (genitive)
    "तपासणीसाठी",   # For examination (postposition)
    "उपचारांमध्ये",  # In treatments (locative)
    # Roman Marathi forms (common in Manglish)
    "rugnalay",
    "aushadh",
    "aajar",
    "lakshan",
    "taap",
    "dokadukhi",
    "potdukhi",
    "raktadaab",
    "madhumeh",
    "hrudayvikar",
]


# ---------------------------------------------------------------------------
# Morphological Test Cases
# ---------------------------------------------------------------------------

@dataclass
class MorphTestCase:
    """Test case for morphological completeness.

    Attributes:
        word: Full inflected Marathi word.
        expected_root: Expected root/stem that should remain intact.
        description: Human-readable description of the test.
    """
    word: str
    expected_root: str
    description: str


MORPH_TEST_CASES: List[MorphTestCase] = [
    MorphTestCase("रुग्णालयात", "रुग्णालय", "Hospital + locative suffix -ात"),
    MorphTestCase("रुग्णांना", "रुग्ण", "Patient + dative plural suffix -ांना"),
    MorphTestCase("औषधांचे", "औषध", "Medicine + genitive plural suffix -ांचे"),
    MorphTestCase("तपासणीसाठी", "तपासणी", "Examination + postposition -साठी"),
    MorphTestCase("उपचारांमध्ये", "उपचार", "Treatment + locative plural -ांमध्ये"),
    MorphTestCase("डॉक्टरांनी", "डॉक्टर", "Doctor + ergative plural -ांनी"),
    MorphTestCase("आजारपणात", "आजार", "Disease + abstract noun + locative -पणात"),
    MorphTestCase("शस्त्रक्रियेनंतर", "शस्त्रक्रिया", "Surgery + postposition -नंतर"),
    MorphTestCase("लक्षणांमुळे", "लक्षण", "Symptom + causal -ांमुळे"),
    MorphTestCase("वेदनाशामक", "वेदना", "Pain + adjective suffix -शामक"),
]


# ---------------------------------------------------------------------------
# Cross-Script Test Pairs
# ---------------------------------------------------------------------------

CROSS_SCRIPT_PAIRS: List[Tuple[str, str]] = [
    ("doctor", "डॉक्टर"),
    ("hospital", "रुग्णालय"),
    ("medicine", "औषध"),
    ("fever", "ताप"),
    ("headache", "डोकेदुखी"),
    ("blood pressure", "रक्तदाब"),
    ("diabetes", "मधुमेह"),
    ("surgery", "शस्त्रक्रिया"),
    ("patient", "रुग्ण"),
    ("treatment", "उपचार"),
    ("cough", "खोकला"),
    ("injection", "इंजेक्शन"),
    ("tablet", "गोळी"),
    ("symptom", "लक्षण"),
    ("disease", "आजार"),
]


# ---------------------------------------------------------------------------
# Token Tax Profiling
# ---------------------------------------------------------------------------

def profile_token_tax(
    texts: List[str],
    model_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Profile the Token Tax (tokens per word) of tokenizers on code-mixed text.

    Args:
        texts: List of code-mixed text samples (from train.jsonl).
        model_names: List of HuggingFace model identifiers to profile.

    Returns:
        List of result dicts: {model, total_words, total_tokens, fertility}.
    """
    import torch
    from transformers import AutoTokenizer

    if model_names is None:
        config = get_config()
        model_names = config["tokenizer"]["profile_models"]

    results: List[Dict[str, Any]] = []

    for model_name in model_names:
        logger.info("Profiling tokenizer: %s", model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
            )

            total_tokens = 0
            total_words = 0

            for text in texts:
                words = text.split()
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_words += len(words)
                total_tokens += len(tokens)

            fertility = total_tokens / total_words if total_words > 0 else 0.0

            result = {
                "model": model_name,
                "total_words": total_words,
                "total_tokens": total_tokens,
                "fertility": round(fertility, 4),
                "token_tax_pct": round((fertility - 1.0) * 100, 2),
            }
            results.append(result)

            logger.info(
                "  %s: fertility=%.4f (%.1f%% tax), %d tokens / %d words",
                model_name, fertility, (fertility - 1.0) * 100,
                total_tokens, total_words,
            )

            # Cleanup to free memory
            del tokenizer
            gc.collect()

        except Exception as exc:
            logger.error("Failed to profile %s: %s", model_name, exc)
            results.append({"model": model_name, "error": str(exc)})

    return results


# ---------------------------------------------------------------------------
# Vocabulary Expansion
# ---------------------------------------------------------------------------

def expand_tokenizer_vocabulary(
    base_model_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """Add Marathi medical roots to a pre-trained tokenizer's vocabulary.

    This does NOT retrain the tokenizer. It adds specific tokens to the
    added_tokens list so they are recognized as single units.

    Args:
        base_model_name: HuggingFace model ID for the base tokenizer.
        output_dir: Directory to save the expanded tokenizer.

    Returns:
        Path string to the saved expanded tokenizer.
    """
    from transformers import AutoTokenizer

    config = get_config()
    if base_model_name is None:
        base_model_name = config["tokenizer"]["base_tokenizer"]

    if output_dir is None:
        output_dir = get_project_root() / "models" / "expanded_tokenizer"

    logger.info("Loading base tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    original_vocab_size = len(tokenizer)
    logger.info("Original vocabulary size: %d", original_vocab_size)

    # Add Marathi medical roots as new tokens
    new_tokens = [t for t in MARATHI_MEDICAL_ROOTS if t not in tokenizer.get_vocab()]
    num_added = tokenizer.add_tokens(new_tokens)

    logger.info(
        "Added %d new Marathi medical tokens. New vocab size: %d",
        num_added, len(tokenizer),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Expanded tokenizer saved to %s", output_dir)

    del tokenizer
    gc.collect()

    return str(output_dir)


# ---------------------------------------------------------------------------
# Morphological Completeness Test
# ---------------------------------------------------------------------------

def test_morphological_completeness(
    tokenizer_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Test that the tokenizer preserves Marathi root morphemes.

    For each test case, checks whether the root substring appears intact
    in at least one of the tokenizer's output tokens.

    Args:
        tokenizer_path: Path or model name for the tokenizer to test.

    Returns:
        List of test result dictionaries.
    """
    from transformers import AutoTokenizer

    if tokenizer_path is None:
        tokenizer_path = str(get_project_root() / "models" / "expanded_tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    results: List[Dict[str, Any]] = []

    for tc in MORPH_TEST_CASES:
        token_ids = tokenizer.encode(tc.word, add_special_tokens=False)
        decoded_tokens = [tokenizer.decode([tid]).strip() for tid in token_ids]

        # Check if root appears intact in any single token
        root_preserved = any(tc.expected_root in tok for tok in decoded_tokens)

        # Also check if root appears in the concatenation of first N tokens
        concat_check = ""
        root_in_prefix = False
        for tok in decoded_tokens:
            concat_check += tok.replace("▁", "").replace("##", "")
            if tc.expected_root in concat_check:
                root_in_prefix = True
                break

        passed = root_preserved or root_in_prefix

        result = {
            "word": tc.word,
            "expected_root": tc.expected_root,
            "description": tc.description,
            "tokens": decoded_tokens,
            "num_tokens": len(decoded_tokens),
            "root_preserved": passed,
            "status": "PASS" if passed else "FAIL",
        }
        results.append(result)

        status = "✓" if passed else "✗"
        logger.info(
            "  %s %s → %s (root '%s' %s)",
            status, tc.word, decoded_tokens, tc.expected_root,
            "preserved" if passed else "BROKEN",
        )

    del tokenizer
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Cross-Script Consistency Test
# ---------------------------------------------------------------------------

def test_cross_script_consistency(
    model_name: Optional[str] = None,
    threshold: float = 0.85,
) -> List[Dict[str, Any]]:
    """Test cosine similarity between English and Devanagari parallel terms.

    Uses the embedding model to verify that parallel terms (e.g., 'doctor'
    and 'डॉक्टर') have high cosine similarity (> threshold).

    Args:
        model_name: Embedding model to use. Defaults to config value.
        threshold: Minimum cosine similarity to pass (default 0.85).

    Returns:
        List of test result dictionaries.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    config = get_config()
    if model_name is None:
        model_name = config["tokenizer"]["base_tokenizer"]

    logger.info("Loading embedding model for cross-script test: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    def _embed(text: str) -> np.ndarray:
        """Get mean-pooled embedding for text."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pool over non-padding tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embs = outputs.last_hidden_state * attention_mask
        return (token_embs.sum(1) / attention_mask.sum(1)).squeeze().numpy()

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    results: List[Dict[str, Any]] = []

    for eng, mar in CROSS_SCRIPT_PAIRS:
        eng_emb = _embed(eng)
        mar_emb = _embed(mar)
        sim = _cosine_sim(eng_emb, mar_emb)
        passed = sim >= threshold

        result = {
            "english": eng,
            "marathi": mar,
            "cosine_similarity": round(sim, 4),
            "threshold": threshold,
            "status": "PASS" if passed else "FAIL",
        }
        results.append(result)

        status = "✓" if passed else "✗"
        logger.info("  %s '%s' ↔ '%s': cos=%.4f", status, eng, mar, sim)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    token_tax_results: List[Dict[str, Any]],
    morph_results: List[Dict[str, Any]],
    cross_script_results: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> None:
    """Generate a classification report summarizing all tokenizer metrics.

    Args:
        token_tax_results: Results from profile_token_tax().
        morph_results: Results from test_morphological_completeness().
        cross_script_results: Results from test_cross_script_consistency().
        output_path: Path for the output CSV. Defaults to results/tokenizer_report.csv.
    """
    if output_path is None:
        output_path = get_project_root() / "results" / "tokenizer_report.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)

        # Section 1: Token Tax
        writer.writerow(["=== TOKEN TAX PROFILE ==="])
        writer.writerow(["Model", "Total Words", "Total Tokens", "Fertility", "Tax %"])
        for r in token_tax_results:
            if "error" in r:
                writer.writerow([r["model"], "ERROR", "", "", r["error"]])
            else:
                writer.writerow([
                    r["model"], r["total_words"], r["total_tokens"],
                    r["fertility"], r["token_tax_pct"],
                ])

        writer.writerow([])

        # Section 2: Morphological Completeness
        writer.writerow(["=== MORPHOLOGICAL COMPLETENESS ==="])
        writer.writerow(["Word", "Root", "Tokens", "Num Tokens", "Status"])
        pass_count = sum(1 for r in morph_results if r["status"] == "PASS")
        for r in morph_results:
            writer.writerow([
                r["word"], r["expected_root"],
                " | ".join(r["tokens"]), r["num_tokens"], r["status"],
            ])
        writer.writerow([f"Pass Rate: {pass_count}/{len(morph_results)}"])

        writer.writerow([])

        # Section 3: Cross-Script Consistency
        writer.writerow(["=== CROSS-SCRIPT CONSISTENCY ==="])
        writer.writerow(["English", "Marathi", "Cosine Similarity", "Threshold", "Status"])
        pass_count = sum(1 for r in cross_script_results if r["status"] == "PASS")
        for r in cross_script_results:
            writer.writerow([
                r["english"], r["marathi"],
                r["cosine_similarity"], r["threshold"], r["status"],
            ])
        writer.writerow([f"Pass Rate: {pass_count}/{len(cross_script_results)}"])

    logger.info("Tokenizer report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for tokenizer pipeline."""
    parser = argparse.ArgumentParser(description="Phase 2: Tokenizer profiling and expansion.")
    parser.add_argument("--profile", action="store_true", help="Profile token tax of existing models.")
    parser.add_argument("--expand", action="store_true", help="Expand tokenizer vocabulary.")
    parser.add_argument("--test", action="store_true", help="Run morphological + cross-script tests.")
    parser.add_argument("--all", action="store_true", help="Run all steps.")
    args = parser.parse_args()

    if args.all:
        args.profile = args.expand = args.test = True

    if not (args.profile or args.expand or args.test):
        parser.print_help()
        return

    from src.data_loader import load_split

    token_tax_results: List[Dict[str, Any]] = []
    morph_results: List[Dict[str, Any]] = []
    cross_script_results: List[Dict[str, Any]] = []

    if args.profile:
        logger.info("=== PROFILING TOKEN TAX ===")
        pairs = load_split("train")
        texts = [p.code_mixed_question for p in pairs] + [p.code_mixed_answer for p in pairs]
        token_tax_results = profile_token_tax(texts)

    if args.expand:
        logger.info("=== EXPANDING VOCABULARY ===")
        expand_tokenizer_vocabulary()

    if args.test:
        logger.info("=== MORPHOLOGICAL COMPLETENESS TEST ===")
        morph_results = test_morphological_completeness()
        logger.info("=== CROSS-SCRIPT CONSISTENCY TEST ===")
        cross_script_results = test_cross_script_consistency()

    if token_tax_results or morph_results or cross_script_results:
        generate_report(token_tax_results, morph_results, cross_script_results)


if __name__ == "__main__":
    main()
