"""
Phase 1: Controllable Code-Mixed (Manglish) Synthetic Data Generation.

Generates medical QA pairs with a mathematically enforced Code-Mixing Index (CMI).
CMI = (N - max(w_i)) / N  where N = total tokens, max(w_i) = tokens in dominant language.

Usage (on Ubuntu):
    python -m src.generate_synthetic_data --target-cmi 0.5 --limit 5000
"""

import os
import re
import gc
import json
import time
import argparse
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src import get_config, get_project_root, setup_logging

logger = setup_logging("generate_synthetic_data")

# ---------------------------------------------------------------------------
# CMI Calculation
# ---------------------------------------------------------------------------

def _is_devanagari_token(token: str) -> bool:
    """Check if a token contains Devanagari characters.

    Args:
        token: A single whitespace-delimited token.

    Returns:
        True if the token contains at least one Devanagari character.
    """
    return any("\u0900" <= ch <= "\u097F" for ch in token)


def _is_latin_token(token: str) -> bool:
    """Check if a token contains Latin alphabetic characters.

    Args:
        token: A single whitespace-delimited token.

    Returns:
        True if the token contains at least one Latin letter.
    """
    return any("a" <= ch.lower() <= "z" for ch in token)


def calculate_cmi(text: str) -> float:
    """Calculate the Code-Mixing Index of a text.

    CMI = (N - max(w_i)) / N
    Where N = total language-bearing tokens and max(w_i) = count of dominant language tokens.
    CMI = 0.0 means monolingual, CMI = 0.5 means perfectly balanced mixing.

    Args:
        text: The code-mixed text to evaluate.

    Returns:
        A float in [0.0, 1.0] representing the CMI score.
    """
    tokens = text.split()
    if not tokens:
        return 0.0

    dev_count = 0
    lat_count = 0
    for tok in tokens:
        clean = re.sub(r"[^\w]", "", tok)
        if not clean:
            continue
        if _is_devanagari_token(clean):
            dev_count += 1
        elif _is_latin_token(clean):
            lat_count += 1

    n_lang = dev_count + lat_count
    if n_lang == 0:
        return 0.0

    dominant = max(dev_count, lat_count)
    return (n_lang - dominant) / n_lang


# ---------------------------------------------------------------------------
# Prompt Templates by CMI Range
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: Dict[str, str] = {
    "light": (
        "You are a bilingual Marathi-English speaker visiting a doctor in Pune.\n"
        "Rewrite the following medical Q&A into LIGHTLY code-mixed Marathi-English "
        "(~70-80%% English, ~20-30%% Marathi in Devanagari script).\n"
        "Keep most sentences in English but sprinkle in natural Marathi connectors, "
        "pronouns, and particles (e.g., 'mala', 'ahe', 'ka', 'tar').\n\n"
        "RULES:\n"
        "1. Preserve ALL medical terms in English.\n"
        "2. Marathi portions MUST use Devanagari script.\n"
        "3. Reflect natural Mumbai/Pune conversational style.\n\n"
        "Input Question: {question}\n"
        "Input Answer: {answer}\n\n"
        "Output ONLY valid JSON:\n"
        '{{"code_mixed_question": "...", "code_mixed_answer": "..."}}'
    ),
    "medium": (
        "You are a bilingual Marathi-English speaker visiting a doctor in Mumbai.\n"
        "Rewrite the following medical Q&A into MODERATELY code-mixed Marathi-English "
        "(~50%% English, ~50%% Marathi in Devanagari script).\n"
        "Alternate freely between Marathi and English within sentences.\n\n"
        "RULES:\n"
        "1. Preserve ALL medical terms in English.\n"
        "2. Conversational grammar should be Marathi (Devanagari).\n"
        "3. Reflect natural Mumbai/Pune conversational style.\n\n"
        "Input Question: {question}\n"
        "Input Answer: {answer}\n\n"
        "Output ONLY valid JSON:\n"
        '{{"code_mixed_question": "...", "code_mixed_answer": "..."}}'
    ),
    "heavy": (
        "You are a Marathi-dominant bilingual visiting a doctor in a rural Maharashtra clinic.\n"
        "Rewrite the following medical Q&A into HEAVILY code-mixed Marathi-English "
        "(~70-80%% Marathi in Devanagari script, ~20-30%% English).\n"
        "Use Marathi as the primary language. Only keep specialized medical terms in English.\n\n"
        "RULES:\n"
        "1. Preserve ONLY technical medical terms in English.\n"
        "2. All conversational parts MUST be Marathi (Devanagari).\n"
        "3. Structure reflects natural rural Maharashtra speech patterns.\n\n"
        "Input Question: {question}\n"
        "Input Answer: {answer}\n\n"
        "Output ONLY valid JSON:\n"
        '{{"code_mixed_question": "...", "code_mixed_answer": "..."}}'
    ),
}


def _select_prompt_tier(target_cmi: float) -> str:
    """Select the appropriate prompt template tier based on target CMI.

    Args:
        target_cmi: Target Code-Mixing Index (0.0–1.0).

    Returns:
        Key into PROMPT_TEMPLATES: 'light', 'medium', or 'heavy'.
    """
    if target_cmi < 0.35:
        return "light"
    elif target_cmi < 0.6:
        return "medium"
    else:
        return "heavy"


# ---------------------------------------------------------------------------
# API Client Wrapper
# ---------------------------------------------------------------------------

class _GroqClient:
    """Thin wrapper around the Groq API with key hot-swapping support."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the Groq client.

        Args:
            api_key: The Groq API key.
            model: Model identifier (e.g., 'llama-3.1-8b-instant').
        """
        from groq import Groq  # Lazy import

        self.model = model
        self._key = api_key
        self._client = Groq(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Call the Groq chat completion API.

        Args:
            prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            Raw text response from the model.

        Raises:
            Exception: On API errors (rate limit, auth, etc.).
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You output valid JSON only. No markdown, no code blocks."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()

    def swap_key(self, new_key: str) -> None:
        """Hot-swap the API key.

        Args:
            new_key: New Groq API key string.
        """
        from groq import Groq

        self._key = new_key
        self._client = Groq(api_key=new_key)
        logger.info("API key swapped: ...%s", new_key[-6:])


# ---------------------------------------------------------------------------
# Save / Resume Helpers
# ---------------------------------------------------------------------------

def _save_dataset(output_path: Path, dataset: List[Dict[str, Any]]) -> None:
    """Persist the generated dataset to disk.

    Args:
        output_path: Path to the output JSON file.
        dataset: List of generated QA dictionaries.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)


def _load_existing(output_path: Path) -> Tuple[List[Dict[str, Any]], set]:
    """Load an existing output file for resume support.

    Args:
        output_path: Path to the output JSON file.

    Returns:
        Tuple of (existing dataset list, set of already-processed original IDs).
    """
    if not output_path.exists():
        return [], set()
    try:
        with open(output_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        ids = {item["original_id"] for item in data}
        logger.info("Resuming from %d saved samples.", len(data))
        return data, ids
    except Exception as exc:
        logger.warning("Could not load existing output (%s). Starting fresh.", exc)
        return [], set()


# ---------------------------------------------------------------------------
# Core Generation Loop
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    input_file: Path,
    output_file: Path,
    target_cmi: float = 0.5,
    limit: int = 5000,
) -> None:
    """Generate code-mixed medical QA data with controllable CMI.

    This function reads raw English medical QA pairs, calls the Groq API
    to generate code-mixed (Manglish) versions, and filters outputs to
    enforce the target CMI threshold.

    Args:
        input_file: Path to the raw source JSON (list of {id, question, answer}).
        output_file: Path for the generated output JSON.
        target_cmi: Target Code-Mixing Index (0.0–1.0).
        limit: Maximum number of samples to generate.
    """
    config = get_config()
    gen_cfg = config["data_generation"]

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Export it or add to .env file."
        )

    client = _GroqClient(api_key, gen_cfg["generation_model"])

    with open(input_file, "r", encoding="utf-8") as fh:
        raw_data: List[Dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d raw records from %s", len(raw_data), input_file)

    # Resume support
    dataset, existing_ids = _load_existing(Path(output_file))

    remaining = limit - len(dataset)
    tier = _select_prompt_tier(target_cmi)
    template = PROMPT_TEMPLATES[tier]

    logger.info(
        "Target: %d | Done: %d | Remaining: %d | CMI tier: %s (target=%.2f)",
        limit, len(dataset), remaining, tier, target_cmi,
    )

    consecutive_errors = 0
    cmi_retries = 0
    max_cmi_retries = 2  # Re-generate if CMI is out of range

    # CMI acceptance window
    cmi_low = max(0.0, target_cmi - 0.2)
    cmi_high = min(1.0, target_cmi + 0.2)

    for item in raw_data:
        if len(dataset) >= limit:
            logger.info("Reached target of %d samples.", limit)
            break

        if consecutive_errors >= gen_cfg["max_consecutive_errors"]:
            logger.error("Too many consecutive errors (%d). Stopping.", consecutive_errors)
            break

        item_id = item.get("id", item.get("original_id", "unknown"))
        if item_id in existing_ids:
            continue

        prompt = template.format(question=item["question"], answer=item["answer"])

        for attempt in range(5):
            try:
                raw_text = client.generate(
                    prompt,
                    temperature=gen_cfg["generation_temperature"],
                    max_tokens=gen_cfg["max_generation_tokens"],
                )

                # Clean markdown wrapping
                for prefix in ("```json", "```"):
                    if raw_text.startswith(prefix):
                        raw_text = raw_text[len(prefix):]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

                res = json.loads(raw_text)
                cm_q = res.get("code_mixed_question", "")
                cm_a = res.get("code_mixed_answer", "")

                # Validate CMI
                q_cmi = calculate_cmi(cm_q)
                a_cmi = calculate_cmi(cm_a)
                avg_cmi = (q_cmi + a_cmi) / 2.0

                if not (cmi_low <= avg_cmi <= cmi_high) and cmi_retries < max_cmi_retries:
                    cmi_retries += 1
                    logger.debug(
                        "CMI %.2f outside [%.2f, %.2f] for item %s — retrying (%d/%d)",
                        avg_cmi, cmi_low, cmi_high, item_id, cmi_retries, max_cmi_retries,
                    )
                    continue

                cmi_retries = 0

                record: Dict[str, Any] = {
                    "original_id": item_id,
                    "original_question": item["question"],
                    "original_answer": item["answer"],
                    "code_mixed_question": cm_q,
                    "code_mixed_answer": cm_a,
                    "cmi_question": round(q_cmi, 4),
                    "cmi_answer": round(a_cmi, 4),
                    "cmi_avg": round(avg_cmi, 4),
                    "cmi_tier": tier,
                    "source": item.get("source", "PubMedQA"),
                }
                dataset.append(record)
                existing_ids.add(item_id)
                consecutive_errors = 0

                if len(dataset) % gen_cfg["save_every_n"] == 0:
                    _save_dataset(Path(output_file), dataset)
                    logger.info("Progress: %d / %d", len(dataset), limit)

                time.sleep(gen_cfg["request_delay_seconds"])
                break  # Success

            except json.JSONDecodeError as exc:
                logger.warning("JSON parse error for item %s: %s", item_id, exc)
                consecutive_errors += 1
                time.sleep(3)
                break

            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate_limit" in err.lower():
                    logger.warning("Rate limited. Waiting 65s...")
                    time.sleep(65)
                elif "401" in err or "invalid_api_key" in err.lower():
                    logger.error("Auth error. Check GROQ_API_KEY.")
                    consecutive_errors = gen_cfg["max_consecutive_errors"]
                    break
                else:
                    logger.warning("API error for item %s: %s", item_id, err[:200])
                    consecutive_errors += 1
                    time.sleep(5)
                    break

    _save_dataset(Path(output_file), dataset)
    logger.info("Done! %d total samples saved to %s", len(dataset), output_file)


# ---------------------------------------------------------------------------
# Split utility
# ---------------------------------------------------------------------------

def create_splits(input_file: Path, output_dir: Path) -> None:
    """Split the generated dataset into train/val/test JSONL files.

    Split ratio: 70% train, 15% val, 15% test.

    Args:
        input_file: Path to the full generated JSON dataset.
        output_dir: Directory to write train.jsonl, val.jsonl, test.jsonl.
    """
    from sklearn.model_selection import train_test_split

    with open(input_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    logger.info("Splitting %d samples (70/15/15).", len(data))

    train, temp = train_test_split(data, test_size=0.30, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for entry in split_data:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("%s: %d samples -> %s", split_name.capitalize(), len(split_data), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate code-mixed medical QA data with controllable CMI."
    )
    parser.add_argument(
        "--target-cmi", type=float, default=0.5,
        help="Target Code-Mixing Index [0.0-1.0]. 0.3=light, 0.5=medium, 0.7=heavy."
    )
    parser.add_argument("--limit", type=int, default=5000, help="Max samples to generate.")
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to raw JSON source. Defaults to data/raw/pubmed_qa_raw.json."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for output JSON. Defaults to data/processed/synthetic_codemix_qa.json."
    )
    parser.add_argument("--split", action="store_true", help="Also create train/val/test splits.")

    args = parser.parse_args()
    root = get_project_root()

    input_file = Path(args.input) if args.input else root / "data" / "raw" / "pubmed_qa_raw.json"
    output_file = Path(args.output) if args.output else root / "data" / "processed" / "synthetic_codemix_qa.json"

    generate_synthetic_data(input_file, output_file, args.target_cmi, args.limit)

    if args.split:
        create_splits(output_file, root / "data" / "splits")


if __name__ == "__main__":
    main()
