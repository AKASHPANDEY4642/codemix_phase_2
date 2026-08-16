"""
Phase 3: Script-Aware Query Processor (Core Novelty).

Standardizes raw Manglish user queries into a UnifiedQuery object through:
    Step A: Unicode NFC normalization + script detection
    Step B: Phonetic dictionary mapping (Roman Marathi → Devanagari)
    Step C: Medical term masking (Aho-Corasick with UMLS)
    Step D: Transliteration (non-masked Latin → Devanagari)
    Step E: Semantic translation (code-mixed → pure English)

Usage:
    from src.query_processor import QueryProcessor
    processor = QueryProcessor()
    result = processor.process("mala stomach pain hotoy ani fever pan ahe")
"""

import gc
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src import get_config, get_project_root, setup_logging
from src.phonetic_normalizer import PhoneticNormalizer
from src.medical_masking import MedicalMasker

logger = setup_logging("query_processor")


# ---------------------------------------------------------------------------
# Output Dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnifiedQuery:
    """The multi-representation output of the query processing pipeline.

    Attributes:
        original_manglish: The raw user input exactly as typed.
        transliterated_marathi: Romanized Marathi portions converted to Devanagari.
        semantic_english: Full semantic translation into pure English.
        extracted_medical_terms: List of extracted medical term dictionaries.
        script_type: Detected script type ('devanagari', 'roman', 'mixed').
        normalized_text: Text after NFC normalization and phonetic mapping.
    """
    original_manglish: str
    transliterated_marathi: str
    semantic_english: str
    extracted_medical_terms: List[Dict[str, str]] = field(default_factory=list)
    script_type: str = "mixed"
    normalized_text: str = ""


# ---------------------------------------------------------------------------
# Script Detection
# ---------------------------------------------------------------------------

def detect_script(text: str) -> str:
    """Classify the script of the text.

    Args:
        text: Input text to classify.

    Returns:
        One of 'devanagari', 'roman', or 'mixed'.
    """
    has_devanagari = False
    has_latin = False

    for ch in text:
        if "\u0900" <= ch <= "\u097F":
            has_devanagari = True
        elif "a" <= ch.lower() <= "z":
            has_latin = True
        if has_devanagari and has_latin:
            return "mixed"

    if has_devanagari:
        return "devanagari"
    return "roman"


def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization and handle Devanagari phonetics.

    Handles anusvara/chandrabindu normalization and nukta standardization.

    Args:
        text: Raw input text.

    Returns:
        NFC-normalized text with standardized whitespace.
    """
    # Step 1: NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Standardize Devanagari variations
    # Normalize chandrabindu + vowel combinations
    # ॅ (candra e) → े (standard e matra) — common in Marathi
    # This is acceptable for search normalization
    text = text.replace("\u0945", "\u0947")  # ॅ → े

    # Step 3: Normalize nukta characters
    # क़ → क (remove nukta for normalized search)
    nukta = "\u093C"  # ़
    # Keep nukta for now — it's important for accuracy
    # But normalize doubled nuktas
    text = re.sub(nukta + "+", nukta, text)

    # Step 4: Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Transliteration
# ---------------------------------------------------------------------------

def _transliterate_to_devanagari(text: str, masked_spans: List[Tuple[int, int]]) -> str:
    """Transliterate Roman text to Devanagari, skipping masked medical terms.

    Uses indic_transliteration library (ITRANS scheme).

    Args:
        text: Input text (may contain <MED>...</MED> tags).
        masked_spans: Not used directly — tags in text handle masking.

    Returns:
        Text with non-masked Roman portions transliterated to Devanagari.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        logger.warning("indic_transliteration not installed. Skipping transliteration.")
        return text

    # Split text around <MED>...</MED> tags
    parts = re.split(r"(<MED>.*?</MED>)", text)
    result_parts: List[str] = []

    for part in parts:
        if part.startswith("<MED>") and part.endswith("</MED>"):
            # Keep medical terms as-is (still tagged)
            result_parts.append(part)
        else:
            # Transliterate non-medical portions word by word
            tokens = part.split()
            transliterated_tokens: List[str] = []

            for token in tokens:
                # Skip if already Devanagari
                if any("\u0900" <= ch <= "\u097F" for ch in token):
                    transliterated_tokens.append(token)
                    continue

                # Skip if purely numeric or punctuation
                clean = re.sub(r"[^\w]", "", token)
                if not clean or clean.isdigit():
                    transliterated_tokens.append(token)
                    continue

                # Transliterate Roman → Devanagari
                try:
                    dev = transliterate(token.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
                    transliterated_tokens.append(dev)
                except Exception:
                    transliterated_tokens.append(token)

            result_parts.append(" ".join(transliterated_tokens))

    return " ".join(result_parts)


# ---------------------------------------------------------------------------
# Semantic Translation (Step E)
# ---------------------------------------------------------------------------

def _translate_to_english(text: str, backend: str = "groq_api") -> str:
    """Translate code-mixed or Devanagari text to pure English.

    Args:
        text: Input text (code-mixed or Devanagari).
        backend: Translation backend ('groq_api' or 'indictrans2').

    Returns:
        Pure English translation of the text.
    """
    config = get_config()
    qp_cfg = config["query_processor"]

    # Remove any remaining MED tags for translation
    clean_text = re.sub(r"<MED>(.*?)</MED>", r"\1", text)

    if backend == "groq_api":
        return _translate_via_groq(clean_text, qp_cfg.get("translation_model", "llama-3.1-8b-instant"))
    elif backend == "indictrans2":
        return _translate_via_indictrans2(clean_text)
    else:
        logger.warning("Unknown translation backend: %s. Returning original.", backend)
        return clean_text


def _translate_via_groq(text: str, model: str) -> str:
    """Translate text to English using Groq API.

    Args:
        text: Input text to translate.
        model: Groq model to use.

    Returns:
        English translation.
    """
    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not set. Returning original text.")
            return text

        client = Groq(api_key=api_key)
        prompt = (
            "Translate the following Marathi-English code-mixed medical text "
            "into clear, pure English. Preserve all medical terminology exactly. "
            "Output ONLY the English translation, nothing else.\n\n"
            f"Text: {text}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical translator specializing in Marathi-English translation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.warning("Groq translation failed: %s. Returning original.", exc)
        return text


def _translate_via_indictrans2(text: str) -> str:
    """Translate text using IndicTrans2 (local model, requires GPU).

    Args:
        text: Input text to translate.

    Returns:
        English translation.
    """
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "ai4bharat/indictrans2-indic-en-1B"
        logger.info("Loading IndicTrans2 model: %s", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

        if torch.cuda.is_available():
            model = model.cuda()

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)

        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cleanup
        model.cpu()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return translation

    except Exception as exc:
        logger.warning("IndicTrans2 translation failed: %s. Returning original.", exc)
        return text


# ---------------------------------------------------------------------------
# Main Query Processor
# ---------------------------------------------------------------------------

class QueryProcessor:
    """The core Script-Aware Query Processing Pipeline.

    Transforms raw Manglish queries into a UnifiedQuery with multiple
    representations for optimal retrieval.
    """

    def __init__(self) -> None:
        """Initialize the query processor with all sub-modules."""
        self._phonetic = PhoneticNormalizer()
        self._masker = MedicalMasker()
        config = get_config()
        self._translation_backend = config["query_processor"]["translation_backend"]
        logger.info("QueryProcessor initialized (translation=%s).", self._translation_backend)

    def process(self, raw_query: str) -> UnifiedQuery:
        """Process a raw Manglish query through the full pipeline.

        Steps:
            A. Unicode NFC normalization + script detection
            B. Phonetic dictionary mapping
            C. Medical term masking (Aho-Corasick)
            D. Transliteration (Roman → Devanagari)
            E. Semantic translation → pure English

        Args:
            raw_query: The raw user input string.

        Returns:
            UnifiedQuery dataclass with all representations.
        """
        logger.debug("Processing query: %s", raw_query[:80])

        # Step A: Normalize and detect script
        normalized = normalize_unicode(raw_query)
        script_type = detect_script(normalized)
        logger.debug("Script detected: %s", script_type)

        # Step B: Phonetic normalization (Roman → Devanagari for known words)
        phonetic_normalized = self._phonetic.normalize_text(normalized)
        logger.debug("After phonetic normalization: %s", phonetic_normalized[:80])

        # Step C: Medical term masking
        masked_text, medical_terms = self._masker.mask(phonetic_normalized)
        logger.debug("Masked %d medical terms.", len(medical_terms))

        # Step D: Transliteration (non-masked Roman → Devanagari)
        if script_type in ("roman", "mixed"):
            transliterated = _transliterate_to_devanagari(masked_text, [])
        else:
            transliterated = masked_text

        # Remove MED tags from transliterated result for clean output
        transliterated_clean = self._masker.unmask(transliterated)
        logger.debug("Transliterated: %s", transliterated_clean[:80])

        # Step E: Semantic translation → pure English
        # Use the phonetic-normalized + masked text for best translation
        semantic_english = _translate_to_english(
            phonetic_normalized,
            backend=self._translation_backend,
        )
        logger.debug("Semantic English: %s", semantic_english[:80])

        return UnifiedQuery(
            original_manglish=raw_query,
            transliterated_marathi=transliterated_clean,
            semantic_english=semantic_english,
            extracted_medical_terms=medical_terms,
            script_type=script_type,
            normalized_text=phonetic_normalized,
        )


# ---------------------------------------------------------------------------
# CLI Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "mala stomach pain hotoy ani fever pan ahe",
        "BP high ahe tar kay karave? doctor ne medicine sangitli",
        "metformin che side effects kay ahet?",
        "माझ्या chest मध्ये pain होतोय, heart attack आहे का?",
        "mla khup tras hotoy headache mule",
    ]

    processor = QueryProcessor()

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Input: {query}")
        result = processor.process(query)
        print(f"Script: {result.script_type}")
        print(f"Transliterated: {result.transliterated_marathi}")
        print(f"English: {result.semantic_english}")
        print(f"Medical terms: {[t['term'] for t in result.extracted_medical_terms]}")
