"""
Unit tests for MedManglish-RAG pipeline components.

Tests cover:
    - CMI calculation
    - Phonetic normalization
    - Medical masking (Aho-Corasick)
    - Script detection
    - Unicode normalization
    - Data loader
    - UMLS CUI extraction

Usage (on Ubuntu):
    python -m pytest tests/test_pipeline.py -v
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest


# ---------------------------------------------------------------------------
# Phase 1: CMI Calculation Tests
# ---------------------------------------------------------------------------

class TestCMICalculation:
    """Tests for Code-Mixing Index calculation."""

    def test_pure_english(self) -> None:
        """Pure English text should have CMI = 0."""
        from src.generate_synthetic_data import calculate_cmi
        assert calculate_cmi("This is pure English text") == 0.0

    def test_pure_devanagari(self) -> None:
        """Pure Devanagari text should have CMI = 0."""
        from src.generate_synthetic_data import calculate_cmi
        assert calculate_cmi("हा शुद्ध मराठी मजकूर आहे") == 0.0

    def test_balanced_mixing(self) -> None:
        """Balanced code-mixed text should have CMI around 0.5."""
        from src.generate_synthetic_data import calculate_cmi
        cmi = calculate_cmi("मला stomach pain होतोय आणि fever आहे")
        assert 0.3 <= cmi <= 0.7

    def test_empty_text(self) -> None:
        """Empty text should return CMI = 0."""
        from src.generate_synthetic_data import calculate_cmi
        assert calculate_cmi("") == 0.0

    def test_numbers_only(self) -> None:
        """Text with only numbers should return CMI = 0."""
        from src.generate_synthetic_data import calculate_cmi
        assert calculate_cmi("123 456 789") == 0.0


# ---------------------------------------------------------------------------
# Phase 3: Phonetic Normalization Tests
# ---------------------------------------------------------------------------

class TestPhoneticNormalizer:
    """Tests for the phonetic normalizer."""

    def test_known_word_mapping(self) -> None:
        """Known words should map to correct Devanagari."""
        from src.phonetic_normalizer import PhoneticNormalizer
        normalizer = PhoneticNormalizer()
        assert normalizer.normalize_word("mala") == "मला"
        assert normalizer.normalize_word("ahe") == "आहे"
        assert normalizer.normalize_word("nahi") == "नाही"

    def test_unknown_word_passthrough(self) -> None:
        """Unknown words should pass through unchanged."""
        from src.phonetic_normalizer import PhoneticNormalizer
        normalizer = PhoneticNormalizer()
        assert normalizer.normalize_word("xyzabc") == "xyzabc"

    def test_devanagari_passthrough(self) -> None:
        """Devanagari text should pass through unchanged."""
        from src.phonetic_normalizer import PhoneticNormalizer
        normalizer = PhoneticNormalizer()
        assert normalizer.normalize_text("हा मराठी आहे") == "हा मराठी आहे"

    def test_mixed_text_normalization(self) -> None:
        """Mixed text should normalize known Marathi words only."""
        from src.phonetic_normalizer import PhoneticNormalizer
        normalizer = PhoneticNormalizer()
        result = normalizer.normalize_text("mala stomach pain hotoy")
        assert "मला" in result
        assert "stomach" in result  # English word unchanged
        assert "pain" in result

    def test_medical_word_normalization(self) -> None:
        """Medical Marathi words should map correctly."""
        from src.phonetic_normalizer import PhoneticNormalizer
        normalizer = PhoneticNormalizer()
        assert normalizer.normalize_word("taap") == "ताप"
        assert normalizer.normalize_word("aushadh") == "औषध"

    def test_custom_mapping(self) -> None:
        """Custom mappings should be addable."""
        from src.phonetic_normalizer import PhoneticNormalizer
        normalizer = PhoneticNormalizer()
        normalizer.add_mapping("customword", "कस्टम")
        assert normalizer.normalize_word("customword") == "कस्टम"


# ---------------------------------------------------------------------------
# Phase 3: Medical Masking Tests
# ---------------------------------------------------------------------------

class TestMedicalMasking:
    """Tests for the Aho-Corasick medical term masker."""

    def test_single_term_masking(self) -> None:
        """A single medical term should be masked."""
        from src.medical_masking import MedicalMasker
        masker = MedicalMasker()
        masked, terms = masker.mask("I have fever")
        assert "<MED>" in masked
        assert any(t["term"].lower() == "fever" for t in terms)

    def test_multi_word_term(self) -> None:
        """Multi-word terms like 'blood pressure' should match."""
        from src.medical_masking import MedicalMasker
        masker = MedicalMasker()
        masked, terms = masker.mask("My blood pressure is high")
        assert any(t["normalized"] == "blood pressure" for t in terms)

    def test_umls_cui_lookup(self) -> None:
        """Medical terms should return correct UMLS CUIs."""
        from src.medical_masking import MedicalMasker
        masker = MedicalMasker()
        assert masker.get_cui("diabetes") == "C0011849"
        assert masker.get_cui("nonexistent") == "UNKNOWN"

    def test_unmasking(self) -> None:
        """Unmasking should remove MED tags cleanly."""
        from src.medical_masking import MedicalMasker
        masker = MedicalMasker()
        assert masker.unmask("<MED>fever</MED> is common") == "fever is common"

    def test_no_terms_found(self) -> None:
        """Text without medical terms should return unchanged."""
        from src.medical_masking import MedicalMasker
        masker = MedicalMasker()
        masked, terms = masker.mask("hello world how are you")
        assert len(terms) == 0


# ---------------------------------------------------------------------------
# Phase 3: Script Detection Tests
# ---------------------------------------------------------------------------

class TestScriptDetection:
    """Tests for Unicode script detection."""

    def test_pure_roman(self) -> None:
        """Pure ASCII text should be detected as 'roman'."""
        from src.query_processor import detect_script
        assert detect_script("this is english text") == "roman"

    def test_pure_devanagari(self) -> None:
        """Pure Devanagari should be detected as 'devanagari'."""
        from src.query_processor import detect_script
        assert detect_script("हा मराठी मजकूर आहे") == "devanagari"

    def test_mixed_script(self) -> None:
        """Mixed script should be detected as 'mixed'."""
        from src.query_processor import detect_script
        assert detect_script("मला stomach pain आहे") == "mixed"


# ---------------------------------------------------------------------------
# Phase 3: Unicode Normalization Tests
# ---------------------------------------------------------------------------

class TestUnicodeNormalization:
    """Tests for Unicode NFC normalization."""

    def test_nfc_normalization(self) -> None:
        """Text should be NFC normalized."""
        from src.query_processor import normalize_unicode
        import unicodedata
        text = "café"
        result = normalize_unicode(text)
        assert result == unicodedata.normalize("NFC", text)

    def test_whitespace_normalization(self) -> None:
        """Multiple spaces should be collapsed."""
        from src.query_processor import normalize_unicode
        assert normalize_unicode("hello   world") == "hello world"

    def test_empty_string(self) -> None:
        """Empty string should return empty."""
        from src.query_processor import normalize_unicode
        assert normalize_unicode("") == ""


# ---------------------------------------------------------------------------
# Phase 1: UMLS CUI Extraction Tests
# ---------------------------------------------------------------------------

class TestUMLSExtraction:
    """Tests for UMLS CUI extraction."""

    def test_known_term(self) -> None:
        """Known medical terms should return CUIs."""
        from src.ingest import extract_umls_cuis
        cuis = extract_umls_cuis("Patient has diabetes and hypertension")
        assert "C0011849" in cuis  # diabetes
        assert "C0020538" in cuis  # hypertension

    def test_no_medical_terms(self) -> None:
        """Non-medical text should return empty list."""
        from src.ingest import extract_umls_cuis
        cuis = extract_umls_cuis("The weather is nice today")
        assert len(cuis) == 0

    def test_multi_word_term(self) -> None:
        """Multi-word terms like 'blood pressure' should be found."""
        from src.ingest import extract_umls_cuis
        cuis = extract_umls_cuis("Monitor blood pressure regularly")
        assert "C0005823" in cuis


# ---------------------------------------------------------------------------
# Phase 1: Semantic Chunking Tests
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    """Tests for medical-aware semantic chunking."""

    def test_basic_chunking(self) -> None:
        """Text should be split into chunks."""
        from src.ingest import semantic_chunk
        text = "First sentence. " * 50 + "Last sentence."
        chunks = semantic_chunk(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_empty_text(self) -> None:
        """Empty text should return empty list."""
        from src.ingest import semantic_chunk
        assert semantic_chunk("") == []

    def test_short_text(self) -> None:
        """Short text should remain as single chunk."""
        from src.ingest import semantic_chunk
        chunks = semantic_chunk("Short text.", chunk_size=400)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Phase 6: Safety Tests
# ---------------------------------------------------------------------------

class TestSafety:
    """Tests for the safety keyword pre-filter."""

    def test_high_risk_keyword(self) -> None:
        """Emergency keywords should trigger HIGH risk."""
        from src.safety import _keyword_risk_check
        assert _keyword_risk_check("I am having a heart attack") == "HIGH"
        assert _keyword_risk_check("suicide") == "HIGH"

    def test_medium_risk_keyword(self) -> None:
        """Medical procedure keywords should trigger MEDIUM risk."""
        from src.safety import _keyword_risk_check
        assert _keyword_risk_check("What is the correct dosage?") == "MEDIUM"

    def test_low_risk_no_keywords(self) -> None:
        """General text should return None (no keyword match)."""
        from src.safety import _keyword_risk_check
        assert _keyword_risk_check("What vitamins should I take?") is None
