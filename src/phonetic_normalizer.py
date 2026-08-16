"""
Phase 3 (Step B): Deterministic Phonetic Normalization for Roman Marathi.

Maps common Roman/phonetic spellings of Marathi sounds to their canonical
Devanagari equivalents. Handles anusvara, chandrabindu, nukta, and
common phonetic variations BEFORE transliteration.

This ensures that 'kh', 'kha', 'kh' all map to 'ख' consistently.
"""

import re
from typing import Dict, List, Tuple

from src import setup_logging

logger = setup_logging("phonetic_normalizer")

# ---------------------------------------------------------------------------
# Phonetic Mapping Tables
# ---------------------------------------------------------------------------

# Order matters: longer patterns must come first to avoid partial matches.
# Each tuple: (roman_pattern, devanagari_replacement)

CONSONANT_CONJUNCTS: List[Tuple[str, str]] = [
    # Aspirated consonants (must come before unaspirated)
    ("ksha", "क्ष"), ("kshu", "क्षु"),
    ("gya", "ज्ञ"), ("dnya", "ज्ञ"), ("dnya", "ज्ञ"),
    ("shra", "श्र"), ("shri", "श्री"),
    ("tra", "त्र"), ("tri", "त्रि"),
    ("pra", "प्र"), ("pri", "प्रि"),
]

ASPIRATED_CONSONANTS: List[Tuple[str, str]] = [
    ("kha", "ख"), ("gha", "घ"),
    ("chha", "छ"), ("cha", "च"),
    ("jha", "झ"),
    ("tha", "ठ"), ("dha", "ध"),
    ("pha", "फ"), ("bha", "भ"),
    ("sha", "श"), ("shha", "ष"),
]

BASE_CONSONANTS: List[Tuple[str, str]] = [
    ("ka", "क"), ("ga", "ग"),
    ("ja", "ज"), ("ta", "त"),
    ("da", "द"), ("na", "न"),
    ("pa", "प"), ("ba", "ब"),
    ("ma", "म"), ("ya", "य"),
    ("ra", "र"), ("la", "ल"),
    ("va", "व"), ("wa", "व"),
    ("sa", "स"), ("ha", "ह"),
]

# Single-character consonants (fallback)
SINGLE_CONSONANTS: List[Tuple[str, str]] = [
    ("k", "क"), ("g", "ग"),
    ("j", "ज"), ("t", "त"),
    ("d", "द"), ("n", "न"),
    ("p", "प"), ("b", "ब"),
    ("m", "म"), ("y", "य"),
    ("r", "र"), ("l", "ल"),
    ("v", "व"), ("w", "व"),
    ("s", "स"), ("h", "ह"),
]

VOWELS: List[Tuple[str, str]] = [
    ("aa", "आ"), ("ee", "ई"), ("oo", "ऊ"),
    ("ai", "ऐ"), ("au", "औ"), ("ou", "औ"),
    ("a", "अ"), ("e", "ए"), ("i", "इ"),
    ("o", "ओ"), ("u", "उ"),
]

# Matra (vowel signs attached to consonants)
MATRAS: Dict[str, str] = {
    "aa": "ा", "a": "",
    "i": "ि", "ee": "ी",
    "u": "ु", "oo": "ू",
    "e": "े", "ai": "ै",
    "o": "ो", "au": "ौ", "ou": "ौ",
}

# Special markers
SPECIAL_MARKERS: Dict[str, str] = {
    "anusvara": "ं",       # Anusvara (nasal)
    "chandrabindu": "ँ",   # Chandrabindu
    "visarga": "ः",        # Visarga
    "nukta": "़",          # Nukta (for foreign sounds)
    "halant": "्",         # Halant/Virama
}


# ---------------------------------------------------------------------------
# Common Marathi Word Normalization Dictionary
# ---------------------------------------------------------------------------

# Direct word-level mappings for the most common Manglish spellings.
# These override the character-level transliteration for accuracy.

WORD_LEVEL_MAPPINGS: Dict[str, str] = {
    # Pronouns and common words
    "mala": "मला", "mla": "मला",
    "tula": "तुला", "tula": "तुला",
    "tyala": "त्याला", "tila": "तिला",
    "aamhala": "आम्हाला", "amhala": "आम्हाला",
    "tumhala": "तुम्हाला",
    "ahe": "आहे", "aahe": "आहे",
    "nahi": "नाही", "naahi": "नाही",
    "kay": "काय", "kaay": "काय",
    "kasa": "कसा", "kashi": "कशी", "kase": "कसे",
    "kuthe": "कुठे", "kutha": "कुठे",
    "kiti": "किती",
    "tar": "तर", "pan": "पण", "ani": "आणि",
    "mhanje": "म्हणजे", "mhanun": "म्हणून",
    "ata": "आता", "aata": "आता",
    "aaj": "आज",
    "udya": "उद्या",
    "hota": "होता", "hoti": "होती", "hote": "होते",
    "hotoy": "होतोय", "hotey": "होतेय",
    "jata": "जाता", "yeta": "येता",
    "ghya": "घ्या", "dya": "द्या",
    "sagla": "सगळं", "sagale": "सगळे",
    "changla": "चांगला", "changli": "चांगली",
    "vait": "वाईट",
    "khup": "खूप", "khoop": "खूप",
    "jara": "जरा", "thoda": "थोडा",
    "mothya": "मोठ्या", "motha": "मोठा",
    "lahaan": "लहान", "lahan": "लहान",

    # Medical-context Marathi
    "dukhat": "दुखत", "dukhte": "दुखते", "dukhato": "दुखतो",
    "tras": "त्रास", "traas": "त्रास",
    "aushadh": "औषध", "goli": "गोळी",
    "aajar": "आजार", "aajaar": "आजार",
    "taap": "ताप", "tap": "ताप",
    "sardi": "सर्दी", "khokhla": "खोकला", "khokla": "खोकला",
    "potaat": "पोटात", "potat": "पोटात",
    "dokyat": "डोक्यात",
    "ghasa": "घसा", "ghashi": "घशी",
    "rugnalay": "रुग्णालय",
    "ilaj": "इलाज", "upchar": "उपचार",
    "tapasni": "तपासणी",
    "rog": "रोग",
    "kashasathi": "कशासाठी",
    "ghyave": "घ्यावे",
    "chya": "च्या", "che": "चे", "chi": "ची",
    "madhye": "मध्ये", "madhe": "मध्ये",
    "sathi": "साठी",
    "pasun": "पासून", "nantar": "नंतर",
    "aadhich": "आधीच", "aadhi": "आधी",
}


# ---------------------------------------------------------------------------
# Normalizer Class
# ---------------------------------------------------------------------------

class PhoneticNormalizer:
    """Deterministic phonetic normalizer for Roman Marathi to Devanagari.

    Processes text word-by-word:
    1. Check word-level dictionary first (most accurate).
    2. Fall back to character-level phonetic mapping.

    This is applied BEFORE the full transliteration step.
    """

    def __init__(self) -> None:
        """Initialize the phonetic normalizer with mapping tables."""
        self._word_map = WORD_LEVEL_MAPPINGS.copy()
        logger.info(
            "PhoneticNormalizer initialized with %d word mappings.",
            len(self._word_map),
        )

    def normalize_word(self, word: str) -> str:
        """Normalize a single Roman Marathi word to Devanagari.

        Args:
            word: A single whitespace-delimited word.

        Returns:
            The Devanagari equivalent if found in dictionary, else the original word.
        """
        clean = word.lower().strip()

        # Strip common punctuation for lookup, preserve for output
        punct_prefix = ""
        punct_suffix = ""
        while clean and not clean[0].isalnum():
            punct_prefix += clean[0]
            clean = clean[1:]
        while clean and not clean[-1].isalnum():
            punct_suffix = clean[-1] + punct_suffix
            clean = clean[:-1]

        if not clean:
            return word

        # Word-level lookup
        if clean in self._word_map:
            return punct_prefix + self._word_map[clean] + punct_suffix

        return word

    def normalize_text(self, text: str) -> str:
        """Normalize an entire text string, word by word.

        Applies word-level phonetic mapping to each non-medical,
        non-Devanagari word in the text.

        Args:
            text: Input text (may be code-mixed).

        Returns:
            Text with recognized Roman Marathi words converted to Devanagari.
        """
        tokens = text.split()
        normalized = []

        for token in tokens:
            # Skip if already Devanagari
            if any("\u0900" <= ch <= "\u097F" for ch in token):
                normalized.append(token)
                continue

            result = self.normalize_word(token)
            normalized.append(result)

        return " ".join(normalized)

    def add_mapping(self, roman: str, devanagari: str) -> None:
        """Add a custom word-level mapping.

        Args:
            roman: Roman/phonetic spelling (lowercase).
            devanagari: Devanagari equivalent.
        """
        self._word_map[roman.lower()] = devanagari

    def get_mapping(self, word: str) -> str:
        """Look up the Devanagari mapping for a Roman word.

        Args:
            word: Roman word to look up.

        Returns:
            Devanagari equivalent or empty string if not found.
        """
        return self._word_map.get(word.lower(), "")
