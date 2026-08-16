"""
Phase 3 (Step C): Medical Term Masking using Aho-Corasick Algorithm.

Wraps English medical terms in protective <MED>...</MED> tags so they
are ignored by the transliterator. Matches against a comprehensive
medical dictionary cross-referenced with UMLS synonyms.
"""

import re
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from src import setup_logging

logger = setup_logging("medical_masking")


# ---------------------------------------------------------------------------
# Comprehensive Medical Vocabulary (UMLS-aligned)
# ---------------------------------------------------------------------------

MEDICAL_VOCABULARY: Dict[str, str] = {
    # --- Symptoms ---
    "fever": "C0015967", "headache": "C0018681", "cough": "C0010200",
    "pain": "C0030193", "stomach pain": "C0000737", "chest pain": "C0008031",
    "back pain": "C0004604", "abdominal pain": "C0000737",
    "nausea": "C0027497", "vomiting": "C0042963", "diarrhea": "C0011991",
    "fatigue": "C0015672", "dizziness": "C0012833",
    "shortness of breath": "C0013404", "breathlessness": "C0013404",
    "sore throat": "C0242429", "runny nose": "C1260880",
    "body ache": "C0281856", "weakness": "C0004093",
    "swelling": "C0013604", "rash": "C0015230", "itching": "C0033774",
    "bleeding": "C0019080", "weight loss": "C1262477",
    "weight gain": "C0043094", "insomnia": "C0917801",
    "anxiety": "C0003467", "depression": "C0011570",
    "palpitations": "C0030252", "constipation": "C0009806",
    "acidity": "C0018834", "gas": "C0016204",
    "joint pain": "C0003862", "muscle pain": "C0231528",

    # --- Diseases ---
    "diabetes": "C0011849", "type 2 diabetes": "C0011860",
    "hypertension": "C0020538", "high blood pressure": "C0020538",
    "low blood pressure": "C0020649",
    "asthma": "C0004096", "tuberculosis": "C0041296",
    "malaria": "C0024530", "dengue": "C0011311",
    "typhoid": "C0041466", "cholera": "C0008354",
    "pneumonia": "C0032285", "bronchitis": "C0006277",
    "arthritis": "C0003864", "cancer": "C0006826",
    "stroke": "C0038454", "heart attack": "C0027051",
    "myocardial infarction": "C0027051",
    "kidney disease": "C0022658", "liver disease": "C0023890",
    "anemia": "C0002871", "thyroid": "C0040136",
    "hypothyroidism": "C0020676", "hyperthyroidism": "C0020550",
    "epilepsy": "C0014544", "migraine": "C0149931",
    "covid": "C5203670", "coronavirus": "C5203670",
    "influenza": "C0021400", "flu": "C0021400",
    "hiv": "C0019693", "aids": "C0001175",
    "hepatitis": "C0019158", "hepatitis b": "C0019163",
    "obesity": "C0028754", "copd": "C0024117",
    "jaundice": "C0022346", "gastritis": "C0017152",
    "ulcer": "C0041582", "kidney stones": "C0022650",
    "urinary tract infection": "C0042029", "uti": "C0042029",

    # --- Medications ---
    "metformin": "C0025598", "paracetamol": "C0000970",
    "acetaminophen": "C0000970", "ibuprofen": "C0020740",
    "aspirin": "C0004057", "amoxicillin": "C0002645",
    "azithromycin": "C0052796", "insulin": "C0021641",
    "atorvastatin": "C0286651", "omeprazole": "C0028978",
    "amlodipine": "C0051696", "losartan": "C0126174",
    "ciprofloxacin": "C0008809", "prednisone": "C0032952",
    "dexamethasone": "C0011777", "cetirizine": "C0055147",
    "montelukast": "C0298130", "pantoprazole": "C0081876",
    "ranitidine": "C0034614", "crocin": "C0000970",
    "dolo": "C0000970", "combiflam": "C0020740",
    "ors": "C0600266",

    # --- Body parts ---
    "blood pressure": "C0005823", "bp": "C0005823",
    "blood sugar": "C0005802",
    "heart": "C0018787", "lung": "C0024109", "liver": "C0023884",
    "kidney": "C0022646", "brain": "C0006104", "stomach": "C0038351",
    "intestine": "C0021853", "bone": "C0262950", "muscle": "C0026845",
    "skin": "C0037267", "eye": "C0015392", "ear": "C0013443",
    "throat": "C0031354", "nose": "C0028429", "blood": "C0005767",

    # --- Procedures ---
    "surgery": "C0543467", "biopsy": "C0005558",
    "ct scan": "C0040405", "mri": "C0024485",
    "x-ray": "C0043299", "xray": "C0043299",
    "ultrasound": "C0041618", "ecg": "C0013798",
    "blood test": "C0005841", "urine test": "C0042014",
    "vaccination": "C0042196", "injection": "C0021485",
    "dialysis": "C0011946", "chemotherapy": "C0013216",
    "radiation therapy": "C0034618", "transplant": "C0040732",
    "endoscopy": "C0014245", "colonoscopy": "C0009378",

    # --- General medical terms ---
    "side effects": "C0879626", "side effect": "C0879626",
    "dose": "C0178602", "dosage": "C0178602",
    "tablet": "C0039225", "capsule": "C0006935",
    "syrup": "C1883063", "ointment": "C0028912",
    "cream": "C0010403", "drops": "C0991568",
    "doctor": "C0031831", "hospital": "C0019994",
    "clinic": "C0002424", "pharmacy": "C0031322",
    "prescription": "C0033081", "diagnosis": "C0011900",
    "treatment": "C0087111", "medicine": "C0013227",
    "vitamin": "C0042890", "antibiotic": "C0003232",
    "antiviral": "C0003451", "painkiller": "C0002771",
    "supplement": "C0242295",
    "oxygen": "C0030054", "icu": "C0021708",
    "opd": "C0002424", "operation": "C0543467",
    "tumor": "C0027651",
}


# ---------------------------------------------------------------------------
# Aho-Corasick Automaton
# ---------------------------------------------------------------------------

class _AhoCorasickNode:
    """A single node in the Aho-Corasick trie."""

    __slots__ = ("children", "fail", "output", "depth")

    def __init__(self) -> None:
        self.children: Dict[str, "_AhoCorasickNode"] = {}
        self.fail: Optional["_AhoCorasickNode"] = None
        self.output: List[str] = []
        self.depth: int = 0


class AhoCorasickMatcher:
    """Aho-Corasick automaton for efficient multi-pattern string matching.

    This is used to find all medical terms in a query in O(n + m + z) time,
    where n = text length, m = total pattern length, z = number of matches.
    """

    def __init__(self, patterns: List[str]) -> None:
        """Build the Aho-Corasick automaton from a list of patterns.

        Args:
            patterns: List of medical terms/phrases to match (case-insensitive).
        """
        self._root = _AhoCorasickNode()
        self._patterns = [p.lower() for p in patterns]
        self._build_trie()
        self._build_failure_links()

    def _build_trie(self) -> None:
        """Insert all patterns into the trie."""
        for pattern in self._patterns:
            node = self._root
            for char in pattern:
                if char not in node.children:
                    child = _AhoCorasickNode()
                    child.depth = node.depth + 1
                    node.children[char] = child
                node = node.children[char]
            node.output.append(pattern)

    def _build_failure_links(self) -> None:
        """Build failure links using BFS (Knuth-Morris-Pratt extension)."""
        queue: deque = deque()
        self._root.fail = self._root

        for child in self._root.children.values():
            child.fail = self._root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                queue.append(child)

                # Follow failure links to find the longest proper suffix
                fail_node = current.fail
                while fail_node != self._root and char not in fail_node.children:
                    fail_node = fail_node.fail

                child.fail = fail_node.children.get(char, self._root)
                if child.fail == child:
                    child.fail = self._root

                # Merge output
                child.output = child.output + child.fail.output

    def find_all(self, text: str) -> List[Tuple[int, int, str]]:
        """Find all occurrences of any pattern in the text.

        Args:
            text: Input text to search.

        Returns:
            List of (start_index, end_index, matched_pattern) tuples.
            Sorted by start index, with longer matches preferred.
        """
        text_lower = text.lower()
        results: List[Tuple[int, int, str]] = []
        node = self._root

        for i, char in enumerate(text_lower):
            while node != self._root and char not in node.children:
                node = node.fail

            node = node.children.get(char, self._root)

            for pattern in node.output:
                start = i - len(pattern) + 1
                # Verify word boundary
                if start > 0 and text_lower[start - 1].isalnum():
                    continue
                end = i + 1
                if end < len(text_lower) and text_lower[end].isalnum():
                    continue
                results.append((start, end, pattern))

        # Sort by start position, prefer longer matches
        results.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Remove overlapping matches (keep longest)
        filtered: List[Tuple[int, int, str]] = []
        last_end = -1
        for start, end, pattern in results:
            if start >= last_end:
                filtered.append((start, end, pattern))
                last_end = end

        return filtered


# ---------------------------------------------------------------------------
# Medical Masker
# ---------------------------------------------------------------------------

class MedicalMasker:
    """Masks English medical terms with <MED>...</MED> tags.

    Uses Aho-Corasick for efficient multi-pattern matching against
    the UMLS-aligned medical vocabulary.
    """

    def __init__(self, extra_terms: Optional[Dict[str, str]] = None) -> None:
        """Initialize the medical masker.

        Args:
            extra_terms: Additional {term: CUI} mappings to include.
        """
        vocab = MEDICAL_VOCABULARY.copy()
        if extra_terms:
            vocab.update(extra_terms)

        self._vocab = vocab
        self._matcher = AhoCorasickMatcher(list(vocab.keys()))
        logger.info("MedicalMasker initialized with %d terms.", len(vocab))

    def mask(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Mask medical terms in the text with <MED> tags.

        Args:
            text: Input text (potentially code-mixed).

        Returns:
            Tuple of:
                - Text with medical terms wrapped in <MED>...</MED> tags.
                - List of extracted medical term dicts: {term, cui, start, end}.
        """
        matches = self._matcher.find_all(text)

        if not matches:
            return text, []

        extracted: List[Dict[str, str]] = []
        # Build the masked text by replacing matched spans
        parts: List[str] = []
        last_end = 0

        for start, end, pattern in matches:
            # Preserve original casing from the text
            original_span = text[start:end]
            cui = self._vocab.get(pattern, "UNKNOWN")

            parts.append(text[last_end:start])
            parts.append(f"<MED>{original_span}</MED>")
            last_end = end

            extracted.append({
                "term": original_span,
                "normalized": pattern,
                "cui": cui,
                "start": str(start),
                "end": str(end),
            })

        parts.append(text[last_end:])
        masked_text = "".join(parts)

        logger.debug("Masked %d medical terms in text.", len(extracted))
        return masked_text, extracted

    def unmask(self, text: str) -> str:
        """Remove <MED>...</MED> tags, restoring original terms.

        Args:
            text: Text with <MED> tags.

        Returns:
            Clean text with tags removed.
        """
        return re.sub(r"<MED>(.*?)</MED>", r"\1", text)

    def get_cui(self, term: str) -> str:
        """Look up the UMLS CUI for a medical term.

        Args:
            term: Medical term to look up.

        Returns:
            UMLS CUI string, or 'UNKNOWN' if not found.
        """
        return self._vocab.get(term.lower(), "UNKNOWN")
