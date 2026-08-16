"""
Phase 6: Live Safety & Risk Classification.

Implements:
    1. Zero-shot classifier for LOW / MEDIUM / HIGH risk labeling
    2. Token-level entropy checks during generation
    3. Hardcoded Marathi disclaimer for HIGH-risk or high-entropy outputs

Usage:
    from src.safety import SafetyFilter
    sf = SafetyFilter()
    risk = sf.classify_risk("mala chest pain hotoy")
"""

import gc
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src import get_config, get_project_root, setup_logging

logger = setup_logging("safety")


# ---------------------------------------------------------------------------
# Keyword-Based Fast Pre-Filter
# ---------------------------------------------------------------------------

# Emergency keywords that always trigger HIGH risk
_HIGH_RISK_KEYWORDS = {
    # English
    "suicide", "suicidal", "kill myself", "want to die", "overdose",
    "heart attack", "stroke", "seizure", "unconscious", "unresponsive",
    "severe bleeding", "can't breathe", "chest pain", "poisoning",
    "anaphylaxis", "allergic reaction severe",
    # Marathi (Devanagari)
    "आत्महत्या", "जीव द्यायचा", "मरायचं", "बेशुद्ध",
    "खूप रक्तस्राव", "श्वास घेता येत नाही",
    # Roman Marathi
    "jiv dyaycha", "maraycha", "beshuddh", "shvas gheta yet nahi",
}

_MEDIUM_RISK_KEYWORDS = {
    "drug interaction", "side effect", "pregnant", "pregnancy",
    "dosage", "dose", "overdose risk", "surgery", "operation",
    "diabetes insulin", "blood thinner", "anticoagulant",
    "गरोदर", "गर्भवती", "शस्त्रक्रिया",
}


def _keyword_risk_check(query: str) -> Optional[str]:
    """Fast keyword-based risk pre-filter.

    Args:
        query: Input query text (any script).

    Returns:
        'HIGH', 'MEDIUM', or None if no keyword match.
    """
    query_lower = query.lower()

    for kw in _HIGH_RISK_KEYWORDS:
        if kw in query_lower or kw in query:
            return "HIGH"

    for kw in _MEDIUM_RISK_KEYWORDS:
        if kw in query_lower or kw in query:
            return "MEDIUM"

    return None


# ---------------------------------------------------------------------------
# Zero-Shot Classifier
# ---------------------------------------------------------------------------

class ZeroShotRiskClassifier:
    """Lightweight zero-shot classifier for medical risk assessment.

    Uses the Groq API with a structured prompt for zero-latency-impact
    classification (single fast API call, no local model loading).
    """

    def __init__(self) -> None:
        """Initialize classifier with configuration."""
        self._config = get_config()
        logger.info("ZeroShotRiskClassifier initialized.")

    def classify(self, query: str) -> str:
        """Classify a query into LOW, MEDIUM, or HIGH risk.

        Args:
            query: Input medical query.

        Returns:
            Risk level string: 'LOW', 'MEDIUM', or 'HIGH'.
        """
        # Fast keyword pre-filter
        keyword_result = _keyword_risk_check(query)
        if keyword_result:
            logger.info("Keyword pre-filter: %s risk for query.", keyword_result)
            return keyword_result

        # Zero-shot via API
        return self._classify_via_api(query)

    def _classify_via_api(self, query: str) -> str:
        """Classify using Groq API (fast, lightweight).

        Args:
            query: Input query.

        Returns:
            Risk level string.
        """
        try:
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                logger.warning("GROQ_API_KEY not set. Defaulting to MEDIUM risk.")
                return "MEDIUM"

            client = Groq(api_key=api_key)

            prompt = (
                "Classify the medical risk level of this patient query.\n\n"
                "CLASSIFICATION RULES:\n"
                "LOW: General health info, diet, vitamins, lifestyle, wellness tips.\n"
                "MEDIUM: Common symptoms, medication questions, side effects, general treatments.\n"
                "HIGH: Emergencies, severe pain, chest pain, breathing difficulty, "
                "drug interactions, life-threatening conditions, suicidal thoughts, overdose.\n\n"
                f'Query: "{query}"\n\n'
                "Respond with ONLY one word: LOW, MEDIUM, or HIGH."
            )

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )

            result = response.choices[0].message.content.strip().upper()

            if "HIGH" in result:
                return "HIGH"
            elif "MEDIUM" in result:
                return "MEDIUM"
            return "LOW"

        except Exception as exc:
            logger.error("Risk classification failed: %s. Defaulting to HIGH.", exc)
            return "HIGH"  # Fail-safe: default to highest risk


# ---------------------------------------------------------------------------
# Token-Level Entropy Check
# ---------------------------------------------------------------------------

def calculate_token_entropy(logprobs: List[float]) -> float:
    """Calculate the average token-level entropy from log probabilities.

    Higher entropy indicates more uncertainty in the model's output.

    Args:
        logprobs: List of log probabilities for each generated token.

    Returns:
        Average entropy value. Higher = more uncertain.
    """
    if not logprobs:
        return 0.0

    entropies: List[float] = []
    for lp in logprobs:
        # Convert log probability to probability
        prob = math.exp(lp)
        if prob > 0 and prob < 1:
            # Shannon entropy for single token: -p * log2(p)
            entropy = -prob * math.log2(prob)
            entropies.append(entropy)

    if not entropies:
        return 0.0

    return sum(entropies) / len(entropies)


def check_generation_uncertainty(
    response_text: str,
    logprobs: Optional[List[float]] = None,
    threshold: Optional[float] = None,
) -> Tuple[bool, float]:
    """Check if a generated response has high uncertainty.

    Uses heuristic checks when log probabilities are not available
    (common with API-based generation).

    Args:
        response_text: The generated text to evaluate.
        logprobs: Optional log probabilities from the model.
        threshold: Entropy threshold for flagging. Defaults to config value.

    Returns:
        Tuple of (is_uncertain, entropy_score).
    """
    config = get_config()
    if threshold is None:
        threshold = config["safety"]["entropy_threshold"]

    # If we have actual logprobs, use them
    if logprobs:
        entropy = calculate_token_entropy(logprobs)
        return entropy > threshold, entropy

    # Heuristic fallback: check for uncertainty markers in text
    uncertainty_markers = [
        r"\b(maybe|perhaps|possibly|might|could be|uncertain)\b",
        r"\b(I'm not sure|I am not sure|I cannot confirm)\b",
        r"\b(consult|see a doctor|seek medical)\b",
        r"(शक्यता|कदाचित|नक्की सांगता येत नाही)",
    ]

    marker_count = 0
    for pattern in uncertainty_markers:
        marker_count += len(re.findall(pattern, response_text, re.IGNORECASE))

    # Normalize to a pseudo-entropy score
    word_count = max(len(response_text.split()), 1)
    pseudo_entropy = marker_count / word_count * 10.0

    return pseudo_entropy > threshold, pseudo_entropy


# ---------------------------------------------------------------------------
# Safety Filter (Main Interface)
# ---------------------------------------------------------------------------

class SafetyFilter:
    """Complete safety filtering system.

    Combines zero-shot risk classification with entropy-based
    uncertainty quantification for runtime medical safety.
    """

    def __init__(self) -> None:
        """Initialize the safety filter."""
        self._classifier = ZeroShotRiskClassifier()
        self._config = get_config()
        logger.info("SafetyFilter initialized.")

    def classify_risk(self, query: str) -> str:
        """Classify the risk level of a user query.

        Args:
            query: Raw user query.

        Returns:
            Risk level: 'LOW', 'MEDIUM', or 'HIGH'.
        """
        return self._classifier.classify(query)

    def check_response_safety(
        self,
        response_text: str,
        logprobs: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Check the safety of a generated response.

        Args:
            response_text: Generated answer text.
            logprobs: Optional token-level log probabilities.

        Returns:
            Dict with 'is_safe', 'entropy', 'flags'.
        """
        is_uncertain, entropy = check_generation_uncertainty(
            response_text, logprobs
        )

        flags: List[str] = []

        if is_uncertain:
            flags.append("HIGH_ENTROPY")

        # Check for dosage information without context backing
        if re.search(r"\d+\s*(mg|ml|mcg|tablet|capsule|dose)", response_text, re.IGNORECASE):
            flags.append("CONTAINS_DOSAGE")

        return {
            "is_safe": len(flags) == 0,
            "entropy": entropy,
            "flags": flags,
        }

    def apply_guardrails(
        self,
        answer: str,
        risk_level: str,
        safety_check: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply safety guardrails to the generated answer.

        Args:
            answer: Raw generated answer.
            risk_level: Classified risk level.
            safety_check: Optional safety check result from check_response_safety.

        Returns:
            Answer with appropriate disclaimers prepended/appended.
        """
        safety_cfg = self._config["safety"]

        # High risk or high entropy → prepend full disclaimer
        if risk_level == "HIGH" or (
            safety_check and "HIGH_ENTROPY" in safety_check.get("flags", [])
        ):
            return f"{safety_cfg['high_risk_disclaimer']}\n\n{answer}"

        # Medium risk → append mild disclaimer
        if risk_level == "MEDIUM":
            disclaimer = (
                "\n\n📋 टीप: हा सामान्य माहितीपूर्ण सल्ला आहे. "
                "कोणतेही औषध घेण्यापूर्वी doctor चा सल्ला घ्या."
            )
            return answer + disclaimer

        # Dosage warning
        if safety_check and "CONTAINS_DOSAGE" in safety_check.get("flags", []):
            dosage_warn = (
                "\n\n⚠️ या उत्तरामध्ये dosage माहिती आहे. "
                "कृपया हे केवळ संदर्भासाठी वापरा आणि doctor चा सल्ला घ्या."
            )
            return answer + dosage_warn

        # Low risk → add gentle info note
        return answer + "\n\n💡 माहिती: हा केवळ सामान्य आरोग्य सल्ला आहे."
