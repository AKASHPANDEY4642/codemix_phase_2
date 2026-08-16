"""
Phase 5: VRAM Orchestration & RAG Pipeline State Machine.

Manages the complete RAG pipeline with strict VRAM budgeting:
    State 1: Load Embedders → Search → Unload
    State 2: Load Reranker → Rerank → Unload
    State 3: Load LLM (local) OR call external API → Generate → Unload

Supports configurable external API endpoints (Groq, OpenAI) for
offloading the LLM generation phase.

Usage:
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    response = pipeline.run("mala stomach pain hotoy")
"""

import gc
import os
import re
import time
import sqlite3
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import get_config, get_project_root, setup_logging

logger = setup_logging("rag_pipeline")


# ---------------------------------------------------------------------------
# Pipeline State Machine
# ---------------------------------------------------------------------------

class PipelineState(Enum):
    """States in the VRAM state machine."""
    IDLE = auto()
    QUERY_PROCESSING = auto()
    EMBEDDING_SEARCH = auto()
    RERANKING = auto()
    GENERATION = auto()
    SAFETY_CHECK = auto()
    COMPLETE = auto()


@dataclass
class PipelineResponse:
    """Complete response from the RAG pipeline.

    Attributes:
        answer: The generated answer text.
        thought_process: The model's reasoning in <thought> block.
        citations: List of cited chunk references.
        risk_level: Safety risk classification (LOW/MEDIUM/HIGH).
        disclaimer: Safety disclaimer text (if applicable).
        language_mode: User's selected language ('manglish', 'english', 'marathi').
        query_info: Processed query information.
        retrieved_chunks: List of retrieved chunk metadata.
        timings: Per-module execution times.
    """
    answer: str = ""
    thought_process: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = "LOW"
    disclaimer: str = ""
    language_mode: str = "manglish"
    query_info: Dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLite Logger
# ---------------------------------------------------------------------------

class SQLiteLogger:
    """Structured logging to SQLite database.

    Captures execution times, language toggles, RRF scores, and full
    interaction history for thesis evaluation.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize the SQLite database.

        Args:
            db_path: Path to the SQLite database file.
        """
        if db_path is None:
            config = get_config()
            db_path = get_project_root() / config["paths"]["sqlite_db"]

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()
        logger.info("SQLite logger initialized: %s", db_path)

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                raw_query TEXT NOT NULL,
                script_type TEXT,
                language_toggle TEXT DEFAULT 'manglish',
                risk_level TEXT,
                answer TEXT,
                thought_process TEXT,
                query_processing_ms REAL,
                hyde_generation_ms REAL,
                bm25_search_ms REAL,
                dense_search_ms REAL,
                rrf_fusion_ms REAL,
                reranking_ms REAL,
                generation_ms REAL,
                safety_check_ms REAL,
                total_ms REAL,
                rrf_scores_json TEXT,
                cited_chunks_json TEXT,
                num_chunks_retrieved INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def log(self, query: str, response: PipelineResponse) -> None:
        """Log a complete query-response interaction.

        Args:
            query: Raw user query.
            response: Complete pipeline response.
        """
        import json

        conn = sqlite3.connect(str(self._db_path))
        conn.execute(
            """INSERT INTO query_logs (
                raw_query, script_type, language_toggle, risk_level,
                answer, thought_process,
                query_processing_ms, hyde_generation_ms, bm25_search_ms,
                dense_search_ms, rrf_fusion_ms, reranking_ms,
                generation_ms, safety_check_ms, total_ms,
                rrf_scores_json, cited_chunks_json, num_chunks_retrieved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                query,
                response.query_info.get("script_type", ""),
                response.language_mode,
                response.risk_level,
                response.answer,
                response.thought_process,
                response.timings.get("query_processing_ms", 0),
                response.timings.get("hyde_generation_ms", 0),
                response.timings.get("bm25_search_ms", 0),
                response.timings.get("dense_search_ms", 0),
                response.timings.get("rrf_fusion_ms", 0),
                response.timings.get("reranking_ms", 0),
                response.timings.get("generation_ms", 0),
                response.timings.get("safety_check_ms", 0),
                response.timings.get("total_ms", 0),
                json.dumps(response.timings.get("rrf_scores", []), ensure_ascii=False),
                json.dumps([c.get("chunk_id") for c in response.retrieved_chunks]),
                len(response.retrieved_chunks),
            ),
        )
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# LLM Generator
# ---------------------------------------------------------------------------

class LLMGenerator:
    """Generates answers using either local LLM or external API.

    Implements the strict prompt engineering with:
    - <thought> block forcing
    - [chunk_id] citation enforcement
    - Dosage safety rules
    - Language mode control
    """

    def __init__(self) -> None:
        """Initialize the generator with config."""
        self._config = get_config()
        self._orch = self._config["orchestration"]
        logger.info(
            "LLMGenerator initialized (provider=%s, model=%s).",
            self._orch["external_llm_provider"],
            self._orch["external_llm_model"],
        )

    def _build_system_prompt(self, language_mode: str = "manglish") -> str:
        """Build the system prompt with safety constraints.

        Args:
            language_mode: One of 'manglish', 'english', 'marathi'.

        Returns:
            The system prompt string.
        """
        lang_instruction = {
            "manglish": (
                "Write your answer in natural code-mixed Marathi-English (Manglish). "
                "Keep ALL medical terms in English. Use Marathi (Devanagari script) "
                "for conversational grammar and connecting words."
            ),
            "english": "Write your entire answer in clear, professional English.",
            "marathi": (
                "Write your entire answer in Marathi (Devanagari script). "
                "You may keep specialized medical terms in English."
            ),
        }.get(language_mode, "Write in Manglish.")

        return (
            "You are a highly accurate, bilingual (Marathi-English) medical AI assistant.\n\n"
            "ABSOLUTE RULES:\n"
            f"1. LANGUAGE: {lang_instruction}\n"
            f"2. SAFETY: {self._orch['dosage_safety_rule']}\n"
            "3. CITATIONS: You MUST cite sources using [chunk_id] notation "
            "(e.g., [42], [105]) at the end of each factual claim. "
            "Only cite chunk IDs from the provided context.\n"
            "4. HONESTY: If the context does not contain enough information, say: "
            "\"मला याबद्दल पुरेशी माहिती नाही. कृपया doctor ला भेटा.\"\n"
            "5. REASONING: Begin your response with a <thought>...</thought> block "
            "explaining your reasoning before giving the final answer.\n"
        )

    def _build_user_prompt(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> str:
        """Build the user prompt with retrieved context.

        Args:
            query: The user query.
            chunks: Retrieved and reranked chunks.

        Returns:
            Formatted user prompt with context.
        """
        context_block = ""
        for chunk in chunks:
            cid = chunk.get("chunk_id", "?")
            source = chunk.get("source", "unknown")
            doc_id = chunk.get("document_id", "?")
            text = chunk.get("text", "")
            umls = ", ".join(chunk.get("umls_cuis", []))
            context_block += (
                f"[{cid}] Source: {source} | Doc: {doc_id}"
                f"{' | UMLS: ' + umls if umls else ''}\n"
                f"{text}\n\n"
            )

        return (
            f"RETRIEVED MEDICAL CONTEXT:\n{context_block}\n"
            f"USER QUESTION:\n{query}\n\n"
            f"ANSWER:"
        )

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        language_mode: str = "manglish",
    ) -> Dict[str, str]:
        """Generate a cited medical answer.

        Args:
            query: User query.
            chunks: Retrieved context chunks.
            language_mode: Output language mode.

        Returns:
            Dict with 'answer', 'thought_process', and 'raw_response'.
        """
        provider = self._orch["external_llm_provider"]

        if provider == "groq":
            return self._generate_via_groq(query, chunks, language_mode)
        elif provider == "openai":
            return self._generate_via_openai(query, chunks, language_mode)
        elif provider == "local":
            return self._generate_local(query, chunks, language_mode)
        else:
            return self._generate_via_groq(query, chunks, language_mode)

    def _generate_via_groq(
        self, query: str, chunks: List[Dict[str, Any]], language_mode: str
    ) -> Dict[str, str]:
        """Generate using Groq API.

        Args:
            query: User query.
            chunks: Context chunks.
            language_mode: Output language.

        Returns:
            Dict with answer components.
        """
        try:
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                return {"answer": "Error: GROQ_API_KEY not set.", "thought_process": "", "raw_response": ""}

            client = Groq(api_key=api_key)
            sys_prompt = self._build_system_prompt(language_mode)
            user_prompt = self._build_user_prompt(query, chunks)

            response = client.chat.completions.create(
                model=self._orch["external_llm_model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._orch["generation_temperature"],
                max_tokens=self._orch["generation_max_tokens"],
            )

            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw)

        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            return {"answer": f"Generation error: {exc}", "thought_process": "", "raw_response": ""}

    def _generate_via_openai(
        self, query: str, chunks: List[Dict[str, Any]], language_mode: str
    ) -> Dict[str, str]:
        """Generate using OpenAI API.

        Args:
            query: User query.
            chunks: Context chunks.
            language_mode: Output language.

        Returns:
            Dict with answer components.
        """
        try:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return {"answer": "Error: OPENAI_API_KEY not set.", "thought_process": "", "raw_response": ""}

            client = OpenAI(api_key=api_key)
            sys_prompt = self._build_system_prompt(language_mode)
            user_prompt = self._build_user_prompt(query, chunks)

            response = client.chat.completions.create(
                model=self._orch.get("external_llm_model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._orch["generation_temperature"],
                max_tokens=self._orch["generation_max_tokens"],
            )

            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw)

        except Exception as exc:
            logger.error("OpenAI generation failed: %s", exc)
            return {"answer": f"Generation error: {exc}", "thought_process": "", "raw_response": ""}

    def _generate_local(
        self, query: str, chunks: List[Dict[str, Any]], language_mode: str
    ) -> Dict[str, str]:
        """Generate using a local LLM (loaded onto GPU, then unloaded).

        Args:
            query: User query.
            chunks: Context chunks.
            language_mode: Output language.

        Returns:
            Dict with answer components.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self._orch["local_llm_model"]
        logger.info("Loading local LLM: %s", model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
        )

        sys_prompt = self._build_system_prompt(language_mode)
        user_prompt = self._build_user_prompt(query, chunks)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self._orch["generation_max_tokens"],
                temperature=self._orch["generation_temperature"],
                do_sample=True,
                top_p=0.9,
            )

        raw = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Unload LLM
        model.cpu()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Local LLM unloaded.")

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> Dict[str, str]:
        """Parse the raw LLM response to extract thought and answer.

        Args:
            raw: Raw LLM output text.

        Returns:
            Dict with 'answer', 'thought_process', 'raw_response'.
        """
        thought = ""
        answer = raw

        # Extract <thought> block
        thought_match = re.search(r"<thought>(.*?)</thought>", raw, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            answer = raw[thought_match.end():].strip()

        return {
            "answer": answer,
            "thought_process": thought,
            "raw_response": raw,
        }


# ---------------------------------------------------------------------------
# Main RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Complete RAG pipeline with VRAM state machine orchestration.

    Coordinates query processing, retrieval, generation, and safety
    with strict sequential model loading/unloading.
    """

    def __init__(self) -> None:
        """Initialize all pipeline components."""
        from src.retriever import HybridRetriever
        from src.safety import SafetyFilter

        self._retriever = HybridRetriever()
        self._generator = LLMGenerator()
        self._safety = SafetyFilter()
        self._logger = SQLiteLogger()
        self._state = PipelineState.IDLE

        logger.info("RAGPipeline initialized.")

    def run(
        self,
        query: str,
        language_mode: str = "manglish",
    ) -> PipelineResponse:
        """Execute the full RAG pipeline.

        Args:
            query: Raw user query (Manglish).
            language_mode: Output language ('manglish', 'english', 'marathi').

        Returns:
            Complete PipelineResponse with answer, citations, timings.
        """
        response = PipelineResponse(language_mode=language_mode)
        total_start = time.time()

        try:
            # State 1 & 2: Retrieval (embedder + reranker loaded/unloaded internally)
            self._state = PipelineState.EMBEDDING_SEARCH
            retrieval_result = self._retriever.retrieve(query)
            response.query_info = retrieval_result["query_info"]
            response.retrieved_chunks = retrieval_result["retrieved_context"]
            response.timings.update(retrieval_result["timings"])

            # State 3: Safety check
            self._state = PipelineState.SAFETY_CHECK
            t0 = time.time()
            risk_level = self._safety.classify_risk(query)
            response.risk_level = risk_level
            response.timings["safety_check_ms"] = (time.time() - t0) * 1000

            # State 4: Generation (LLM loaded/unloaded or API called)
            self._state = PipelineState.GENERATION
            t0 = time.time()
            gen_result = self._generator.generate(
                query, response.retrieved_chunks, language_mode
            )
            response.answer = gen_result["answer"]
            response.thought_process = gen_result["thought_process"]
            response.timings["generation_ms"] = (time.time() - t0) * 1000

            # Apply safety guardrails
            if risk_level == "HIGH":
                config = get_config()
                response.disclaimer = config["safety"]["high_risk_disclaimer"]
                response.answer = f"{response.disclaimer}\n\n{response.answer}"

            # Extract citations
            response.citations = self._extract_citations(response.answer, response.retrieved_chunks)

            self._state = PipelineState.COMPLETE

        except Exception as exc:
            logger.error("Pipeline error: %s", exc)
            response.answer = f"Pipeline error: {exc}"
            self._state = PipelineState.IDLE

        response.timings["total_ms"] = (time.time() - total_start) * 1000

        # Log to SQLite
        try:
            self._logger.log(query, response)
        except Exception as exc:
            logger.warning("SQLite logging failed: %s", exc)

        self._state = PipelineState.IDLE
        return response

    def _extract_citations(
        self, answer: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract cited chunk references from the answer text.

        Args:
            answer: Generated answer with [chunk_id] citations.
            chunks: Retrieved chunk metadata.

        Returns:
            List of cited chunk metadata dicts.
        """
        cited_ids = set(re.findall(r"\[(\d+)\]", answer))
        cited = []
        for chunk in chunks:
            cid = str(chunk.get("chunk_id", ""))
            if cid in cited_ids:
                cited.append(chunk)
        return cited
