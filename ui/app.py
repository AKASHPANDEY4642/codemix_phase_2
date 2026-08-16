"""
Phase 5: Gradio Frontend Application.

Provides a full-featured medical QA chatbot interface with:
    - Chat history
    - "Thought Process" accordion
    - "Cited Sources" panel
    - Language toggle (Marathi / English / Code-Mixed)
    - Risk level indicator

Usage (on Ubuntu):
    python -m ui.app
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr

from src import get_config, setup_logging

logger = setup_logging("ui")

# ---------------------------------------------------------------------------
# Pipeline (lazy-loaded)
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    """Lazy-load the RAG pipeline.

    Returns:
        Initialized RAGPipeline instance.
    """
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline
        logger.info("Loading RAG pipeline...")
        _pipeline = RAGPipeline()
        logger.info("Pipeline ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Chat Handler
# ---------------------------------------------------------------------------

def process_query(
    user_message: str,
    history: List[Dict[str, str]],
    language_mode: str,
) -> Tuple[List[Dict[str, str]], str, str, str]:
    """Process a user query and return chat response with metadata.

    Args:
        user_message: Raw user input.
        history: Chat history as list of {role, content} dicts.
        language_mode: Selected language ('Manglish', 'English', 'मराठी').

    Returns:
        Tuple of (updated_history, thought_process, citations_text, risk_badge).
    """
    if not user_message.strip():
        return history, "", "", ""

    # Map UI language labels to internal codes
    lang_map = {
        "Manglish (मराठी + English)": "manglish",
        "Pure English": "english",
        "शुद्ध मराठी": "marathi",
    }
    lang_code = lang_map.get(language_mode, "manglish")

    try:
        pipeline = _get_pipeline()
        result = pipeline.run(user_message, lang_code)

        # Format the answer
        answer = result.answer

        # Risk badge
        risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        risk_badge = f"{risk_colors.get(result.risk_level, '⚪')} **Risk: {result.risk_level}**"

        # Thought process
        thought = result.thought_process if result.thought_process else "No reasoning trace available."

        # Citations
        citations_parts: List[str] = []
        for i, chunk in enumerate(result.retrieved_chunks, 1):
            cid = chunk.get("chunk_id", "?")
            source = chunk.get("source", "unknown")
            doc_id = chunk.get("document_id", "?")
            text_preview = chunk.get("text", "")[:200].replace("\n", " ")
            umls = ", ".join(chunk.get("umls_cuis", []))
            score = chunk.get("rerank_score", 0)

            citations_parts.append(
                f"**[{cid}]** {source} (Doc: {doc_id})"
                f"{' | UMLS: ' + umls if umls else ''}"
                f" | Score: {score:.4f}\n"
                f"> {text_preview}..."
            )

        citations_text = "\n\n".join(citations_parts) if citations_parts else "No citations available."

        # Timing info
        total_ms = result.timings.get("total_ms", 0)
        timing_note = f"\n\n---\n⏱️ Total: {total_ms:.0f}ms"

        # Update history
        history = history or []
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer + timing_note})

        return history, thought, citations_text, risk_badge

    except Exception as exc:
        logger.error("UI processing error: %s", exc)
        error_msg = f"⚠️ Error processing query: {exc}"
        history = history or []
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": error_msg})
        return history, "", "", "🔴 **Error**"


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    """Create the Gradio application.

    Returns:
        Configured Gradio Blocks application.
    """
    with gr.Blocks(
        title="MedManglish-RAG",
        theme=gr.themes.Soft(),
    ) as app:
        # Header
        gr.Markdown(
            "# 🩺 MedManglish-RAG\n"
            "**A Script-Aware RAG Framework for Marathi-English Code-Mixed Medical QA**\n\n"
            "⚠️ *This system is for research purposes only. "
            "Not a substitute for professional medical advice.*"
        )

        with gr.Row():
            # Main chat column
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Medical AI Assistant",
                    height=500,
                    type="messages",
                    show_copy_button=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a medical question (e.g., 'mala stomach pain hotoy ani fever pan ahe')...",
                        show_label=False,
                        scale=6,
                        container=False,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    language_toggle = gr.Radio(
                        choices=[
                            "Manglish (मराठी + English)",
                            "Pure English",
                            "शुद्ध मराठी",
                        ],
                        value="Manglish (मराठी + English)",
                        label="🌐 Response Language",
                        interactive=True,
                    )
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                # Example queries
                gr.Examples(
                    examples=[
                        "mala stomach pain hotoy ani fever pan ahe",
                        "metformin che side effects kay ahet?",
                        "BP high ahe tar kay karave?",
                        "vitamins kashasathi ghyave?",
                        "mla chest madhye khup pain hotoy",
                        "diabetes sathi konti medicine changali ahe?",
                    ],
                    inputs=msg_input,
                    label="📝 Try these examples",
                )

            # Sidebar
            with gr.Column(scale=2):
                risk_display = gr.Markdown(
                    "⚪ **Risk: N/A**",
                    label="Risk Level",
                )

                with gr.Accordion("🧠 Thought Process", open=False):
                    thought_display = gr.Markdown(
                        "Submit a query to see the AI's reasoning.",
                    )

                with gr.Accordion("📚 Cited Sources", open=True):
                    citations_display = gr.Markdown(
                        "Submit a query to see cited sources.",
                    )

        # Event handlers
        def _on_submit(msg, history, lang):
            return process_query(msg, history, lang) + ("",)

        submit_btn.click(
            fn=_on_submit,
            inputs=[msg_input, chatbot, language_toggle],
            outputs=[chatbot, thought_display, citations_display, risk_display, msg_input],
        )

        msg_input.submit(
            fn=_on_submit,
            inputs=[msg_input, chatbot, language_toggle],
            outputs=[chatbot, thought_display, citations_display, risk_display, msg_input],
        )

        clear_btn.click(
            fn=lambda: ([], "Submit a query to see reasoning.", "Submit a query to see cited sources.", "⚪ **Risk: N/A**"),
            outputs=[chatbot, thought_display, citations_display, risk_display],
        )

    return app


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
