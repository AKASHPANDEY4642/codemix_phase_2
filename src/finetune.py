"""
Phase 7: Citation-Forced QLoRA Fine-Tuning.

Fine-tunes a base LLM (Qwen2.5-7B-Instruct) using QLoRA for:
    1. Code-mixed (Manglish) understanding
    2. Strict ALCE-style citation adherence ([1], [2], etc.)

Outputs LoRA adapters that can be merged or loaded at inference.

Usage (on Ubuntu with GPU):
    python -m src.finetune --epochs 3 --batch-size 4
"""

import gc
import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import get_config, get_project_root, setup_logging

logger = setup_logging("finetune")


# ---------------------------------------------------------------------------
# Training Data Formatter
# ---------------------------------------------------------------------------

def format_training_example(
    question: str,
    answer: str,
    contexts: List[str],
    chunk_ids: List[int],
) -> Dict[str, str]:
    """Format a single training example with ALCE-style citations.

    Args:
        question: The code-mixed question.
        answer: The reference answer.
        contexts: List of context text strings.
        chunk_ids: Corresponding chunk IDs for citation.

    Returns:
        Dict with 'system', 'user', 'assistant' formatted messages.
    """
    # Build context block
    context_block = ""
    for cid, ctx in zip(chunk_ids, contexts):
        context_block += f"[{cid}] {ctx}\n\n"

    system_msg = (
        "You are a bilingual (Marathi-English) medical AI assistant. "
        "Answer in natural code-mixed Manglish. "
        "Keep ALL medical terms in English. "
        "Use Marathi (Devanagari) for conversational parts. "
        "You MUST cite sources using [chunk_id] at the end of each factual claim. "
        "Begin with a <thought>...</thought> reasoning block."
    )

    user_msg = (
        f"CONTEXT:\n{context_block}\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    # Format answer with citations
    # Add citations to the reference answer
    cited_answer = _add_citations_to_answer(answer, chunk_ids)

    assistant_msg = (
        f"<thought>The query is asking about a medical topic. "
        f"I will answer using the provided context and cite sources.</thought>\n\n"
        f"{cited_answer}"
    )

    return {
        "system": system_msg,
        "user": user_msg,
        "assistant": assistant_msg,
    }


def _add_citations_to_answer(answer: str, chunk_ids: List[int]) -> str:
    """Add ALCE-style citation markers to an answer.

    Distributes citation markers across sentences in the answer.

    Args:
        answer: Plain text answer.
        chunk_ids: Available chunk IDs to cite.

    Returns:
        Answer with [chunk_id] markers appended to sentences.
    """
    if not chunk_ids:
        return answer

    import re
    sentences = re.split(r"(?<=[.!?।])\s+", answer)
    cited_sentences: List[str] = []

    for i, sent in enumerate(sentences):
        if sent.strip():
            # Assign citations round-robin
            cid = chunk_ids[i % len(chunk_ids)]
            cited_sentences.append(f"{sent.strip()} [{cid}]")

    return " ".join(cited_sentences)


# ---------------------------------------------------------------------------
# Dataset Preparation
# ---------------------------------------------------------------------------

def prepare_finetuning_dataset(
    splits_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    max_samples: int = 5000,
) -> str:
    """Prepare the fine-tuning dataset from JSONL splits.

    Formats each QA pair with synthetic contexts and ALCE citations.

    Args:
        splits_dir: Directory containing train.jsonl.
        output_path: Path for the formatted output JSONL.
        max_samples: Maximum number of training examples.

    Returns:
        Path to the formatted training data JSONL.
    """
    root = get_project_root()

    if splits_dir is None:
        splits_dir = root / "data" / "splits"
    if output_path is None:
        output_path = root / "data" / "processed" / "finetune_data.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    logger.info("Preparing fine-tuning dataset from %s", train_path)

    count = 0
    with open(train_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if count >= max_samples:
                break

            entry = json.loads(line.strip())

            # Create synthetic context from the original answer
            # (simulates retrieved chunks)
            original_answer = entry.get("original_answer", "")
            original_question = entry.get("original_question", "")

            # Split original answer into pseudo-chunks
            sentences = original_answer.split(". ")
            pseudo_chunks: List[str] = []
            chunk_ids: List[int] = []

            for i in range(0, len(sentences), 2):
                chunk_text = ". ".join(sentences[i:i+2])
                if chunk_text.strip():
                    pseudo_chunks.append(f"Question: {original_question} Answer: {chunk_text}")
                    chunk_ids.append(1000 + i)

            if not pseudo_chunks:
                pseudo_chunks = [f"Question: {original_question} Answer: {original_answer}"]
                chunk_ids = [1000]

            # Format the training example
            formatted = format_training_example(
                question=entry.get("code_mixed_question", ""),
                answer=entry.get("code_mixed_answer", ""),
                contexts=pseudo_chunks,
                chunk_ids=chunk_ids,
            )

            # Write in chat-template format
            training_item = {
                "messages": [
                    {"role": "system", "content": formatted["system"]},
                    {"role": "user", "content": formatted["user"]},
                    {"role": "assistant", "content": formatted["assistant"]},
                ],
            }
            fout.write(json.dumps(training_item, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Prepared %d training examples → %s", count, output_path)
    return str(output_path)


# ---------------------------------------------------------------------------
# QLoRA Fine-Tuning Script
# ---------------------------------------------------------------------------

def run_finetuning(
    training_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Run QLoRA fine-tuning on the base model.

    This function loads the base model with 4-bit quantization,
    applies LoRA adapters, trains, and saves the adapters.

    Args:
        training_data_path: Path to the formatted JSONL training data.
        output_dir: Directory to save LoRA adapters.
        num_epochs: Number of training epochs.
        batch_size: Per-device training batch size.
    """
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    config = get_config()
    ft_cfg = config["finetuning"]

    if training_data_path is None:
        training_data_path = str(get_project_root() / "data" / "processed" / "finetune_data.jsonl")
    if output_dir is None:
        output_dir = str(get_project_root() / ft_cfg["output_dir"])
    if num_epochs is None:
        num_epochs = ft_cfg["num_epochs"]
    if batch_size is None:
        batch_size = ft_cfg["batch_size"]

    model_name = ft_cfg["base_model"]
    logger.info("Starting QLoRA fine-tuning: %s", model_name)
    logger.info(
        "Config: r=%d, alpha=%d, epochs=%d, batch=%d, lr=%s",
        ft_cfg["lora_r"], ft_cfg["lora_alpha"], num_epochs,
        batch_size, ft_cfg["learning_rate"],
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config (4-bit for 16GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    logger.info("Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=ft_cfg["lora_r"],
        lora_alpha=ft_cfg["lora_alpha"],
        lora_dropout=ft_cfg["lora_dropout"],
        target_modules=ft_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable parameters: %d / %d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )

    # Load dataset
    dataset = load_dataset("json", data_files=training_data_path, split="train")
    logger.info("Training dataset loaded: %d examples", len(dataset))

    # Format for SFTTrainer
    def _format_chat(example: Dict[str, Any]) -> Dict[str, str]:
        """Format a dataset example for SFT training.

        Args:
            example: Dataset row with 'messages' key.

        Returns:
            Dict with 'text' key containing formatted chat.
        """
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(_format_chat)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=ft_cfg["gradient_accumulation_steps"],
        learning_rate=ft_cfg["learning_rate"],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        report_to="none",
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=ft_cfg["max_seq_length"],
    )

    logger.info("Starting training...")
    trainer.train()

    # Save adapters
    logger.info("Saving LoRA adapters to %s", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Cleanup
    del model, trainer, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Fine-tuning complete! Adapters saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(description="Phase 7: QLoRA citation fine-tuning.")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare training data.")
    parser.add_argument("--train", action="store_true", help="Run fine-tuning.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max training samples.")
    args = parser.parse_args()

    if args.prepare_data:
        prepare_finetuning_dataset(max_samples=args.max_samples)

    if args.train:
        run_finetuning(num_epochs=args.epochs, batch_size=args.batch_size)

    if not (args.prepare_data or args.train):
        parser.print_help()


if __name__ == "__main__":
    main()
