"""
train_sft.py

Supervised Fine-Tuning (SFT) for a causal LM using TRL's SFTTrainer + PEFT LoRA.

Key points:
- Dataset format expected: [{"instruction": "...", "output": "..."}]
- We build a prompt/target pair and do teacher-forcing:
    input_ids = prompt_ids + target_ids
    labels    = [-100 for prompt] + target_ids
- LoRA adapters are saved every epoch into:
    ./sft_results/<dataset_name>/<model_suffix>/checkpoint-epoch-<E>

This adapter path is later consumed by vLLM inference via LoRARequest.
"""

import os
import json
import argparse
import logging
from functools import partial

import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(prog="SFT training with LoRA adapters")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=30)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    return parser.parse_args()


# =========================================================
# Data formatting / tokenization
# =========================================================
def format_prompt(example):
    """
    Convert raw sample to a standardized (prompt, target) format.

    Raw expected keys:
      - instruction
      - output
    """
    return {"prompt": example["instruction"], "target": example["output"]}


def tokenize_with_labels(example, tokenizer):
    """
    Tokenize prompt+target for causal LM SFT with label masking.

    - prompt tokens do NOT contribute to loss => label = -100
    - target tokens contribute to loss => label = target_ids
    """
    prompt_tokens = tokenizer(example["prompt"].rstrip("\n\t"), add_special_tokens=False)
    target_tokens = tokenizer(example["target"].rstrip("\n\t"), add_special_tokens=False)

    input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]
    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(prompt_tokens["input_ids"]) + target_tokens["input_ids"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# =========================================================
# Model / Tokenizer loading
# =========================================================
def get_model_and_tokenizer(model_name: str):
    """
    Load base model + tokenizer from HF (or local path).
    Uses device_map="auto" for convenience (multi-GPU mapping).

    Note: this is training-time HF model, NOT vLLM.
    """
    hf_token = os.getenv("HF_API_KEY")
    cache_dir = "/assets/hub"

    logger.info(f"Loading base model: {model_name}")

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        token=hf_token,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Ensure padding is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Pad token set to EOS token: {tokenizer.eos_token}")

    tokenizer.padding_side = "left"
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


# =========================================================
# Callback: save LoRA adapter by epoch
# =========================================================
class PeftSavingCallback(TrainerCallback):
    """
    TRL/HF Trainer callback that saves PEFT adapter weights after each checkpoint save.

    We force an epoch-based folder name:
      checkpoint-epoch-<E>
    and store adapter weights into that folder.

    This produces adapter checkpoints that can be directly loaded by vLLM LoRARequest.
    """

    def on_save(self, args, state, control, **kwargs):
        epoch_folder = f"checkpoint-epoch-{int(state.epoch)}"
        checkpoint_path = os.path.join(args.output_dir, epoch_folder)

        model = kwargs.get("model", None)
        if isinstance(model, PeftModel):
            logger.info(f"Saving PEFT adapter at epoch {int(state.epoch)} to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)

        return control


# =========================================================
# Training entry
# =========================================================
def train_sft(args):
    """
    End-to-end SFT training:
      1) Load base model/tokenizer
      2) Load JSON dataset and split train/eval
      3) Tokenize with labels
      4) Apply LoRA
      5) Run TRL SFTTrainer
      6) Save adapters per epoch
    """
    model_suffix = args.model_name.split("/")[-1]
    dataset_name = os.path.basename(args.data_path).replace(".json", "")

    split_ratio = 0.99  # 99% train, 1% eval (small monitor set)
    split_tag = f"{int(split_ratio * 100)}"

    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # Save split files for reproducibility
    dataset_dir = f"./sft_training_data/{dataset_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    train_split_path = os.path.join(dataset_dir, f"train_split_{split_tag}.json")
    eval_split_path = os.path.join(dataset_dir, f"eval_split_{split_tag}.json")

    # -------------------------------------------------
    # Load or create splits
    # -------------------------------------------------
    if os.path.exists(train_split_path) and os.path.exists(eval_split_path):
        logger.info("Loading existing dataset splits...")
        with open(train_split_path, "r", encoding="utf-8") as f:
            train_dataset = Dataset.from_list(json.load(f))
        with open(eval_split_path, "r", encoding="utf-8") as f:
            eval_dataset = Dataset.from_list(json.load(f))
    else:
        logger.info("Creating new dataset splits...")
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        logger.info(f"Raw dataset size: {len(raw)}")

        dataset = Dataset.from_list(raw).map(format_prompt)

        split_dataset = dataset.train_test_split(train_size=split_ratio, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Eval size: {len(eval_dataset)}")

        with open(train_split_path, "w", encoding="utf-8") as f:
            json.dump(train_dataset.to_list(), f, indent=2, ensure_ascii=False)
        with open(eval_split_path, "w", encoding="utf-8") as f:
            json.dump(eval_dataset.to_list(), f, indent=2, ensure_ascii=False)

        logger.info("Splits saved successfully.")

    # -------------------------------------------------
    # Tokenization with labels
    # -------------------------------------------------
    tokenize_fn = partial(tokenize_with_labels, tokenizer=tokenizer)

    train_dataset = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_fn, remove_columns=eval_dataset.column_names)

    logger.info("Tokenization complete.")

    # -------------------------------------------------
    # Apply LoRA
    # -------------------------------------------------
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
    )

    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied.")
    model.print_trainable_parameters()

    # -------------------------------------------------
    # Training arguments
    # -------------------------------------------------
    output_dir = f"./sft_results/{dataset_name}/{model_suffix}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",  # will trigger callback each epoch
        fp16=False,
        report_to="none",
    )

    logger.info(f"Starting training. Output dir: {output_dir}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[PeftSavingCallback()],
    )

    trainer.train()
    logger.info("Training finished successfully.")


if __name__ == "__main__":
    args = get_args()
    train_sft(args)