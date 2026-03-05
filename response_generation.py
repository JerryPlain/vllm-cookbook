"""
response_generation.py

Generate model responses on privacy benchmarks using vLLM.

Features:
- Supports vLLM Tensor Parallelism (TP) via tensor_parallel_size = --num_gpus
- Supports LoRA adapter inference via enable_lora + LoRARequest
- Stable run_id derived from adapter path to avoid overwriting experiments
- Skip-if-eval-exists logic (if downstream eval already computed)

Output:
  model_responses/<dataset_name>/<run_id>/response_by_<model_alias>.json

Then it can optionally call response_evaluation.evaluate(...) to run the judge.
"""

import os
import json
import gc
import argparse
import logging
from functools import partial

import torch
from datasets import Dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

from response_evaluation import evaluate


# =========================================================
# Utility: stable run_id from adapter path
# =========================================================
def make_run_id(adapter_path: str) -> str:
    """
    Build a stable run_id from the adapter checkpoint path.

    Example:
      ./results/alignment_data_HQ_Feb_2026_English_V4/Qwen2.5-7B-Instruct/checkpoint-epoch-5
    => run_id:
      alignment_data_HQ_Feb_2026_English_V4__checkpoint-epoch-5
    """
    if not adapter_path:
        return "base"

    p = adapter_path.rstrip("/")
    checkpoint_name = os.path.basename(p)
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
    return f"{dataset_name}__{checkpoint_name}"


def get_eval_output_path(eval_root: str, dataset_name: str, run_id: str, target_model_name: str) -> str:
    """
    This is the exact eval json path that evaluate() will create.
    Used for skip-if-exists logic.

    Structure:
      eval_results/<dataset_name>/<run_id>/eval_results_target=<target_model>.json
    """
    return os.path.join(eval_root, dataset_name, run_id, f"eval_results_target={target_model_name}.json")


# =========================================================
# Prompt builder
# =========================================================
def build_vllm_prompt(example: dict, tokenizer) -> dict:
    """
    Convert a dataset example into a string prompt.
    If tokenizer has a chat_template, we format it as chat; otherwise raw question.
    """
    q = example.get("question", "")
    try:
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": q}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = q
    except Exception:
        prompt = q
    return {"prompt": prompt}


# =========================================================
# Dataset Registry
# =========================================================
class BaseAssessmentDataset:
    def __init__(self):
        self.raw_data = None
        self.dataset = None

    def load(self):
        raise NotImplementedError


class MultiOpenSourceV1(BaseAssessmentDataset):
    def load(self):
        path = "./eval_datasets/alignment_data_v2_privacy_leakage.json"
        logging.info(f"Loading MultiOpenSourceV1 from {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())
        self.raw_data = raw
        self.dataset = Dataset.from_list(raw)


# Add your other dataset classes here...
DATASET_REGISTRY = {
    "multi_opensource_v1": MultiOpenSourceV1,
    # ...
}


class AssessmentDataset:
    """Thin wrapper to instantiate a dataset class and expose raw_data + dataset."""
    def __init__(self, dataset_name: str):
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset_name: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
        obj = DATASET_REGISTRY[dataset_name]()
        obj.load()
        self.raw_data = obj.raw_data
        self.dataset = obj.dataset


# =========================================================
# CLI args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Run vLLM inference")
    p.add_argument("--dataset_name", type=str, required=True, choices=DATASET_REGISTRY.keys())
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output_dir", type=str, default="./model_responses")
    p.add_argument("--adapter_path", type=str, default="")
    p.add_argument("--num_gpus", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--eval_root", type=str, default="./eval_results")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# =========================================================
# Main generation pipeline
# =========================================================
def response_generation(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    model_alias = args.model.split("/")[-1]

    # -----------------------------------------------------
    # Skip-all shortcut: if eval result already exists, we can skip everything.
    # -----------------------------------------------------
    run_id = make_run_id(args.adapter_path)
    expected_eval_path = get_eval_output_path(args.eval_root, args.dataset_name, run_id, model_alias)
    if os.path.exists(expected_eval_path) and not args.overwrite:
        logging.info(f"[SKIP] Eval result already exists: {expected_eval_path}")
        return None

    logging.info("Initializing vLLM engine...")

    # vLLM Tensor Parallelism:
    # tensor_parallel_size == number of GPUs participating in TP.
    # IMPORTANT: num_attention_heads must be divisible by tensor_parallel_size.
    llm = LLM(
        model=args.model,
        download_dir="/assets/hub",
        tensor_parallel_size=args.num_gpus,
        pipeline_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=8192,
        disable_custom_all_reduce=True,
        seed=42,
        enable_lora=True if args.adapter_path else False,
    )

    logging.info("vLLM initialized.")
    tokenizer = llm.get_tokenizer()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # -----------------------------------------------------
    # Load dataset and build prompts
    # -----------------------------------------------------
    dataset_cls = AssessmentDataset(args.dataset_name)
    dataset = dataset_cls.dataset
    logging.info(f"Total samples: {len(dataset)}")

    _build_prompt = partial(build_vllm_prompt, tokenizer=tokenizer)
    formatted = dataset.map(_build_prompt, num_proc=args.num_proc)

    # Filter out invalid prompts to avoid vLLM crash
    formatted = formatted.filter(lambda x: isinstance(x.get("prompt"), str) and len(x["prompt"].strip()) > 0)
    logging.info(f"After filtering empty prompts: {len(formatted)} samples")

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        n=1,
        temperature=0.0,
    )

    # Optional LoRA injection
    lora_request = None
    if args.adapter_path:
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=args.adapter_path,
        )

    outputs = llm.generate(
        formatted["prompt"],
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    logging.info("Inference completed.")

    # -----------------------------------------------------
    # Save responses
    # -----------------------------------------------------
    dataset_run_dir = os.path.join(args.output_dir, args.dataset_name, run_id)
    os.makedirs(dataset_run_dir, exist_ok=True)

    results = {}
    for i, out in enumerate(outputs):
        sample = dict(dataset_cls.raw_data[i])
        sample["response"] = out.outputs[0].text
        results[i] = sample

    output_path = os.path.join(dataset_run_dir, f"response_by_{model_alias}.json")
    if os.path.exists(output_path) and not args.overwrite:
        logging.info(f"[SKIP] Response file already exists: {output_path}")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved responses to: {output_path}")

    # -----------------------------------------------------
    # Cleanup GPU memory (important for sequential runs)
    # -----------------------------------------------------
    destroy_model_parallel()
    destroy_distributed_environment()

    del llm, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_path


if __name__ == "__main__":
    # Recommended env vars for stable vLLM runtime in your setup
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["OMP_NUM_THREADS"] = "4"
    # NOTE: you typically set CUDA_VISIBLE_DEVICES in the shell command, not hardcode here.

    args = parse_args()

    output_response_path = response_generation(args)
    if output_response_path is None:
        raise SystemExit(0)

    # -----------------------------------------------------
    # Chain: run judge evaluation automatically after generation
    # This is where Qwen3Guard (judge model) gets plugged in.
    # -----------------------------------------------------
    evaluate(
        input_path=output_response_path,
        output_dir=args.eval_root,
        model_path="./Qwen3-4B-Instruct-2507_V1",  # <-- Qwen3Guard model path
        batch_size=1,
        quantization=False,
        max_new_tokens=2560,
        system_prompt_path="system_prompt_response_evaluation_20260109.txt",
        dataset_name=args.dataset_name,
        target_model_name=args.model.split("/")[-1],
    )