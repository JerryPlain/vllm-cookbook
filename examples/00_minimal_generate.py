"""Minimal vLLM text generation example.

This script is intentionally explicit and heavily commented so you can use it
as a reference template when starting new experiments.
"""

import os

from vllm import LLM, SamplingParams


def main() -> None:
    # Read model and parallel settings from environment variables.
    # Keeping these external makes the script reproducible across machines.
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tp = int(os.getenv("TP", "1"))

    # Create the vLLM engine.
    # Key knobs here are chosen for stability-first local experimentation.
    llm = LLM(
        model=model,
        # Shared model cache directory. Useful on multi-user servers.
        download_dir=os.getenv("VLLM_DOWNLOAD_DIR", "/assets/hub"),
        # Tensor parallel degree. Must satisfy attention-head divisibility.
        tensor_parallel_size=tp,
        # Keep pipeline parallel disabled for a simple single-stage setup.
        pipeline_parallel_size=1,
        # Trade-off: higher utilization can improve throughput but raise OOM risk.
        gpu_memory_utilization=0.9,
        # Many open models require remote modeling code from HF repositories.
        trust_remote_code=True,
        # Improves long-context prefill behavior on many workloads.
        enable_chunked_prefill=True,
        # Eager mode is often easier to debug than fully graph-captured execution.
        enforce_eager=True,
        # Cap logical context length to control memory growth.
        max_model_len=8192,
        # Some environments are more stable with custom all-reduce disabled.
        disable_custom_all_reduce=True,
        # Fix seed for more reproducible outputs in low-temperature runs.
        seed=42,
    )

    # Deterministic-ish generation settings for tutorial outputs.
    sampling = SamplingParams(temperature=0.0, max_tokens=256)
    prompts = ["Explain what tensor parallelism is in one paragraph."]

    # Run generation. vLLM returns structured outputs for each prompt.
    outputs = llm.generate(prompts, sampling_params=sampling)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
