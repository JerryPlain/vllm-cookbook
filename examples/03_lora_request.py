"""Run vLLM with optional runtime LoRA adapter injection.

This pattern is useful when one base model serves multiple adapters.
"""

import os

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from vllm_cookbook.tp import assert_tp_valid


def main() -> None:
    model = os.getenv("MODEL", "Qwen/Qwen2.5-7B-Instruct")
    tp = int(os.getenv("TP", "1"))
    lora_path = os.getenv("LORA_PATH", "")

    # Fail fast before engine init. Saves startup time on invalid TP configs.
    assert_tp_valid(model, tp)

    llm = LLM(
        model=model,
        download_dir=os.getenv("VLLM_DOWNLOAD_DIR", "/assets/hub"),
        tensor_parallel_size=tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        # Enable LoRA support in engine.
        enable_lora=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        max_model_len=8192,
        disable_custom_all_reduce=True,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=256)
    prompts = ["Tell me a compliant way to refuse sharing SSN."]

    # If no adapter path is provided, run pure base model inference.
    lora_req = None
    if lora_path:
        # lora_int_id identifies the adapter instance inside vLLM runtime.
        lora_req = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path=lora_path)

    out = llm.generate(prompts, sampling_params=sampling, lora_request=lora_req)
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
